#!/usr/bin/env python3
"""Minimal OpenPI -> Franka FR3 live inference loop.

This script intentionally keeps the control path close to offline replay:

1. Read one live RealSense observation and current Franka joint/gripper state.
2. Send DROID-shaped observation to an already-running OpenPI websocket server.
3. Receive an absolute joint action horizon, use the first --chunk-size actions.
4. Stream those absolute joint targets directly with panda-py joint control.
5. Repeat from a fresh observation after the chunk is consumed.

It keeps the policy/execution path simple, but applies a small joint-step guard
before sending absolute targets to Franka so zero-shot outliers do not trigger a
libfranka reflex. Optional sidecars are --max-target-distance for absolute
target guarding and --record-video-subdir for two-camera RealSense video
recording.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _inject_local_paths() -> None:
    script_path = Path(__file__).resolve()
    embodied_root = script_path.parents[2]
    extra_paths = [embodied_root]
    openpi_client_root = os.environ.get("OPENPI_CLIENT_ROOT")
    if openpi_client_root:
        extra_paths.append(Path(openpi_client_root).expanduser())
    for path in extra_paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_inject_local_paths()

from openpi_client import image_tools  # noqa: E402
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402
from rdeploy.robot.controller import ControlType  # noqa: E402
from rdeploy.robot.controller.franka_fr3.sdk_controller import (  # noqa: E402
    FrankaFR3ControllerConfig,
)
from rdeploy.robot.robot.franka_fr3_realsense_single_arm_sdk import (  # noqa: E402
    FrankaFR3RealSenseSdkRobot,
    FrankaFR3RealSenseSdkRobotConfig,
)
from rdeploy.robot.sensor.camera.realsense_sdk import (  # noqa: E402
    MultiRealSenseCameraConfig,
)
from rdeploy.robot.sensor.non_blocking_wrapper import NonBlockingSensorWrapper  # noqa: E402
from rdeploy.utils.logger_utils import logger  # noqa: E402


POLICY_EXTERIOR_KEY = "observation/exterior_image_1_left"
POLICY_WRIST_KEY = "observation/wrist_image_left"
POLICY_JOINT_KEY = "observation/joint_position"
POLICY_GRIPPER_KEY = "observation/gripper_position"

EXTERIOR_ROLE = "exterior"
WRIST_ROLE = "wrist"
DEFAULT_VIDEO_ROOT = Path("outputs/real_deploy")
HOME_Q = np.asarray([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.0], dtype=np.float64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-host", default="127.0.0.1")
    parser.add_argument("--openpi-port", type=int, default=8010)
    parser.add_argument("--prompt", default=None)

    parser.add_argument("--robot-hostname", required=True)
    parser.add_argument("--controller-name", default="franka_fr3_single_arm_controller")
    parser.add_argument("--camera-sensor-name", default="franka_fr3_realsense_camera")

    parser.add_argument("--control-freq", type=float, default=15.0)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--exterior-serial", required=True)
    parser.add_argument("--wrist-serial", required=True)
    parser.add_argument(
        "--exterior-usb-port",
        default="",
        help="Expected RealSense physical USB path token for the exterior camera.",
    )
    parser.add_argument(
        "--wrist-usb-port",
        default="",
        help="Expected RealSense physical USB path token for the wrist camera.",
    )
    parser.add_argument(
        "--skip-camera-port-check",
        action="store_true",
        help="Skip RealSense serial-to-physical-USB-port validation.",
    )
    parser.add_argument(
        "--exterior-rotate", type=int, default=0, choices=[0, 90, 180, 270]
    )
    parser.add_argument(
        "--wrist-rotate", type=int, default=0, choices=[0, 90, 180, 270]
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Number of returned OpenPI actions to execute before requesting a new observation.",
    )
    parser.add_argument(
        "--max-target-distance",
        type=float,
        default=0.25,
        help="Guard if max(abs(raw_target_q-current_q)) exceeds this many rad. Set <=0 to disable.",
    )
    parser.add_argument(
        "--far-target-action",
        choices=["drop", "abort"],
        default="drop",
        help="What to do when raw target exceeds --max-target-distance.",
    )
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.05,
        help="Clip each sent joint target around the current q by this many rad. Set <=0 to disable.",
    )
    parser.add_argument(
        "--slowdown",
        type=float,
        default=1.0,
        help="Multiplier applied to the clipped joint step. Must be in (0, 1].",
    )
    parser.add_argument(
        "--joint-filter-alpha",
        type=float,
        default=1.0,
        help="Low-pass filter alpha for joint targets. 1 disables filtering.",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--warmup-observations", type=int, default=3)

    parser.add_argument(
        "--gripper-type", default="robotiq", choices=["robotiq", "panda_hand", "none"]
    )
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--gripper-max-open", type=float, default=0.08)
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.35,
        help="OpenPI gripper output >= threshold means closed. DROID convention: 0=open, 1=closed.",
    )

    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument(
        "--move-home-before-start",
        action="store_true",
        help="Move to HOME_Q with raw.reset() before inference. Disabled by default.",
    )
    parser.add_argument("--log-actions", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--record-video-subdir",
        default=None,
        help=(
            "Enable two-camera inference video recording under "
            "--record-video-root/SUBDIR."
        ),
    )
    parser.add_argument(
        "--record-video-root",
        default=str(DEFAULT_VIDEO_ROOT),
        help="Root directory for inference videos.",
    )
    parser.add_argument(
        "--record-video-prefix",
        default=None,
        help="Video filename prefix. Defaults to run_YYYYmmdd_HHMMSS.",
    )
    parser.add_argument(
        "--record-video-fps",
        type=float,
        default=0.0,
        help="Output video FPS. Set <=0 to use --camera-fps.",
    )
    parser.add_argument(
        "--record-video-codec",
        default="mp4v",
        help="OpenCV fourcc codec for mp4 recording.",
    )
    return parser.parse_args()


def _realsense_physical_port_matches(actual: str, expected: str) -> bool:
    expected = expected.strip()
    if not expected:
        return True
    tokens = {expected}
    if expected.startswith("usb-0:"):
        tokens.add(f"1-{expected.removeprefix('usb-0:')}")
    elif expected.startswith("1-"):
        tokens.add(f"usb-0:{expected.removeprefix('1-')}")
    return any(token in actual for token in tokens)


def _validate_realsense_physical_ports(args: argparse.Namespace) -> None:
    if args.skip_camera_port_check:
        logger.info("Skipping RealSense physical USB port validation")
        return

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is required for RealSense port validation"
        ) from exc

    def _device_info(device: Any, info: Any) -> str:
        try:
            if hasattr(device, "supports") and not device.supports(info):
                return ""
            return str(device.get_info(info))
        except RuntimeError:
            return ""

    ctx = rs.context()
    devices: dict[str, dict[str, str]] = {}
    for device in ctx.devices:
        serial = _device_info(device, rs.camera_info.serial_number)
        if not serial:
            continue
        devices[serial] = {
            "name": _device_info(device, rs.camera_info.name),
            "physical_port": _device_info(device, rs.camera_info.physical_port),
            "usb_type": _device_info(device, rs.camera_info.usb_type_descriptor),
        }

    expected = {
        EXTERIOR_ROLE: (args.exterior_serial, args.exterior_usb_port),
        WRIST_ROLE: (args.wrist_serial, args.wrist_usb_port),
    }
    errors = []
    for role, (serial, expected_port) in expected.items():
        info = devices.get(serial)
        if info is None:
            errors.append(
                f"{role}: serial {serial} not found; available={sorted(devices)}"
            )
            continue

        actual_port = info["physical_port"]
        usb_type = info["usb_type"] or "unknown"
        logger.info(
            "RealSense role={} serial={} physical_port={} usb_type={}",
            role,
            serial,
            actual_port,
            usb_type,
        )
        if expected_port and not _realsense_physical_port_matches(
            actual_port, expected_port
        ):
            errors.append(
                f"{role}: serial {serial} expected USB port {expected_port}, got {actual_port}"
            )
        if usb_type and usb_type != "unknown" and not usb_type.startswith("3"):
            logger.warning(
                "RealSense role={} serial={} is on USB {}; USB 3.x is recommended",
                role,
                serial,
                usb_type,
            )

    if errors:
        raise RuntimeError(
            "RealSense physical USB port mismatch:\n" + "\n".join(errors)
        )


def _as_mapping(name: str, value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        mapped = value.to_dict()
        if isinstance(mapped, Mapping):
            return dict(mapped)
    if hasattr(value, "items") and callable(value.items):
        return dict(value.items())
    raise TypeError(f"{name} must be mapping-like, got {type(value).__name__}")


def _rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return image
    return np.rot90(image, k=degrees // 90)


def _sanitize_filename_component(value: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip()
    )
    return cleaned.strip("._-") or "run"


def _resolve_record_video_dir(args: argparse.Namespace) -> Path | None:
    if not args.record_video_subdir:
        return None

    subdir = Path(str(args.record_video_subdir).strip())
    if subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError("--record-video-subdir must be a relative subdirectory")
    if not subdir.parts:
        raise ValueError("--record-video-subdir must not be empty")
    return Path(args.record_video_root).expanduser() / subdir


def _unique_video_prefix(output_dir: Path, requested_prefix: str | None) -> str:
    base = _sanitize_filename_component(
        requested_prefix or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    for idx in range(1000):
        prefix = base if idx == 0 else f"{base}_{idx:03d}"
        if not any(
            (output_dir / f"{prefix}_{role}.mp4").exists()
            for role in (EXTERIOR_ROLE, WRIST_ROLE)
        ):
            return prefix
    raise RuntimeError(f"Could not find a free video prefix in {output_dir}")


def _as_rgb_uint8_frame(image: np.ndarray) -> np.ndarray:
    frame = image_tools.convert_to_uint8(np.asarray(image))
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"Expected image shape HxWx3/4, got {frame.shape}")
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    return np.ascontiguousarray(frame)


def _has_required_color_frames(camera_data: Mapping[str, Any]) -> bool:
    color = camera_data.get("color")
    if not isinstance(color, Mapping):
        return False
    return EXTERIOR_ROLE in color and WRIST_ROLE in color


def _wait_for_nonblocking_camera(
    camera: NonBlockingSensorWrapper,
    *,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        data = camera.read()
        if isinstance(data, Mapping) and _has_required_color_frames(data):
            return
        time.sleep(0.05)
    raise RuntimeError(
        f"Timed out waiting for non-blocking RealSense frames "
        f"({EXTERIOR_ROLE}, {WRIST_ROLE})"
    )


class _RealSenseVideoRecorder:
    """Records latest non-blocking RealSense RGB frames to one mp4 per role."""

    def __init__(
        self,
        *,
        output_dir: Path,
        prefix: str,
        fps: float,
        codec: str,
        exterior_rotate: int,
        wrist_rotate: int,
    ):
        if fps <= 0:
            raise ValueError("Video FPS must be positive")
        if len(codec) != 4:
            raise ValueError("--record-video-codec must be a 4-character fourcc")

        self.output_dir = output_dir
        self.prefix = prefix
        self.fps = float(fps)
        self.codec = codec
        self.paths = {
            EXTERIOR_ROLE: output_dir / f"{prefix}_{EXTERIOR_ROLE}.mp4",
            WRIST_ROLE: output_dir / f"{prefix}_{WRIST_ROLE}.mp4",
        }
        self._rotations = {
            EXTERIOR_ROLE: exterior_rotate,
            WRIST_ROLE: wrist_rotate,
        }
        self._writers: dict[str, Any] = {}
        self._frame_counts = {EXTERIOR_ROLE: 0, WRIST_ROLE: 0}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._exception: BaseException | None = None
        self._cv2: Any | None = None

    def start(self, camera: NonBlockingSensorWrapper) -> None:
        if self._thread is not None:
            raise RuntimeError("Video recorder is already started")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV (cv2) is required for --record-video-subdir"
            ) from exc

        self._cv2 = cv2
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            args=(camera,),
            name="minimal_franka_realsense_video_recorder",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Recording RealSense videos to {} with prefix={} fps={} codec={}",
            self.output_dir,
            self.prefix,
            self.fps,
            self.codec,
        )

    def check(self) -> None:
        if self._exception is not None:
            raise RuntimeError("RealSense video recorder failed.") from self._exception

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
        if self.paths:
            logger.info(
                "Stopped RealSense video recording: exterior_frames={} wrist_frames={} files={}",
                self._frame_counts[EXTERIOR_ROLE],
                self._frame_counts[WRIST_ROLE],
                {role: str(path) for role, path in self.paths.items()},
            )

    def _run(self, camera: NonBlockingSensorWrapper) -> None:
        try:
            period = 1.0 / self.fps
            while not self._stop.is_set():
                loop_start = time.perf_counter()
                data = camera.read()
                if isinstance(data, Mapping):
                    self._write_camera_data(data)
                elapsed = time.perf_counter() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)
        except BaseException as exc:  # noqa: BLE001
            self._exception = exc
            self._stop.set()
            logger.warning("RealSense video recorder stopped: {}", exc)

    def _write_camera_data(self, camera_data: Mapping[str, Any]) -> None:
        color = camera_data.get("color")
        if not isinstance(color, Mapping):
            return
        for role in (EXTERIOR_ROLE, WRIST_ROLE):
            image = color.get(role)
            if image is None:
                continue
            frame = _as_rgb_uint8_frame(
                _rotate_image(np.asarray(image), self._rotations[role])
            )
            self._write_frame(role, frame)

    def _write_frame(self, role: str, rgb_frame: np.ndarray) -> None:
        if self._cv2 is None:
            raise RuntimeError("OpenCV writer backend is not initialized")

        writer = self._writers.get(role)
        if writer is None:
            height, width = rgb_frame.shape[:2]
            fourcc = self._cv2.VideoWriter_fourcc(*self.codec)
            writer = self._cv2.VideoWriter(
                str(self.paths[role]),
                fourcc,
                self.fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self.paths[role]}")
            self._writers[role] = writer

        bgr_frame = self._cv2.cvtColor(rgb_frame, self._cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)
        self._frame_counts[role] += 1


def _controller_state_from_observation(
    observation: dict[str, Any],
    controller_name: str,
) -> np.ndarray:
    controllers = _as_mapping(
        "observation['controllers']", observation.get("controllers")
    )
    controller_obs = _as_mapping(
        f"observation['controllers']['{controller_name}']",
        controllers.get(controller_name),
    )
    state = np.asarray(controller_obs.get("state"), dtype=np.float32)
    if state.shape != (9,):
        raise ValueError(f"Expected controller state shape (9,), got {state.shape}")
    return state


def _camera_images_from_observation(
    observation: dict[str, Any],
    camera_sensor_name: str,
) -> dict[str, np.ndarray]:
    sensors = _as_mapping("observation['sensors']", observation.get("sensors"))
    camera_obs = _as_mapping(
        f"observation['sensors']['{camera_sensor_name}']",
        sensors.get(camera_sensor_name),
    )
    color = _as_mapping(
        f"observation['sensors']['{camera_sensor_name}']['color']",
        camera_obs.get("color"),
    )
    return {name: np.asarray(image) for name, image in color.items()}


def _fetch_controller_state(
    robot: FrankaFR3RealSenseSdkRobot,
    controller_name: str,
) -> np.ndarray:
    state_dict = robot.controllers[controller_name].get_state()
    if not isinstance(state_dict, dict):
        raise TypeError(f"Controller get_state() returned {type(state_dict).__name__}")
    state = np.asarray(state_dict.get("state"), dtype=np.float32)
    if state.shape != (9,):
        raise ValueError(f"Expected controller state shape (9,), got {state.shape}")
    return state


def _gripper_width_to_closedness(width: float, max_open: float) -> np.ndarray:
    open_fraction = np.clip(width / max_open, 0.0, 1.0)
    return np.asarray([1.0 - open_fraction], dtype=np.float32)


def _build_policy_request(
    observation: dict[str, Any],
    *,
    controller_name: str,
    camera_sensor_name: str,
    gripper_max_open: float,
    exterior_rotate: int,
    wrist_rotate: int,
    prompt: str | None,
) -> tuple[dict[str, Any], np.ndarray]:
    state = _controller_state_from_observation(observation, controller_name)
    images = _camera_images_from_observation(observation, camera_sensor_name)

    if EXTERIOR_ROLE not in images or WRIST_ROLE not in images:
        raise KeyError(
            f"Need camera roles {EXTERIOR_ROLE!r}, {WRIST_ROLE!r}; got {sorted(images)}"
        )

    exterior = image_tools.convert_to_uint8(
        _rotate_image(images[EXTERIOR_ROLE], exterior_rotate)
    )
    wrist = image_tools.convert_to_uint8(
        _rotate_image(images[WRIST_ROLE], wrist_rotate)
    )
    gripper_width = float(state[7] + state[8])

    request = {
        POLICY_EXTERIOR_KEY: exterior,
        POLICY_WRIST_KEY: wrist,
        POLICY_JOINT_KEY: state[:7].astype(np.float32),
        POLICY_GRIPPER_KEY: _gripper_width_to_closedness(
            gripper_width, gripper_max_open
        ),
    }
    if prompt:
        request["prompt"] = prompt
    return request, state


def _infer_action_chunk(
    policy: WebsocketClientPolicy,
    request: dict[str, Any],
    chunk_size: int,
) -> np.ndarray:
    response = policy.infer(request)
    if "actions" not in response:
        raise KeyError(f"OpenPI response missing 'actions': {response.keys()}")
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    actions = np.atleast_2d(actions)
    if actions.shape[-1] < 8:
        raise ValueError(f"Expected OpenPI action dim >=8, got {actions.shape}")
    if len(actions) < chunk_size:
        raise ValueError(
            f"OpenPI returned horizon {len(actions)} < chunk_size {chunk_size}"
        )
    return actions[:chunk_size, :8].copy()


def _action_to_direct_joint_command(
    action: np.ndarray,
    *,
    gripper_threshold: float,
    gripper_max_open: float,
) -> tuple[np.ndarray, bool, float, float]:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (8,):
        raise ValueError(f"Expected action shape (8,), got {action.shape}")
    policy_gripper = float(action[7])
    closed = policy_gripper >= gripper_threshold
    # DROID/OpenPI: 0=open, 1=closed.
    droid_binary_cmd = 1.0 if closed else 0.0
    # Open-width convention: 1=open, 0=closed.
    spacemouse_width_percent = 0.0 if closed else 1.0
    finger_width = 0.0 if closed else gripper_max_open / 2.0
    direct = np.concatenate(
        [action[:7].astype(np.float64), np.asarray([finger_width, finger_width])]
    )
    return direct, closed, droid_binary_cmd, spacemouse_width_percent


def _limit_direct_joint_command(
    direct_action: np.ndarray,
    *,
    current_q: np.ndarray,
    previous_target_q: np.ndarray | None,
    max_joint_step: float,
    slowdown: float,
    joint_filter_alpha: float,
) -> tuple[np.ndarray, float, float]:
    limited = np.asarray(direct_action, dtype=np.float64).copy()
    current_q = np.asarray(current_q, dtype=np.float64)
    target_q = limited[:7].copy()
    raw_distance = float(np.max(np.abs(target_q - current_q)))

    if max_joint_step > 0:
        target_q = np.clip(
            target_q,
            current_q - max_joint_step,
            current_q + max_joint_step,
        )

    if slowdown < 1.0:
        target_q = current_q + slowdown * (target_q - current_q)

    alpha = float(np.clip(joint_filter_alpha, 0.0, 1.0))
    if previous_target_q is not None and alpha < 1.0:
        target_q = (1.0 - alpha) * previous_target_q + alpha * target_q
        if max_joint_step > 0:
            effective_step = max_joint_step * min(1.0, max(0.0, slowdown))
            target_q = np.clip(
                target_q,
                current_q - effective_step,
                current_q + effective_step,
            )

    limited[:7] = target_q
    sent_distance = float(np.max(np.abs(target_q - current_q)))
    return limited, raw_distance, sent_distance


def _send_binary_gripper_command(raw: Any, closed: bool, step_index: int) -> None:
    gripper = getattr(raw, "gripper", None)
    if gripper is None:
        return

    last_closed = getattr(raw, "_minimal_last_gripper_closed", None)
    last_time = float(getattr(raw, "_minimal_last_gripper_command_time", 0.0))
    now = time.monotonic()
    if last_closed == closed and now - last_time < 0.5:
        return

    droid_binary_cmd = 1.0 if closed else 0.0
    spacemouse_width_percent = 0.0 if closed else 1.0
    state_name = "close" if closed else "open"
    logger.info(
        "Sending gripper {} at step {} droid_bin={} spacemouse_width_percent={}",
        state_name,
        step_index,
        droid_binary_cmd,
        spacemouse_width_percent,
    )

    if closed and hasattr(gripper, "close"):
        gripper.close(block=False)
    elif not closed and hasattr(gripper, "open"):
        gripper.open(block=False)
    elif hasattr(gripper, "set_gripper_position"):
        gripper.set_gripper_position(droid_binary_cmd)

    raw.gripper_width = 0.0 if closed else 0.08
    raw._minimal_last_gripper_closed = closed
    raw._minimal_last_gripper_command_time = now


class _JointTargetStreamer:
    """Continuously sends the latest joint target while policy inference blocks."""

    def __init__(self, controller: Any, frequency: float):
        self._raw = controller.controller
        self._period = 1.0 / float(frequency)
        self._lock = threading.Lock()
        self._target_q: np.ndarray | None = None
        self._stop = threading.Event()
        self._exception: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="minimal_franka_joint_streamer",
            daemon=True,
        )

    def start(self, initial_q: np.ndarray) -> None:
        self.set_target(initial_q)
        self._thread.start()

    def set_target(self, target_q: np.ndarray) -> None:
        target_q = np.asarray(target_q, dtype=np.float64).reshape(-1)
        if target_q.shape != (7,):
            raise ValueError(f"Expected target_q shape (7,), got {target_q.shape}")
        with self._lock:
            self._target_q = target_q.copy()

    def check(self) -> None:
        if self._exception is not None:
            raise RuntimeError("Franka joint streamer failed.") from self._exception

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                loop_start = time.perf_counter()
                with self._lock:
                    target_q = None if self._target_q is None else self._target_q.copy()
                if target_q is not None:
                    if not self._raw.ctx.ok():
                        raise RuntimeError("Franka joint control context is not ok.")
                    self._raw.controller.set_control(target_q)
                elapsed = time.perf_counter() - loop_start
                if elapsed < self._period:
                    time.sleep(self._period - elapsed)
        except BaseException as exc:  # noqa: BLE001
            self._exception = exc
            self._stop.set()
            logger.warning("Franka joint streamer stopped: {}", exc)


def _send_direct_joint_action(
    controller: Any,
    streamer: _JointTargetStreamer,
    direct_action: np.ndarray,
    *,
    closed: bool,
    step_index: int,
) -> None:
    raw = controller.controller
    streamer.check()
    streamer.set_target(direct_action[:7])
    _send_binary_gripper_command(raw, closed, step_index)


def _stop_franka_without_opening_gripper(controller: Any) -> None:
    raw = getattr(controller, "controller", None)
    panda = getattr(raw, "panda", None)
    if panda is not None:
        panda.get_robot().stop()
    elif raw is not None and hasattr(raw, "end"):
        raw.end()


def _move_to_home(controller: Any) -> None:
    raw = controller.controller
    raw.init_pose = HOME_Q.astype(float).tolist()
    logger.info(
        "Moving to required home q={} before inference...",
        np.array2string(HOME_Q, precision=4, suppress_small=True),
    )
    raw.reset()


def _build_robot(args: argparse.Namespace) -> FrankaFR3RealSenseSdkRobot:
    gripper_type = None if args.gripper_type == "none" else args.gripper_type
    return FrankaFR3RealSenseSdkRobot(
        FrankaFR3RealSenseSdkRobotConfig(
            controller_config=FrankaFR3ControllerConfig(
                name=args.controller_name,
                hostname=args.robot_hostname,
                fps=max(1, int(round(args.control_freq))),
                gripper_type=gripper_type,
                gripper_port=args.gripper_port,
                control_type=ControlType.JOINT,
                reset=False,
            ),
            camera_config=MultiRealSenseCameraConfig(
                name=args.camera_sensor_name,
                cameras={
                    EXTERIOR_ROLE: args.exterior_serial,
                    WRIST_ROLE: args.wrist_serial,
                },
                image_width=args.camera_width,
                image_height=args.camera_height,
                fps=args.camera_fps,
                enable_depth=False,
            ),
        )
    )


def main() -> None:
    args = _parse_args()
    if args.control_freq <= 0:
        raise ValueError("--control-freq must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.gripper_max_open <= 0:
        raise ValueError("--gripper-max-open must be positive")
    if args.record_video_fps < 0:
        raise ValueError("--record-video-fps must be >= 0")
    if args.max_joint_step < 0:
        raise ValueError("--max-joint-step must be non-negative")
    if not (0.0 < args.slowdown <= 1.0):
        raise ValueError("--slowdown must be in (0, 1]")
    if not (0.0 <= args.joint_filter_alpha <= 1.0):
        raise ValueError("--joint-filter-alpha must be in [0, 1]")

    record_video_dir = _resolve_record_video_dir(args)
    record_video_fps = float(args.record_video_fps or args.camera_fps)
    if record_video_dir is not None and record_video_fps <= 0:
        raise ValueError("Recording FPS must be positive")
    _validate_realsense_physical_ports(args)
    robot = _build_robot(args)
    policy = WebsocketClientPolicy(host=args.openpi_host, port=args.openpi_port)
    controller = robot.controllers[args.controller_name]
    period = 1.0 / float(args.control_freq)
    request_index = 0
    step_index = 0
    streamer: _JointTargetStreamer | None = None
    camera_wrapper: NonBlockingSensorWrapper | None = None
    raw_camera_sensor: Any | None = None
    video_recorder: _RealSenseVideoRecorder | None = None
    previous_target_q: np.ndarray | None = None

    try:
        logger.info(
            "Minimal OpenPI Franka inference: server={}:{} host={} chunk_size={} hz={} record_video_dir={}",
            args.openpi_host,
            args.openpi_port,
            args.robot_hostname,
            args.chunk_size,
            args.control_freq,
            record_video_dir,
        )
        robot.set_up()
        if record_video_dir is not None:
            raw_camera_sensor = robot.sensors.get(args.camera_sensor_name)
            if raw_camera_sensor is None:
                raise KeyError(
                    f"Camera sensor {args.camera_sensor_name!r} not found; "
                    f"available={sorted(robot.sensors)}"
                )
            camera_wrapper = NonBlockingSensorWrapper(
                raw_camera_sensor,
                buffer_size=1,
                read_timeout=max(1.0, 2.0 / max(1, int(args.camera_fps))),
            )
            if not camera_wrapper.start():
                raise RuntimeError("Failed to start non-blocking RealSense capture")
            _wait_for_nonblocking_camera(camera_wrapper, timeout_s=5.0)
            robot.sensors[args.camera_sensor_name] = camera_wrapper
            logger.info(
                "Using non-blocking RealSense capture for inference and video recording"
            )

        if args.move_home_before_start:
            _move_to_home(controller)
            startup_q = HOME_Q.copy()
        else:
            startup_state = _fetch_controller_state(robot, args.controller_name)
            startup_q = startup_state[:7].astype(np.float64)
            logger.info(
                "Skipping home/reset before inference; starting from current q={}",
                np.array2string(startup_q, precision=4, suppress_small=True),
            )

        streamer = _JointTargetStreamer(controller, args.control_freq)
        streamer.start(startup_q)
        logger.info(
            "Started continuous joint target streamer at {} Hz", args.control_freq
        )

        for _ in range(max(0, args.warmup_observations)):
            streamer.check()
            robot.get_observation()

        metadata = policy.get_server_metadata()
        streamer.check()
        logger.info(
            "Connected to OpenPI server metadata_keys={}", sorted(metadata.keys())
        )

        if not args.auto_start:
            input("Press Enter to start minimal OpenPI inference...")

        if record_video_dir is not None:
            assert camera_wrapper is not None
            video_prefix = _unique_video_prefix(
                record_video_dir, args.record_video_prefix
            )
            video_recorder = _RealSenseVideoRecorder(
                output_dir=record_video_dir,
                prefix=video_prefix,
                fps=record_video_fps,
                codec=args.record_video_codec,
                exterior_rotate=args.exterior_rotate,
                wrist_rotate=args.wrist_rotate,
            )
            video_recorder.start(camera_wrapper)

        while args.max_steps <= 0 or step_index < args.max_steps:
            streamer.check()
            if video_recorder is not None:
                video_recorder.check()
            obs = robot.get_observation()
            request, request_state = _build_policy_request(
                obs,
                controller_name=args.controller_name,
                camera_sensor_name=args.camera_sensor_name,
                gripper_max_open=args.gripper_max_open,
                exterior_rotate=args.exterior_rotate,
                wrist_rotate=args.wrist_rotate,
                prompt=args.prompt,
            )

            infer_start = time.perf_counter()
            chunk = _infer_action_chunk(policy, request, args.chunk_size)
            streamer.check()
            latency = time.perf_counter() - infer_start
            logger.info(
                "request={} obs_q={} obs_grip={:.3f} chunk_shape={} latency={:.3f}s",
                request_index,
                np.array2string(request_state[:7], precision=4, suppress_small=True),
                float(request[POLICY_GRIPPER_KEY][0]),
                tuple(chunk.shape),
                latency,
            )

            request_index += 1
            for chunk_step, action in enumerate(chunk):
                if args.max_steps > 0 and step_index >= args.max_steps:
                    break

                if video_recorder is not None:
                    video_recorder.check()
                loop_start = time.perf_counter()
                current_state = _fetch_controller_state(robot, args.controller_name)
                (
                    direct_action,
                    closed,
                    droid_gripper_cmd,
                    spacemouse_width_percent,
                ) = _action_to_direct_joint_command(
                    action,
                    gripper_threshold=args.gripper_threshold,
                    gripper_max_open=args.gripper_max_open,
                )
                direct_action, raw_target_distance, target_distance = (
                    _limit_direct_joint_command(
                        direct_action,
                        current_q=current_state[:7],
                        previous_target_q=previous_target_q,
                        max_joint_step=args.max_joint_step,
                        slowdown=args.slowdown,
                        joint_filter_alpha=args.joint_filter_alpha,
                    )
                )

                if args.max_target_distance > 0:
                    if raw_target_distance > args.max_target_distance:
                        message = (
                            f"Absolute target too far at step {step_index}: "
                            f"{raw_target_distance:.3f} rad > {args.max_target_distance:.3f} rad."
                        )
                        if args.far_target_action == "abort":
                            raise RuntimeError(
                                f"{message} Check OpenPI config/action semantics."
                            )
                        logger.warning(
                            "{} Dropping action and requesting a fresh chunk from current state.",
                            message,
                        )
                        assert streamer is not None
                        streamer.set_target(current_state[:7])
                        previous_target_q = current_state[:7].copy()
                        step_index += 1
                        break

                if (
                    args.log_actions
                    and args.log_every > 0
                    and step_index % args.log_every == 0
                ):
                    logger.info(
                        "step={} chunk_step={} q={} raw={} target={} raw_dq_max={:.4f} dq_max={:.4f} "
                        "gripper_closed={} droid_gripper_cmd={} spacemouse_width_percent={}",
                        step_index,
                        chunk_step,
                        np.array2string(
                            current_state[:7], precision=4, suppress_small=True
                        ),
                        np.array2string(action, precision=4, suppress_small=True),
                        np.array2string(
                            direct_action, precision=4, suppress_small=True
                        ),
                        raw_target_distance,
                        target_distance,
                        closed,
                        droid_gripper_cmd,
                        spacemouse_width_percent,
                    )

                assert streamer is not None
                _send_direct_joint_action(
                    controller,
                    streamer,
                    direct_action,
                    closed=closed,
                    step_index=step_index,
                )
                previous_target_q = direct_action[:7].copy()

                step_index += 1
                elapsed = time.perf_counter() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if video_recorder is not None:
            video_recorder.stop()
        if streamer is not None:
            streamer.stop()
        try:
            _stop_franka_without_opening_gripper(controller)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to stop Franka cleanly: {}", exc)
        if camera_wrapper is not None:
            camera_wrapper.stop()
        if raw_camera_sensor is not None:
            robot.sensors[args.camera_sensor_name] = raw_camera_sensor
        try:
            robot.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to close robot cleanly: {}", exc)
        if hasattr(policy, "_ws"):
            try:
                policy._ws.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
