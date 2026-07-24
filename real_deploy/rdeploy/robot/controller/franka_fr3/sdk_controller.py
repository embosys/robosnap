"""Franka FR3 joint controller backed by panda-py."""

from dataclasses import dataclass
from typing import Any, Optional

from rdeploy.robot.controller import Controller, ControllerConfig, ControlType
from rdeploy.utils.logger_utils import logger


@dataclass
class FrankaFR3ControllerConfig(ControllerConfig):
    hostname: str = ""
    fps: int = 30
    gripper_type: Optional[str] = "robotiq"
    control_type: ControlType = ControlType.JOINT
    gripper_port: Optional[str] = "/dev/ttyUSB0"
    reset: bool = True


class FrankaFR3Controller(Controller):
    """Joint-position controller used by the OpenPI deployment loop."""

    def __init__(self, config: FrankaFR3ControllerConfig):
        super().__init__(config)
        if not isinstance(config, FrankaFR3ControllerConfig):
            raise TypeError(
                f"Expected FrankaFR3ControllerConfig, got {type(config).__name__}"
            )

        if not config.hostname:
            raise ValueError("Franka hostname must be configured")

        control_type = (
            config.control_type.value
            if isinstance(config.control_type, ControlType)
            else str(config.control_type).lower()
        )
        if control_type != ControlType.JOINT.value:
            raise ValueError(
                "This deployment entry supports joint control only, "
                f"got {config.control_type}"
            )

        from rdeploy.robot.controller.franka_fr3._joint_control import (
            FrankaJointController,
        )

        self.config = config
        self.controller = FrankaJointController(
            hostname=config.hostname,
            fps=config.fps,
            gripper_port=config.gripper_port,
            gripper_type=config.gripper_type,
            reset=config.reset,
        )
        self._initialized = False

    def set_up(self) -> None:
        self._initialized = True
        logger.info("Franka panda-py joint controller initialized")

    def reset(self) -> None:
        if not self._initialized:
            self.set_up()
        self.controller.reset()

    def get_state(self) -> Optional[dict]:
        if not self._initialized:
            self.set_up()
        try:
            return self.controller.get_obs(
                read_gripper=self.config.gripper_type != "robotiq"
            )
        except Exception as exc:
            logger.warning(
                "Live gripper read failed; using cached gripper state: {}", exc
            )
            try:
                return self.controller.get_obs(read_gripper=False)
            except Exception as fallback_exc:
                logger.warning("Failed to read Franka state: {}", fallback_exc)
                return None

    def _apply_action(
        self, action: Any, action_type: Optional[ControlType] = None
    ) -> None:
        if not self._initialized:
            self.set_up()
        if action_type not in (None, ControlType.JOINT, "joint"):
            raise ValueError(f"Unsupported action type: {action_type}")
        self.controller.apply_action(action, type="joint")

    def help(self) -> str:
        return (
            "Franka joint action: 9 values "
            "[q0, q1, q2, q3, q4, q5, q6, left_finger, right_finger]"
        )
