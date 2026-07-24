import time

import numpy as np

from rdeploy.utils.logger_utils import logger

try:
    import panda_py
    from panda_py import controllers, libfranka
except ImportError:
    panda_py = None
    controllers = None
    libfranka = None
    logger.warning("panda_py is not installed; Franka initialization is unavailable.")

PI = np.pi
HOME_JOINTS = [0, -PI / 4, 0, -3 * PI / 4, 0, PI / 2, PI / 4 - PI / 4]


class FrankaJointController:
    def __init__(
        self,
        hostname,
        fps=30,
        init_pose=None,
        gripper_type="panda_hand",
        gripper_port="/dev/ttyUSB0",
        reset=True,
    ):
        if panda_py is None or libfranka is None:
            raise ImportError(
                "panda_py is not installed. Please install it to use FrankaJointController."
            )
        self.gripper_type = gripper_type
        if gripper_type == "panda_hand":
            self.gripper = libfranka.Gripper(hostname)
            self.gripper.gripper_speed = 0.2
            self.gripper.gripper_force = 5.0
        elif gripper_type == "robotiq":
            from rdeploy.robot.controller.gripper.robotiq_2f85 import (
                RobotiqCGripper,
                RobotiqCGripperConfig,
            )

            gripper_config = RobotiqCGripperConfig(
                name="robotiq_gripper",
                port=gripper_port,
            )
            self.gripper = RobotiqCGripper(config=gripper_config)
            self.gripper.set_up()
        else:
            self.gripper = None
        self.panda = panda_py.Panda(hostname)
        try:
            self.panda.recover_from_errors()
        except Exception as exc:
            logger.warning("Franka error recovery before joint control failed: {}", exc)
        self.controller = controllers.JointPosition(
            stiffness=[600.0, 600.0, 600.0, 500.0, 250.0, 150.0, 50.0]
        )
        self.panda.enable_logging(int(1e2))
        if init_pose is not None:
            self.init_pose = init_pose
        else:
            self.init_pose = HOME_JOINTS
        if reset:
            self.init_robot()
            if self.gripper is not None:
                self.gripper.open(block=False)
        self.panda.start_controller(self.controller)
        self.gripper_width = 0.08
        self.ctx = self.panda.create_context(frequency=fps)
        self._last_qpos = None
        self.fps = fps

        self._last_gripper_command = None
        self._last_gripper_command_time = 0.0
        self._gripper_command_repeat_s = 1.0

    def _read_joint_positions(self):
        """Read the latest joint positions from the robot state.

        Prefer the most recent RobotState snapshot over the logging buffer so
        closed-loop policies condition on the actual live arm state.
        """
        try:
            robot_state = self.panda.get_state()
            q = np.asarray(robot_state.q, dtype=np.float64)
            if q.shape == (7,):
                return q
        except Exception:
            pass

        return np.asarray(self.panda.get_log()["q"][-1], dtype=np.float64)

    def init_robot(self):
        self.panda.move_to_joint_position(self.init_pose)
        if self.gripper is not None:
            if self.gripper_type == "robotiq":
                self.gripper.open(block=False)
            else:
                self.gripper.homing()

    def reset_joint(self):
        self.panda.move_to_joint_position(self.init_pose)

    def reset(self):
        self.init_robot()
        self.panda.start_controller(self.controller)
        self.gripper_width = 0.08
        self.ctx = self.panda.create_context(frequency=self.fps)
        self._last_qpos = None
        self._last_gripper_command = None
        self._last_gripper_command_time = 0.0

    def get_robot_state(self, read_gripper=False):
        """
        Get the real robot state.
        """
        if self.gripper is not None and read_gripper:
            try:
                if self.gripper_type == "robotiq":
                    gripper_width = self.gripper.get_current_width()
                else:
                    gripper_state = self.gripper.read_once()
                    gripper_width = gripper_state.width
            except Exception as exc:
                logger.warning(
                    "Failed to read live gripper width, using cached width: {}",
                    exc,
                )
                gripper_width = self.gripper_width
        else:
            gripper_width = self.gripper_width
        self.gripper_width = gripper_width

        gripper_qpos = gripper_width

        self._last_qpos = self._read_joint_positions()

        robot_qpos = np.concatenate(
            [self._last_qpos, [gripper_qpos / 2.0], [gripper_qpos / 2.0]]
        )

        obs = robot_qpos
        assert obs.shape == (9,), f"incorrect obs shape, {obs.shape}"

        return obs

    def get_obs(self, read_gripper=False):
        """
        Get the real robot observation.
        """
        state = self.get_robot_state(read_gripper=read_gripper)

        return {"state": state}

    def _clip_action(self, action, delta):
        action[:7] = np.clip(
            action[:7], self._last_qpos - delta, self._last_qpos + delta
        )
        return action

    def apply_action(self, action, type="joint", read_gripper=False):
        try:
            if type == "joint":
                action = np.array(action)
                assert action.shape == (9,), f"incorrect action shape, {action.shape}"
                gripper_width = sum(action[7:])
                if gripper_width > 0.04:
                    gripper_width = 0.08
                else:
                    gripper_width = 0.0
                action = self._clip_action(action, delta=0.10)
                if not self.ctx.ok():
                    raise RuntimeError(
                        "Franka joint control context is not ok; recover robot errors before replay."
                    )
                self.controller.set_control(action[:7])
                if self.gripper is not None and self.gripper_type == "robotiq":
                    command = "open" if gripper_width > 0.04 else "close"
                    now = time.monotonic()
                    should_send = (
                        command != self._last_gripper_command
                        or now - self._last_gripper_command_time
                        >= self._gripper_command_repeat_s
                    )
                    if should_send:
                        logger.info("Sending Robotiq gripper command: {}", command)
                        if command == "open":
                            self.gripper.open(block=False)
                        else:
                            self.gripper.close(block=False)
                        self._last_gripper_command = command
                        self._last_gripper_command_time = now
                    self.gripper_width = gripper_width
                elif (
                    self.gripper is not None
                    and abs(gripper_width - self.gripper_width) > 0.01
                ):
                    self.gripper.grasp(
                        width=gripper_width,
                        speed=0.1,
                        force=20,
                        epsilon_outer=0.08,
                    )
                    self.gripper_width = gripper_width
            else:
                raise ValueError(f"Unsupported action type: {type}")

        except Exception as e:
            print(e)
            raise
        finally:
            self._last_qpos = self._read_joint_positions()

    def end(self):
        if self.gripper is not None:
            if self.gripper_type == "robotiq":
                self.gripper.open(block=True)
            else:
                self.gripper.homing()

        self.panda.get_robot().stop()

    def activate_guiding_mode(self):
        self.panda.teaching_mode(active=True)

    def deactivate_guiding_mode(self):
        self.panda.teaching_mode(active=False)
