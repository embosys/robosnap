"""Franka FR3 robot composed with two RealSense cameras."""

from dataclasses import dataclass, field

from rdeploy.robot.controller import ControlType
from rdeploy.robot.controller.franka_fr3.sdk_controller import (
    FrankaFR3Controller,
    FrankaFR3ControllerConfig,
)
from rdeploy.robot.robot.robot import Robot, RobotConfig
from rdeploy.robot.sensor.camera.realsense_sdk import (
    MultiRealSenseCamera,
    MultiRealSenseCameraConfig,
)


@dataclass
class FrankaFR3RealSenseSdkRobotConfig(RobotConfig):
    name: str = "franka_fr3_realsense_sdk"
    controller_config: FrankaFR3ControllerConfig = field(
        default_factory=lambda: FrankaFR3ControllerConfig(
            name="franka_fr3_single_arm_controller",
            hostname="",
            fps=30,
            gripper_type="robotiq",
            control_type=ControlType.JOINT,
            reset=False,
        )
    )
    camera_config: MultiRealSenseCameraConfig = field(
        default_factory=lambda: MultiRealSenseCameraConfig(
            name="franka_fr3_realsense_camera",
            fps=30,
            enable_depth=False,
        )
    )


class FrankaFR3RealSenseSdkRobot(Robot):
    def __init__(self, config: FrankaFR3RealSenseSdkRobotConfig):
        super().__init__(config=config)
        controller_config = config.controller_config
        camera_config = config.camera_config
        self.controllers = {
            controller_config.name: FrankaFR3Controller(controller_config)
        }
        self.sensors = {camera_config.name: MultiRealSenseCamera(camera_config)}
