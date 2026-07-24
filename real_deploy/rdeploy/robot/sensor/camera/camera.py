"""Camera interfaces used by the RealSense deployment client."""

from abc import abstractmethod
from dataclasses import dataclass

from rdeploy.robot.sensor.sensor import Sensor, SensorConfig, SensorOutput


@dataclass
class CameraConfig(SensorConfig):
    sensor_type: str = "camera"


@dataclass
class CameraOutput(SensorOutput):
    pass


class Camera(Sensor):
    @abstractmethod
    def _read_data(self) -> CameraOutput:
        pass
