"""Robot composition used by the Franka deployment client."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from rdeploy.utils.logger_utils import logger


@dataclass
class RobotConfig:
    name: str = ""
    return_observation_hook: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


class Robot:
    def __init__(self, config: RobotConfig):
        self.name = config.name or self.__class__.__name__
        self.controllers: Dict[str, Any] = {}
        self.sensors: Dict[str, Any] = {}
        self._is_setup = False
        self.return_observation_hook = config.return_observation_hook

    def set_up(self) -> None:
        for controller in self.controllers.values():
            controller.set_up()
        for sensor in self.sensors.values():
            sensor.initialize()
        self._is_setup = True
        logger.info("Robot {} is ready", self.name)

    def get_observation(self) -> Dict[str, Any]:
        controller_data = {
            name: controller.get_state()
            for name, controller in self.controllers.items()
        }
        sensor_data = {}
        for name, sensor in self.sensors.items():
            reading = sensor.read()
            if reading is not None:
                sensor_data[name] = reading
        observation = {
            "controllers": controller_data,
            "sensors": sensor_data,
        }
        if self.return_observation_hook:
            observation = self.return_observation_hook(observation)
        return observation

    def reset(self) -> None:
        for controller in self.controllers.values():
            controller.reset()

    def close(self) -> None:
        for controller in self.controllers.values():
            close = getattr(controller, "close", None)
            if close is not None:
                close()
        for sensor in self.sensors.values():
            sensor.close()
        self._is_setup = False
