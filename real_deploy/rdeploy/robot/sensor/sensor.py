"""Sensor interfaces used by the RealSense deployment client."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Optional

from rdeploy.utils.logger_utils import logger


@dataclass
class SensorOutput:
    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def items(self):
        return self.to_dict().items()

    def to_dict(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass
class SensorConfig:
    name: str
    sensor_type: str = "sensor"
    return_data_hook: Optional[Callable[[Any], Any]] = None


class Sensor(ABC):
    def __init__(self, config: SensorConfig):
        self.name = config.name or self.__class__.__name__
        self.sensor_type = config.sensor_type
        self._is_initialized = False
        self._return_data_hook = config.return_data_hook

    @abstractmethod
    def _read_data(self) -> Any:
        pass

    def read(self) -> Any:
        if not self._is_initialized:
            logger.warning(
                "Sensor {} is not initialized. Call initialize() first.", self.name
            )
            return None
        try:
            data = self._read_data()
            if self._return_data_hook:
                data = self._return_data_hook(data)
            return data
        except Exception as exc:
            logger.error("Error reading from sensor {}: {}", self.name, exc)
            return None

    def initialize(self) -> bool:
        self._is_initialized = True
        return True

    def close(self) -> None:
        self._is_initialized = False

    def is_initialized(self) -> bool:
        return self._is_initialized
