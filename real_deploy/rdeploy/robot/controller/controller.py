"""Controller interfaces used by the Franka deployment client."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional


class ControlType(Enum):
    JOINT = "joint"


@dataclass
class ControllerConfig:
    name: str
    control_type: ControlType = ControlType.JOINT
    gripper_type: Optional[str] = None
    before_apply_action_hook: Optional[Callable[..., Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Controller(ABC):
    def __init__(self, config: ControllerConfig):
        self.name = config.name
        self.config = config
        self.before_apply_action_hook = config.before_apply_action_hook

    @abstractmethod
    def set_up(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_state(self) -> Any:
        pass

    @abstractmethod
    def _apply_action(
        self, action: Any, action_type: Optional[ControlType] = None
    ) -> None:
        pass

    def apply_action(
        self, action: Any, action_type: Optional[ControlType] = None
    ) -> None:
        if self.before_apply_action_hook:
            action = self.before_apply_action_hook(action, action_type)
        return self._apply_action(action, action_type)

    @abstractmethod
    def help(self) -> str:
        pass
