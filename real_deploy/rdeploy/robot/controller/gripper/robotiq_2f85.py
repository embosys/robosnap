from dataclasses import dataclass
from typing import Any, Optional

from rdeploy.robot.controller import Controller, ControllerConfig
from rdeploy.utils.logger_utils import logger

try:
    from pymodbus.client.sync import ModbusSerialClient as ModbusClient
except ImportError:
    ModbusClient = None
    logger.warning("pymodbus is not installed; Robotiq initialization is unavailable.")


@dataclass
class RobotiqCGripperConfig(ControllerConfig):
    """Configuration for Robotiq C Gripper controller.

    Args:
        name: Name of the gripper controller.
        port: Serial port path, default is "/dev/ttyUSB0".
        baudrate: Baud rate for serial communication, default is 115200.
        method: Modbus communication method, default is "rtu".
        stopbits: Number of stop bits, default is 1.
        bytesize: Number of data bits, default is 8.
        gripper_speed: Gripper speed (0.0-1.0), default is 0.8.
        gripper_force: Gripper force (0.0-1.0), default is 0.2.
        timeout: Communication timeout in seconds, default is 0.2.
        extra: Additional configuration parameters as dictionary.
    """

    name: str = "robotiq_gripper"
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    method: str = "rtu"
    stopbits: int = 1
    bytesize: int = 8
    gripper_speed: float = 0.8
    gripper_force: float = 0.2
    timeout: float = 0.2


class RobotiqCGripper(Controller):
    """Robotiq C Gripper controller.

    This class provides an interface to control Robotiq C-series grippers
    using Modbus communication.
    """

    def __init__(self, config: RobotiqCGripperConfig):
        """Initialize Robotiq C Gripper controller.

        Args:
            config: Controller configuration. Must be RobotiqCGripperConfig instance.

        Raises:
            TypeError: If config is not an instance of RobotiqCGripperConfig.
            ImportError: If pymodbus is not installed.
        """
        super().__init__(config)
        if not isinstance(config, RobotiqCGripperConfig):
            raise TypeError(
                f"Expected RobotiqCGripperConfig, got {type(config).__name__}"
            )

        if ModbusClient is None:
            raise ImportError(
                "pymodbus is not installed. Please install it to use RobotiqCGripper."
            )

        self.config: RobotiqCGripperConfig = config

        self.gripper_speed = config.gripper_speed
        self.gripper_force = config.gripper_force
        self.client = ModbusClient(
            method=config.method,
            port=config.port,
            stopbits=config.stopbits,
            bytesize=config.bytesize,
            baudrate=config.baudrate,
            timeout=config.timeout,
        )
        self._initialized = False

    def set_up(self) -> None:
        """Initialize the gripper connection."""
        if self._initialized:
            return

        self.wait_for_connection()
        self._initialized = True
        logger.info("Robotiq C Gripper controller initialized")

    def reset(self) -> None:
        """Reset gripper to open position."""
        if not self._initialized:
            self.set_up()

        self.open(block=True)

    def get_state(self) -> Optional[dict]:
        """Get current gripper state.

        Returns:
            Dictionary containing gripper state information, or None if error.
        """
        if not self._initialized:
            try:
                self.set_up()
            except Exception:
                return None

        try:
            width = self.get_current_width()
            is_grasped = self.is_object_grasped()
            return {
                "gripper_width": width,
                "is_grasped": bool(is_grasped),
            }
        except Exception as e:
            logger.warning(f"get_state encountered an error: {e}")
            return None

    def _apply_action(self, action: Any, action_type: Optional[str] = None) -> None:
        """Apply action to the gripper.

        Args:
            action: Action to apply. Can be:
                - float: Gripper position (0.0-1.0), where 0.0 is open, 1.0 is closed
                - str: "open" or "close"
            action_type: Type of action (not used, kept for compatibility).
        """
        if not self._initialized:
            self.set_up()

        try:
            if isinstance(action, str):
                if action.lower() == "open":
                    self.open(block=False)
                elif action.lower() == "close":
                    self.close(block=False)
                else:
                    logger.warning(f"Unknown action string: {action}")
            elif isinstance(action, (int, float)):
                # Normalize to 0.0-1.0 range
                position = max(0.0, min(1.0, float(action)))
                self.set_gripper_position(position)
            else:
                logger.warning(f"Unknown action type: {type(action)}")
        except Exception as e:
            logger.warning(f"apply_action encountered an error: {e}")

    def help(self) -> str:
        """Return help message with action format and configuration details."""
        help_msg = f"""RobotiqCGripper - Controller for Robotiq C-series gripper

Action Formats:
  - Float (0.0-1.0): Gripper position, where 0.0 is fully open, 1.0 is fully closed
  - String: "open" or "close" to open or close the gripper

Configuration:
  - Port: {self.config.port}
  - Baudrate: {self.config.baudrate}
  - Gripper speed: {self.config.gripper_speed}
  - Gripper force: {self.config.gripper_force}
  - Timeout: {self.config.timeout}
"""
        return help_msg

    def __repr__(self) -> str:
        """Return string representation of the controller."""
        return f"RobotiqCGripper(config={self.config})"

    def wait_for_connection(self, timeout=-1):
        """Wait for gripper connection.

        Args:
            timeout: Connection timeout in seconds. -1 means no timeout.
        """
        self.client.connect()

    def _get_bit(self, number, bit) -> bool:
        return (number >> bit) & 1

    def _read_and_check_input_register(self, address) -> int:
        feedback = self.client.read_input_registers(
            address=address, count=1, unit=0x0009
        )
        register_value = feedback.registers[0]
        return register_value

    def set_gripper_position(self, position):
        position_value = int(position * 255)
        speed_value = int(self.gripper_speed * 255)
        force_value = int(self.gripper_force * 255)
        bytes = [0b00001001, 0, 0, position_value, speed_value, force_value]
        values = []
        if len(bytes) % 2 != 0:
            bytes.append(0)
        for i in range(0, int(len(bytes) / 2)):
            values.append((bytes[2 * i] << 8) + bytes[2 * i + 1])
        self.client.write_registers(0x03E8, values=values, unit=0x0009)

    def is_object_grasped(self) -> bool:
        movement_register = self._read_and_check_input_register(address=0x07D0)
        bit_15 = self._get_bit(movement_register, 15)
        bit_14 = self._get_bit(movement_register, 14)

        if bit_15 == 1 and bit_14 == 0:
            return 1.0
        else:
            return 0.0

    def open(self, block=False):
        self.set_gripper_position(position=0.1)  # 0.1

    def close(self, block=False):
        self.set_gripper_position(position=0.9)  # 1

    def get_current_width(self):
        return self._read_and_check_input_register(address=0x07D4) / 255.0
