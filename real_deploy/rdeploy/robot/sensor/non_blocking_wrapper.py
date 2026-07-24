"""Non-blocking sensor wrapper for all sensor types."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from copy import deepcopy
from threading import Event, Thread
from typing import Any, Dict, Optional

import numpy as np

from rdeploy.robot.sensor.sensor import Sensor
from rdeploy.utils.logger_utils import logger


class NonBlockingSensorWrapper:
    """Wrapper class that provides non-blocking access to any Sensor instance.

    This class wraps a Sensor instance and provides non-blocking data access
    by running data capture in a background thread and storing data in
    a buffer.

    Example:
        # Create a sensor
        sensor = RealSenseCamera(config=config)
        sensor.initialize()

        # Wrap it for non-blocking access with timeout
        non_blocking_sensor = NonBlockingSensorWrapper(
            sensor,
            buffer_size=1,
            read_timeout=5.0  # Timeout each read operation after 5 seconds
        )
        non_blocking_sensor.start()

        # Non-blocking access
        data = non_blocking_sensor.read()  # Returns latest data immediately
        # Extract specific keys from data as needed
        if data:
            color = data.get("color")
            depth = data.get("depth")

        # Cleanup
        non_blocking_sensor.stop()
        sensor.close()
    """

    def __init__(
        self,
        sensor: Sensor,
        buffer_size: int = 1,
        read_timeout: Optional[float] = None,
    ):
        """Initialize the non-blocking sensor wrapper.

        Args:
            sensor: Sensor instance to wrap. Must be initialized before use.
            buffer_size: Maximum number of data frames to keep in buffer (default: 1, only latest).
            read_timeout: Timeout in seconds for each read operation. If None, no timeout (default: None).
                If a read operation takes longer than this timeout, it will be skipped and a warning logged.
        """
        if not isinstance(sensor, Sensor):
            raise TypeError(f"Expected Sensor instance, got {type(sensor)}")

        self.sensor = sensor
        self.buffer_size = buffer_size
        self.read_timeout = read_timeout

        # Data buffer (only keep latest data by default)
        self.data_buffer: deque = deque(maxlen=buffer_size)

        # Threading
        self._data_thread: Optional[Thread] = None
        self._exit_event: Optional[Event] = None
        self._keep_running = False
        self._is_running = False
        self._executor: Optional[ThreadPoolExecutor] = None

    def start(self) -> bool:
        """Start the background data capture thread.

        Returns:
            True if thread started successfully, False otherwise.
        """
        if self._is_running:
            logger.warning(
                f"Non-blocking wrapper for {self.sensor.name} is already running"
            )
            return True

        if not self.sensor.is_initialized():
            logger.error(
                f"Sensor {self.sensor.name} is not initialized. Call sensor.initialize() first."
            )
            return False

        try:
            self._keep_running = True
            self._exit_event = Event()
            self._executor = ThreadPoolExecutor(max_workers=1)
            self._data_thread = Thread(target=self._update_data, daemon=True)
            self._data_thread.start()
            self._is_running = True
            logger.info(f"Started non-blocking data capture for {self.sensor.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start data thread for {self.sensor.name}: {e}")
            return False

    def stop(self) -> None:
        """Stop the background data capture thread."""
        if not self._is_running:
            return

        self._keep_running = False
        if self._exit_event is not None:
            self._exit_event.set()

        if self._data_thread is not None and self._data_thread.is_alive():
            self._data_thread.join(timeout=2.0)
            if self._data_thread.is_alive():
                logger.warning(
                    f"Data thread for {self.sensor.name} did not stop gracefully"
                )
            else:
                logger.info(f"Stopped non-blocking data capture for {self.sensor.name}")

        self._data_thread = None
        self._is_running = False
        self.data_buffer.clear()

        # Shutdown executor
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def is_running(self) -> bool:
        """Check if the non-blocking wrapper is running.

        Returns:
            True if running, False otherwise.
        """
        return self._is_running

    def _update_data(self) -> None:
        """Background thread function to continuously capture sensor data."""
        try:
            while not self._exit_event.is_set() and self._keep_running:
                try:
                    # Read data from sensor using read() interface (blocking call)
                    # Apply timeout if specified
                    if self.read_timeout is not None and self._executor is not None:
                        future = self._executor.submit(self.sensor.read)
                        try:
                            sensor_data = future.result(timeout=self.read_timeout)
                        except FutureTimeoutError:
                            logger.warning(
                                f"{self.sensor.name} read operation timed out after {self.read_timeout}s, skipping..."
                            )
                            continue
                    else:
                        sensor_data = self.sensor.read()

                    if sensor_data is not None:
                        self.data_buffer.append(sensor_data)

                except RuntimeError as e:
                    if "timeout" in str(e).lower():
                        logger.debug(
                            f"{self.sensor.name} data wait timeout, retrying..."
                        )
                    else:
                        logger.error(f"{self.sensor.name} error in data capture: {e}")
                        # Continue running even on error
                except Exception as e:
                    logger.error(f"{self.sensor.name} exception in data capture: {e}")
                    if not self._exit_event.is_set():
                        # Continue running even on error
                        pass
        except Exception as e:
            logger.error(f"{self.sensor.name} data update thread error: {e}")

    def read(self) -> Optional[Dict[str, Any]]:
        """Read data from the sensor in non-blocking mode.

        Returns:
            Dictionary containing sensor data (e.g., 'color', 'depth', 'imu', etc.),
            or None if no data available.
        """
        if not self._is_running:
            logger.warning(
                f"Non-blocking wrapper for {self.sensor.name} is not running. Call start() first."
            )
            return None

        try:
            return self._safe_deepcopy(self.data_buffer[-1])
        except IndexError:
            return None

    @staticmethod
    def _safe_deepcopy(data: Any) -> Optional[Dict[str, Any]]:
        """Deep-copy payload safely (including numpy arrays & dict-like outputs)."""

        def _copy_obj(obj: Any) -> Any:
            if obj is None:
                return None
            if isinstance(obj, np.ndarray):
                return obj.copy()
            if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
                return _copy_obj(obj.to_dict())
            if isinstance(obj, dict):
                return {k: _copy_obj(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_copy_obj(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_copy_obj(v) for v in obj)
            return deepcopy(obj)

        copied = _copy_obj(data)
        if copied is None:
            return None
        if isinstance(copied, dict):
            return copied
        # Unknown payload type -> drop
        return None

    def has_data(self) -> bool:
        """Check if there is data available in the buffer.

        Returns:
            True if data is available, False otherwise.
        """
        return len(self.data_buffer) > 0

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return (
            f"NonBlockingSensorWrapper("
            f"sensor={self.sensor.name}, "
            f"running={self._is_running}, "
            f"buffer_size={len(self.data_buffer)}/{self.buffer_size})"
        )
