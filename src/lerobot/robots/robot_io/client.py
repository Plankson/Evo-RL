from __future__ import annotations

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

from lerobot.robots.robot_io.serialization import (
    deserialize_message,
    normalize_action_for_transport,
    serialize_message,
)

logger = logging.getLogger(__name__)


class RobotIOClient:
    """Robot proxy that communicates with Robot IO Server via ZMQ."""

    name = "robot_io_client"

    def __init__(
        self,
        observation_address: str,
        action_address: str,
        metadata_address: str,
        observation_timeout_s: float = 5.0,
    ):
        import zmq

        self._zmq = zmq
        self.observation_address = observation_address
        self.action_address = action_address
        self.metadata_address = metadata_address
        self.observation_timeout_s = observation_timeout_s

        self._context: Any = None
        self._observation_socket: Any = None
        self._action_socket: Any = None
        self._metadata_socket: Any = None

        self._latest_observation: dict[str, Any] | None = None
        self._latest_envelope: dict[str, Any] | None = None
        self._observation_lock = Lock()
        self._new_observation_event = Event()

        self._receiver_stop = Event()
        self._receiver_thread: Thread | None = None

        self._connected = False
        self.id = "robot-io-client"
        self.robot_type = "remote"
        self.action_features: dict[str, Any] = {}
        self.observation_features: dict[str, Any] = {}
        self.cameras: dict[str, Any] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self._connected:
            return

        zmq = self._zmq
        self._context = zmq.Context()

        self._metadata_socket = self._context.socket(zmq.REQ)
        self._metadata_socket.setsockopt(zmq.LINGER, 0)
        self._metadata_socket.connect(self.metadata_address)
        self._metadata_socket.send(serialize_message({"cmd": "metadata"}))
        metadata = deserialize_message(self._metadata_socket.recv())

        self.name = metadata["name"]
        self.robot_type = metadata.get("robot_type", self.name)
        self.action_features = metadata["action_features"]
        self.observation_features = metadata["observation_features"]
        camera_keys = metadata.get("camera_keys", [])
        self.cameras = {key: None for key in camera_keys}

        self._action_socket = self._context.socket(zmq.PUSH)
        self._action_socket.setsockopt(zmq.LINGER, 0)
        self._action_socket.connect(self.action_address)

        self._observation_socket = self._context.socket(zmq.SUB)
        self._observation_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._observation_socket.setsockopt(zmq.CONFLATE, 1)
        self._observation_socket.setsockopt(zmq.LINGER, 0)
        self._observation_socket.connect(self.observation_address)

        self._receiver_stop.clear()
        self._receiver_thread = Thread(target=self._observation_receiver_loop, daemon=True, name="RobotIOObsRecv")
        self._receiver_thread.start()

        deadline = time.monotonic() + self.observation_timeout_s
        while time.monotonic() < deadline:
            if self._new_observation_event.wait(timeout=0.05):
                self._connected = True
                logger.info("RobotIOClient connected to %s", self.observation_address)
                return

        raise TimeoutError(
            f"RobotIOClient timed out waiting for observations from {self.observation_address}. "
            "Ensure robot_io_server is running."
        )

    def _observation_receiver_loop(self) -> None:
        zmq = self._zmq
        while not self._receiver_stop.is_set():
            if self._observation_socket is None:
                return
            try:
                payload = self._observation_socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue

            envelope = deserialize_message(payload)
            with self._observation_lock:
                self._latest_envelope = envelope
                self._latest_observation = envelope["observation"]
            self._new_observation_event.set()

    def get_observation(self, timeout_s: float | None = None, latest: bool = True) -> dict[str, Any]:
        del latest
        if not self._connected:
            raise RuntimeError("RobotIOClient is not connected")

        if self._latest_observation is None:
            wait_timeout = self.observation_timeout_s if timeout_s is None else timeout_s
            has_observation = self._new_observation_event.wait(timeout=wait_timeout)
            if not has_observation:
                raise TimeoutError(
                    f"No observation received in {wait_timeout:.2f}s from {self.observation_address}"
                )

        with self._observation_lock:
            if self._latest_observation is None:
                raise RuntimeError("Observation receiver did not cache any frame")
            return self._latest_observation

    def send_action(self, action: Any) -> Any:
        if not self._connected:
            raise RuntimeError("RobotIOClient is not connected")
        normalized_action = normalize_action_for_transport(action)
        assert self._action_socket is not None
        self._action_socket.send(serialize_message({"action": normalized_action, "t_client": time.time()}))
        return normalized_action

    def disconnect(self) -> None:
        if not self._connected and self._context is None:
            return

        self._receiver_stop.set()
        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=1.0)

        if self._metadata_socket is not None:
            self._metadata_socket.close()
            self._metadata_socket = None
        if self._observation_socket is not None:
            self._observation_socket.close()
            self._observation_socket = None
        if self._action_socket is not None:
            self._action_socket.close()
            self._action_socket = None

        if self._context is not None:
            self._context.term()
            self._context = None

        self._connected = False
