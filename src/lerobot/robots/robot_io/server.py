from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.robot_io.serialization import deserialize_message, serialize_message

logger = logging.getLogger(__name__)


@dataclass
class RobotIOServerConfig:
    robot: Any
    observation_publish_address: str = "tcp://127.0.0.1:5555"
    action_pull_address: str = "tcp://127.0.0.1:5556"
    metadata_rep_address: str = "tcp://127.0.0.1:5557"
    frequency: float = 30.0
    print_fps: bool = False
    stale_action_timeout_s: float | None = None


class RobotIOServer:
    def __init__(self, cfg: RobotIOServerConfig):
        import zmq

        self.cfg = cfg
        self._zmq = zmq
        self.robot = make_robot_from_config(cfg.robot)

        self._context = zmq.Context()

        self._obs_socket = self._context.socket(zmq.PUB)
        self._obs_socket.setsockopt(zmq.SNDHWM, 5)
        self._obs_socket.setsockopt(zmq.LINGER, 0)
        self._obs_socket.bind(cfg.observation_publish_address)

        self._action_socket = self._context.socket(zmq.PULL)
        self._action_socket.setsockopt(zmq.RCVHWM, 5)
        self._action_socket.setsockopt(zmq.LINGER, 0)
        self._action_socket.bind(cfg.action_pull_address)

        self._meta_socket = self._context.socket(zmq.REP)
        self._meta_socket.setsockopt(zmq.LINGER, 0)
        self._meta_socket.bind(cfg.metadata_rep_address)

        self._latest_action_lock = threading.Lock()
        self._latest_action: Any = None
        self._latest_action_time_s: float | None = None

        self._stop_event = threading.Event()
        self._action_thread: threading.Thread | None = None
        self._metadata_thread: threading.Thread | None = None

        self._seq = 0
        self._obs_count = 0
        self._action_recv_count = 0
        self._action_apply_count = 0
        self._dropped_stale_actions = 0

    def _metadata_loop(self) -> None:
        zmq = self._zmq
        camera_keys = list(getattr(self.robot, "cameras", {}).keys())
        metadata = {
            "name": self.robot.name,
            "robot_type": getattr(self.robot, "robot_type", self.robot.name),
            "action_features": self.robot.action_features,
            "observation_features": self.robot.observation_features,
            "camera_keys": camera_keys,
        }
        while not self._stop_event.is_set():
            try:
                request = self._meta_socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.005)
                continue
            _ = deserialize_message(request)
            self._meta_socket.send(serialize_message(metadata))

    def _action_loop(self) -> None:
        zmq = self._zmq
        while not self._stop_event.is_set():
            try:
                payload = self._action_socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue

            message = deserialize_message(payload)
            with self._latest_action_lock:
                self._latest_action = message["action"]
                self._latest_action_time_s = message.get("t_client", time.time())
            self._action_recv_count += 1

    def _take_latest_action(self) -> Any:
        with self._latest_action_lock:
            action = self._latest_action
            action_time_s = self._latest_action_time_s
            self._latest_action = None
            self._latest_action_time_s = None

        if action is None:
            return None

        if self.cfg.stale_action_timeout_s is None:
            return action

        if action_time_s is None:
            return action

        age_s = time.time() - action_time_s
        if age_s > self.cfg.stale_action_timeout_s:
            self._dropped_stale_actions += 1
            return None

        return action

    def run(self) -> None:
        logger.info(
            "Robot IO Server starting | obs=%s action=%s meta=%s freq=%.2fHz",
            self.cfg.observation_publish_address,
            self.cfg.action_pull_address,
            self.cfg.metadata_rep_address,
            self.cfg.frequency,
        )

        self.robot.connect()
        logger.info("Robot initialized: %s", self.robot.name)

        self._action_thread = threading.Thread(target=self._action_loop, daemon=True, name="RobotIOActionRecv")
        self._metadata_thread = threading.Thread(target=self._metadata_loop, daemon=True, name="RobotIOMeta")
        self._action_thread.start()
        self._metadata_thread.start()

        loop_period_s = 1.0 / self.cfg.frequency
        last_print_t = time.monotonic()

        try:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()
                observation = self.robot.get_observation()
                envelope = {
                    "seq": self._seq,
                    "t_server": time.time(),
                    "observation": observation,
                }
                self._obs_socket.send(serialize_message(envelope))
                self._seq += 1
                self._obs_count += 1

                latest_action = self._take_latest_action()
                if latest_action is not None:
                    _ = self.robot.send_action(latest_action)
                    self._action_apply_count += 1

                if self.cfg.print_fps:
                    now = time.monotonic()
                    if now - last_print_t >= 1.0:
                        logger.info(
                            "obs_hz=%d action_recv=%d action_apply=%d stale_drop=%d",
                            self._obs_count,
                            self._action_recv_count,
                            self._action_apply_count,
                            self._dropped_stale_actions,
                        )
                        self._obs_count = 0
                        self._action_recv_count = 0
                        self._action_apply_count = 0
                        self._dropped_stale_actions = 0
                        last_print_t = now

                elapsed = time.perf_counter() - t0
                sleep_s = loop_period_s - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()

        if self._action_thread is not None:
            self._action_thread.join(timeout=1.0)
        if self._metadata_thread is not None:
            self._metadata_thread.join(timeout=1.0)

        self.robot.disconnect()

        self._obs_socket.close()
        self._action_socket.close()
        self._meta_socket.close()
        self._context.term()
        logger.info("Robot IO Server shutdown complete")
