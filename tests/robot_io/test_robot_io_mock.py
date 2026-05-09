from __future__ import annotations

import threading
import time

from lerobot.robots.robot_io import RobotIOClient, RobotIOServer, RobotIOServerConfig
from tests.mocks.mock_robot import MockRobotConfig


def test_robot_io_server_client_roundtrip_mock_robot():
    server_cfg = RobotIOServerConfig(
        robot=MockRobotConfig(n_motors=2, random_values=False, static_values=[1.0, -2.0]),
        observation_publish_address="tcp://127.0.0.1:15655",
        action_pull_address="tcp://127.0.0.1:15656",
        metadata_rep_address="tcp://127.0.0.1:15657",
        frequency=20.0,
    )
    server = RobotIOServer(server_cfg)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    client = RobotIOClient(
        observation_address="tcp://127.0.0.1:15655",
        action_address="tcp://127.0.0.1:15656",
        metadata_address="tcp://127.0.0.1:15657",
        observation_timeout_s=3.0,
    )

    try:
        client.connect()
        obs = client.get_observation(timeout_s=3.0)
        assert "motor_1.pos" in obs
        assert "motor_2.pos" in obs

        sent = client.send_action({"motor_1.pos": 3.5, "motor_2.pos": -1.5})
        assert sent["motor_1.pos"] == 3.5

        time.sleep(0.2)
        assert server._action_apply_count > 0
    finally:
        client.disconnect()
        server.shutdown()
        server_thread.join(timeout=2.0)
