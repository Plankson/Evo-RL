#!/usr/bin/env python

import logging
from dataclasses import dataclass

from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_openarm_follower,
    bi_piper_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    omx_follower,
    openarm_follower,
    piper_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.robots.robot_io import RobotIOServer, RobotIOServerConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class RobotIOServerCLIConfig:
    robot: RobotConfig
    obs_pub_address: str = "tcp://127.0.0.1:5555"
    action_pull_address: str = "tcp://127.0.0.1:5556"
    meta_rep_address: str = "tcp://127.0.0.1:5557"
    frequency: float = 30.0
    print_fps: bool = False
    dry_run: bool = False
    stale_action_timeout_s: float | None = None


@parser.wrap()
def run_server(cfg: RobotIOServerCLIConfig) -> None:
    init_logging()
    server_cfg = RobotIOServerConfig(
        robot=cfg.robot,
        observation_publish_address=cfg.obs_pub_address,
        action_pull_address=cfg.action_pull_address,
        metadata_rep_address=cfg.meta_rep_address,
        frequency=cfg.frequency,
        print_fps=cfg.print_fps,
        stale_action_timeout_s=cfg.stale_action_timeout_s,
    )
    server = RobotIOServer(server_cfg)

    if cfg.dry_run:
        logging.info("Dry-run enabled; configuration parsed and server initialized.")
        server.shutdown()
        return

    try:
        server.run()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, shutting down server.")
        server.shutdown()


def main() -> None:
    register_third_party_plugins()
    run_server()


if __name__ == "__main__":
    main()
