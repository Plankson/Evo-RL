#!/usr/bin/env bash
set -euo pipefail

lerobot-setup-can --mode=setup --interfaces=can_left,can_back_left,can_right,can_back_right

python scripts/robot_io_server.py \
  --robot.type=bi_piper_follower \
  --robot.id=my_bi_piper_follower \
  --robot.left_arm_config.port=can_left \
  --robot.right_arm_config.port=can_right \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --robot.left_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "243322070942", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "243722071316", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "239622301704", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --obs_pub_address=tcp://127.0.0.1:5555 \
  --action_pull_address=tcp://127.0.0.1:5556 \
  --meta_rep_address=tcp://127.0.0.1:5557 \
  --frequency=30 \
  --print_fps=true
