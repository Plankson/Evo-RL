#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSE_PATH="${1:-${SCRIPT_DIR}/reset_state/default.json}"
DURATION_S="${2:-5}"

PYTHON_BIN="${PYTHON_BIN:-/home/agilex/miniconda3/envs/evo-rl/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python3"
fi
CONDA_ENV_DIR="$(dirname "$(dirname "${PYTHON_BIN}")")"
SETUP_CAN_BIN="${SETUP_CAN_BIN:-/home/agilex/miniconda3/envs/evo-rl/bin/lerobot-setup-can}"
if [ ! -x "${SETUP_CAN_BIN}" ]; then
  SETUP_CAN_BIN="lerobot-setup-can"
fi

can_ready() {
  local iface="$1"
  ip -details link show "${iface}" 2>/dev/null \
    | grep -q "state ERROR-ACTIVE" \
    && ip -details link show "${iface}" 2>/dev/null | grep -q "bitrate 1000000"
}

if can_ready can_left && can_ready can_right; then
  echo "Follower CAN interfaces are already active."
else
  echo "Configuring follower CAN interfaces..."
  "${SETUP_CAN_BIN}" --mode=setup --interfaces=can_left,can_right
fi

echo "Reset pose: ${POSE_PATH}"
echo "Duration: ${DURATION_S}s"

LD_LIBRARY_PATH="${CONDA_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}" \
"${PYTHON_BIN}" - "${POSE_PATH}" "${DURATION_S}" <<'PY'
from pathlib import Path
import json
import sys
import time

from lerobot.robots.bi_piper_follower.config_bi_piper_follower import BiPiperFollowerConfig
from lerobot.robots.bi_piper_follower.bi_piper_follower import BiPiperFollower
from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfigBase

pose_path = Path(sys.argv[1])
duration_s = float(sys.argv[2])


def load_reset_pose(path: Path) -> dict[str, float]:
    with open(path) as f:
        payload = json.load(f)
    joint_pos_raw = payload["joint_pos"] if isinstance(payload, dict) and "joint_pos" in payload else payload
    if not isinstance(joint_pos_raw, dict):
        raise ValueError(f"Invalid reset pose payload in {path}: expected dict, got {type(joint_pos_raw)}")
    joint_pos = {str(key): float(value) for key, value in joint_pos_raw.items() if str(key).endswith(".pos")}
    if not joint_pos:
        raise ValueError(f"Invalid reset pose payload in {path}: no '.pos' joints found.")
    return joint_pos


def extract_joint_pos(observation: dict) -> dict[str, float]:
    return {key: float(value) for key, value in observation.items() if key.endswith(".pos")}


def slow_move_robot_to_pose(robot, target_pose: dict[str, float], duration_s: float) -> None:
    joint_keys = [key for key in robot.action_features if key.endswith(".pos") and key in target_pose]
    if not joint_keys:
        raise ValueError("No matching '.pos' joints found for reset pose.")

    current_pose = extract_joint_pos(robot.get_observation())
    start_pose = {key: current_pose.get(key, float(target_pose[key])) for key in joint_keys}
    goal_pose = {key: float(target_pose[key]) for key in joint_keys}

    step_dt_s = 0.05
    steps = max(int(duration_s / step_dt_s), 1)
    for idx in range(1, steps + 1):
        alpha = idx / steps
        action = {key: start_pose[key] + (goal_pose[key] - start_pose[key]) * alpha for key in joint_keys}
        robot.send_action(action)
        time.sleep(step_dt_s)


cfg = BiPiperFollowerConfig(
    id="my_bi_piper_follower",
    left_arm_config=PiperFollowerConfigBase(port="can_left", require_calibration=False),
    right_arm_config=PiperFollowerConfigBase(port="can_right", require_calibration=False),
)

robot = BiPiperFollower(cfg)
robot.connect()
try:
    target_pose = load_reset_pose(pose_path)
    print(f"Moving bi_piper_follower to {pose_path} over {duration_s:.1f}s")
    slow_move_robot_to_pose(robot=robot, target_pose=target_pose, duration_s=duration_s)
    print("Done.")
finally:
    robot.disconnect()
PY
