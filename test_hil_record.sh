#!/usr/bin/env bash
set -euo pipefail

lerobot-setup-can --mode=setup --interfaces=can_left,can_back_left,can_right,can_back_right

# PROMPT="hang clothes on the hanger"
PROMPT="fold clothes"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PROMPT_SLUG="$(printf '%s' "$PROMPT" | tr '[:upper:]' '[:lower:]' | sed 's/[^[:alnum:]]\+/_/g; s/^_//; s/_$//')"
DATASET_NAME="${PROMPT_SLUG}_${TIMESTAMP}"
DATASET_BASE_DIR="${HOME}/evorl_dataset"
DATASET_ROOT="${DATASET_BASE_DIR}/${DATASET_NAME}"
DATASET_REPO_ID="ACE_ROBOTICS/eval_${DATASET_NAME}"

mkdir -p "${DATASET_BASE_DIR}"

SUFFIX=1
while [ -e "${DATASET_ROOT}" ]; do
  DATASET_NAME="${PROMPT_SLUG}_${TIMESTAMP}_${SUFFIX}"
  DATASET_ROOT="${DATASET_BASE_DIR}/${DATASET_NAME}"
  DATASET_REPO_ID="ACE_ROBOTICS/eval_${DATASET_NAME}"
  SUFFIX=$((SUFFIX + 1))
done

echo "Saving dataset to: ${DATASET_ROOT}"

args=(
  --robot.type=bi_piper_follower
  --robot.id=my_bi_piper_follower
  --robot.left_arm_config.port=can_left
  --robot.right_arm_config.port=can_right
  --robot.left_arm_config.require_calibration=false
  --robot.right_arm_config.require_calibration=false
  --robot.left_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "243322070942", width: 640, height: 480, fps: 30, warmup_s: 2}}'
  --robot.right_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "243722071316", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "239622301704", width: 640, height: 480, fps: 30, warmup_s: 2}}'
  --teleop.type=bi_piper_leader
  --teleop.id=my_bi_piper_leader
  --teleop.left_arm_config.port=can_back_left
  --teleop.right_arm_config.port=can_back_right
  --teleop.left_arm_config.require_calibration=false
  --teleop.right_arm_config.require_calibration=false
  --policy.type=remote_client
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.single_task="${PROMPT}"
  --dataset.num_episodes=30
  --dataset.episode_time_s=200
  --dataset.reset_time_s=20
  --dataset.push_to_hub=false
  --display_data=true
  --play_sounds=false
  --policy.policy_name=ace_policy
  --policy.host=103.237.28.254
  --policy.port=3336
  --policy.chunk_size=50
  --policy.n_action_steps=24
)

lerobot-human-inloop-record "${args[@]}"
