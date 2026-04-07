#!/usr/bin/env bash
set -euo pipefail

lerobot-setup-can --mode=setup --interfaces=can_left,can_back_left,can_right,can_back_right

# PROMPT="prompt"
# PROMPT="hang clothes on the hanger"
PROMPT="fold clothes"

# POLICY_NAME="pi0"
POLICY_NAME="ace_policy"

TAG="v2_load"
TESTMODE="false"

for arg in "$@"; do
  case "$arg" in
    tag=*)
      TAG="${arg#tag=}"
      ;;
    --tag=*)
      TAG="${arg#--tag=}"
      ;;
    testmode=*)
      TESTMODE="${arg#testmode=}"
      ;;
    --testmode=*)
      TESTMODE="${arg#--testmode=}"
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      echo "Usage: $0 [tag=<value>] [testmode=true|false]" >&2
      exit 1
      ;;
  esac
done

TESTMODE="$(printf '%s' "${TESTMODE}" | tr '[:upper:]' '[:lower:]')"
if [ "${TESTMODE}" != "true" ] && [ "${TESTMODE}" != "false" ]; then
  echo "Invalid testmode: ${TESTMODE}. Use true or false." >&2
  exit 1
fi

DAY_FOLDER="$(date +%m%d)"
PROMPT_SLUG="$(printf '%s' "$PROMPT" | tr '[:upper:]' '[:lower:]' | sed 's/[^[:alnum:]]\+/_/g; s/^_//; s/_$//')"
TAG_SLUG="$(printf '%s' "$TAG" | tr '[:upper:]' '[:lower:]' | sed 's/[^[:alnum:]]\+/_/g; s/^_//; s/_$//')"

DATASET_NAME="${PROMPT_SLUG}"
if [ -n "${TAG_SLUG}" ]; then
  DATASET_NAME="${PROMPT_SLUG}_${TAG_SLUG}"
fi

if [ "${TESTMODE}" = "true" ]; then
  DATASET_BASE_DIR="${TMPDIR:-/tmp}/evorl_dataset_testmode/${POLICY_NAME}/${DAY_FOLDER}"
  DATASET_ROOT="${DATASET_BASE_DIR}/${DATASET_NAME}"
  trap 'rm -rf "${DATASET_ROOT}"' EXIT
else
  DATASET_BASE_DIR="${HOME}/evorl_dataset/${POLICY_NAME}/${DAY_FOLDER}"
  DATASET_ROOT="${DATASET_BASE_DIR}/${DATASET_NAME}"
fi

DATASET_REPO_ID="ACE_ROBOTICS/${POLICY_NAME}_${DAY_FOLDER}_${DATASET_NAME}"
mkdir -p "${DATASET_ROOT}"

echo "Saving dataset to: ${DATASET_ROOT}"
if [ -n "${TAG}" ]; then
  echo "Recording tag: ${TAG}"
fi
if [ "${TESTMODE}" = "true" ]; then
  echo "Test mode enabled: this run will not persist any saved data."
fi

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
  --dataset.num_episodes=20
  --dataset.episode_time_s=200
  --dataset.push_to_hub=false
  --display_data=true
  --play_sounds=false
  --test_mode="${TESTMODE}"
  --policy.policy_name="${POLICY_NAME}"
  --policy.host=103.237.28.254
  --policy.port=10888
  --policy.chunk_size=50
  --policy.n_action_steps=36
)

lerobot-human-inloop-record "${args[@]}"


