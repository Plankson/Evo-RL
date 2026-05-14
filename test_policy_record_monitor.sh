#!/usr/bin/env bash
set -euo pipefail

lerobot-setup-can --mode=setup --interfaces=can_left,can_right

# PROMPT="wipe the table with the towel"
PROMPT="fold clothes"
# PROMPT="Zip up the zipper of the clothes"
# PROMPT="hang clothes on the hanger"
# PROMPT="PUT THE CUBES INTO BUCKET"
# PROMPT="PUSH OBJECTS WITH MARKER"
# PROMPT="POUR WATER FROM ONE CUP INTO ANOTHER CUP"
# PROMPT="WIPE THE TABLE WITH THE TOWEL"
# PROMPT="BAG ITEMS INTO PAPER BAG"
# PROMPT="PUT THE PEN INTO THE PEN HOLDER"

POLICY_NAME="pi05"
# POLICY_NAME="ace_policy"
# PORT=9991
# PORT=8080
TAG="policy_only"
TESTMODE="true"

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
  rm -rf "${DATASET_ROOT}"
  trap 'rm -rf "${DATASET_ROOT}"' EXIT
else
  DATASET_BASE_DIR="${HOME}/evorl_dataset/${POLICY_NAME}/${DAY_FOLDER}"
  DATASET_ROOT="${DATASET_BASE_DIR}/${DATASET_NAME}"
fi

DATASET_REPO_ID="ACE_ROBOTICS/${POLICY_NAME}_${DAY_FOLDER}_${DATASET_NAME}"
mkdir -p "${DATASET_BASE_DIR}"

echo "Saving dataset to: ${DATASET_ROOT}"
echo "Policy-only record: follower arms + remote policy, no leader/master arms."
if [ -n "${TAG}" ]; then
  echo "Recording tag: ${TAG}"
fi
if [ "${TESTMODE}" = "true" ]; then
  echo "Test mode enabled: this run will not persist any saved data."
fi

args=(
  --distributed_robot_io=true \
  --robot_io_obs_address=tcp://127.0.0.1:5555 \
  --robot_io_action_address=tcp://127.0.0.1:5556 \
  --robot_io_meta_address=tcp://127.0.0.1:5557 \
  --robot.type=bi_piper_follower
  --robot.id=my_bi_piper_follower
  --robot.left_arm_config.port=can_left
  --robot.right_arm_config.port=can_right
  --robot.left_arm_config.require_calibration=false
  --robot.right_arm_config.require_calibration=false
  --robot.left_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "243322070942", width: 640, height: 480, fps: 30, warmup_s: 2}}'
  --robot.right_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "243722071316", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "239622301704", width: 640, height: 480, fps: 30, warmup_s: 2}}'
  --policy.policy_name="${POLICY_NAME}"
  --policy.chunk_size=50
  --policy.n_action_steps=24
  --policy.type=remote_monitor \
  --policy.predictor_remote.host=103.237.28.254 \
  --policy.predictor_remote.port=9991 \
#   --policy.detector_remote.host=103.237.28.254 \
#   --policy.detector_remote.port=51888 \
  --local_detector.monitor_config=pi05_monitor_debug \
  --local_detector.monitor_dir=/home/agilex/evorl-ljy/monitor.msgpack \
  --local_detector.detector_conformal_path=/home/agilex/evorl-ljy/safe_score \
  --local_detector.detector_head_dir=/home/agilex/evorl-ljy/detector.msgpack \
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.single_task="${PROMPT}"
  --dataset.num_episodes=20
  --dataset.episode_time_s=200
  --dataset.reset_time_s=0
  --dataset.push_to_hub=false
  --display_data=true
  --play_sounds=false
  --test_mode="${TESTMODE}"
)

lerobot-record-monitor-local-detector "${args[@]}"
