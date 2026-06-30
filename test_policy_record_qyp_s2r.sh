#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

lerobot-setup-can --mode=setup --interfaces=can_left,can_right

# PROMPT="wipe the table with the towel"
# PROMPT="fold clothes"
# PROMPT="Zip up the zippers of the clothes"
# PROMPT="hang clothes on the hanger"
# PROMPT="PUT THE CUBES INTO BUCKET"
# PROMPT="PUSH OBJECTS WITH MARKER"
# PROMPT="POUR WATER FROM ONE CUP INTO ANOTHER CUP"
# PROMPT="WIPE THE TABLE WITH THE TOWEL"
PROMPT="BAG ITEMS INTO PAPER BAG"
# PROMPT="PUT THE PEN INTO THE PEN HOLDER"

POLICY_NAME="openvla-oft"
# POLICY_NAME="ace_policy"
PORT=8886
# PORT=8080
TAG="policy_only"
TESTMODE="true"
START_STATE="${START_STATE:-flat}"
RESET_POSE_PATH=""
CALIBRATION_DIR="${CALIBRATION_DIR:-${TMPDIR:-/tmp}/evorl_openvla_no_calib}"
CONTROL_FPS="${CONTROL_FPS:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-5}"
PIPER_SPEED_RATIO="${PIPER_SPEED_RATIO:-20}"
PIPER_HIGH_FOLLOW="${PIPER_HIGH_FOLLOW:-false}"
OPENVLA_OFT_SMOOTH_ACTIONS="${OPENVLA_OFT_SMOOTH_ACTIONS:-true}"
OPENVLA_OFT_INTERPOLATE_START_STEPS="${OPENVLA_OFT_INTERPOLATE_START_STEPS:-10}"
OPENVLA_OFT_MAX_JOINT_DELTA_RAD="${OPENVLA_OFT_MAX_JOINT_DELTA_RAD:-0.01}"
OPENVLA_OFT_MAX_GRIPPER_DELTA="${OPENVLA_OFT_MAX_GRIPPER_DELTA:-0.002}"

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
    start_state=*)
      START_STATE="${arg#start_state=}"
      ;;
    --start_state=*)
      START_STATE="${arg#--start_state=}"
      ;;
    reset_pose_path=*)
      RESET_POSE_PATH="${arg#reset_pose_path=}"
      ;;
    --reset_pose_path=*)
      RESET_POSE_PATH="${arg#--reset_pose_path=}"
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      echo "Usage: $0 [tag=<value>] [testmode=true|false] [start_state=reset|flat] [reset_pose_path=<path>]" >&2
      exit 1
      ;;
  esac
done

TESTMODE="$(printf '%s' "${TESTMODE}" | tr '[:upper:]' '[:lower:]')"
if [ "${TESTMODE}" != "true" ] && [ "${TESTMODE}" != "false" ]; then
  echo "Invalid testmode: ${TESTMODE}. Use true or false." >&2
  exit 1
fi

START_STATE="$(printf '%s' "${START_STATE}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
if [ -z "${RESET_POSE_PATH}" ]; then
  case "${START_STATE}" in
    reset)
      RESET_POSE_PATH="${SCRIPT_DIR}/reset_state/reset.json"
      ;;
    flat|default)
      RESET_POSE_PATH="${SCRIPT_DIR}/reset_state/default.json"
      ;;
    *)
      echo "Invalid start_state: ${START_STATE}. Use reset or flat, or pass reset_pose_path=<path>." >&2
      exit 1
      ;;
  esac
fi

if [ ! -f "${RESET_POSE_PATH}" ]; then
  echo "Reset pose file not found: ${RESET_POSE_PATH}" >&2
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
echo "Start state: ${START_STATE}"
echo "Reset pose path: ${RESET_POSE_PATH}"
echo "Calibration dir: ${CALIBRATION_DIR}"
echo "Control fps: ${CONTROL_FPS}"
echo "Policy action steps: ${N_ACTION_STEPS}"
echo "Piper speed ratio: ${PIPER_SPEED_RATIO}"
echo "Piper high follow: ${PIPER_HIGH_FOLLOW}"
echo "OpenVLA-OFT smoothing: ${OPENVLA_OFT_SMOOTH_ACTIONS}"
echo "OpenVLA-OFT max joint delta rad: ${OPENVLA_OFT_MAX_JOINT_DELTA_RAD}"
if [ "${TESTMODE}" = "true" ]; then
  echo "Test mode enabled: this run will not persist any saved data."
fi

args=(
  --robot.type=bi_piper_follower
  --robot.id=my_bi_piper_follower
  --robot.left_arm_config.port=can_left
  --robot.right_arm_config.port=can_right
  --robot.calibration_dir="${CALIBRATION_DIR}"
  --robot.left_arm_config.require_calibration=false
  --robot.right_arm_config.require_calibration=false
  --robot.left_arm_config.speed_ratio="${PIPER_SPEED_RATIO}"
  --robot.right_arm_config.speed_ratio="${PIPER_SPEED_RATIO}"
  --robot.left_arm_config.high_follow="${PIPER_HIGH_FOLLOW}"
  --robot.right_arm_config.high_follow="${PIPER_HIGH_FOLLOW}"
  --robot.left_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "243322070942", width: 640, height: 480, fps: 30, warmup_s: 2}}'
  --robot.right_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "243722071316", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "239622301704", width: 640, height: 480, fps: 30, warmup_s: 2}}'
  --policy.type=remote_client
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.single_task="${PROMPT}"
  --dataset.num_episodes=20
  --dataset.episode_time_s=200
  --dataset.reset_time_s=0
  --dataset.fps="${CONTROL_FPS}"
  --dataset.push_to_hub=false
  --policy_only_reset_pose_path="${RESET_POSE_PATH}"
  --policy_only_reset_duration_s=5
  --display_data=true
  --play_sounds=false
  --test_mode="${TESTMODE}"
  --policy.policy_name="${POLICY_NAME}"
  --policy.host=103.237.28.254
  --policy.port="${PORT}"
  --policy.chunk_size=25
  --policy.n_action_steps="${N_ACTION_STEPS}"
  --policy.openvla_oft_smooth_actions="${OPENVLA_OFT_SMOOTH_ACTIONS}"
  --policy.openvla_oft_interpolate_start_steps="${OPENVLA_OFT_INTERPOLATE_START_STEPS}"
  --policy.openvla_oft_max_joint_delta_rad="${OPENVLA_OFT_MAX_JOINT_DELTA_RAD}"
  --policy.openvla_oft_max_gripper_delta="${OPENVLA_OFT_MAX_GRIPPER_DELTA}"
)

lerobot-record "${args[@]}"
