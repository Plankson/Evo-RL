#!/usr/bin/env bash

set -euo pipefail

REPO_ID="${REPO_ID:-ace_fold_cloth_v2}"
RUN_NAME="${RUN_NAME:-value_$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VALUE_DTYPE="${VALUE_DTYPE:-bfloat16}"
VALUE_PUSH_TO_HUB="${VALUE_PUSH_TO_HUB:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/value_train/${RUN_NAME}}"
USE_MULTI_GPU="${USE_MULTI_GPU:-false}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-}"
NUM_PROCESSES="${NUM_PROCESSES:-}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

train_args=(
  "--dataset.repo_id=${REPO_ID}"
  "--value.type=pistar06"
  "--value.dtype=${VALUE_DTYPE}"
  "--value.push_to_hub=${VALUE_PUSH_TO_HUB}"
  "--batch_size=${BATCH_SIZE}"
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${RUN_NAME}"
  "--wandb.enable=${WANDB_ENABLE}"
)

if [[ "${VALUE_PUSH_TO_HUB}" == "true" ]]; then
  : "${VALUE_REPO_ID:?VALUE_REPO_ID must be set when VALUE_PUSH_TO_HUB=true}"
  train_args+=("--value.repo_id=${VALUE_REPO_ID}")
fi

train_args+=("$@")

cmd=()

if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
  cmd+=("env" "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}")
fi

if [[ "${USE_MULTI_GPU}" == "true" ]]; then
  : "${NUM_PROCESSES:?NUM_PROCESSES must be set when USE_MULTI_GPU=true}"
  cmd+=(
    accelerate
    launch
    --multi_gpu
    "--num_processes=${NUM_PROCESSES}"
    "--mixed_precision=${MIXED_PRECISION}"
    "$(which lerobot-value-train)"
  )
else
  cmd+=(lerobot-value-train)
fi

cmd+=("${train_args[@]}")

printf 'Running command:\n%s\n' "${cmd[*]}"
exec "${cmd[@]}"
