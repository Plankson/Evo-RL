# 多卡训
USE_MULTI_GPU=true \
CUDA_VISIBLE_DEVICES_VALUE=0,1,2,3,4,5,6,7 \
NUM_PROCESSES=8 \
BATCH_SIZE=32 \
REPO_ID=ace_fold_cloth_v2 \
RUN_NAME=value_ace_fold_cloth_v2 \
WANDB_ENABLE=true \
OUTPUT_DIR=outputs/value_train/value_ace_fold_cloth_v2 \
bash scripts/train_value.sh \
--steps=40000
sleep inf # for debug!