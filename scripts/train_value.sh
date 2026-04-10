# initial for ACP
export HF_HOME=/data/dataset/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=/tmp/qingyunpeng_vla_cache/datasets
export WANDB_API_KEY=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
source /data/users/kongyilun/miniconda3/bin/activate
conda activate evo-rl
cd /data/users/qingyunpeng/code/Evo-RL


# 多卡训
USE_MULTI_GPU=true \
CUDA_VISIBLE_DEVICES_VALUE=0,1,2,3,4,5,6,7 \
NUM_PROCESSES=8 \
BATCH_SIZE=32 \
REPO_ID=ace_fold_cloth_v2 \
RUN_NAME=value_fcv2 \
WANDB_ENABLE=true \
OUTPUT_DIR=outputs/value_train/value_ace_fold_cloth_v2 \
bash scripts/train_v.sh \
--steps=40000
sleep inf # for debug!

#多机多卡
NUM_MACHINES=2 \
MACHINE_RANK=$SENSECORE_PYTORCH_NODE_RANK \
MAIN_PROCESS_IP=$MASTER_ADDR \
MAIN_PROCESS_PORT=29500 \
NUM_PROCESSES=4 \
CUDA_VISIBLE_DEVICES_VALUE=0,1,2,3 \
BATCH_SIZE=64 \
REPO_ID=ace_fold_cloth_v2 \
RUN_NAME=value_fcv2 \
WANDB_ENABLE=true \
OUTPUT_DIR=outputs/value_train/value_multinode \
bash scripts/train_value_multinode.sh --steps=40000
