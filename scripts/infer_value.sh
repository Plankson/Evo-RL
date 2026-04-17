# initial for ACP
export HF_HOME=/data/dataset/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=/tmp/qingyunpeng_vla_cache/datasets
export WANDB_API_KEY=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
source /data/users/kongyilun/miniconda3/bin/activate
conda activate evo-rl
cd /data/users/qingyunpeng/code/Evo-RL

# DATASET_REPO_ID=/data/dataset/data_fold_cloth_1110
DATASET_REPO_ID=/data/dataset/rollout_lerobot/rollout_lerobot_relabeled
# DATASET_REPO_ID=/data/lnj/ace_data/20260311_lerobot_test


INCLUDE_STATE_IN_PROMPT=false 
# CHECKPOINT_PATH=/data/users/qingyunpeng/code/Evo-RL/outputs/value_train/value_ace_fold_cloth_v3/checkpoints/020000
# CHECKPOINT_PATH=/data/users/qingyunpeng/code/Evo-RL/outputs/value_train/value_ace_fold_cloth_v3/checkpoints/080000
CHECKPOINT_PATH=/data/users/qingyunpeng/code/Evo-RL/outputs/value_train/value_ace_fold_cloth_v4/checkpoints/044000

DATASET_TOLERANCE_S=0.0002
CUDA_VISIBLE_DEVICES=0 \
lerobot-value-infer \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.tolerance_s=${DATASET_TOLERANCE_S} \
  --inference.checkpoint_path=${CHECKPOINT_PATH} \
  --runtime.device=cuda \
  --runtime.batch_size=256 \
  --runtime.image_resize_shape=[480,640] \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field=complementary_info.value_fcv2 \
  --acp.advantage_field=complementary_info.advantage_fcv2 \
  --acp.indicator_field=complementary_info.acp_indicator_fcv2 \
  --output_dir=/data/users/qingyunpeng/code/Evo-RL/outputs/value_infer/value_fcv4_on_vlarollout \
  --job_name=value_fcv2.infer \
  --rename_map='{"observation.images.left": "observation.images.camera1", "observation.images.top": "observation.images.camera2"}' \
  --viz.enable=true \
  --dataset.episodes=[120,121,133]
