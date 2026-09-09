# Real-robot Evo-RL server / client

## 结论：config 已统一到 `openpi_local`

现在 server 不再依赖 `openpi-main` 的 Python import。已经在
`/data/users/liujingyuan/workspace/openpi_local/src/openpi/training/config.py`
中加入了四个 policy config：

```text
pi05_jax_fc_v0
pi05_jax_vase_and_flowers_fc_v0
pi05_jax_stack_plates
pi05_jax_cube_v0
```

它们只描述 policy；monitor 仍然是独立 config。不要把 policy config 和
monitor config 写成同一个名字。

```text
task           policy.config                         monitor.config
arrange_flower pi05_jax_vase_and_flowers_fc_v0       pi05_real_robot_arrange_flower_predictor
fold_cloth     pi05_jax_fc_v0                         pi05_concat_fold_cloth_obs1_interval1
stack_plates   pi05_jax_stack_plates                  pi05_real_robot_stack_plates_predictor
pick_cubes     pi05_jax_cube_v0                        pi05_real_robot_pick_cubes_predictor
```

`policy.dir` 是 policy checkpoint，`monitor.dir` 是 predictor monitor checkpoint。
它们分别加载，不能互换。

## 1. Server 启动

在 server 机器上：

```bash
cd /data/users/liujingyuan/workspace/openpi_local
source /data/anaconda3/etc/profile.d/conda.sh
conda activate pose
export OPENPI_ROOT=/data/users/liujingyuan/workspace/openpi_local
export PYTHONPATH="$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1
export DINOV2_LARGE_LOCAL_PATH=/data/dataset/hub/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c

# Optional Pi-Prob/SAFE baseline monitors. Keep disabled unless all paths below
# are supplied; the normal detector/predictor path does not require them.
export PI_PROB_ROOT=/data/users/liujingyuan/workspace/pi_prob
export SAFE_ROOT=/data/users/liujingyuan/third_party/SAFE
export REAL_ROBOT_INFERENCE_ONLY=1
export REAL_ROBOT_RUNTIME_REPO=trossen
```

先确认 policy 和 monitor 在同一个注册表中：

```bash
python - <<'PY'
from openpi.training import config
names = [
    "pi05_jax_vase_and_flowers_fc_v0",
    "pi05_real_robot_arrange_flower_predictor",
]
for name in names:
    cfg = config.get_config(name)
    print(name, type(cfg.model).__name__, type(cfg.monitor).__name__)
PY
```

如果这里报 `Config not found`，说明 `PYTHONPATH` 没有指向
`openpi_local/src`，不要再把 `openpi-main/src` 拼到前面。

### arrange_flower 示例

```bash
python scripts/serve_policy_with_monitor.py \
  --host=10.119.16.248 \
  --policy_port=8088 \
  --serve_mode=PREDICTOR_ONLY \
  --default_prompt="arrange the flower" \
  --policy.config=pi05_jax_vase_and_flowers_fc_v0 \
  --policy.dir=/data/users/qingyunpeng/code/openpi-main/checkpoints/pi05_jax_vase_and_flowers_fc_v0/SFT_ppi05_jax_vase_and_flowers_fc_v0/50000 \
  --monitor.config=pi05_real_robot_arrange_flower_predictor \
  --monitor.dir=/data/users/liujingyuan/data/monitor_experiments/real_robot_predictor/checkpoints/pi05_real_robot_arrange_flower_predictor/60_real_robot_predictor_arrange_flower_seed42/obs1_interval1/monitor/00009999.msgpack \
  --monitor.predictor_band_path=/data/users/liujingyuan/data/monitor_experiments/real_robot_or_bands/arrange_flower/predictor_band.npz \
  --monitor.task_stats_path=/data/users/liujingyuan/data/monitor_experiments/real_robot_or_bands/arrange_flower/task_stats.json \
  --monitor.task_index=0 \
  --monitor.band_alpha=0.02 \
  --history_len_prediction=1 \
  --conformal_timestep_frequency=30 \
  --action_selection.effective_action_steps=36 \
  --action_selection.effective_action_dims=14
```

其它任务只替换以下参数：

```text
stack_plates:
  --default_prompt="stack plates"
  --policy.config=pi05_jax_stack_plates
  --monitor.config=pi05_real_robot_stack_plates_predictor
  --monitor.dir=.../pi05_real_robot_stack_plates_predictor/.../monitor/00009999.msgpack
  --monitor.predictor_band_path=.../real_robot_or_bands/stack_plates/predictor_band.npz
  --monitor.task_stats_path=.../real_robot_or_bands/stack_plates/task_stats.json

pick_cubes:
  --default_prompt="pick cubes"
  --policy.config=pi05_jax_cube_v0
  --monitor.config=pi05_real_robot_pick_cubes_predictor
  --monitor.dir=.../pi05_real_robot_pick_cubes_predictor/.../monitor/00009999.msgpack
  --monitor.predictor_band_path=.../real_robot_or_bands/pick_cubes/predictor_band.npz
  --monitor.task_stats_path=.../real_robot_or_bands/pick_cubes/task_stats.json

fold_cloth:
  --default_prompt="fold clothes"
  --policy.config=pi05_jax_fc_v0
  --monitor.config=pi05_concat_fold_cloth_obs1_interval1
  --monitor.dir=.../pi05_concat_fold_cloth_obs1_interval1/.../monitor/00009999.msgpack
  --monitor.predictor_band_path=.../real_robot_or_bands/fold_cloth/predictor_band.npz
  --monitor.task_stats_path=.../real_robot_or_bands/fold_cloth/task_stats.json
```

## 2. Client：完整启动命令

### 端口说明

client 里的：

```text
--policy.predictor_remote.port=63334
```

只需要一个 remote port，因为当前 client 使用的是 `PREDICTOR_ONLY` server：
`BrainServer` 在这个端口一次完成 policy action 和 predictor risk。monitor 不会
再单独通过 websocket 暴露给 client。

server 端对应关系是：

```text
--policy_port=63334   -> BrainServer：policy action + predictor risk
--monitor_port=63335  -> DetectionServer：只有 serve_mode=BOTH/DETECTOR_ONLY 时才启动
```

本方案使用 local detector，所以不连接 `63335`：detector checkpoint 在 client
本地加载，client 不需要 `--policy.detector_remote.host/port`。只有把 detector 也
改成远程服务时，才启动：

```bash
--serve_mode=BOTH --policy_port=63334 --monitor_port=63335
```

并在 client 中额外配置 `--policy.detector_remote.host` 和
`--policy.detector_remote.port`。

先在 robot 机器的另一个终端启动 RobotIO server（它独占真实机器人、CAN 和相机）：

```bash
conda activate evork-ljy
lerobot-setup-can --mode=setup --interfaces=can_left,can_right

python scripts/robot_io_server.py \
  --robot.type=bi_piper_follower \
  --robot.id=my_bi_piper_follower \
  --robot.left_arm_config.port=can_left \
  --robot.right_arm_config.port=can_right \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --robot.left_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "346522074444", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "346522074314", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "239622301704", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --obs_pub_address=tcp://127.0.0.1:5555 \
  --action_pull_address=tcp://127.0.0.1:5556 \
  --meta_rep_address=tcp://127.0.0.1:5557 \
  --frequency=30 \
  --print_fps=true
```

然后在 robot 机器的另一个终端启动 Evo-RL client。下面是
`arrange_flower` 的完整命令，`<SERVER_IP>` 替换成 predictor server 的 IP：

```bash
conda activate evorl-ljy

export OPENPI_ROOT=/home/agilex/evorl-ljy/pizero
export PYTHONPATH="$PWD/src:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1
export DINOV2_LARGE_LOCAL_PATH=dinov2-large
export REAL_ROBOT_INFERENCE_ONLY=1
export REAL_ROBOT_RUNTIME_REPO=trossen

lerobot-record-monitor-local-detector \
  --distributed_robot_io=true \
  --robot_io_obs_address=tcp://127.0.0.1:5555 \
  --robot_io_action_address=tcp://127.0.0.1:5556 \
  --robot_io_meta_address=tcp://127.0.0.1:5557 \
  --robot.type=bi_piper_follower \
  --robot.id=my_bi_piper_follower \
  --robot.left_arm_config.port=can_left \
  --robot.right_arm_config.port=can_right \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --robot.left_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "346522074444", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "346522074314", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "239622301704", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --policy.type=remote_monitor \
  --policy.predictor_remote.policy_name=pi05 \
  --policy.predictor_remote.chunk_size=50 \
  --policy.predictor_remote.n_action_steps=24 \
  --policy.predictor_remote.host=103.237.28.254 \
  --policy.predictor_remote.port=63333 \
  --local_detector.monitor_config=pi05_real_robot_arrange_flower_evorl_detector \
  --local_detector.monitor_dir=detector/arrange_flower_detector.msgpack \
  --local_detector.detector_conformal_path=bands/arrange_flower_bands \
  --local_detector.task_stats_path=stats/arrange_flower.json \
  --local_detector.task_index=0 \
  --local_detector.task_length_subset=all \
  --local_detector.history_len_detection=1 \
  --local_detector.conformal_timestep_frequency=30 \
  --dataset.repo_id=arrange_flower \
  --dataset.root=~/evorl_dataset/arrange_flower \
  --dataset.single_task="arrange flower" \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=200 \
  --dataset.reset_time_s=0 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false \
  --test_mode=false
```

client 不传 `predictor_band`；predictor risk 在 server 端计算。本地 detector 使用
`detector checkpoint + detector_safe_score + task_stats.json`。

detector 参数对应关系：

```text
arrange_flower:
  detector_config=pi05_real_robot_arrange_flower_evorl_detector
  detector_checkpoint=.../real_robot_arrange_flower_evorl_detector/.../monitor/00004000.msgpack
  detector_band=.../real_robot_or_bands/arrange_flower/detector_safe_score

stack_plates:
  detector_config=pi05_real_robot_stack_plates_evorl_detector
  detector_checkpoint=.../real_robot_stack_plates_evorl_detector/.../monitor/00002000.msgpack
  detector_band=.../real_robot_or_bands/stack_plates/detector_safe_score

pick_cubes:
  detector_config=pi05_real_robot_pick_cubes_evorl_detector
  detector_checkpoint=.../real_robot_pick_cubes_evorl_detector/.../monitor/00004000.msgpack
  detector_band=.../real_robot_or_bands/pick_cubes/detector_safe_score

fold_cloth:
  detector_config=pi05_real_robot_temporal_bce_detector
  detector_checkpoint=.../real_robot_temporal_bce_detector/.../monitor/00009999.msgpack
  detector_band=.../real_robot_or_bands/fold_cloth/detector_safe_score
```

stack/pick/fold 使用上面同一条 client 命令，只替换以下字段：

```text
stack_plates:
  --policy.predictor_remote.port=9991
  --local_detector.monitor_config=pi05_real_robot_stack_plates_evorl_detector
  --local_detector.monitor_dir=.../pi05_real_robot_stack_plates_evorl_detector/.../monitor/00002000.msgpack
  --local_detector.detector_conformal_path=.../real_robot_or_bands/stack_plates/detector_safe_score
  --local_detector.task_stats_path=.../real_robot_or_bands/stack_plates/task_stats.json
  --dataset.single_task="stack plates"
  --dataset.root=$HOME/evorl_dataset/stack_plates

pick_cubes:
  --local_detector.monitor_config=pi05_real_robot_pick_cubes_evorl_detector
  --local_detector.monitor_dir=.../pi05_real_robot_pick_cubes_evorl_detector/.../monitor/00004000.msgpack
  --local_detector.detector_conformal_path=.../real_robot_or_bands/pick_cubes/detector_safe_score
  --local_detector.task_stats_path=.../real_robot_or_bands/pick_cubes/task_stats.json
  --dataset.single_task="pick cubes"
  --dataset.root=$HOME/evorl_dataset/pick_cubes

fold_cloth:
  --local_detector.monitor_config=pi05_real_robot_temporal_bce_detector
  --local_detector.monitor_dir=.../pi05_real_robot_temporal_bce_detector/.../monitor/00009999.msgpack
  --local_detector.detector_conformal_path=.../real_robot_or_bands/fold_cloth/detector_safe_score
  --local_detector.task_stats_path=.../real_robot_or_bands/fold_cloth/task_stats.json
  --dataset.single_task="fold clothes"
  --dataset.root=$HOME/evorl_dataset/fold_cloth
```

`detector_head_dir` 对当前完整 detector checkpoint 不需要传；只有 backbone 和
head 分开保存时才使用它。`task_stats_path` 用来把实际运行 timestep 映射到
训练得到的 episode-length band；因此 detector 端也应传入，不能省略。

## 3. 不能做的事情

不要这样启动：

```bash
export PYTHONPATH=/data/users/qingyunpeng/code/openpi-main/src:...
```

然后期待 `openpi_local` 的 monitor config 自动出现。两个目录都提供
`openpi` 包时，Python 只会加载排在最前面的一个包，不能自动合并两个
`training.config` 注册表。当前解决方案就是使用已经补齐 policy aliases 的
`openpi_local`，让 policy 和 monitor 在同一份 config.py 中解析。

启动成功的关键日志应包含：

```text
Starting Brain Server ... on port 9991
Local detector warmup complete
```

## 4. Policy server 上的 Pi-Prob baseline（可选）

Pi-Prob baseline 和 predictor monitor 在同一个 policy server 进程中运行。每次
请求复用 policy 的输入，在返回包中增加：

```text
baseline_risks.indep
baseline_risks.logpzo
baseline_risks.rnd
baseline_risks.accel
```

每一项都包含 `score`、`threshold`、`is_dangerous` 和 `timestep`。其中 indep/logpzo
使用 Pi0.5 的 `get_vlm_embedding()` 特征，rnd 额外使用 policy action chunk，accel
运行 deterministic flow path 后计算 smoothness。

server 脚本和 adapter：

```text
/data/users/liujingyuan/workspace/openpi_local/scripts/serve_policy_with_monitor.py
/data/users/liujingyuan/workspace/openpi_local/scripts/pi_prob_baseline_runtime.py
```

在 server 命令中额外加入（以 arrange_flower 为例）：

```bash
--baselines.enabled=true \
--baselines.device=cuda \
--baselines.feature_dim=2048 \
--baselines.feature_pool=mean_h \
--baselines.action_dim=14 \
--baselines.action_horizon=36 \
--baselines.alpha=0.1 \
--baselines.reference_episode_length=1000 \
--baselines.indep.config_path=/data/users/liujingyuan/workspace/pi_prob/outputs/manifest_baselines/arrange_flower/seed_42/indep/train_logs/config.yaml \
--baselines.indep.checkpoint_path=/data/users/liujingyuan/workspace/pi_prob/outputs/manifest_baselines/arrange_flower/seed_42/indep/train_logs/model_final.ckpt \
--baselines.indep.band_path=/PATH/TO/arrange_flower/indep/classify_cp_functional__model_bands.json \
--baselines.logpzo.config_path=/data/users/liujingyuan/workspace/pi_prob/outputs/manifest_baselines/arrange_flower/seed_42/logpzo/train_logs/config.yaml \
--baselines.logpzo.checkpoint_path=/data/users/liujingyuan/workspace/pi_prob/outputs/manifest_baselines/arrange_flower/seed_42/logpzo/train_logs/model_final.ckpt \
--baselines.logpzo.band_path=/PATH/TO/arrange_flower/logpzo/classify_cp_functional__model_bands.json \
--baselines.rnd.config_path=/data/users/liujingyuan/workspace/pi_prob/outputs/manifest_baselines/arrange_flower/seed_42/rnd/train_logs/config.yaml \
--baselines.rnd.checkpoint_path=/data/users/liujingyuan/workspace/pi_prob/outputs/manifest_baselines/arrange_flower/seed_42/rnd/train_logs/model_final.ckpt \
--baselines.rnd.band_path=/PATH/TO/arrange_flower/rnd/classify_cp_functional__model_bands.json \
--baselines.accel_band_path=/PATH/TO/arrange_flower/accel/classify_cp_functional__model_bands.json \
--baselines.accel_prefix_steps=9 \
--baselines.fm_num_steps=10
```

当前 baseline 实验目录里已有 indep/logpzo/rnd checkpoint；accel 还没有拟合 band，
所以 `accel_band_path` 暂时不要传，或保持空字符串。baseline 未配置时，server
完全不加载 PyTorch/SAFE 模型，不影响原来的 predictor+detector 部署。

baseline adapter 读取的 band 可以是 JSON 文件，也可以是包含以下文件的目录：

```text
json/classify_cp_functional__model_bands.json
```

## 5. 可视化已保存的 LeRobot monitor episode

`visualize_real_robot_monitor_dataset.py` 读取标准 LeRobot v3 数据集，不读取
HDF5。数据集目录必须直接包含 `meta/info.json`、`data/` 和 `videos/`，并且需要
包含 recorder 写入的 `complementary_info.predictor_*` 和
`complementary_info.detector_*` 字段。

脚本路径：

```text
/data/users/liujingyuan/workspace/Evo-RL/scripts/visualize_real_robot_monitor_dataset.py
```

运行单个 episode：

```bash
cd /data/users/liujingyuan/workspace/Evo-RL
source /data/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

PYTHONPATH=$PWD/src \
MPLCONFIGDIR=/tmp/mpl \
python scripts/visualize_real_robot_monitor_dataset.py \
  --dataset-path /PATH/TO/lerobot_dataset \
  --episode 0 \
  --output /PATH/TO/episode_0_monitor.mp4 \
  --fps 30
```

脚本会自动选择包含 `right_front`、`front`、`cam_high`、`head` 或 `global` 的
相机字段。需要指定相机时加入：

```bash
--camera-key observation.images.right_front
```

输出视频布局为：左侧真实头部/前置相机，右上 predictor risk 与 predictor band，
右下 detector risk 与 detector band。横轴使用 `control_timestep`，标题中同时显示
对应的 predictor/detector prediction timestep。
