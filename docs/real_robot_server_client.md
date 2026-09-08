# Evo-RL 真机 Server / Client 启动

当前部署分成两个进程：

```text
openpi_local server
  └─ 远程 predictor：action + predictor risk，监听 9991

Evo-RL client（真机）
  ├─ websocket 连接 predictor server
  └─ 本地加载 detector checkpoint + detector conformal band
```

`task_stats.json` 和 `predictor_band.npz` 只在 server 端使用；client 端只需要 detector checkpoint 和 `detector_safe_score/`。

## 1. Server：启动 predictor

在 predictor server 机器上执行：

```bash
cd /data/users/liujingyuan/workspace/openpi_local

export PYTHONPATH="/data/users/liujingyuan/workspace/openpi_local/src:/data/users/liujingyuan/workspace/openpi_local/packages/openpi-client/src:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1
```

将 `/PATH/TO/PI05_POLICY_CKPT` 替换成实际的 Pi05 policy checkpoint。`--policy.dir` 是 policy checkpoint，`--monitor.dir` 是 predictor monitor checkpoint，两者不是同一个文件。

### arrange_flower

```bash
python scripts/serve_policy_with_monitor.py \
  --host=0.0.0.0 \
  --policy_port=9991 \
  --monitor_port=9992 \
  --serve_mode=predictor_only \
  --default_prompt="arrange flower" \
  --policy.config=pi05 \
  --policy.dir=/PATH/TO/PI05_POLICY_CKPT \
  --monitor.config=pi05_real_robot_arrange_flower_predictor \
  --monitor.dir=/data/users/liujingyuan/data/monitor_experiments/real_robot_predictor/checkpoints/pi05_real_robot_arrange_flower_predictor/60_real_robot_predictor_arrange_flower_seed42/obs1_interval1/monitor/00009999.msgpack \
  --monitor.predictor_band_path=/data/users/liujingyuan/data/monitor_experiments/real_robot_or_bands/arrange_flower/predictor_band.npz \
  --monitor.task_stats_path=/data/users/liujingyuan/data/monitor_experiments/real_robot_or_bands/arrange_flower/task_stats.json \
  --monitor.task_index=0 \
  --monitor.band_alpha=0.02 \
  --monitor.history_len_prediction=1 \
  --monitor.conformal_timestep_frequency=30
```

### 其他任务的 server 参数

只替换上面命令中的以下三组参数，其他参数保持不变：

#### stack_plates

```text
--default_prompt="stack plates"
--monitor.config=pi05_real_robot_stack_plates_predictor
--monitor.dir=/data/users/liujingyuan/data/monitor_experiments/real_robot_predictor/checkpoints/pi05_real_robot_stack_plates_predictor/60_real_robot_predictor_stack_plates_seed42/obs1_interval1/monitor/00009999.msgpack
--monitor.predictor_band_path=/data/users/liujingyuan/data/monitor_experiments/real_robot_or_bands/stack_plates/predictor_band.npz
--monitor.task_stats_path=/data/users/liujingyuan/data/monitor_experiments/real_robot_or_bands/stack_plates/task_stats.json
```

#### pick_cubes

```text
--default_prompt="pick cubes"
--monitor.config=pi05_real_robot_pick_cubes_predictor
--monitor.dir=/data/users/liujingyuan/data/monitor_experiments/real_robot_predictor/checkpoints/pi05_real_robot_pick_cubes_predictor/60_real_robot_predictor_pick_cubes_seed42/obs1_interval1/monitor/00009999.msgpack
--monitor.predictor_band_path=/data/users/liujingyuan/data/monitor_experiments/real_robot_or_bands/pick_cubes/predictor_band.npz
--monitor.task_stats_path=/data/users/liujingyuan/data/monitor_experiments/real_robot_or_bands/pick_cubes/task_stats.json
```

#### fold_cloth

fold_cloth 的 OR band 需要 predictor `collected_arrays.npz` 生成完成后才可使用。路径准备好后使用：

```text
--default_prompt="fold clothes"
--monitor.config=pi05_concat_fold_cloth_obs1_interval1
--monitor.dir=/data/users/liujingyuan/data/monitor_experiments/real_robot_predictor/checkpoints/pi05_concat_fold_cloth_obs1_interval1/60_real_robot_predictor_fold_cloth_seed42/obs1_interval1/monitor/00009999.msgpack
--monitor.predictor_band_path=/data/users/liujingyuan/data/monitor_experiments/real_robot_or_bands/fold_cloth/predictor_band.npz
--monitor.task_stats_path=/data/users/liujingyuan/data/monitor_experiments/real_robot_or_bands/fold_cloth/task_stats.json
```

Server 正常启动后应看到：

```text
Starting Brain Server ... on port 9991
```

## 2. Client：真机启动 local detector + remote predictor

在真机 Evo-RL 机器上执行。下面以 `arrange_flower` 为例；CAN、RobotIO 和相机配置沿用当前双臂真机配置。

```bash
cd /data/users/liujingyuan/workspace/Evo-RL

export OPENPI_ROOT=/data/users/liujingyuan/workspace/openpi_local
export PYTHONPATH="$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1

lerobot-setup-can --mode=setup --interfaces=can_left,can_right

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
  --robot.left_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "243322070942", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: intelrealsense, serial_number_or_name: "243722071316", width: 640, height: 480, fps: 30, warmup_s: 2}, front: {type: intelrealsense, serial_number_or_name: "239622301704", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --policy.type=remote_monitor \
  --policy.policy_name=pi05 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=24 \
  --policy.predictor_remote.host=<SERVER_IP> \
  --policy.predictor_remote.port=9991 \
  --local_detector.monitor_config=pi05_real_robot_arrange_flower_evorl_detector \
  --local_detector.monitor_dir=../detector/arrange_flower_detector.msgpack \
  --local_detector.detector_conformal_path=../bands/arrange_flower_bands \
  --local_detector.task_stats_path=../stats/arrange_flower.json \
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

`<SERVER_IP>` 是运行 server 的机器 IP。client 不传 `--monitor.predictor_band_path` 或 `--monitor.task_stats_path`。

## 3. Client 的四个 detector 组合

只替换 client 命令中的 `local_detector` 三行：

### arrange_flower

```text
--local_detector.monitor_config=pi05_real_robot_arrange_flower_evorl_detector
--local_detector.monitor_dir=/data/users/liujingyuan/data/monitor_experiments/real_robot_detector/checkpoints/pi05_real_robot_arrange_flower_evorl_detector/56_real_robot_temporal_bce_detector_arrange_flower_seed42/obs1/monitor/00004000.msgpack
--local_detector.detector_conformal_path=/data/users/liujingyuan/data/monitor_experiments/real_robot_or_bands/arrange_flower/detector_safe_score
```

### stack_plates

```text
--local_detector.monitor_config=pi05_real_robot_stack_plates_evorl_detector
--local_detector.monitor_dir=/data/users/liujingyuan/data/monitor_experiments/real_robot_detector/checkpoints/pi05_real_robot_stack_plates_evorl_detector/56_real_robot_temporal_bce_detector_stack_plates_seed42/obs1/monitor/00002000.msgpack
--local_detector.detector_conformal_path=/data/users/liujingyuan/data/monitor_experiments/real_robot_or_bands/stack_plates/detector_safe_score
```

### pick_cubes

```text
--local_detector.monitor_config=pi05_real_robot_pick_cubes_evorl_detector
--local_detector.monitor_dir=/data/users/liujingyuan/data/monitor_experiments/real_robot_detector/checkpoints/pi05_real_robot_pick_cubes_evorl_detector/56_real_robot_temporal_bce_detector_pick_cubes_seed42/obs1/monitor/00004000.msgpack
--local_detector.detector_conformal_path=/data/users/liujingyuan/data/monitor_experiments/real_robot_or_bands/pick_cubes/detector_safe_score
```

### fold_cloth

```text
--local_detector.monitor_config=pi05_real_robot_temporal_bce_detector
--local_detector.monitor_dir=/data/users/liujingyuan/data/monitor_experiments/real_robot_detector/checkpoints/pi05_real_robot_temporal_bce_detector/56_real_robot_temporal_bce_detector_fold_cloth_seed42/obs1/monitor/00009999.msgpack
--local_detector.detector_conformal_path=/data/users/liujingyuan/data/monitor_experiments/real_robot_or_bands/fold_cloth/detector_safe_score
```

## 4. 启动检查

server：

```bash
python -c "import openpi, f_token, flax, jax; print('server imports ok')"
```

client：

```bash
python -c "import openpi, f_token, flax, jax; print('client imports ok')"
```

client 正常启动日志：

```text
[DETECTOR-LOCAL] Worker started.
Local detector warmup complete
Detector is ready. Starting robot execution loop.
```

如果 detector 初始化失败，检查：

```bash
ls -l <detector_checkpoint.msgpack>
ls -l <detector_safe_score>/metrics_summary.json
ls -l <detector_safe_score>/json/classify_cp_functional__model_bands.json
```
