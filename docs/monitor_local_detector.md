# Local Detector + Remote Predictor (Evo-RL)

This flow runs:
- Detector: local in the monitor script process (PiZero loading pipeline)
- Predictor: remote OpenPI websocket endpoint (existing `remote_monitor` config)

## Script

Use:

```bash
python -m lerobot.scripts.lerobot_human_inloop_record_monitor_local_detector \
  --policy.type=remote_monitor \
  --policy.predictor_remote.host=<predictor_host> \
  --policy.predictor_remote.port=<predictor_port> \
  --local_detector.monitor_config=<pizero_monitor_config_name> \
  --local_detector.monitor_dir=<monitor_checkpoint_dir> \
  --local_detector.detector_conformal_path=<detector_conformal_json> \
  <existing monitor/robot/teleop args>
```

For policy-only recording aligned with `lerobot-record` (no teleop), use:

```bash
python -m lerobot.scripts.lerobot_record_monitor_local_detector \
  --policy.type=remote_monitor \
  --policy.predictor_remote.host=<predictor_host> \
  --policy.predictor_remote.port=<predictor_port> \
  --local_detector.monitor_config=<pizero_monitor_config_name> \
  --local_detector.monitor_dir=<monitor_checkpoint_dir> \
  --local_detector.detector_conformal_path=<detector_conformal_json> \
  <existing lerobot-record robot/dataset args>
```

## Notes

- `local_detector.detector_head_dir` is optional.
- `distributed_robot_io=true` is supported in this script as in the monitor script.
- If local detector dependencies (`openpi`, `f_token`, `jax`) are unavailable, startup fails fast.

## Episode visualization

In local detector mode, each finished episode triggers an async renderer **process** that writes detector score-band videos to:

- default: `outputs/monitor_local_detector_videos/episode_XXXXXX.mp4`

Control via:
- `--local_detector.render_episode_video=true|false`
- `--local_detector.render_output_dir=<dir>`
- `--local_detector.render_fps=<int>`
