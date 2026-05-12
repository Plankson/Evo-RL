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

## Notes

- `local_detector.detector_head_dir` is optional.
- `distributed_robot_io=true` is supported in this script as in the monitor script.
- If local detector dependencies (`openpi`, `f_token`, `jax`) are unavailable, startup fails fast.
