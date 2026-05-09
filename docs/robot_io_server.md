# Robot IO Server Architecture

## Why this exists

In real-world runs, a robot instance owns exclusive hardware resources (camera pipelines, motor buses, serial/CAN handles, camera background threads). Running multiple processes that each create a robot and call `get_observation()`/`send_action()` is unsafe and can break hardware access.

## Ownership model

- Exactly one process owns the real robot object: **Robot IO Server**.
- Other processes (monitor/policy/frontend/logger) use a network client proxy and never initialize real hardware.

## Transport and protocol (v1)

- Observation stream: ZMQ `PUB` (server) -> `SUB` (clients)
- Action stream: ZMQ `PULL` (server) <- `PUSH` (clients)
- Metadata handshake: ZMQ `REP` (server) <-> `REQ` (clients)
- Serialization: Python `pickle` for trusted local IPC only.

Observation envelope:
- `seq`: monotonically increasing sequence id
- `t_server`: server timestamp (seconds)
- `observation`: robot observation dict from `robot.get_observation()`

## Start server

```bash
python scripts/robot_io_server.py \
  --robot.type=<existing_robot_type> \
  --obs_pub_address=tcp://127.0.0.1:5555 \
  --action_pull_address=tcp://127.0.0.1:5556 \
  --meta_rep_address=tcp://127.0.0.1:5557 \
  --frequency=30 \
  --print_fps=true
```

## Start monitor client with distributed mode

```bash
python -m lerobot.scripts.lerobot_human_inloop_record_monitor \
  --distributed_robot_io=true \
  --robot_io_obs_address=tcp://127.0.0.1:5555 \
  --robot_io_action_address=tcp://127.0.0.1:5556 \
  --robot_io_meta_address=tcp://127.0.0.1:5557 \
  <existing monitor args>
```

## Current limitations

- Pickle transport is for trusted local/private networks only.
- Multiple action producers are not arbitrated; current policy is latest action wins.
- No auth/encryption in v1.
