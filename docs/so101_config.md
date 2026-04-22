# SO101 config

## MotorsBus ports

MotorBus ports are machine-specific and may change across computers, USB ports, or adapters.
Do **not** rely on hardcoded `/dev/tty...` values from someone else's machine.

For this repo, the expected workflow is:
1. detect the current leader/follower ports
2. write them to `config/so101_ports.json`
3. use the project scripts, which read that file automatically

### Detect and write ports config

This interactively detects the USB ports for the leader and follower arms and writes
`config/so101_ports.json`.

```bash
python scripts/detect_ports.py
```

If you want to write to a different file:
```bash
python scripts/detect_ports.py --output config/so101_ports_other.json
```

If the file already exists and you want to overwrite without being prompted:
```bash
python scripts/detect_ports.py --force
```

How it works:
- start with both leader and follower connected
- the script asks you to disconnect the leader arm and detects which port disappeared
- then it asks you to reconnect the leader
- then it repeats the same process for the follower
- finally it saves the detected ports as JSON


## Project scripts

### Setup motors
```bash
python scripts/setup_motor.py follower
python scripts/setup_motor.py leader
```

### Calibrate motors
```bash
python scripts/calibrate_motor.py follower
python scripts/calibrate_motor.py leader
```

These scripts read `config/so101_ports.json`, so they should be run **after**
`python scripts/detect_ports.py`.

### Teleoperation

Without camera:
```bash
python scripts/teleop.py
```

With camera:
```bash
python scripts/teleop.py --camera
```

### Equivalent raw LeRobot CLI commands

If needed, the corresponding raw LeRobot commands are:

Follower setup:
```bash
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=<follower_port>
```

Leader setup:
```bash
lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=<leader_port>
```

Follower calibration:
```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=<follower_port> \
    --robot.id=follower
```

Leader calibration:
```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=<leader_port> \
    --teleop.id=leader
```

### Record teleoperation dataset

Create a new dataset in this project under `data/lerobot/<dataset-repo-id>`:
```bash
python scripts/teleop_record.py \
    --camera \
    --num-episodes 2 \
    --task "test recording"
```

Use a different dataset name:
```bash
python scripts/teleop_record.py \
    --camera \
    --dataset-repo-id local/so101_teleop_test2 \
    --num-episodes 2 \
    --task "test recording"
```

Resume/appended recording on an existing dataset:
```bash
python scripts/teleop_record.py \
    --camera \
    --dataset-repo-id local/so101_teleop \
    --resume \
    --num-episodes 2 \
    --task "test recording"
```

Notes:
- default dataset root: `data/lerobot`
- default dataset repo id: `local/so101_teleop`
- if the dataset directory already exists and you want to keep adding episodes, use `--resume`
- if you want a fresh dataset, use a new `--dataset-repo-id` or delete the old dataset directory first

### Upload recorded dataset to Hugging Face

Set your Hugging Face token in the environment first:
```bash
export HF_TOKEN=hf_xxx
```

Then push after recording:
```bash
python scripts/teleop_record.py \
    --camera \
    --dataset-repo-id <hf_username>/so101_teleop \
    --num-episodes 2 \
    --task "test recording" \
    --push-to-hub
```

Push to a private dataset repo:
```bash
python scripts/teleop_record.py \
    --camera \
    --dataset-repo-id <hf_username>/so101_teleop_private \
    --num-episodes 2 \
    --task "test recording" \
    --push-to-hub \
    --private
```

Resume locally and then push:
```bash
python scripts/teleop_record.py \
    --camera \
    --dataset-repo-id <hf_username>/so101_teleop \
    --resume \
    --num-episodes 2 \
    --task "test recording" \
    --push-to-hub
```

Upload notes:
- `--push-to-hub` requires `--dataset-repo-id` to be a real Hugging Face repo id like `<hf_username>/<dataset_name>`
- the default `local/so101_teleop` is only for local testing
- token lookup order: `HF_TOKEN`, then `HUGGINGFACE_HUB_TOKEN`


## TODO

- Move SO-101 calibration files into a project-local directory instead of the default temporary/cache location.
- Do this by setting the same `calibration_dir` consistently across calibration and runtime scripts, rather than only moving files manually after calibration.
- Update at least:
  - `scripts/calibrate_motor.py`
  - `scripts/teleop.py`
  - `scripts/teleop_record.py`
