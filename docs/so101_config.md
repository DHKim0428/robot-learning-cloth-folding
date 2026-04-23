# SO101 config

## Recording on the Spark machine

Before running any robot script on the Spark machine, run these two commands in the terminal you'll use:

```bash
newgrp dialout
conda activate lerobot
```

`newgrp dialout` gives your shell access to the USB serial ports (leader/follower arms) without logging out. You need to run it once per terminal session.

Run `export HF_TOKEN=<your_token>` before running the recording script.

### Record and push to the shared team dataset

```bash
DISPLAY=:1 python scripts/teleop_record.py \
    --camera \
    --episode-time-sec 120 \
    --return-move-time-sec 4 \
    --num-episodes 2 \
    --dataset-repo-id robot-learning-team43/so101_teleop_private \
    --push-to-hub \
    --resume
```

- `--resume` appends your episode to what the team has already recorded — keep it.
- `DISPLAY=:1` is needed on the Spark machine to connect to the display for camera/rerun visualization.
- Make sure your Hugging Face account is a member of the `robot-learning-team43` org, otherwise the push will fail with a 403.

When you see `Reset the environment` printed in the terminal between episodes, that is your cue to reset the towel to the starting position before the next episode begins.

**Tip:** start with `--num-episodes 1` to verify everything looks good, then bump the number once you're confident.

**Keyboard controls during recording:**
- `→` right arrow — finish the episode early (before the time limit)
- `←` left arrow — discard and re-record the current episode
- `Esc` — stop the session; previously saved episodes are kept and pushed. The episode currently in progress will be saved as a partial — if you want to avoid that, press `←` first to clear the current buffer, then `Esc`.

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

Calibration files are now read from the project-local directory:
- `config/calibration/robots/so_follower/follower.json`
- `config/calibration/teleoperators/so_leader/leader.json`

So if you need to inspect or carefully patch the current calibration JSON by hand, use those files rather than the default Hugging Face cache path.

### Save a reusable home pose

Move the follower arm to a safe pose manually, then save that pose:
```bash
python scripts/save_home_pose.py
```

This writes `config/so101_home_pose.json`.

### Save a reusable final pose

Move the follower arm to the pose you want to use when the full recording session ends, then save it:
```bash
python scripts/save_final_pose.py
```

This writes `config/so101_final_pose.json`.

### Teleoperation

Without camera:
```bash
python scripts/teleop.py
```

With camera:
```bash
python scripts/teleop.py --camera
```

### Replay a recorded episode

Replay episode 0 from the default local dataset:
```bash
python scripts/replay_episode.py --episode 0
```

Replay from a different local dataset:
```bash
python scripts/replay_episode.py \
    --dataset-repo-id local/so101_teleop_test2 \
    --episode 0
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
- by default, `teleop_record.py` returns to the saved home pose in `config/so101_home_pose.json` after each saved episode
- if you have not saved a home pose yet, run `python scripts/save_home_pose.py` first
- if `config/so101_final_pose.json` exists, `teleop_record.py` also moves to that final pose when the full recording session exits
- if you want the old behavior instead, use `--return-pose-source initial`
- if needed, return-to-pose can be disabled with `--no-return-to-initial-pose` or adjusted with `--return-move-time-sec`
- if the dataset directory already exists and you want to keep adding episodes, use `--resume`
- if `--resume` is used with a real Hugging Face repo id and the local dataset directory does not exist yet, the script will initialize the local dataset root from the Hub metadata and then append new episodes locally
- if `--resume` is used with a real Hugging Face repo id but no Hub repo exists yet (for example, a previous run was interrupted before push), the script falls back to creating a new local dataset instead of failing with a raw Hub traceback
- if you want a fresh dataset, use a new `--dataset-repo-id` or delete the old dataset directory first

Example using a saved home pose:
```bash
python scripts/teleop_record.py \
    --camera \
    --return-pose-source home \
    --num-episodes 2 \
    --task "test recording"
```

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

Resume from a Hub dataset that is not yet present locally:
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


