# Team 43 — ETH Robot Learning Cloth Folding

Submission repository for Team 43's cloth-folding project.

Hugging Face: https://huggingface.co/robot-learning-team43

## Policies

- Basic eval: `robot-learning-team43/smolvla_HQ`
- Bonus eval: `robot-learning-team43/molmoact2_HQ_extended_020000`

## Setup

Clone with submodules:

```bash
git clone https://github.com/DHKim0428/robot-learning-cloth-folding.git --recursive
cd robot-learning-cloth-folding
```

### Basic eval environment

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y ffmpeg -c conda-forge
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -e "lerobot[feetech,training,viz,dataset,diffusion,async,smolvla]" \
    --extra-index-url https://download.pytorch.org/whl/cu128
pip install pynput
```

### Bonus eval environment

```bash
git submodule sync
git submodule update --init --recursive
uv sync
```

## Run evaluation locally

These scripts assume the policy and robot run on the same machine with a CUDA GPU.
Override `ROBOT_PORT`, `CAMERA_INDEX`, or `POLICY_PATH` if needed.

Run calibration once before evaluation. If you already calibrated for basic eval, you do not need to repeat it for bonus eval.

Basic eval, from the `lerobot` conda environment:

```bash
conda activate lerobot
python scripts/calibrate_motor.py
bash run_eval_basic.sh
```

Bonus eval, after `uv sync`:

```bash
uv run bash run_eval_bonus.sh
```

If bonus eval is the first run after setup, calibrate once with `uv run python scripts/calibrate_motor.py` before running the script.

Defaults:

- Robot port: `/dev/ttyACM1`
- Camera index: `0`
- FPS: `30`

Override the robot port from the CLI if needed:

```bash
ROBOT_PORT=/dev/ttyACM0 bash run_eval_basic.sh
ROBOT_PORT=/dev/ttyACM0 uv run bash run_eval_bonus.sh
```

MolmoAct2 needs a larger GPU; we used a GPU server when local VRAM was insufficient.

## Remote GPU / Brev inference

Use this setup when the policy is too large to run on the robot machine. Start the policy server on the GPU machine or Brev instance, then start the robot client on the machine connected to the robot.

Set `SERVER_ADDRESS` on the robot machine to the GPU machine address, for example `<GPU_IP>:8080`. If you use Brev port forwarding, run this on the robot machine and use `SERVER_ADDRESS=127.0.0.1:8080`:

```bash
brev port-forward <brev-instance-name> -p 8080:8080
```

If the robot is not on `/dev/ttyACM1`, override `ROBOT_PORT` on the robot machine.

### SmolVLA

Use the `lerobot` conda environment for SmolVLA.

On the GPU machine or Brev instance:

```bash
conda activate lerobot
HOST=0.0.0.0 PORT=8080 bash shell/rtc_policy_server_smolvla.sh
```

On the robot machine:

```bash
conda activate lerobot
SERVER_ADDRESS=<GPU_IP>:8080 bash shell/rtc_robot_client_smolvla.sh
```

Override the robot port if needed:

```bash
ROBOT_PORT=/dev/ttyACM0 SERVER_ADDRESS=<GPU_IP>:8080 bash shell/rtc_robot_client_smolvla.sh
```

### MolmoAct2

Use `uv run` for MolmoAct2 after `uv sync`.

#### Option A: custom Team 43 server/client

GPU machine or Brev instance:

```bash
HOST=0.0.0.0 PORT=8080 uv run bash shell/molmoact_policy_server.sh
```

Robot machine:

```bash
SERVER_ADDRESS=<GPU_IP>:8080 uv run bash shell/molmoact_robot_client.sh
```

Override the robot port if needed:

```bash
ROBOT_PORT=/dev/ttyACM0 SERVER_ADDRESS=<GPU_IP>:8080 uv run bash shell/molmoact_robot_client.sh
```

#### Option B: LeRobot async inference

GPU machine or Brev instance:

```bash
uv run python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080
```

Robot machine:

```bash
# Linux/macOS; edit SERVER_ADDRESS/ROBOT_PORT in the script if needed.
uv run bash shell/async_client_molmoact2.sh

# Windows
shell\async_client_molmoact2.bat
```

## Training scripts

The following training scripts are included for reference in case they are needed.

SmolVLA training uses the `lerobot` conda environment:

```bash
conda activate lerobot
bash shell/train_smolvla_HQ.sh
```

MolmoAct2 training uses `uv run` after `uv sync`:

```bash
uv run bash shell/train_molmoac2.sh
```
