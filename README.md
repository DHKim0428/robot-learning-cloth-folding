# Team 43 — ETH Robot Learning Cloth Folding

Submission repository for Team 43's cloth-folding project.

Hugging Face: https://huggingface.co/robot-learning-team43

## Policies

- Basic eval: `robot-learning-team43/smolvla_HQ`
- Bonus eval: `robot-learning-team43/molmo_b16_lora_reward_10000`

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
source .venv/bin/activate
```

## Run evaluation locally

These scripts assume the policy and robot run on the same machine with a CUDA GPU.
Override `ROBOT_PORT`, `CAMERA_INDEX`, or `POLICY_PATH` if needed.

```bash
bash run_eval_basic.sh
bash run_eval_bonus.sh
```

Defaults:

- Robot port: `/dev/ttyACM1`
- Camera index: `0`
- FPS: `30`

MolmoAct2 needs a larger GPU; we used a GPU server when local VRAM was insufficient.

## Remote GPU / Brev inference

Run policy server on the GPU machine and robot client on the robot machine.

### SmolVLA

GPU machine:

```bash
HOST=0.0.0.0 PORT=8080 bash shell/rtc_policy_server_smolvla.sh
```

Robot machine:

```bash
SERVER_ADDRESS=<GPU_IP>:8080 bash shell/rtc_robot_client_smolvla.sh
```

### MolmoAct2

GPU machine:

```bash
HOST=0.0.0.0 PORT=8080 bash shell/molmoact_policy_server.sh
```

Robot machine:

```bash
SERVER_ADDRESS=<GPU_IP>:8080 bash shell/molmoact_robot_client.sh
```

## Training scripts

Training is not required for evaluation. The submitted checkpoints are on Hugging Face.

```bash
bash shell/train_smolvla_HQ.sh
bash shell/train_molmoac2.sh
```
