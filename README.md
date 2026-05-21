# Team 43 — ETH Robot Learning Cloth Folding

Repository for Team 43's ETH Robot Learning project on cloth folding.
Below are the setup and usage instructions for the codebase, including training and inference for the SmolVLA and MolmoAct2 policy used in the demo and bonus task, respectively.

**Hugging Face models and datasets:** https://huggingface.co/robot-learning-team43

## Code and environment setup
```bash
git clone https://github.com/DHKim0428/robot-learning-cloth-folding.git --recursive
cd robot-learning-cloth-folding
```

### SmolVLA - Policy during demo
```bash
# Install conda first then:
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install ffmpeg -c conda-forge
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -e "lerobot[feetech,training,viz,dataset,diffusion,async,smolvla]" \
    --extra-index-url https://download.pytorch.org/whl/cu128
pip install pynput
```

### MolmoAct2 - Policy for bonus task
```bash
## Setup submodule
git fetch origin
git submodule sync
git submodule update --init

## Setup env (install uv then)
#curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```
## Training
### MolmoAct2
The following script was used to train [`robot-learning-team43/molmoact2_HQ_extended_020000`](https://huggingface.co/robot-learning-team43/molmoact2_HQ_extended_020000) using a H100 GPU in Nvidia Brev. Both the checkpoint and dataset ([`robot-learning-team43/so101_HQ_merged_diverse_pos`](robot-learning-team43/so101_HQ_merged_diverse_pos)) are openly available on Hugging Face.
```bash
bash shell/train_molmoact2.sh
```
## Inference
### MolmoAct2
Run the following script to use the trained model [`robot-learning-team43/molmoact2_HQ_extended_020000`](https://huggingface.co/robot-learning-team43/molmoact2_HQ_extended_020000). Adjust port and camera index as needed.
```bash
bash shell/rollout_molmoact2.sh
```

## Repository structure
- `config/` — local configuration templates and port files
- `data/lerobot/` — local LeRobot-format recordings (gitignored)
- `docs/` — project notes, setup notes, and decisions
- `papers/` — optional paper notes and references
- `scripts/` — python scripts for robot setup, teleoperation, and data collection
- `shell/` — shell scripts for various tasks like training and rollout

## SO-101 data collection
For day-to-day robot setup and data collection, see **[docs/so101_config.md](docs/so101_config.md)**.
