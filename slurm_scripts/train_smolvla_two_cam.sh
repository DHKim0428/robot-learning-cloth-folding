#!/bin/bash
#SBATCH --job-name=smolvla_two_cam
#SBATCH --partition=gpu.120h
#SBATCH --gpus=quadro_rtx_6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=120:00:00
#SBATCH --output=logs/smolvla_two_cam_%j.out
#SBATCH --error=logs/smolvla_two_cam_%j.err

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_DIR/logs"

module load stack/2025-06 gcc/12.2.0
module load ffmpeg/7.0.2

export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_LEROBOT_HOME=/cluster/scratch/spanwar/cache/huggingface

source /cluster/work/igp_psr/spanwar/envs/lerobot/bin/activate

cd "$REPO_DIR"

nvidia-smi

srun bash shell/train_smolvla_two_cam.sh
