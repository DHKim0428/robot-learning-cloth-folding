#!/usr/bin/env bash
# Train a Diffusion Policy on the SO-101 cloth-folding dataset.
#
# Defaults: frozen DINOv2 backbone, horizon=64, n_action_steps=32, lr=1e-4,
# batch=32, drop `bad` episodes, 100k steps, save every 10k steps.
# Pass extra flags to override (e.g. --backbone resnet18, --episodes 5 --num-steps 5000 --batch-size 8).
# To resume: bash shell/train_diffusion.sh --resume diffusion-policy/outputs/diff_<timestamp>
#
# Reminder: run inside the `lerobot` conda env.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then
    echo "[warn] CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}' (expected 'lerobot')"
    echo "       run \`conda activate lerobot\` first if you hit import errors."
fi

OUT_BASE="$REPO_ROOT/diffusion-policy/outputs"
mkdir -p "$OUT_BASE"

echo "[hint] tail TensorBoard with: tensorboard --logdir $OUT_BASE --port 6007"
echo

exec python diffusion-policy/train.py \
    --episode-filter \
    --backbone dinov2 \
    --horizon 64 \
    --n-action-steps 32 \
    --num-steps 100000 \
    --batch-size 32 \
    --lr 1e-4 \
    "$@"
