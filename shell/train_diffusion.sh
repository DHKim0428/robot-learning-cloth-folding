#!/usr/bin/env bash
# Native LeRobot training wrapper for the single end-to-end SO-101 DiffusionPolicy.
#
# This intentionally uses `lerobot-train --policy.type=diffusion` rather than
# the experimental custom trainer in `diffusion_model/train_custom.py`, so we
# get LeRobot's native sampler, optimizer/scheduler presets, checkpoint format,
# resume behavior, and optional Hub/W&B integration.
#
# Pass any extra LeRobot CLI args through this wrapper, e.g.
#   bash shell/train_diffusion.sh --steps=50000 --batch_size=8
#   bash shell/train_diffusion.sh --dataset.episodes='[0,2,5]' --policy.horizon=64

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then
    echo "[warn] CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}' (expected 'lerobot')"
    echo "       run \`conda activate lerobot\` first if you hit import errors."
fi

DATASET_REPO_ID="${DATASET_REPO_ID:-robot-learning-team43/so101_teleop_private}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="$REPO_ROOT/outputs/train/diffusion_so101_${RUN_TS}"

echo "[hint] LeRobot native output dir: $OUT_BASE"
echo "[hint] dataset repo id: $DATASET_REPO_ID"
echo "[hint] private HF dataset access requires: hf auth login  # or HF_TOKEN"
echo

exec lerobot-train \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --policy.type=diffusion \
    --policy.device=cuda \
    --policy.n_obs_steps=2 \
    --policy.horizon=64 \
    --policy.n_action_steps=32 \
    --output_dir="$OUT_BASE" \
    --job_name=diffusion_so101 \
    --policy.push_to_hub=false \
    --wandb.enable=false \
    "$@"
