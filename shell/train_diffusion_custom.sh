#!/usr/bin/env bash
# Experimental custom PyTorch training loop for DiffusionPolicy.
#
# Prefer `shell/train_diffusion.sh` for normal runs. This wrapper is kept for
# debugging because it mirrors the ACT_model training style and exposes the
# custom code in `diffusion_model/train_custom.py`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then
    echo "[warn] CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}' (expected 'lerobot')"
    echo "       run \`conda activate lerobot\` first if you hit import errors."
fi

OUT_BASE="$REPO_ROOT/diffusion_model/outputs"
mkdir -p "$OUT_BASE"

echo "[hint] custom trainer TensorBoard: tensorboard --logdir $OUT_BASE --port 6006"
echo "[hint] private HF dataset access requires: hf auth login  # or HF_TOKEN"
echo

exec python diffusion_model/train_custom.py "$@"
