#!/usr/bin/env bash
# Experimental custom DiffusionPolicy trainer with relative arm actions.
#
# Native `shell/train_diffusion.sh` remains the absolute-action baseline. This
# wrapper uses `diffusion_model/train_custom.py --relative-actions`; arm joints
# become deltas from current observation.state, while the gripper stays absolute.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then
    echo "[warn] CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}' (expected 'lerobot')"
    echo "       run \`conda activate lerobot\` first if you hit import errors."
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/diffusion_model/outputs/diffusion_relative_${RUN_TS}}"
mkdir -p "$(dirname "$OUT_DIR")"

echo "[hint] relative custom output dir: $OUT_DIR"
echo "[hint] TensorBoard: tensorboard --logdir $REPO_ROOT/diffusion_model/outputs --port 6006"
echo "[hint] gripper remains absolute via --relative-exclude-joints gripper"
echo "[hint] video backend: ${DATASET_VIDEO_BACKEND:-pyav} (override with DATASET_VIDEO_BACKEND=... or --dataset.video_backend=...)"
echo "[hint] private HF dataset access requires: hf auth login  # or HF_TOKEN"
echo

exec python diffusion_model/train_custom.py \
    --relative-actions \
    --relative-exclude-joints gripper \
    --dataset.video_backend "${DATASET_VIDEO_BACKEND:-pyav}" \
    --output-dir "$OUT_DIR" \
    "$@"
