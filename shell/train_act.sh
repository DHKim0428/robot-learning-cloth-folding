#!/usr/bin/env bash
# Train an ACT policy on the SO-101 cloth-folding dataset.
#
# Defaults match plan: drop `bad` episodes, chunk_size=100, 100k steps.
# Pass extra flags to override (e.g. --num-steps 5000, --batch-size 4).
#
# Reminder: run inside the `lerobot` conda env.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then
    echo "[warn] CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}' (expected 'lerobot')"
    echo "       run \`conda activate lerobot\` first if you hit import errors."
fi

OUT_BASE="$REPO_ROOT/ACT_model/outputs"
mkdir -p "$OUT_BASE"

echo "[hint] tail TensorBoard with: tensorboard --logdir $OUT_BASE --port 6006"
echo

exec python ACT_model/train.py \
    --episode-filter \
    "$@"
