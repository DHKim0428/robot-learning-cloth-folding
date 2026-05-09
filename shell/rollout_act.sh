#!/usr/bin/env bash
# Run a trained ACT policy on the SO-101 follower arm.
#
# Usage:
#   bash shell/rollout_act.sh ACT_model/outputs/<run>/policy_act.pt [--dry-run] [extra args]
#
# Reminders:
#   - run `newgrp dialout` once per terminal so the USB ports are accessible
#   - run `conda activate lerobot` first

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <checkpoint.pt> [extra args]" >&2
    exit 2
fi

CHECKPOINT="$1"
shift

if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then
    echo "[warn] CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}' (expected 'lerobot')"
fi

if ! groups | tr ' ' '\n' | grep -qx dialout; then
    echo "[warn] you may need to run \`newgrp dialout\` first to access the USB serial ports."
fi

exec python ACT_model/rollout.py \
    --checkpoint "$CHECKPOINT" \
    --ports-config "$REPO_ROOT/config/so101_ports.json" \
    --home-pose "$REPO_ROOT/config/so101_home_pose.json" \
    "$@"
