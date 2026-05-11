#!/usr/bin/env bash
# Run the ACT cloth-folding policy on the SO-101 follower arm.
#
# Usage:
#   bash shell/rollout_act.sh [--duration 60] [--task "fold the cloth"] [extra args]
#
# Reminders:
#   - run `newgrp dialout` once per terminal so USB ports are accessible
#   - HF_TOKEN must be set to pull the private model from the Hub

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

POLICY_REPO="robot-learning-team43/act_cloth_folding_05_11"

# Download model from Hub (cached after first run)
echo "[init] fetching model from Hub: $POLICY_REPO"
CHECKPOINT=$(python -c "
from huggingface_hub import snapshot_download
import pathlib
path = snapshot_download('$POLICY_REPO')
pts = list(pathlib.Path(path).glob('*.pt'))
if not pts:
    raise FileNotFoundError(f'No .pt file found in {path}')
print(pts[0])
")
echo "[init] checkpoint: $CHECKPOINT"

if ! groups | tr ' ' '\n' | grep -qx dialout; then
    echo "[warn] you may need to run \`newgrp dialout\` first to access the USB serial ports."
fi

exec python ACT_model/rollout.py \
    --checkpoint "$CHECKPOINT" \
    --ports-config "$REPO_ROOT/config/so101_ports.json" \
    --home-pose "$REPO_ROOT/config/so101_home_pose.json" \
    "$@"
