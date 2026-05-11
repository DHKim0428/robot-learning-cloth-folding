#!/usr/bin/env bash
# Run the ACT cloth-folding policy on the SO-101 follower arm.
#
# Usage:
#   bash shell/rollout_act.sh [--duration 60] [--task "fold the cloth"] [extra lerobot-rollout args]
#
# Reminders:
#   - run `newgrp dialout` once per terminal so USB ports are accessible
#   - HF_TOKEN must be set to pull the private model from the Hub

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

POLICY="robot-learning-team43/act_cloth_folding_05_11"
PORTS_CONFIG="$REPO_ROOT/config/so101_ports.json"
CALIBRATION_DIR="$REPO_ROOT/config/calibration"
DURATION=60
TASK="fold the cloth"

# Parse named overrides before passing remainder to lerobot-rollout.
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration) DURATION="$2"; shift 2 ;;
        --task)     TASK="$2";     shift 2 ;;
        *)          EXTRA_ARGS+=("$1"); shift ;;
    esac
done

FOLLOWER_PORT=$(python -c "import json; print(json.load(open('$PORTS_CONFIG'))['follower'])")

if ! groups | tr ' ' '\n' | grep -qx dialout; then
    echo "[warn] you may need to run \`newgrp dialout\` first to access the USB serial ports."
fi

exec lerobot-rollout \
    --strategy.type=base \
    --policy.path="$POLICY" \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id=follower \
    --robot.calibration_dir="$CALIBRATION_DIR/robots/so_follower" \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="$TASK" \
    --duration="$DURATION" \
    "${EXTRA_ARGS[@]}"
