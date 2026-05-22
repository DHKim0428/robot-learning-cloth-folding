#!/usr/bin/env bash
# Run the final SmolVLA HQ RTC rollout on the SO-101 follower.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-follower}"
CALIBRATION_DIR="${CALIBRATION_DIR:-config/calibration/robots/so_follower}"
CAMERA_NAME="${CAMERA_NAME:-camera1}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
FPS="${FPS:-30}"
POLICY_PATH="${POLICY_PATH:-robot-learning-team43/smolvla_HQ}"
TASK="${TASK:-SO101 teleoperation task}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-10}"

python scripts/goto_start_pose2.py --port "$ROBOT_PORT"

exec lerobot-rollout \
    --policy.path="$POLICY_PATH" \
    --robot.type=so101_follower \
    --robot.port="$ROBOT_PORT" \
    --robot.id="$ROBOT_ID" \
    --robot.calibration_dir="$CALIBRATION_DIR" \
    --robot.cameras="{\"$CAMERA_NAME\": {\"type\": \"opencv\", \"index_or_path\": $CAMERA_INDEX, \"width\": $CAMERA_WIDTH, \"height\": $CAMERA_HEIGHT, \"fps\": $FPS}}" \
    --fps="$FPS" \
    --task="$TASK" \
    --inference.type=rtc \
    --inference.rtc.execution_horizon="$EXECUTION_HORIZON" \
    "$@"
