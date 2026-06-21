#!/usr/bin/env bash
# Run the two-camera SmolVLA HQ RTC rollout on the SO-101 follower.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-follower}"
CALIBRATION_DIR="${CALIBRATION_DIR:-config/calibration/robots/so_follower}"
TOP_CAMERA_NAME="${TOP_CAMERA_NAME:-camera1}"
TOP_CAMERA_INDEX="${TOP_CAMERA_INDEX:-2}"
WRIST_CAMERA_NAME="${WRIST_CAMERA_NAME:-camera2}"
WRIST_CAMERA_INDEX="${WRIST_CAMERA_INDEX:-4}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
FPS="${FPS:-30}"
if [ "$#" -gt 0 ]; then
    POLICY_PATH="$1"
    shift
elif [ -z "${POLICY_PATH:-}" ]; then
    if [ -d "checkpoints/smolvla_HQ_two_cam" ]; then
        POLICY_PATH="checkpoints/smolvla_HQ_two_cam"
    else
        POLICY_PATH="klrshak/smolvla_HQ_two_cam"
    fi
fi
TASK="${TASK:-SO101 teleoperation task}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-10}"

python scripts/goto_start_pose2.py --port "$ROBOT_PORT"

exec lerobot-rollout \
    --policy.path="$POLICY_PATH" \
    --robot.type=so101_follower \
    --robot.port="$ROBOT_PORT" \
    --robot.id="$ROBOT_ID" \
    --robot.calibration_dir="$CALIBRATION_DIR" \
    --robot.cameras="{\"$TOP_CAMERA_NAME\": {\"type\": \"opencv\", \"index_or_path\": $TOP_CAMERA_INDEX, \"width\": $CAMERA_WIDTH, \"height\": $CAMERA_HEIGHT, \"fps\": $FPS, \"fourcc\": \"MJPG\"}, \"$WRIST_CAMERA_NAME\": {\"type\": \"opencv\", \"index_or_path\": $WRIST_CAMERA_INDEX, \"width\": $CAMERA_WIDTH, \"height\": $CAMERA_HEIGHT, \"fps\": $FPS, \"fourcc\": \"MJPG\"}}" \
    --fps="$FPS" \
    --task="$TASK" \
    --inference.type=rtc \
    --inference.rtc.execution_horizon="$EXECUTION_HORIZON" \
    "$@"
