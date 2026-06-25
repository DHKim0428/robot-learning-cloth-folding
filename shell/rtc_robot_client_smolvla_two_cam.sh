#!/usr/bin/env bash
# Run the two-camera SO-101 RTC robot client matching run_eval_two_cam.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-follower}"
CALIBRATION_DIR="${CALIBRATION_DIR:-config/calibration/robots/so_follower}"
TOP_CAMERA_NAME="${TOP_CAMERA_NAME:-camera1}"
TOP_CAMERA_INDEX="${TOP_CAMERA_INDEX:-0}"
WRIST_CAMERA_NAME="${WRIST_CAMERA_NAME:-camera2}"
WRIST_CAMERA_INDEX="${WRIST_CAMERA_INDEX:-2}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
FPS="${FPS:-30}"
REFILL_THRESHOLD="${REFILL_THRESHOLD:-5}"
TASK="SO101 teleoperation task"

python scripts/goto_start_pose2.py --port "$ROBOT_PORT"

exec python scripts/rtc_robot_client.py \
    --server-address "$SERVER_ADDRESS" \
    --robot-type so101_follower \
    --robot-port "$ROBOT_PORT" \
    --robot-id "$ROBOT_ID" \
    --calibration-dir "$CALIBRATION_DIR" \
    --camera-name "$TOP_CAMERA_NAME" \
    --camera-index "$TOP_CAMERA_INDEX" \
    --camera2-name "$WRIST_CAMERA_NAME" \
    --camera2-index "$WRIST_CAMERA_INDEX" \
    --camera-width "$CAMERA_WIDTH" \
    --camera-height "$CAMERA_HEIGHT" \
    --fps "$FPS" \
    --task "$TASK" \
    --refill-threshold "$REFILL_THRESHOLD" \
    "$@"
