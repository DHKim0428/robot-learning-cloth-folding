#!/usr/bin/env bash
# Run the final MolmoAct2 rollout on the SO-101 follower.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-follower}"
CALIBRATION_DIR="${CALIBRATION_DIR:-config/calibration/robots/so_follower}"
CAMERA_NAME="${CAMERA_NAME:-front}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
FPS="${FPS:-30}"
if [ "$#" -gt 0 ]; then
    POLICY_PATH="$1"
    shift
elif [ -z "${POLICY_PATH:-}" ]; then
    if [ -d "checkpoints/molmoact2_HQ_extended_020000" ]; then
        POLICY_PATH="checkpoints/molmoact2_HQ_extended_020000"
    elif [ -d "checkpoints/molmoact2_general" ]; then
        POLICY_PATH="checkpoints/molmoact2_general"
    else
        POLICY_PATH="robot-learning-team43/molmoact2_HQ_extended_020000"
    fi
fi
TASK="${TASK:-Fold the towel diagonally twice}"

python scripts/goto_start_pose.py --port "$ROBOT_PORT"

exec lerobot-rollout \
    --policy.path="$POLICY_PATH" \
    --robot.type=so101_follower \
    --robot.port="$ROBOT_PORT" \
    --robot.id="$ROBOT_ID" \
    --robot.calibration_dir="$CALIBRATION_DIR" \
    --robot.cameras="{\"$CAMERA_NAME\": {\"type\": \"opencv\", \"index_or_path\": $CAMERA_INDEX, \"width\": $CAMERA_WIDTH, \"height\": $CAMERA_HEIGHT, \"fps\": $FPS}}" \
    --fps="$FPS" \
    --task="$TASK" \
    --policy.inference_action_mode=continuous \
    --policy.model_dtype=bfloat16 \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.num_inference_steps=4 \
    --policy.enable_inference_cuda_graph=true \
    "$@"
