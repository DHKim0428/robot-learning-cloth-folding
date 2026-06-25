#!/usr/bin/env bash
# Start the custom RTC policy server for the two-camera SmolVLA HQ policy.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
if [ -z "${POLICY_PATH:-}" ]; then
    if [ -d "checkpoints/smolvla_HQ_two_cam" ]; then
        POLICY_PATH="checkpoints/smolvla_HQ_two_cam"
    else
        POLICY_PATH="klrshak/smolvla_HQ_two_cam"
    fi
fi
DEVICE="${DEVICE:-cuda}"
FPS="${FPS:-30}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-10}"
MAX_GUIDANCE_WEIGHT="${MAX_GUIDANCE_WEIGHT:-10.0}"
MAX_ACTIONS_PER_RESPONSE="${MAX_ACTIONS_PER_RESPONSE:-10}"

exec python scripts/rtc_policy_server.py \
    --host "$HOST" \
    --port "$PORT" \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --fps "$FPS" \
    --execution-horizon "$EXECUTION_HORIZON" \
    --max-guidance-weight "$MAX_GUIDANCE_WEIGHT" \
    --max-actions-per-response "$MAX_ACTIONS_PER_RESPONSE" \
    "$@"
