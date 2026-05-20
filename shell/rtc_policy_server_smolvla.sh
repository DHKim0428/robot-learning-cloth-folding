#!/usr/bin/env bash
# Start the custom RTC policy server matching rollout_smolvla_dohyung.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
POLICY_PATH="${POLICY_PATH:-robot-learning-team43/smolvla_HQ}"
DEVICE="${DEVICE:-cuda}"
FPS="${FPS:-30}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-10}"
MAX_GUIDANCE_WEIGHT="${MAX_GUIDANCE_WEIGHT:-10.0}"

exec python scripts/rtc_policy_server.py \
    --host "$HOST" \
    --port "$PORT" \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --fps "$FPS" \
    --execution-horizon "$EXECUTION_HORIZON" \
    --max-guidance-weight "$MAX_GUIDANCE_WEIGHT" \
    "$@"
