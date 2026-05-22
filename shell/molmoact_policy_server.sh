#!/usr/bin/env bash
# Start the non-RTC MolmoAct2 chunked policy server matching rollout_molmoact.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
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
DEVICE="${DEVICE:-cuda}"
INFERENCE_ACTION_MODE="${INFERENCE_ACTION_MODE:-continuous}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
CHUNK_SIZE="${CHUNK_SIZE:-30}"
N_ACTION_STEPS="${N_ACTION_STEPS:-30}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-4}"
ENABLE_INFERENCE_CUDA_GRAPH="${ENABLE_INFERENCE_CUDA_GRAPH:-true}"

exec python scripts/molmoact_policy_server.py \
    --host "$HOST" \
    --port "$PORT" \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --inference-action-mode "$INFERENCE_ACTION_MODE" \
    --model-dtype "$MODEL_DTYPE" \
    --chunk-size "$CHUNK_SIZE" \
    --n-action-steps "$N_ACTION_STEPS" \
    --num-inference-steps "$NUM_INFERENCE_STEPS" \
    --enable-inference-cuda-graph "$ENABLE_INFERENCE_CUDA_GRAPH" \
    "$@"
