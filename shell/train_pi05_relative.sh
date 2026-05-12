#!/usr/bin/env bash
# Native LeRobot π0.5 fine-tuning wrapper with relative actions enabled.
#
# This follows the LeRobot π0.5 docs:
#   1) recompute dataset ACTION stats in relative-action space
#   2) train with --policy.use_relative_actions=true
#
# Important: the gripper stays absolute via relative_exclude_joints=["gripper"].
# That means arm joints are trained as deltas relative to observation.state, while
# gripper commands remain direct absolute open/close targets.
#
# Typical smoke run:
#   bash shell/train_pi05_relative.sh --steps=20 --batch_size=1 --num_workers=0
#
# If relative stats were already prepared, skip the preprocessing copy:
#   PREPARE_RELATIVE_STATS=false bash shell/train_pi05_relative.sh ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then
    echo "[warn] CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}' (expected 'lerobot')"
    echo "       run \`conda activate lerobot\` first if you hit import errors."
fi

SOURCE_DATASET_REPO_ID="${SOURCE_DATASET_REPO_ID:-robot-learning-team43/so101_teleop_private}"
RELATIVE_DATASET_REPO_ID="${RELATIVE_DATASET_REPO_ID:-robot-learning-team43/so101_teleop_private_pi05_relative}"

# Use the local downloaded dataset if present; otherwise let LeRobot resolve from
# the HF cache. Set SOURCE_DATASET_ROOT explicitly to force another path.
DEFAULT_SOURCE_ROOT="$REPO_ROOT/data/so101_teleop_private"
if [[ -z "${SOURCE_DATASET_ROOT+x}" ]]; then
    SOURCE_DATASET_ROOT=""
fi
if [[ -z "$SOURCE_DATASET_ROOT" && -d "$DEFAULT_SOURCE_ROOT" ]]; then
    SOURCE_DATASET_ROOT="$DEFAULT_SOURCE_ROOT"
fi

# Keep the recomputed-stats copy local by default, so the original dataset is not
# modified. Set RELATIVE_DATASET_ROOT="" to use LeRobot's HF cache location.
if [[ -z "${RELATIVE_DATASET_ROOT+x}" ]]; then
    RELATIVE_DATASET_ROOT="$REPO_ROOT/data/so101_teleop_private_pi05_relative"
fi

PREPARE_RELATIVE_STATS="${PREPARE_RELATIVE_STATS:-true}"
PUSH_RELATIVE_STATS_TO_HUB="${PUSH_RELATIVE_STATS_TO_HUB:-false}"
STATS_NUM_WORKERS="${STATS_NUM_WORKERS:-0}"

PI05_CHUNK_SIZE="${PI05_CHUNK_SIZE:-50}"
RELATIVE_EXCLUDE_JOINTS="${RELATIVE_EXCLUDE_JOINTS:-['gripper']}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="$REPO_ROOT/outputs/train/pi05_relative_so101_${RUN_TS}"

if [[ "$PREPARE_RELATIVE_STATS" == "true" ]]; then
    echo "[prep] source repo              : $SOURCE_DATASET_REPO_ID"
    echo "[prep] relative-stats repo      : $RELATIVE_DATASET_REPO_ID"
    echo "[prep] source root              : ${SOURCE_DATASET_ROOT:-<LeRobot HF cache>}"
    echo "[prep] relative-stats root      : ${RELATIVE_DATASET_ROOT:-<LeRobot HF cache>}"
    echo "[prep] chunk_size               : $PI05_CHUNK_SIZE"
    echo "[prep] relative_exclude_joints  : $RELATIVE_EXCLUDE_JOINTS"
    echo "[prep] push_to_hub              : $PUSH_RELATIVE_STATS_TO_HUB"
    echo

    PREP_CMD=(
        lerobot-edit-dataset
        --repo_id "$SOURCE_DATASET_REPO_ID"
        --new_repo_id "$RELATIVE_DATASET_REPO_ID"
        --operation.type recompute_stats
        --operation.relative_action true
        --operation.chunk_size "$PI05_CHUNK_SIZE"
        --operation.relative_exclude_joints "$RELATIVE_EXCLUDE_JOINTS"
        --operation.num_workers "$STATS_NUM_WORKERS"
        --push_to_hub "$PUSH_RELATIVE_STATS_TO_HUB"
    )
    if [[ -n "$SOURCE_DATASET_ROOT" ]]; then
        PREP_CMD+=(--root "$SOURCE_DATASET_ROOT")
    fi
    if [[ -n "$RELATIVE_DATASET_ROOT" ]]; then
        PREP_CMD+=(--new_root "$RELATIVE_DATASET_ROOT")
    fi

    echo "[prep] running: ${PREP_CMD[*]}"
    "${PREP_CMD[@]}"
    echo
fi

echo "[hint] LeRobot native output dir: $OUT_BASE"
echo "[hint] training dataset repo id  : $RELATIVE_DATASET_REPO_ID"
echo "[hint] training dataset root     : ${RELATIVE_DATASET_ROOT:-<LeRobot HF cache>}"
echo "[hint] relative actions          : true; gripper remains absolute"
echo "[hint] private HF dataset access requires: hf auth login  # or HF_TOKEN"
echo

TRAIN_CMD=(
    lerobot-train
    --dataset.repo_id "$RELATIVE_DATASET_REPO_ID"
    --policy.type pi05
    --policy.pretrained_path lerobot/pi05_base
    --policy.use_relative_actions true
    --policy.relative_exclude_joints '["gripper"]'
    --policy.chunk_size "$PI05_CHUNK_SIZE"
    --policy.n_action_steps "$PI05_CHUNK_SIZE"
    --policy.device cuda
    --policy.dtype bfloat16
    --policy.compile_model true
    --policy.gradient_checkpointing true
    --policy.freeze_vision_encoder false
    --policy.train_expert_only false
    --output_dir "$OUT_BASE"
    --job_name pi05_relative_so101
    --policy.push_to_hub false
    --wandb.enable false
    --steps "${STEPS:-3000}"
    --batch_size "${BATCH_SIZE:-4}"
)
if [[ -n "$RELATIVE_DATASET_ROOT" ]]; then
    TRAIN_CMD+=(--dataset.root "$RELATIVE_DATASET_ROOT")
fi

exec "${TRAIN_CMD[@]}" "$@"
