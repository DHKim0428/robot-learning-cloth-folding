#!/usr/bin/env bash
# scripts/brev_train.sh — Headless A100 training pipeline for Brev
#
# USAGE
#   tmux new -s train               # strongly recommended: survives SSH disconnect
#   HF_TOKEN=<token> bash scripts/brev_train.sh [extra train.py flags]
#   # Extra flags are forwarded to diffusion-policy/train.py, e.g.:
#   #   NUM_STEPS=300000 bash scripts/brev_train.sh --backbone resnet18
#
# REQUIRED env var
#   HF_TOKEN   — HF token with read (dataset) + write (model repo) access
#
# OPTIONAL env vars
#   HF_CKPT_REPO   default: robot-learning-team43/diffusion-policy-checkpoints
#   NUM_STEPS      default: 100000
#   BATCH_SIZE     default: 256
#   NUM_WORKERS    default: 8
#   SAVE_EVERY     default: 10000
#   LEROBOT_COMMIT default: fc6c94c82a4624bdfeffffc7a30dd00c67b2065c
#   VERBOSE        default: 0  — set to 1 to show full pip output

set -euo pipefail
exec > >(tee -a brev_train.log) 2>&1

# ---------------------------------------------------------------------------
# 1. Validate + defaults
# ---------------------------------------------------------------------------
: "${HF_TOKEN:?HF_TOKEN is required}"
HF_CKPT_REPO="${HF_CKPT_REPO:-robot-learning-team43/diffusion-policy-checkpoints}"
NUM_STEPS="${NUM_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
LEROBOT_COMMIT="${LEROBOT_COMMIT:-fc6c94c82a4624bdfeffffc7a30dd00c67b2065c}"
VERBOSE="${VERBOSE:-0}"
PIP_QUIET=$( [ "$VERBOSE" = "1" ] && echo "" || echo "--quiet" )

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
echo "[config] $(date) — HF_CKPT_REPO=$HF_CKPT_REPO NUM_STEPS=$NUM_STEPS BATCH_SIZE=$BATCH_SIZE"

export HF_TOKEN HF_CKPT_REPO

# ---------------------------------------------------------------------------
# 2. Create venv + install lerobot (idempotent)
# ---------------------------------------------------------------------------
VENV_DIR="$HOME/lerobot-env"
PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

if [ ! -d "$VENV_DIR" ]; then
    echo "[setup] creating venv at $VENV_DIR"
    # lerobot requires Python >=3.12; prefer python3.12 explicitly
    PY3=$(command -v python3.12 || command -v python3 || true)
    : "${PY3:?Could not find python3.12 — install it first: sudo apt install python3.12 python3.12-venv}"
    "$PY3" -m venv "$VENV_DIR"
fi

LEROBOT_DIR="$(dirname "$REPO_ROOT")/lerobot"
if [ ! -d "$LEROBOT_DIR" ]; then
    echo "[setup] cloning lerobot"
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi
git -C "$LEROBOT_DIR" checkout "$LEROBOT_COMMIT"

echo "[setup] installing lerobot[diffusion] + peft"
"$PIP" install -e "$LEROBOT_DIR[diffusion]" $PIP_QUIET --use-deprecated=legacy-resolver
"$PIP" install peft $PIP_QUIET --use-deprecated=legacy-resolver

# ---------------------------------------------------------------------------
# 3. HF write preflight (fail fast before any GPU time)
# ---------------------------------------------------------------------------
echo "[preflight] testing HF write access..."
"$PY" <<'PYEOF'
from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_CKPT_REPO"]
api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
api.upload_file(
    path_or_fileobj=b"write_test",
    path_in_repo=".write_test",
    repo_id=repo_id,
    repo_type="model",
)
api.delete_file(path_in_repo=".write_test", repo_id=repo_id, repo_type="model")
print("[preflight] HF write access confirmed for", repo_id)
PYEOF

# ---------------------------------------------------------------------------
# 4. Download dataset
# ---------------------------------------------------------------------------
echo "[dataset] downloading robot-learning-team43/so101_teleop_private..."
"$PY" <<'PYEOF'
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="robot-learning-team43/so101_teleop_private",
    repo_type="dataset",
    local_dir="data/so101_teleop",
    token=os.environ["HF_TOKEN"],
)
print("[dataset] download complete → data/so101_teleop")
PYEOF

# ---------------------------------------------------------------------------
# 5. Pre-set output dir + start background checkpoint watcher
# ---------------------------------------------------------------------------
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
OUTPUT_DIR="$REPO_ROOT/diffusion-policy/outputs/diff_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

export HF_TOKEN HF_CKPT_REPO OUTPUT_DIR

(
  while true; do
    sleep 900
    "$PY" <<'PYEOF' || true
from huggingface_hub import upload_folder
import os

output_dir = os.environ["OUTPUT_DIR"]
if not os.path.exists(output_dir) or not os.listdir(output_dir):
    print("[watcher] output dir empty, skipping")
else:
    upload_folder(
        folder_path=output_dir,
        repo_id=os.environ["HF_CKPT_REPO"],
        repo_type="model",
        path_in_repo=os.path.basename(output_dir),
        token=os.environ["HF_TOKEN"],
        commit_message="[brev watcher] checkpoint sync",
    )
    print("[watcher] synced", output_dir)
PYEOF
  done
) &
WATCHER_PID=$!
trap "kill $WATCHER_PID 2>/dev/null || true" EXIT
echo "[watcher] background upload process started (PID=$WATCHER_PID, poll every 15 min)"

# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------
echo "[train] starting at $(date)"
"$PY" diffusion-policy/train.py \
    --output-dir "$OUTPUT_DIR" \
    --episode-filter \
    --backbone dinov2 \
    --horizon 64 \
    --n-action-steps 32 \
    --num-steps "$NUM_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --save-every "$SAVE_EVERY" \
    --lr 1e-4 \
    --device cuda \
    "$@"
echo "[train] completed at $(date)"

# ---------------------------------------------------------------------------
# 7. Kill watcher + final upload
# ---------------------------------------------------------------------------
kill $WATCHER_PID 2>/dev/null || true
trap - EXIT

echo "[upload] final sync of $OUTPUT_DIR..."
"$PY" <<'PYEOF'
from huggingface_hub import upload_folder
import os

output_dir = os.environ["OUTPUT_DIR"]
upload_folder(
    folder_path=output_dir,
    repo_id=os.environ["HF_CKPT_REPO"],
    repo_type="model",
    path_in_repo=os.path.basename(output_dir),
    token=os.environ["HF_TOKEN"],
    commit_message=f"[brev final] {os.path.basename(output_dir)}",
)
print("[upload] done →", os.environ["HF_CKPT_REPO"] + "/" + os.path.basename(output_dir))
PYEOF

# ---------------------------------------------------------------------------
# 8. Shutdown
# ---------------------------------------------------------------------------
echo "[done] $(date) — shutting down in 60 s (Ctrl-C to abort)"
sleep 60
sudo shutdown -h now
