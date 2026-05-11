#!/usr/bin/env bash
# Create a filtered LeRobot dataset from the Team 43 HF dataset.
#
# This wrapper is intentionally non-destructive: it always writes to a
# `--new_repo_id` and never deletes episodes from the source dataset in place.
#
# Usage:
#   bash shell/filter_diffusion_dataset.sh bad
#   bash shell/filter_diffusion_dataset.sh meh robot-learning-team43/so101_teleop_private_filtered_meh
#
# Environment overrides:
#   SRC_REPO_ID=...       source dataset repo id
#   PUSH_TO_HUB=true      push the filtered dataset to the Hub
#   DRY_RUN=true          print the command without running it

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-bad}"
SRC_REPO_ID="${SRC_REPO_ID:-robot-learning-team43/so101_teleop_private}"

case "$MODE" in
    bad)
        DEFAULT_DST_REPO_ID="robot-learning-team43/so101_teleop_private_filtered_bad"
        ;;
    meh)
        DEFAULT_DST_REPO_ID="robot-learning-team43/so101_teleop_private_filtered_meh"
        ;;
    *)
        echo "usage: $0 [bad|meh] [new_repo_id]" >&2
        exit 2
        ;;
esac

DST_REPO_ID="${2:-$DEFAULT_DST_REPO_ID}"

EPISODES="$(python - "$MODE" "$REPO_ROOT/config/episode_filter.toml" <<'PY'
import json
import sys
import tomllib
from pathlib import Path

mode = sys.argv[1]
path = Path(sys.argv[2])

with path.open("rb") as f:
    config = tomllib.load(f)

episodes = config.get("episodes", {})
ignored = set(int(ep) for ep in episodes.get("bad", []))
if mode == "meh":
    ignored.update(int(ep) for ep in episodes.get("meh", []))

print(json.dumps(sorted(ignored)))
PY
)"

if [[ "$EPISODES" == "[]" ]]; then
    echo "[error] no episodes found for mode=$MODE in config/episode_filter.toml" >&2
    exit 1
fi

echo "[filter] source repo : $SRC_REPO_ID"
echo "[filter] output repo : $DST_REPO_ID"
echo "[filter] mode        : $MODE"
echo "[filter] delete eps  : $EPISODES"
echo

CMD=(
    lerobot-edit-dataset
    --repo_id "$SRC_REPO_ID"
    --new_repo_id "$DST_REPO_ID"
    --operation.type delete_episodes
    --operation.episode_indices "$EPISODES"
)

if [[ "${PUSH_TO_HUB:-false}" == "true" ]]; then
    CMD+=(--push_to_hub true)
    echo "[filter] push_to_hub : true"
else
    echo "[filter] push_to_hub : false (local/cache only)"
fi

echo "[filter] running: ${CMD[*]}"
if [[ "${DRY_RUN:-false}" == "true" ]]; then
    exit 0
fi

exec "${CMD[@]}"
