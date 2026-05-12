#!/usr/bin/env bash
# Train native LeRobot DiffusionPolicy with episodes selected from episode_filter.toml.
#
# This avoids materializing a filtered dataset with `lerobot-edit-dataset`, which
# can fail if the source dataset's video/episode metadata is inconsistent.
#
# Usage:
#   bash shell/train_diffusion_filtered.sh bad --steps=50000 --batch_size=8
#   bash shell/train_diffusion_filtered.sh meh --steps=50000 --batch_size=8

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-bad}"
if [[ "$MODE" != "bad" && "$MODE" != "meh" ]]; then
    echo "usage: $0 [bad|meh] [extra train args]" >&2
    exit 2
fi
shift || true

DATASET_REPO_ID="${DATASET_REPO_ID:-robot-learning-team43/so101_teleop_private}"

KEEP_EPISODES="$(python - "$MODE" "$DATASET_REPO_ID" "$REPO_ROOT/config/episode_filter.toml" <<'PY'
import json
import sys
import tomllib
from pathlib import Path

from lerobot.datasets import LeRobotDatasetMetadata

mode = sys.argv[1]
repo_id = sys.argv[2]
filter_path = Path(sys.argv[3])

meta = LeRobotDatasetMetadata(repo_id)

with filter_path.open("rb") as f:
    config = tomllib.load(f)

episodes = config.get("episodes", {})
ignored = set(int(ep) for ep in episodes.get("bad", []))
if mode == "meh":
    ignored.update(int(ep) for ep in episodes.get("meh", []))

keep = [idx for idx in range(meta.total_episodes) if idx not in ignored]
print(json.dumps(keep))
PY
)"

echo "[filter] dataset repo : $DATASET_REPO_ID"
echo "[filter] mode         : $MODE"
echo "[filter] keep episodes: $KEEP_EPISODES"
echo

DATASET_REPO_ID="$DATASET_REPO_ID" \
    exec bash shell/train_diffusion.sh \
    --dataset.episodes="$KEEP_EPISODES" \
    "$@"
