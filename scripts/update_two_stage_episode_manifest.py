import argparse
import json
import re
import tomllib
from pathlib import Path

from script_utils import DEFAULT_TWO_STAGE_DATASET_ROOT, PROJECT_ROOT


DEFAULT_DATASET_REPO_ID = "local/so101_two_stage"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "config" / "two_stage_episode_manifest.toml"
TASK_STAGE_RE = re.compile(r"\bstage\s+([12])\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count two-stage LeRobot episodes and append missing default entries "
            "to the two-stage episode manifest."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_TWO_STAGE_DATASET_ROOT,
        help="Base directory containing the two-stage LeRobot dataset.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=DEFAULT_DATASET_REPO_ID,
        help="Dataset repo id under --dataset-root.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Episode manifest TOML path.",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Only print the dataset episode count and manifest coverage; do not write anything.",
    )
    return parser.parse_args()


def dataset_path(args: argparse.Namespace) -> Path:
    return args.dataset_root / args.dataset_repo_id


def load_existing_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}

    with path.open("rb") as f:
        return tomllib.load(f)


def existing_episode_indices(manifest: dict[str, object]) -> set[int]:
    episodes = manifest.get("episodes", {})
    if not isinstance(episodes, dict):
        raise ValueError("Manifest [episodes] table must be a TOML table.")

    indices: set[int] = set()
    for key in episodes:
        try:
            indices.add(int(key))
        except ValueError as e:
            raise ValueError(f"Invalid episode key in manifest: {key!r}") from e
    return indices


def read_total_episodes_from_info(root: Path) -> int | None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    total = info.get("total_episodes")
    return int(total) if total is not None else None


def read_episode_rows(root: Path) -> dict[int, dict[str, object]]:
    episodes_root = root / "meta" / "episodes"
    rows: dict[int, dict[str, object]] = {}
    parquet_paths = sorted(episodes_root.rglob("*.parquet"))
    if not parquet_paths:
        return rows

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "pyarrow is required to read LeRobot episode metadata. Run this script "
            "inside the `lerobot` environment."
        ) from e

    for path in parquet_paths:
        table = pq.read_table(path, columns=["episode_index", "tasks"])
        data = table.to_pydict()
        for episode_index, tasks in zip(data["episode_index"], data["tasks"], strict=True):
            rows[int(episode_index)] = {"tasks": list(tasks or [])}
    return rows


def infer_stage(episode_index: int, episode_rows: dict[int, dict[str, object]]) -> int | None:
    row = episode_rows.get(episode_index)
    if not row:
        return None

    tasks = row.get("tasks", [])
    if not isinstance(tasks, list):
        return None

    for task in tasks:
        match = TASK_STAGE_RE.search(str(task))
        if match:
            return int(match.group(1))
    return None


def dataset_episode_indices(root: Path) -> list[int]:
    episode_rows = read_episode_rows(root)
    if episode_rows:
        return sorted(episode_rows)

    total = read_total_episodes_from_info(root)
    if total is None:
        raise FileNotFoundError(f"No LeRobot episode metadata found under: {root}")
    return list(range(total))


def count_stages(
    episode_indices: list[int],
    episode_rows: dict[int, dict[str, object]],
) -> dict[str, int]:
    counts = {"total": len(episode_indices), "stage_1": 0, "stage_2": 0, "unknown": 0}
    for episode_index in episode_indices:
        stage = infer_stage(episode_index, episode_rows)
        if stage == 1:
            counts["stage_1"] += 1
        elif stage == 2:
            counts["stage_2"] += 1
        else:
            counts["unknown"] += 1
    return counts


def print_stage_counts(counts: dict[str, int]) -> None:
    print(f"Total episodes: {counts['total']}")
    print(f"Stage 1 episodes: {counts['stage_1']}")
    print(f"Stage 2 episodes: {counts['stage_2']}")
    if counts["unknown"]:
        print(f"Unknown-stage episodes: {counts['unknown']}")


def manifest_header() -> str:
    return (
        "# Two-stage folding episode manifest.\n"
        "#\n"
        "# This file is append-only by convention. Do not rewrite existing episode\n"
        "# entries; update individual values only when intentionally relabeling data.\n"
        "#\n"
        "# Each episode entry should look like:\n"
        "#\n"
        "# [episodes.\"0\"]\n"
        "# stage = 1\n"
        "# quality = \"clean\"\n"
        "# notes = \"\"\n"
        "\n"
        "[episodes]\n"
    )


def format_episode_entry(
    episode_index: int,
    *,
    stage: int | None,
    quality: str = "clean",
) -> str:
    lines = [
        "",
        f"[episodes.\"{episode_index}\"]",
    ]
    if stage is not None:
        lines.append(f"stage = {stage}")
    else:
        lines.append("# stage = 1")
    lines.extend(
        [
            f'quality = "{quality}"',
            'notes = ""',
        ]
    )
    return "\n".join(lines) + "\n"


def append_missing_entries(
    manifest_path: Path,
    missing_indices: list[int],
    episode_rows: dict[int, dict[str, object]],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        manifest_path.write_text(manifest_header(), encoding="utf-8")

    with manifest_path.open("a", encoding="utf-8") as f:
        for episode_index in missing_indices:
            f.write(
                format_episode_entry(
                    episode_index,
                    stage=infer_stage(episode_index, episode_rows),
                )
            )


def main() -> None:
    args = parse_args()
    root = dataset_path(args)
    episode_rows = read_episode_rows(root)
    episode_indices = sorted(episode_rows) if episode_rows else dataset_episode_indices(root)
    stage_counts = count_stages(episode_indices, episode_rows)

    if args.count:
        print_stage_counts(stage_counts)
        return

    manifest = load_existing_manifest(args.manifest)
    existing_indices = existing_episode_indices(manifest)
    missing_indices = [idx for idx in episode_indices if idx not in existing_indices]

    print(f"Dataset: {root}")
    print_stage_counts(stage_counts)
    print(f"Episodes in manifest: {len(existing_indices)}")
    print(f"Missing manifest entries: {len(missing_indices)}")

    if not missing_indices:
        print("Manifest already covers all current episodes. No changes made.")
        return

    append_missing_entries(args.manifest, missing_indices, episode_rows)
    print(f"Appended {len(missing_indices)} entries to: {args.manifest}")


if __name__ == "__main__":
    main()
