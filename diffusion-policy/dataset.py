"""LeRobotDataset wrapper for Diffusion Policy training.

Builds the right ``delta_timestamps`` for action chunking and applies the
project-wide episode filter (``config/episode_filter.toml``) by passing an
explicit ``episodes=[...]`` list to ``LeRobotDataset``.

The ``DiffusionConfig`` exposes the same ``observation_delta_indices`` /
``action_delta_indices`` / ``image_features`` attributes that ACT does, so
this module is the ACT version verbatim modulo the docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Reuse the TOML parser from nn-approach-eleni instead of duplicating it.
sys.path.insert(0, str(PROJECT_ROOT / "nn-approach-eleni"))
from features import load_episode_filter as _load_episode_filter  # noqa: E402


def load_episode_filter(mode: str | None) -> set[int]:
    """Wrapper that resolves the filter TOML at the project-level path."""
    return _load_episode_filter(mode, path=PROJECT_ROOT / "config" / "episode_filter.toml")


def _make_delta_timestamps(delta_indices, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def build_delta_timestamps(cfg, fps: int) -> dict[str, list[float]]:
    """Build the ``delta_timestamps`` dict that feeds diffusion action chunking."""
    delta = {"action": _make_delta_timestamps(cfg.action_delta_indices, fps)}
    delta |= {
        k: _make_delta_timestamps(cfg.observation_delta_indices, fps)
        for k in cfg.input_features
    }
    return delta


def select_episodes(
    meta: LeRobotDatasetMetadata,
    episode_filter_mode: str | None,
    episodes_whitelist: list[int] | None,
) -> list[int] | None:
    """Resolve the final list of episode indices to load.

    - If ``episodes_whitelist`` is provided, use it verbatim (handy for overfit tests).
    - Otherwise, take all episodes minus the filter set.
    - Returns None when no filtering is needed (LeRobotDataset then loads all).
    """
    if episodes_whitelist is not None:
        return list(episodes_whitelist)

    ignored = load_episode_filter(episode_filter_mode)
    if not ignored:
        return None

    return [i for i in range(meta.total_episodes) if i not in ignored]


def _scalar(value) -> int:
    """Convert pyarrow / numpy / torch-ish scalar values to plain ints."""
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _episode_bounds(meta: LeRobotDatasetMetadata) -> dict[int, tuple[int, int]]:
    """Return canonical absolute frame bounds for each episode.

    Some local caches can contain stale parquet files after recording or syncing.
    ``info.json`` still carries the canonical totals, so only the first
    ``total_episodes`` metadata rows are trusted here.
    """
    bounds: dict[int, tuple[int, int]] = {}
    for row_idx in range(min(meta.total_episodes, len(meta.episodes))):
        episode = meta.episodes[row_idx]
        ep_idx = _scalar(episode["episode_index"])
        if ep_idx in bounds:
            continue
        bounds[ep_idx] = (
            _scalar(episode["dataset_from_index"]),
            _scalar(episode["dataset_to_index"]),
        )
    return bounds


def _sanitize_loaded_rows(dataset: LeRobotDataset) -> None:
    """Drop stale rows from the loaded HF table without mutating files on disk."""
    reader = dataset.reader
    if reader is None or reader.hf_dataset is None:
        return

    hf_dataset = reader.hf_dataset
    bounds = _episode_bounds(dataset.meta)
    total_tasks = len(dataset.meta.tasks)
    episode_indices = hf_dataset.data.column("episode_index").to_pylist()
    absolute_indices = hf_dataset.data.column("index").to_pylist()
    task_indices = hf_dataset.data.column("task_index").to_pylist()

    valid_rows: list[int] = []
    seen_absolute_indices: set[int] = set()
    for rel_idx, (ep_value, abs_value, task_value) in enumerate(
        zip(episode_indices, absolute_indices, task_indices, strict=True)
    ):
        ep_idx = _scalar(ep_value)
        abs_idx = _scalar(abs_value)
        task_idx = _scalar(task_value)
        ep_bounds = bounds.get(ep_idx)
        if ep_bounds is None:
            continue
        ep_start, ep_end = ep_bounds
        if not (ep_start <= abs_idx < ep_end):
            continue
        if not (0 <= task_idx < total_tasks):
            continue
        if abs_idx in seen_absolute_indices:
            continue
        seen_absolute_indices.add(abs_idx)
        valid_rows.append(rel_idx)

    if len(valid_rows) == len(hf_dataset):
        return

    removed = len(hf_dataset) - len(valid_rows)
    reader.hf_dataset = hf_dataset.select(valid_rows)
    reader._build_index_mapping()
    print(
        f"[warn] dropped {removed} stale/invalid rows from local dataset cache view "
        f"({len(reader.hf_dataset)} valid rows remain)"
    )


def build_dataset(
    repo_id: str,
    dataset_root: Path,
    cfg,
    episode_filter_mode: str | None,
    episodes_whitelist: list[int] | None = None,
) -> tuple[LeRobotDataset, LeRobotDatasetMetadata]:
    meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    delta_timestamps = build_delta_timestamps(cfg, meta.fps)
    episodes = select_episodes(meta, episode_filter_mode, episodes_whitelist)

    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
    )
    _sanitize_loaded_rows(dataset)
    return dataset, meta
