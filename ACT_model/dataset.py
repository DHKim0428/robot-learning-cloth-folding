"""LeRobotDataset wrapper for ACT training.

Builds the right `delta_timestamps` for action chunking and applies the
project-wide episode filter (`config/episode_filter.toml`) by passing an
explicit `episodes=[...]` list to `LeRobotDataset`.
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
    """Build the `delta_timestamps` dict that feeds ACT-style action chunking."""
    delta = {"action": _make_delta_timestamps(cfg.action_delta_indices, fps)}
    delta |= {
        k: _make_delta_timestamps(cfg.observation_delta_indices, fps)
        for k in cfg.image_features
    }
    return delta


def select_episodes(
    meta: LeRobotDatasetMetadata,
    episode_filter_mode: str | None,
    episodes_whitelist: list[int] | None,
) -> list[int] | None:
    """Resolve the final list of episode indices to load.

    - If `episodes_whitelist` is provided, use it verbatim (handy for overfit tests).
    - Otherwise, take all episodes minus the filter set.
    - Returns None when no filtering is needed (LeRobotDataset then loads all).
    """
    if episodes_whitelist is not None:
        return list(episodes_whitelist)

    ignored = load_episode_filter(episode_filter_mode)
    if not ignored:
        return None

    return [i for i in range(meta.total_episodes) if i not in ignored]


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
    return dataset, meta
