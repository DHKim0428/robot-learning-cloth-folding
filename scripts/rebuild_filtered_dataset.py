#!/usr/bin/env python
"""Rebuild a clean filtered LeRobot dataset by replaying frames into a new dataset.

This is a workaround for source datasets whose episode subset metadata works for
full training but breaks LeRobot's `delete_episodes` or `EpisodeAwareSampler`
paths after filtering. Unlike `lerobot-edit-dataset delete_episodes`, this script
creates a fresh dataset and lets LeRobot assign new contiguous episode/frame
indices from zero.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILTER_PATH = PROJECT_ROOT / "config" / "episode_filter.toml"
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "so101_teleop_private"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "lerobot_rebuilt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a clean filtered SO-101 LeRobot dataset."
    )
    parser.add_argument(
        "--source-repo-id",
        default="robot-learning-team43/so101_teleop_private",
        help="Source LeRobot dataset repo id.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Exact local source dataset directory containing meta/, data/, videos/.",
    )
    parser.add_argument(
        "--target-repo-id",
        default="robot-learning-team43/so101_teleop_private_rebuilt_meh",
        help="Repo id stored in the rebuilt dataset metadata.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=None,
        help=(
            "Exact output dataset directory. Default: "
            "data/lerobot_rebuilt/<target-repo-id>."
        ),
    )
    parser.add_argument(
        "--episode-filter",
        nargs="?",
        const="bad",
        choices=["bad", "meh"],
        default="meh",
        help="Quality filter to apply from config/episode_filter.toml.",
    )
    parser.add_argument("--filter-path", type=Path, default=DEFAULT_FILTER_PATH)
    parser.add_argument(
        "--video-backend",
        default="pyav",
        help="Video backend used for reading source frames and writing output metadata.",
    )
    parser.add_argument(
        "--vcodec",
        default="libsvtav1",
        help="Video codec for the rebuilt dataset.",
    )
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit source episode whitelist; overrides --episode-filter.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for smoke testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete target-root first if it already exists.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push rebuilt dataset to Hugging Face Hub after finalize().",
    )
    return parser.parse_args()


def load_ignored(mode: str | None, path: Path) -> set[int]:
    if mode is None:
        return set()
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    episodes = cfg.get("episodes", {})
    ignored = set(int(ep) for ep in episodes.get("bad", []))
    if mode == "meh":
        ignored.update(int(ep) for ep in episodes.get("meh", []))
    return ignored


def resolve_target_root(target_repo_id: str, target_root: Path | None) -> Path:
    if target_root is not None:
        return target_root
    return DEFAULT_OUTPUT_ROOT / target_repo_id


def strip_auto_features(features: dict[str, Any]) -> dict[str, Any]:
    auto_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    return {key: value for key, value in features.items() if key not in auto_keys}


def to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def image_to_hwc_uint8(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        img = value.detach().cpu()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)
        if img.dtype != torch.uint8:
            img = (img.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        return img.numpy()

    img = np.asarray(value)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img.astype(np.float32), 0.0, 1.0)
        img = np.rint(img * 255.0).astype(np.uint8)
    return img


def build_frame(item: dict[str, Any], feature_keys: list[str], camera_keys: set[str]) -> dict[str, Any]:
    frame: dict[str, Any] = {}
    for key in feature_keys:
        if key in camera_keys:
            frame[key] = image_to_hwc_uint8(item[key])
        else:
            frame[key] = to_numpy(item[key])
    frame["task"] = item.get("task", "SO101 teleoperation task")
    return frame


def main() -> None:
    args = parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_root = args.source_root
    if not (source_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Invalid source dataset root: {source_root}")

    target_root = resolve_target_root(args.target_repo_id, args.target_root)
    if target_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Target root already exists: {target_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(target_root)

    meta_only = LeRobotDataset(args.source_repo_id, root=source_root)
    if args.episodes is None:
        ignored = load_ignored(args.episode_filter, args.filter_path)
        episodes = [idx for idx in range(meta_only.meta.total_episodes) if idx not in ignored]
    else:
        episodes = list(args.episodes)

    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    print(f"[rebuild] source repo : {args.source_repo_id}")
    print(f"[rebuild] source root : {source_root}")
    print(f"[rebuild] target repo : {args.target_repo_id}")
    print(f"[rebuild] target root : {target_root}")
    print(f"[rebuild] episodes    : {episodes}")
    print(f"[rebuild] count       : {len(episodes)}")

    features = strip_auto_features(meta_only.features)
    camera_keys = set(meta_only.meta.camera_keys)
    feature_keys = list(features.keys())

    target = LeRobotDataset.create(
        repo_id=args.target_repo_id,
        fps=meta_only.fps,
        features=features,
        root=target_root,
        use_videos=True,
        video_backend=args.video_backend,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
        vcodec=args.vcodec,
    )

    failures: list[tuple[int, str]] = []
    for source_ep in episodes:
        try:
            source = LeRobotDataset(
                args.source_repo_id,
                root=source_root,
                episodes=[source_ep],
                video_backend=args.video_backend,
            )
            for idx in tqdm(range(len(source)), desc=f"episode {source_ep}"):
                item = source[idx]
                target.add_frame(build_frame(item, feature_keys, camera_keys))
            target.save_episode()
        except Exception as exc:  # noqa: BLE001
            failures.append((source_ep, repr(exc)))
            print(f"[warn] failed episode {source_ep}: {exc!r}", file=sys.stderr)

    target.finalize()
    print(f"[done] rebuilt dataset at {target_root}")
    print(f"[done] episodes saved: {target.meta.total_episodes}")
    print(f"[done] frames saved  : {target.meta.total_frames}")

    if failures:
        print("[warn] failed source episodes:")
        for ep, msg in failures:
            print(f"  - {ep}: {msg}")

    if args.push_to_hub:
        target.push_to_hub()
        print(f"[done] pushed to hub as {args.target_repo_id}")


if __name__ == "__main__":
    main()
