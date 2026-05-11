"""Build a LeRobot DiffusionPolicy from dataset metadata.

This mirrors `ACT_model/build_policy.py`, but wraps LeRobot's built-in
`lerobot.policies.diffusion` implementation instead of defining a custom
policy. The defaults intentionally stay close to LeRobot's DiffusionConfig
defaults; CLI flags only override values when explicitly provided.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.policies.diffusion.processor_diffusion import (
    make_diffusion_pre_post_processors,
)
from lerobot.utils.feature_utils import dataset_to_policy_features

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from dataset import HF_DATASET_ACCESS_HINT, load_metadata  # noqa: E402


def _optional_tuple(values: list[int] | None) -> tuple[int, int] | None:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError(f"Expected exactly two dimensions, got: {values}")
    return int(values[0]), int(values[1])


def build_diffusion_policy(
    dataset_meta: LeRobotDatasetMetadata,
    *,
    device: str = "cuda",
    n_obs_steps: int | None = None,
    horizon: int | None = None,
    n_action_steps: int | None = None,
    vision_backbone: str | None = None,
    noise_scheduler_type: str | None = None,
    num_train_timesteps: int | None = None,
    num_inference_steps: int | None = None,
    resize_shape: tuple[int, int] | None = None,
    crop_ratio: float | None = None,
    use_group_norm: bool | None = None,
    pretrained_backbone_weights: str | None = None,
    do_mask_loss_for_padding: bool | None = None,
) -> tuple[DiffusionPolicy, Any, Any, DiffusionConfig]:
    """Construct DiffusionConfig + DiffusionPolicy + processors.

    Returns `(policy, preprocessor, postprocessor, cfg)`.
    """
    features = dataset_to_policy_features(dataset_meta.features)
    output_features = {
        key: feature
        for key, feature in features.items()
        if feature.type is FeatureType.ACTION
    }
    input_features = {
        key: feature for key, feature in features.items() if key not in output_features
    }

    cfg_kwargs: dict[str, Any] = {
        "input_features": input_features,
        "output_features": output_features,
        "device": device,
    }
    optional_overrides = {
        "n_obs_steps": n_obs_steps,
        "horizon": horizon,
        "n_action_steps": n_action_steps,
        "vision_backbone": vision_backbone,
        "noise_scheduler_type": noise_scheduler_type,
        "num_train_timesteps": num_train_timesteps,
        "num_inference_steps": num_inference_steps,
        "resize_shape": resize_shape,
        "crop_ratio": crop_ratio,
        "use_group_norm": use_group_norm,
        "pretrained_backbone_weights": pretrained_backbone_weights,
        "do_mask_loss_for_padding": do_mask_loss_for_padding,
    }
    cfg_kwargs.update(
        {key: value for key, value in optional_overrides.items() if value is not None}
    )

    cfg = DiffusionConfig(**cfg_kwargs)
    policy = DiffusionPolicy(cfg)
    preprocessor, postprocessor = make_diffusion_pre_post_processors(
        cfg, dataset_stats=dataset_meta.stats
    )
    return policy, preprocessor, postprocessor, cfg


def add_diffusion_config_args(parser: argparse.ArgumentParser) -> None:
    """Add optional DiffusionConfig overrides while preserving defaults."""
    parser.add_argument("--n-obs-steps", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--vision-backbone", default=None)
    parser.add_argument(
        "--noise-scheduler-type",
        choices=["DDPM", "DDIM"],
        default=None,
    )
    parser.add_argument("--num-train-timesteps", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--resize-shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Optional image resize override. Omit to keep LeRobot default.",
    )
    parser.add_argument("--crop-ratio", type=float, default=None)
    parser.add_argument(
        "--use-group-norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional override. Omit to keep LeRobot default.",
    )
    parser.add_argument("--pretrained-backbone-weights", default=None)
    parser.add_argument(
        "--do-mask-loss-for-padding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional override. Omit to keep LeRobot default.",
    )


def diffusion_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_obs_steps": args.n_obs_steps,
        "horizon": args.horizon,
        "n_action_steps": args.n_action_steps,
        "vision_backbone": args.vision_backbone,
        "noise_scheduler_type": args.noise_scheduler_type,
        "num_train_timesteps": args.num_train_timesteps,
        "num_inference_steps": args.num_inference_steps,
        "resize_shape": _optional_tuple(args.resize_shape),
        "crop_ratio": args.crop_ratio,
        "use_group_norm": args.use_group_norm,
        "pretrained_backbone_weights": args.pretrained_backbone_weights,
        "do_mask_loss_for_padding": args.do_mask_loss_for_padding,
    }


def serializable_diffusion_config(cfg: DiffusionConfig) -> dict[str, Any]:
    """Persist only the knobs needed to rebuild the checkpoint architecture."""
    keys = [
        "n_obs_steps",
        "horizon",
        "n_action_steps",
        "vision_backbone",
        "noise_scheduler_type",
        "num_train_timesteps",
        "num_inference_steps",
        "resize_shape",
        "crop_ratio",
        "use_group_norm",
        "pretrained_backbone_weights",
        "do_mask_loss_for_padding",
    ]
    payload: dict[str, Any] = {}
    for key in keys:
        value = getattr(cfg, key)
        if isinstance(value, tuple):
            value = list(value)
        payload[key] = value
    return payload


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-repo-id",
        default="robot-learning-team43/so101_teleop_private",
        help="Hugging Face LeRobot dataset repo id.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional local dataset root override. Omit to use the HF cache/download path.",
    )
    parser.add_argument("--device", default="cpu")
    add_diffusion_config_args(parser)
    args = parser.parse_args()

    try:
        meta = load_metadata(args.dataset_repo_id, args.dataset_root)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load dataset metadata for {args.dataset_repo_id!r}.\n"
            f"{HF_DATASET_ACCESS_HINT}"
        ) from exc

    policy, pre, _post, cfg = build_diffusion_policy(
        meta,
        device=args.device,
        **diffusion_kwargs_from_args(args),
    )
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"DiffusionPolicy built — {n_params/1e6:.2f} M params")
    print(f"  input_features: {list(cfg.input_features.keys())}")
    print(f"  output_features: {list(cfg.output_features.keys())}")
    print(
        "  "
        f"n_obs_steps={cfg.n_obs_steps}, horizon={cfg.horizon}, "
        f"n_action_steps={cfg.n_action_steps}"
    )
    print(
        "  "
        f"scheduler={cfg.noise_scheduler_type}, "
        f"train_steps={cfg.num_train_timesteps}, "
        f"inference_steps={cfg.num_inference_steps}"
    )
    print(
        "  "
        f"vision_backbone={cfg.vision_backbone}, "
        f"resize_shape={cfg.resize_shape}, use_group_norm={cfg.use_group_norm}"
    )

    # Synthetic smoke forward: shapes mirror our current SO-101 dataset.
    bsz = 2
    batch = {
        "observation.state": torch.zeros(bsz, cfg.n_obs_steps, 6),
        "observation.images.front": torch.zeros(bsz, cfg.n_obs_steps, 3, 480, 640),
        "action": torch.zeros(bsz, cfg.horizon, 6),
        "action_is_pad": torch.zeros(bsz, cfg.horizon, dtype=torch.bool),
    }
    batch = pre(batch)
    loss, loss_dict = policy.forward(batch)
    print(
        f"forward OK — loss={float(loss.detach().cpu()):.4f} "
        f"loss_dict={loss_dict}"
    )


if __name__ == "__main__":
    _main()
