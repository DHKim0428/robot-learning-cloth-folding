"""Build a Diffusion Policy + pre/post processors from LeRobot dataset metadata.

Thin wrapper around ``lerobot.policies.diffusion.{DiffusionConfig,DiffusionPolicy}``.
For ``backbone="dinov2"`` (default) the default ResNet18 RGB encoder is
constructed first (so lerobot's U-Net is sized with the correct
``global_cond_dim``) and then replaced with a frozen DINOv2 encoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.configs import FeatureType, NormalizationMode
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.policies.diffusion.processor_diffusion import (
    make_diffusion_pre_post_processors,
)
from lerobot.utils.feature_utils import dataset_to_policy_features

# Allow `python diffusion-policy/build_policy.py` to import the sibling encoder
# module without packaging tricks.
import sys
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from encoder import DINOv2RgbEncoder  # noqa: E402


def _build_normalization_mapping(backbone: str) -> dict[str, NormalizationMode]:
    """Return the per-feature normalization mapping for ``DiffusionConfig``.

    DINOv2 applies its own ImageNet normalization inside ``forward``, so the
    pre-processor must not also normalize images for that backbone.
    """
    if backbone == "dinov2":
        return {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    return {
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
    }


def build_diffusion_policy(
    dataset_meta: LeRobotDatasetMetadata,
    backbone: str = "dinov2",
    lora_depth: int = 0,
    lora_rank: int = 8,
    horizon: int = 64,
    n_action_steps: int = 32,
    n_obs_steps: int = 2,
    noise_scheduler_type: str = "DDIM",
    num_train_timesteps: int = 100,
    num_inference_steps: int = 10,
    spatial_softmax_num_keypoints: int = 32,
    device: str = "cuda",
):
    """Construct DiffusionConfig + DiffusionPolicy + pre/post processors.

    Returns ``(policy, preprocessor, postprocessor, cfg)``.

    Args:
        backbone: ``"dinov2"`` (default, frozen ViT-S/14 + LoRA gate) or
            ``"resnet18"`` (ImageNet-pretrained, matches
            ``FUTURE_DIFFUSION_NOTES.md`` baseline).
        lora_depth: When ``backbone="dinov2"`` and ``>0``, attach LoRA
            adapters to the last ``lora_depth`` transformer blocks.
            Ignored for the resnet18 backbone.
        lora_rank: LoRA rank when ``lora_depth > 0``.
    """
    if backbone not in {"dinov2", "resnet18"}:
        raise ValueError(f"Unknown backbone {backbone!r}")

    features = dataset_to_policy_features(dataset_meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    # ``DiffusionConfig.__post_init__`` rejects non-resnet ``vision_backbone``
    # values, so we always pass "resnet18" here. For backbone="dinov2" the
    # ResNet18 RGB encoder built by lerobot is overwritten further down.
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        vision_backbone="resnet18",
        pretrained_backbone_weights=(
            "ResNet18_Weights.IMAGENET1K_V1" if backbone == "resnet18" else None
        ),
        use_group_norm=False,
        spatial_softmax_num_keypoints=spatial_softmax_num_keypoints,
        use_separate_rgb_encoder_per_camera=True,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        noise_scheduler_type=noise_scheduler_type,
        num_train_timesteps=num_train_timesteps,
        num_inference_steps=num_inference_steps,
        normalization_mapping=_build_normalization_mapping(backbone),
        device=device,
    )
    policy = DiffusionPolicy(cfg)

    if backbone == "dinov2":
        # ``policy.diffusion.rgb_encoder`` is an ``nn.ModuleList`` because we
        # set ``use_separate_rgb_encoder_per_camera=True``. Replace each entry
        # with an independent DINOv2 encoder (currently single-camera, but
        # this stays correct if a wrist/scene camera is added later).
        rgb_encoder = policy.diffusion.rgb_encoder
        if not isinstance(rgb_encoder, torch.nn.ModuleList):
            raise RuntimeError(
                "Expected policy.diffusion.rgb_encoder to be an nn.ModuleList. "
                "Got " + type(rgb_encoder).__name__
            )
        for i in range(len(rgb_encoder)):
            rgb_encoder[i] = DINOv2RgbEncoder(
                num_keypoints=spatial_softmax_num_keypoints,
                lora_depth=lora_depth,
                lora_rank=lora_rank,
            )

    preprocessor, postprocessor = make_diffusion_pre_post_processors(
        cfg, dataset_stats=dataset_meta.stats
    )
    return policy, preprocessor, postprocessor, cfg


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-repo-id", default="robot-learning-team43/so101_teleop_private"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "so101_teleop",
    )
    parser.add_argument("--backbone", default="dinov2", choices=["dinov2", "resnet18"])
    parser.add_argument("--lora-depth", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--n-action-steps", type=int, default=32)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    policy, pre, post, cfg = build_diffusion_policy(
        meta,
        backbone=args.backbone,
        lora_depth=args.lora_depth,
        lora_rank=args.lora_rank,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        n_obs_steps=args.n_obs_steps,
        device=args.device,
    )

    n_params = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"DiffusionPolicy built — {n_params/1e6:.2f} M params total, "
          f"{n_trainable/1e6:.2f} M trainable")
    print(f"  backbone={args.backbone} lora_depth={args.lora_depth}")
    print(f"  input_features: {list(cfg.input_features.keys())}")
    print(f"  output_features: {list(cfg.output_features.keys())}")
    print(f"  horizon={cfg.horizon} n_action_steps={cfg.n_action_steps} "
          f"n_obs_steps={cfg.n_obs_steps}")
    print(f"  noise_scheduler={cfg.noise_scheduler_type} "
          f"num_inference_steps={cfg.num_inference_steps}")

    bsz = 2
    img_h, img_w = next(iter(cfg.image_features.values())).shape[1:]
    img_key = next(iter(cfg.image_features))
    batch = {
        "observation.state": torch.zeros(bsz, cfg.n_obs_steps, 6),
        img_key: torch.zeros(bsz, cfg.n_obs_steps, 3, img_h, img_w),
        "action": torch.zeros(bsz, cfg.horizon, 6),
        "action_is_pad": torch.zeros(bsz, cfg.horizon, dtype=torch.bool),
    }
    loss, loss_dict = policy.forward(batch)
    print(f"forward OK — loss={loss.item():.4f} loss_dict={loss_dict}")


if __name__ == "__main__":
    _main()
