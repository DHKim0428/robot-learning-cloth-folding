"""Build an ACT policy + pre/post processors from LeRobot dataset metadata.

Thin wrapper around `lerobot.policies.act.{ACTConfig,ACTPolicy}` so that
`train.py` and `rollout.py` share the exact same construction logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.utils.feature_utils import dataset_to_policy_features


def build_act_policy(
    dataset_meta: LeRobotDatasetMetadata,
    chunk_size: int = 100,
    n_action_steps: int | None = None,
    vision_backbone: str = "resnet18",
    use_vae: bool = True,
    device: str = "cuda",
):
    """Construct ACTConfig + ACTPolicy + pre/post processors for the given dataset.

    Returns (policy, preprocessor, postprocessor, cfg).
    """
    features = dataset_to_policy_features(dataset_meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps if n_action_steps is not None else chunk_size,
        vision_backbone=vision_backbone,
        use_vae=use_vae,
        device=device,
    )
    policy = ACTPolicy(cfg)
    preprocessor, postprocessor = make_act_pre_post_processors(
        cfg, dataset_stats=dataset_meta.stats
    )
    return policy, preprocessor, postprocessor, cfg


def make_delta_timestamps(delta_indices, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


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
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    policy, pre, post, cfg = build_act_policy(
        meta, chunk_size=args.chunk_size, device=args.device
    )

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"ACTPolicy built — {n_params/1e6:.2f} M params")
    print(f"  input_features: {list(cfg.input_features.keys())}")
    print(f"  output_features: {list(cfg.output_features.keys())}")
    print(f"  chunk_size={cfg.chunk_size}, n_action_steps={cfg.n_action_steps}")
    print(f"  use_vae={cfg.use_vae}, vision_backbone={cfg.vision_backbone}")

    # one forward pass on a synthetic batch as a smoke test
    bsz = 2
    batch = {
        "observation.state": torch.zeros(bsz, 6),
        "observation.images.front": torch.zeros(bsz, 3, 480, 640),
        "action": torch.zeros(bsz, cfg.chunk_size, 6),
        "action_is_pad": torch.zeros(bsz, cfg.chunk_size, dtype=torch.bool),
    }
    batch = pre(batch)
    loss, loss_dict = policy.forward(batch)
    print(f"forward OK — loss={loss.item():.4f} loss_dict={loss_dict}")


if __name__ == "__main__":
    _main()
