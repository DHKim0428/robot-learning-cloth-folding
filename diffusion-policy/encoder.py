"""DINOv2 RGB encoder used as a drop-in for lerobot's ``DiffusionRgbEncoder``.

Phase 1 keeps the DINOv2 ViT-S/14 backbone frozen and trains only a small
SpatialSoftmax + linear head. Phase 2 (``lora_depth > 0``) attaches LoRA
adapters to the last N transformer blocks via ``peft``; the module is built
to be inert at ``lora_depth=0`` so the same code runs both phases.

The output ``feature_dim`` is ``num_keypoints * 2`` — matching what
``DiffusionRgbEncoder`` exposes — so the swapped encoder is interchangeable
with the default ResNet18 path and the U-Net's ``global_cond_dim`` is
unaffected.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SpatialSoftmax(nn.Module):
    """Spatial soft-argmax pooling.

    Maps a ``(B, C, H, W)`` feature map to ``(B, num_kp * 2)`` after the
    caller flattens the returned ``(B, num_kp, 2)`` keypoint tensor — same
    contract as lerobot's internal ``SpatialSoftmax``.
    """

    def __init__(self, input_shape, num_kp: int | None = None):
        super().__init__()
        assert len(input_shape) == 3, "input_shape must be (C, H, W)"
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w),
            np.linspace(-1.0, 1.0, self._in_h),
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        return expected_xy.view(-1, self._out_c, 2)


class DINOv2RgbEncoder(nn.Module):
    """Frozen DINOv2 ViT-S/14 + SpatialSoftmax head.

    The backbone is loaded once from ``torch.hub`` and frozen. When
    ``lora_depth > 0`` we install LoRA adapters on the QKV projections of
    the last N transformer blocks via ``peft`` — those adapter weights
    become the only backbone params with ``requires_grad=True``.

    Forward pipeline (matches the ``DiffusionRgbEncoder`` contract: input in
    ``[0, 1]`` float, output ``(B, feature_dim)``):

        resize → augment (train-only) → ImageNet normalize → backbone
        → reshape patches to feature map → SpatialSoftmax → Linear + ReLU
    """

    INPUT_SIZE = 224  # 14 * 16, required by patch_size=14
    PATCH_GRID = 16
    EMBED_DIM = 384
    NUM_BLOCKS = 12

    def __init__(
        self,
        num_keypoints: int = 32,
        lora_depth: int = 0,
        lora_rank: int = 8,
    ) -> None:
        super().__init__()

        self.lora_depth = lora_depth
        self.lora_rank = lora_rank

        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        for p in backbone.parameters():
            p.requires_grad_(False)

        if lora_depth > 0:
            if lora_depth > self.NUM_BLOCKS:
                raise ValueError(
                    f"lora_depth={lora_depth} exceeds NUM_BLOCKS={self.NUM_BLOCKS}"
                )
            from peft import LoraConfig, get_peft_model

            first = self.NUM_BLOCKS - lora_depth
            cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                target_modules=["qkv"],
                layers_to_transform=list(range(first, self.NUM_BLOCKS)),
                bias="none",
            )
            backbone = get_peft_model(backbone, cfg)

        self.backbone = backbone

        self.resize = torchvision.transforms.Resize(
            (self.INPUT_SIZE, self.INPUT_SIZE), antialias=True
        )
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.aug = torchvision.transforms.Compose(
            [
                torchvision.transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                ),
                torchvision.transforms.RandomGrayscale(p=0.05),
                torchvision.transforms.RandomApply(
                    [
                        torchvision.transforms.GaussianBlur(
                            kernel_size=5, sigma=(0.1, 1.0)
                        )
                    ],
                    p=0.2,
                ),
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2)],
                    p=0.2,
                ),
                torchvision.transforms.RandomErasing(
                    p=0.1, scale=(0.01, 0.05), ratio=(0.3, 3.3), value=0.0
                ),
            ]
        )

        self.pool = SpatialSoftmax(
            (self.EMBED_DIM, self.PATCH_GRID, self.PATCH_GRID), num_kp=num_keypoints
        )
        self.feature_dim = num_keypoints * 2
        self.out = nn.Linear(num_keypoints * 2, num_keypoints * 2)
        self.relu = nn.ReLU()

    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Run the (possibly LoRA-wrapped) backbone and grab the patch tokens.

        DINOv2's ``forward_features`` returns a dict with ``x_norm_patchtokens``
        of shape ``(B, num_patches, embed_dim)``.
        """
        out = self.backbone.forward_features(x)
        return out["x_norm_patchtokens"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Order matters: every augmentation (including RandomErasing) runs in
        # the un-normalized [0, 1] pixel space, *before* the ImageNet
        # Normalize. RandomErasing uses ``value=0.0`` so erased patches read
        # as black pixels rather than a large negative-mean feature spike
        # after normalization.
        x = self.resize(x)
        if self.training:
            x = self.aug(x)
        x = self.normalize(x)

        if self.lora_depth == 0:
            with torch.no_grad():
                patch_tokens = self._extract_patch_tokens(x)
        else:
            patch_tokens = self._extract_patch_tokens(x)

        b = patch_tokens.shape[0]
        feature_map = patch_tokens.permute(0, 2, 1).reshape(
            b, self.EMBED_DIM, self.PATCH_GRID, self.PATCH_GRID
        )
        kp = torch.flatten(self.pool(feature_map), start_dim=1)
        return self.relu(self.out(kp))
