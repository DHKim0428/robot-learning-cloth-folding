# Future Diffusion Notes

Notes for future diffusion-policy implementation work.

## LeRobot PR #3202

Reference: <https://github.com/huggingface/lerobot/pull/3202>

Title: `feat(policy): use pretrained vision encoder weights by default for diffusion and vqbet`

Status when reviewed: open.

This PR changes LeRobot defaults for Diffusion Policy and VQBeT. The motivation is that LeRobot's ACT baseline has usually been easier to train on real tasks, while many users have struggled to make Diffusion Policy work beyond simple PushT-style tasks. The suspected issue is that Diffusion Policy and VQBeT used randomly initialized image encoders by default, while ACT commonly uses ImageNet-pretrained ResNet weights.

## Proposed DiffusionConfig Default Changes

```python
horizon: 16 -> 64
n_action_steps: 8 -> 32
pretrained_backbone_weights: None -> "ResNet18_Weights.IMAGENET1K_V1"
use_group_norm: True -> False
use_separate_rgb_encoder_per_camera: False -> True
```

## Proposed VQBeTConfig Default Changes

```python
pretrained_backbone_weights: None -> "ResNet18_Weights.IMAGENET1K_V1"
use_group_norm: True -> False
```

## Why It Matters

For real robot policies, pretrained visual features may make diffusion training much more stable and faster than starting the vision encoder from random weights. This also makes Diffusion Policy defaults more comparable to ACT defaults.

The `use_group_norm=False` change is tied to pretrained weights. Replacing BatchNorm in a pretrained ResNet with GroupNorm would effectively break the pretrained normalization assumptions, so pretrained ResNet weights should be used with BatchNorm intact.

The horizon/action-step change is especially relevant for cloth folding:

- Old default at 30 fps:
  - `horizon=16` is about 0.53 s.
  - `n_action_steps=8` is about 0.27 s.
- New proposed default at 30 fps:
  - `horizon=64` is about 2.13 s.
  - `n_action_steps=32` is about 1.07 s.

The longer default is more plausible for real SO-101 towel manipulation than the short PushT-oriented default.

## Caveat

This is not guaranteed to be universally optimal. A reviewer noted that the original Diffusion Policy paper found pretrained vision useful in simulation, while their best real-world models were trained end-to-end from scratch. Treat this PR as a pragmatic LeRobot default improvement and a strong baseline to test, not as proof that pretrained vision is always best.

## Implications For This Repo

For our first LeRobot diffusion baseline, start with:

- ImageNet-pretrained ResNet18 vision encoder.
- `use_group_norm=False`.
- `horizon=64`.
- `n_action_steps=32`.
- LeRobot v3 dataset loading through `LeRobotDataset` / `LeRobotDatasetMetadata`.
- Current observation names:
  - `observation.state`
  - `observation.images.front`
- Current action name:
  - `action`

For the current single-front-camera setup, `use_separate_rgb_encoder_per_camera=True` likely has little effect. If we add wrist plus scene/external cameras later, keep separate encoders as a serious default to test.

## Suggested Experiments

Run controlled comparisons before committing to one config:

1. One-episode overfit with pretrained ResNet18, `horizon=64`, `n_action_steps=32`.
2. Same setup with `pretrained_backbone_weights=None`.
3. Same setup with shorter `horizon=16`, `n_action_steps=8`.
4. If compute allows, test frozen vs finetuned visual backbone or lower LR for the visual backbone.
5. Always run dry-rollout / action sanity checks before sending actions to the SO-101.

Track the following:

- train loss and validation loss
- action smoothness
- one-episode overfit quality
- predicted action ranges after unnormalization
- live dry-run behavior
- hardware rollout success for grasping first, then single fold
