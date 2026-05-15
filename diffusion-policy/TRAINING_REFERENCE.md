# Diffusion Policy — Training Reference

## Default hyperparameters

These are the defaults set by `shell/train_diffusion.sh` / `diffusion-policy/train.py`.

| Parameter | Value | Notes |
|---|---|---|
| Backbone | DINOv2 ViT-S/14 (frozen) | `--backbone dinov2` |
| Batch size | 32 | `--batch-size 32` |
| Learning rate | 1e-4 | `--lr 1e-4` |
| Weight decay | 1e-6 | `--weight-decay 1e-6` |
| Num steps | 100,000 | `--num-steps 100000` |
| Horizon | 64 frames (~2.1 s @ 30 fps) | `--horizon 64` |
| n\_action\_steps | 32 (~1.07 s executed per inference) | `--n-action-steps 32` |
| n\_obs\_steps | 2 | `--n-obs-steps 2` |
| Noise scheduler | DDIM | `--noise-scheduler DDIM` |
| Num train timesteps | 100 | `--num-train-timesteps 100` |
| Num inference steps | 10 | `--num-inference-steps 10` |
| U-Net down\_dims | (512, 1024, 2048) | lerobot default |
| diffusion\_step\_embed\_dim | 128 | lerobot default |
| Spatial softmax keypoints | 32 | `--spatial-softmax-num-keypoints 32` |
| Use separate RGB encoder | True | per-camera encoder |
| Use group norm | False | required with pretrained ResNet |
| DataLoader workers | 4 | `--num-workers 4` |
| Save checkpoint every | 5,000 steps | `--save-every 5000` |
| Log every | 100 steps | `--log-every 100` |

## Model size

Measured with `python diffusion-policy/build_policy.py` (backbone=dinov2, default config).

| Component | Params | Weights (fp32) | Trainable |
|---|---|---|---|
| DINOv2 ViT-S/14 (frozen) | 22.1 M | 84 MB | No (0.02 M head only) |
| U-Net (lerobot ConditionalUNet1d) | 251.8 M | 961 MB | Yes |
| **Total** | **273.8 M** | **1,045 MB** | 91.9% |

The U-Net is large because lerobot's default `down_dims=(512, 1024, 2048)` is sized for
high-DoF tasks. With `action_dim=6` this is over-parameterized — see ablation ideas below.

## GPU memory breakdown (batch_size=32, fp32)

| Component | Memory | Scales with batch? |
|---|---|---|
| Model weights | 1,045 MB | No |
| Gradients (trainable only) | 960 MB | No |
| AdamW optimizer states (m + v) | 1,921 MB | No |
| CUDA / PyTorch allocator overhead | ~800 MB | No |
| U-Net activations (1D, action seq) | ~24 MB | Yes (~0.75 MB/sample) |
| DINOv2 activations (transient, no\_grad) | ~24 MB | Yes (freed after encoder) |
| **Total estimate** | **~4.8 GB** | |

**Key insight:** the U-Net operates on tiny 1D sequences `(batch, 6, 64)`, so activation
memory is negligible (~0.75 MB/sample). The fixed costs (weights + grads + optimizer)
dominate at **4.6 GB** regardless of batch size.

## Batch size guide by GPU

| GPU VRAM | Recommended batch size | Notes |
|---|---|---|
| 8 GB (local, e.g. RTX 3070/4070) | **32** | ~4.8 GB used, ~3.2 GB headroom |
| 24 GB (e.g. RTX 3090/4090) | 64–128 | comfortable |
| 80 GB H100 | 128–256 | use for final training runs |

On 8 GB VRAM, activations are so small that batch_size=32 fits with ~3 GB to spare.
You can try 64 too. Training throughput (steps/sec) will be the main bottleneck on a
local GPU, not memory.

## Quick start commands

```bash
# Standard training — drop bad episodes, all defaults
bash shell/train_diffusion.sh

# Override batch size for local 8 GB GPU
bash shell/train_diffusion.sh --batch-size 32

# Single-episode overfit sanity check (should drive loss near zero)
python diffusion-policy/train.py --episodes 5 --num-steps 5000 --batch-size 8

# ResNet18 baseline (FUTURE_DIFFUSION_NOTES.md recommended starting point)
bash shell/train_diffusion.sh --backbone resnet18
```

## Ablation ideas (from FUTURE_DIFFUSION_NOTES.md)

- Pretrained vs scratch backbone
- Frozen DINOv2 vs LoRA fine-tuned (`--lora-depth 4`)
- Long horizon (64) vs short (16)
- Reduced U-Net: `down_dims=(256, 512, 1024)` cuts model from 251 M → ~38 M params
- DDIM vs DDPM noise schedule
