"""Train a Diffusion Policy on the SO-101 cloth-folding dataset.

Mirrors ``ACT_model/train.py`` but builds a ``DiffusionPolicy`` (with a
frozen DINOv2 encoder by default) and uses a cosine LR schedule.

Run from the repo root:
    python diffusion-policy/train.py --episode-filter --num-steps 100000
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Make sibling modules importable regardless of cwd.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from build_policy import build_diffusion_policy  # noqa: E402
from dataset import build_dataset  # noqa: E402

PROJECT_ROOT = _THIS_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diffusion Policy trainer for SO-101 cloth folding."
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="robot-learning-team43/so101_teleop_private",
        help="LeRobot dataset repo id.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "so101_teleop",
        help="Local LeRobot dataset root (where parquet + videos live).",
    )
    parser.add_argument(
        "--episode-filter",
        nargs="?",
        const="bad",
        choices=["bad", "meh"],
        default=None,
        help=(
            "Enable episode filtering. Use without a value to ignore bad "
            "episodes; use 'meh' to ignore bad and meh episodes."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Optional whitelist of episode indices (overrides --episode-filter).",
    )
    parser.add_argument(
        "--backbone", default="dinov2", choices=["dinov2", "resnet18"]
    )
    parser.add_argument("--lora-depth", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--n-action-steps", type=int, default=32)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument(
        "--noise-scheduler", default="DDIM", choices=["DDIM", "DDPM"]
    )
    parser.add_argument("--num-train-timesteps", type=int, default=100)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--spatial-softmax-num-keypoints", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: diffusion-policy/outputs/diff_<timestamp>).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="Resume from a previous run directory (e.g. diffusion-policy/outputs/diff_20260514_212018).",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--image-log-every", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.resume is not None:
        out = Path(args.resume)
        if not out.is_dir():
            raise FileNotFoundError(f"--resume path not found: {out}")
        return out
    if args.output_dir is not None:
        out = args.output_dir
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = PROJECT_ROOT / "diffusion-policy" / "outputs" / f"diff_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _summarize_args(args: argparse.Namespace, out_dir: Path) -> None:
    payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out_dir / "args.json").write_text(json.dumps(payload, indent=2))


def _rotate_checkpoints(out_dir: Path, keep_first: int = 3, keep_last: int = 3) -> None:
    """Delete step_* dirs outside the first/last keep window."""
    dirs = sorted(out_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    to_keep = set(dirs[:keep_first] + dirs[-keep_last:])
    for d in dirs:
        if d not in to_keep:
            shutil.rmtree(d)
            print(f"[ckpt] deleted old checkpoint {d.name}")


def _save_checkpoint(
    out_dir: Path,
    policy,
    cfg,
    step: int,
    current_loss: float,
    best_loss: float,
) -> float:
    """Save policy weights, rotate old checkpoints, and track best loss.

    We deliberately do NOT call ``policy.save_pretrained`` here: after the
    DINOv2 backbone swap, the persisted ``DiffusionConfig`` still claims
    ``vision_backbone="resnet18"`` and the saved state-dict keys belong to
    ``DINOv2RgbEncoder``, so ``DiffusionPolicy.from_pretrained`` would
    silently reconstruct the wrong architecture. The deployment contract
    is: rebuild via ``build_diffusion_policy(... args matching args.json)``
    and then ``load_state_dict`` from one of these ``.pt`` files.

    Returns the updated best_loss.
    """
    state_dict = policy.state_dict()

    ckpt_dir = out_dir / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, ckpt_dir / "policy_diffusion.pt")

    torch.save(state_dict, out_dir / "policy_diffusion.pt")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "step": step,
                "horizon": cfg.horizon,
                "n_action_steps": cfg.n_action_steps,
                "n_obs_steps": cfg.n_obs_steps,
            },
            indent=2,
        )
    )
    print(f"[ckpt] saved to {ckpt_dir}")

    if current_loss < best_loss:
        best_dir = out_dir / "best"
        best_dir.mkdir(exist_ok=True)
        torch.save(state_dict, best_dir / "policy_diffusion.pt")
        print(f"[ckpt] new best loss {current_loss:.4f} (was {best_loss:.4f}) → saved to best/")
        best_loss = current_loss

    _rotate_checkpoints(out_dir)
    return best_loss


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = _resolve_output_dir(args)
    _summarize_args(args, out_dir)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    print(f"[init] output dir: {out_dir}")
    print(f"[init] tensorboard --logdir {out_dir / 'tb'}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type != args.device:
        print(f"[init] requested device={args.device} unavailable; falling back to {device}")

    if args.episodes is not None and len(args.episodes) == 1:
        print(f"[overfit] single-episode mode on episode {args.episodes[0]}")

    from lerobot.datasets import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    print(
        f"[init] dataset {args.dataset_repo_id}: "
        f"{meta.total_episodes} episodes, {meta.total_frames} frames, fps={meta.fps}"
    )

    policy, preprocessor, postprocessor, cfg = build_diffusion_policy(
        meta,
        backbone=args.backbone,
        lora_depth=args.lora_depth,
        lora_rank=args.lora_rank,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        n_obs_steps=args.n_obs_steps,
        noise_scheduler_type=args.noise_scheduler,
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        spatial_softmax_num_keypoints=args.spatial_softmax_num_keypoints,
        device=str(device),
    )
    policy.train()
    policy.to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(
        f"[init] DiffusionPolicy: {n_params/1e6:.2f} M params total, "
        f"{n_trainable/1e6:.2f} M trainable | backbone={args.backbone} "
        f"lora_depth={args.lora_depth}"
    )

    dataset, _ = build_dataset(
        args.dataset_repo_id,
        args.dataset_root,
        cfg,
        episode_filter_mode=args.episode_filter,
        episodes_whitelist=args.episodes,
    )
    print(
        f"[init] training on {dataset.num_episodes} episodes "
        f"({dataset.num_frames} frames) — filter={args.episode_filter}, "
        f"whitelist={args.episodes}"
    )

    pin_memory = device.type == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )

    start_step = 0
    if args.resume is not None:
        policy_pt = out_dir / "policy_diffusion.pt"
        if not policy_pt.exists():
            raise FileNotFoundError(f"No checkpoint found in {out_dir} (expected policy_diffusion.pt)")
        policy.load_state_dict(torch.load(policy_pt, map_location=device))
        cfg_json = out_dir / "config.json"
        if cfg_json.exists():
            start_step = json.loads(cfg_json.read_text())["step"]
        print(f"[resume] loaded weights from step {start_step} (optimizer starts cold — expect brief loss spike)")

    if start_step > 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps,
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )

    step = start_step
    best_loss = float("inf")
    last_loss = float("inf")
    last_log_t = time.perf_counter()
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, loss_dict = policy.forward(batch)
            last_loss = loss.item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            if step % args.log_every == 0:
                now = time.perf_counter()
                steps_per_s = args.log_every / max(now - last_log_t, 1e-9)
                last_log_t = now

                # DiffusionPolicy.forward returns (loss, None) — no per-head
                # breakdown is exposed, so log loss/total and loss/diffusion
                # from loss.item() and only iterate loss_dict if non-empty.
                loss_value = loss.item()
                writer.add_scalar("loss/total", loss_value, step)
                writer.add_scalar("loss/diffusion", loss_value, step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)
                writer.add_scalar("throughput/steps_per_s", steps_per_s, step)
                extra = ""
                if loss_dict:
                    for k, v in loss_dict.items():
                        writer.add_scalar(f"loss/{k}", v, step)
                    extra = " " + " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                print(
                    f"[step {step:>7d}] loss={loss_value:.4f}{extra}"
                    f" | {steps_per_s:.1f} step/s"
                )

            if args.image_log_every > 0 and step % args.image_log_every == 0:
                for img_key in cfg.image_features:
                    if img_key in batch:
                        img = batch[img_key][0].detach().cpu().float()
                        if img.dim() == 4:
                            img = img[0]
                        img = img.clamp(0.0, 1.0)
                        writer.add_image(f"input/{img_key}", img, step)
                        break

            if step > 0 and step % args.save_every == 0:
                best_loss = _save_checkpoint(out_dir, policy, cfg, step, last_loss, best_loss)

            step += 1
            if step >= args.num_steps:
                done = True
                break

    _save_checkpoint(out_dir, policy, cfg, step, last_loss, best_loss)
    writer.flush()
    writer.close()
    print(f"[done] final checkpoint at {out_dir}")


if __name__ == "__main__":
    main()
