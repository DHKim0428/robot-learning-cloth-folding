"""Experimental custom trainer for SO-101 DiffusionPolicy.

Prefer `shell/train_diffusion.sh`, which uses the native LeRobot `lerobot-train`
workflow. This file is kept for debugging because it mirrors the ACT_model
training loop and supports `config/episode_filter.toml` directly.

Run from the repo root:
    python diffusion_model/train_custom.py --episode-filter --num-steps 50000
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - depends on local env extras
    class SummaryWriter:  # type: ignore[no-redef]
        """No-op fallback when tensorboard is not installed."""

        def __init__(self, *args, **kwargs):
            print("[warn] tensorboard is not installed; TensorBoard logging disabled.")

        def add_scalar(self, *args, **kwargs):
            return None

        def add_image(self, *args, **kwargs):
            return None

        def flush(self):
            return None

        def close(self):
            return None

try:
    import wandb
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    wandb = None

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from build_policy import (  # noqa: E402
    add_diffusion_config_args,
    build_diffusion_config,
    build_diffusion_policy,
    diffusion_kwargs_from_args,
    serializable_diffusion_config,
)
from dataset import HF_DATASET_ACCESS_HINT, build_dataset, load_metadata  # noqa: E402
from relative_actions import compute_relative_action_stats  # noqa: E402

PROJECT_ROOT = _THIS_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LeRobot DiffusionPolicy on SO-101 cloth folding demos."
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="robot-learning-team43/so101_teleop_private",
        help="Hugging Face LeRobot dataset repo id.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional local dataset root override. Omit to use HF cache/download.",
    )
    parser.add_argument(
        "--dataset-video-backend",
        "--dataset.video_backend",
        dest="dataset_video_backend",
        default="pyav",
        choices=["pyav", "torchcodec"],
        help=(
            "Video decoder backend for LeRobotDataset. Defaults to pyav to "
            "match our stable CLI smoke-test path and avoid torchcodec/FFmpeg "
            "shared-library issues on some servers."
        ),
    )
    parser.add_argument(
        "--episode-filter",
        nargs="?",
        const="bad",
        choices=["bad", "meh"],
        default=None,
        help=(
            "Enable quality filtering. Use without a value to ignore bad "
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
    add_diffusion_config_args(parser)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument("--num-steps", "--steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: diffusion_model/outputs/diffusion_<timestamp>).",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--image-log-every", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--relative-actions",
        action="store_true",
        help=(
            "Train custom diffusion on arm-joint deltas relative to the current "
            "observation.state. Gripper remains absolute by default."
        ),
    )
    parser.add_argument(
        "--relative-exclude-joints",
        nargs="*",
        default=["gripper"],
        help="Action-name tokens to keep absolute when --relative-actions is enabled.",
    )
    parser.add_argument(
        "--relative-stats-workers",
        type=int,
        default=0,
        help="Reserved for relative-action stats computation; currently vectorized single-process.",
    )
    parser.add_argument("--wandb-enable", "--wandb.enable", action="store_true")
    parser.add_argument("--wandb-project", "--wandb.project", default="lerobot")
    parser.add_argument("--wandb-entity", "--wandb.entity", default=None)
    parser.add_argument("--wandb-name", "--wandb.name", default=None)
    parser.add_argument("--wandb-mode", "--wandb.mode", default=None)
    parser.add_argument("--wandb-notes", "--wandb.notes", default=None)
    return parser.parse_args()


def _resolve_output_dir(arg: Path | None, *, relative_actions: bool = False) -> Path:
    if arg is not None:
        out = arg
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "diffusion_relative" if relative_actions else "diffusion"
        out = PROJECT_ROOT / "diffusion_model" / "outputs" / f"{prefix}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _summarize_args(args: argparse.Namespace, out_dir: Path) -> None:
    payload = {key: _jsonable(value) for key, value in vars(args).items()}
    (out_dir / "args.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_checkpoint(
    out_dir: Path,
    policy,
    preprocessor,
    postprocessor,
    cfg,
    step: int,
) -> None:
    """Save both a LeRobot bundle and a plain state dict."""
    ckpt_dir = out_dir / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    preprocessor.save_pretrained(ckpt_dir)
    postprocessor.save_pretrained(ckpt_dir)
    torch.save(policy.state_dict(), ckpt_dir / "policy_diffusion.pt")

    torch.save(policy.state_dict(), out_dir / "policy_diffusion.pt")
    config_payload = {
        "step": step,
        "policy_type": "diffusion",
        "diffusion_config": serializable_diffusion_config(cfg),
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )
    print(f"[ckpt] saved to {ckpt_dir}")


def _optimizer_defaults(cfg, args: argparse.Namespace) -> tuple[float, float]:
    lr = args.lr if args.lr is not None else float(getattr(cfg, "optimizer_lr", 1e-4))
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else float(getattr(cfg, "optimizer_weight_decay", 1e-6))
    )
    return lr, weight_decay


def _loss_dict_items(loss_dict) -> list[tuple[str, float]]:
    if not loss_dict:
        return []
    return [(str(key), float(value)) for key, value in loss_dict.items()]


def _wandb_config(args: argparse.Namespace, cfg, out_dir: Path) -> dict[str, Any]:
    payload = {key: _jsonable(value) for key, value in vars(args).items()}
    payload["output_dir"] = str(out_dir)
    payload["policy_type"] = "diffusion"
    payload["diffusion_config"] = serializable_diffusion_config(cfg)
    return payload


def _init_wandb(args: argparse.Namespace, cfg, out_dir: Path):
    if not args.wandb_enable:
        return None
    if wandb is None:
        print("[warn] wandb is not installed; W&B logging disabled.")
        return None

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or out_dir.name,
        mode=args.wandb_mode,
        notes=args.wandb_notes,
        config=_wandb_config(args, cfg, out_dir),
        dir=str(out_dir),
    )
    print(f"[init] wandb run: {run.url if getattr(run, 'url', None) else run.name}")
    return run


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = _resolve_output_dir(args.output_dir, relative_actions=args.relative_actions)
    _summarize_args(args, out_dir)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    print(f"[init] output dir: {out_dir}")
    print(f"[init] tensorboard --logdir {out_dir / 'tb'}")

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    if device.type != args.device:
        print(f"[init] requested device={args.device} unavailable; falling back to {device}")

    try:
        meta = load_metadata(args.dataset_repo_id, args.dataset_root)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load dataset metadata for {args.dataset_repo_id!r}.\n"
            f"{HF_DATASET_ACCESS_HINT}"
        ) from exc

    print(
        f"[init] dataset {args.dataset_repo_id}: "
        f"{meta.total_episodes} episodes, {meta.total_frames} frames, fps={meta.fps}"
    )

    cfg = build_diffusion_config(
        meta,
        device=str(device),
        **diffusion_kwargs_from_args(args),
    )

    dataset, _ = build_dataset(
        args.dataset_repo_id,
        args.dataset_root,
        cfg,
        episode_filter_mode=args.episode_filter,
        episodes_whitelist=args.episodes,
        video_backend=args.dataset_video_backend,
    )
    print(
        f"[init] training on {dataset.num_episodes} episodes "
        f"({dataset.num_frames} frames) — filter={args.episode_filter}, "
        f"whitelist={args.episodes}, video_backend={args.dataset_video_backend}"
    )

    dataset_stats = meta.stats
    if args.relative_actions:
        print(
            "[init] computing relative-action stats in memory "
            f"(exclude_joints={args.relative_exclude_joints})"
        )
        dataset_stats = dict(meta.stats)
        dataset_stats["action"] = compute_relative_action_stats(
            dataset,
            action_delta_indices=list(cfg.action_delta_indices),
            exclude_joints=args.relative_exclude_joints,
            num_workers=args.relative_stats_workers,
        )
        action_stats = dataset_stats["action"]
        print(
            "[init] relative action stats: "
            f"min={action_stats['min'].round(4).tolist()} "
            f"max={action_stats['max'].round(4).tolist()}"
        )

    policy, preprocessor, postprocessor, cfg = build_diffusion_policy(
        meta,
        dataset_stats=dataset_stats,
        use_relative_actions=args.relative_actions,
        relative_exclude_joints=args.relative_exclude_joints,
        device=str(device),
        **diffusion_kwargs_from_args(args),
    )
    cfg.training_episode_filter = args.episode_filter
    cfg.training_episodes = args.episodes
    policy.train()
    policy.to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(
        f"[init] DiffusionPolicy: {n_params/1e6:.2f} M params, "
        f"n_obs_steps={cfg.n_obs_steps}, horizon={cfg.horizon}, "
        f"n_action_steps={cfg.n_action_steps}"
    )
    if args.relative_actions:
        print(
            "[init] relative actions enabled: arm joints are deltas from "
            f"current state; excluded joints stay absolute: {cfg.relative_exclude_joints}"
        )

    wandb_run = _init_wandb(args, cfg, out_dir)

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

    lr, weight_decay = _optimizer_defaults(cfg, args)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    print(f"[init] optimizer AdamW lr={lr:g} weight_decay={weight_decay:g}")

    step = 0
    last_log_t = time.perf_counter()
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, loss_dict = policy.forward(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            if step % args.log_every == 0:
                now = time.perf_counter()
                steps_per_s = args.log_every / max(now - last_log_t, 1e-9)
                last_log_t = now

                writer.add_scalar("loss/total", loss.item(), step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)
                writer.add_scalar("throughput/steps_per_s", steps_per_s, step)
                loss_items = _loss_dict_items(loss_dict)
                for key, value in loss_items:
                    writer.add_scalar(f"loss/{key}", value, step)
                if wandb_run is not None:
                    wandb.log(
                        {
                            "loss/total": loss.item(),
                            "lr": optimizer.param_groups[0]["lr"],
                            "throughput/steps_per_s": steps_per_s,
                            **{f"loss/{key}": value for key, value in loss_items},
                        },
                        step=step,
                    )
                extra = " ".join(f"{key}={value:.4f}" for key, value in loss_items)
                if extra:
                    extra = " " + extra
                print(
                    f"[step {step:>7d}] loss={loss.item():.4f}"
                    f"{extra} | {steps_per_s:.1f} step/s"
                )

            if args.image_log_every > 0 and step % args.image_log_every == 0:
                for img_key in cfg.image_features:
                    if img_key in batch:
                        img = batch[img_key][0].detach().cpu().float()
                        if img.dim() == 4:
                            img = img[-1]  # (T, C, H, W) -> latest (C, H, W)
                        img = img.clamp(0.0, 1.0)
                        writer.add_image(f"input/{img_key}", img, step)
                        break

            if step > 0 and step % args.save_every == 0:
                _save_checkpoint(out_dir, policy, preprocessor, postprocessor, cfg, step)

            step += 1
            if step >= args.num_steps:
                done = True
                break

    _save_checkpoint(out_dir, policy, preprocessor, postprocessor, cfg, step)
    writer.flush()
    writer.close()
    if wandb_run is not None:
        wandb.finish()
    print(f"[done] final checkpoint at {out_dir}")


if __name__ == "__main__":
    main()
