"""Train an ACT policy on the SO-101 cloth-folding dataset.

MVP single-shot trainer: full-episode demos, no fold-phase split. Logs
to TensorBoard and saves checkpoints under
`ACT_model/outputs/<timestamp>/`.

Run from the repo root:
    python ACT_model/train.py --episode-filter --num-steps 100000
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
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

from build_policy import build_act_policy  # noqa: E402
from dataset import build_dataset  # noqa: E402

PROJECT_ROOT = _THIS_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACT MVP trainer for SO-101 cloth folding."
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
    # NOTE: spec mirrors nn-approach-eleni/train_policy.py:24-31 so the flag
    # behaves identically across the two training tracks.
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
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--vision-backbone", default="resnet18")
    parser.add_argument("--no-vae", action="store_true", help="Disable the CVAE objective.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ACT_model/outputs/<timestamp>).",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--image-log-every", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _resolve_output_dir(arg: Path | None) -> Path:
    if arg is not None:
        out = arg
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = PROJECT_ROOT / "ACT_model" / "outputs" / f"act_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _summarize_args(args: argparse.Namespace, out_dir: Path) -> None:
    payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out_dir / "args.json").write_text(json.dumps(payload, indent=2))


def _save_checkpoint(
    out_dir: Path,
    policy,
    preprocessor,
    postprocessor,
    cfg,
    step: int,
) -> None:
    """Saves both the lerobot bundle and a plain `policy_act.pt` state dict."""
    ckpt_dir = out_dir / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    preprocessor.save_pretrained(ckpt_dir)
    postprocessor.save_pretrained(ckpt_dir)
    torch.save(policy.state_dict(), ckpt_dir / "policy_act.pt")

    # Also keep a `latest` mirror at the top of out_dir for easy rollout.
    latest_state = out_dir / "policy_act.pt"
    torch.save(policy.state_dict(), latest_state)
    (out_dir / "config.json").write_text(json.dumps({"step": step, "chunk_size": cfg.chunk_size}, indent=2))
    print(f"[ckpt] saved to {ckpt_dir}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = _resolve_output_dir(args.output_dir)
    _summarize_args(args, out_dir)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    print(f"[init] output dir: {out_dir}")
    print(f"[init] tensorboard --logdir {out_dir / 'tb'}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type != args.device:
        print(f"[init] requested device={args.device} unavailable; falling back to {device}")

    # Build metadata first so we can construct ACTConfig with the right shapes.
    from lerobot.datasets import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    print(
        f"[init] dataset {args.dataset_repo_id}: "
        f"{meta.total_episodes} episodes, {meta.total_frames} frames, fps={meta.fps}"
    )

    policy, preprocessor, postprocessor, cfg = build_act_policy(
        meta,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        vision_backbone=args.vision_backbone,
        use_vae=not args.no_vae,
        device=str(device),
    )
    policy.train()
    policy.to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[init] ACTPolicy: {n_params/1e6:.2f} M params, chunk={cfg.chunk_size}")

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

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

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
                for k, v in loss_dict.items():
                    writer.add_scalar(f"loss/{k}", v, step)
                print(
                    f"[step {step:>7d}] loss={loss.item():.4f} "
                    + " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                    + f" | {steps_per_s:.1f} step/s"
                )

            if args.image_log_every > 0 and step % args.image_log_every == 0:
                # Log the first image of the first sample (CHW float in [0,1]).
                for img_key in cfg.image_features:
                    if img_key in batch:
                        img = batch[img_key][0].detach().cpu().float()
                        if img.dim() == 4:
                            img = img[0]  # (T, C, H, W) -> (C, H, W)
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
    print(f"[done] final checkpoint at {out_dir}")


if __name__ == "__main__":
    main()
