"""Roll out a custom DiffusionPolicy trained with relative arm actions.

This entrypoint is intentionally separate from `rollout.py` so absolute-action
native/custom diffusion checkpoints are not confused with relative-action custom
checkpoints. Dry-run remains the default; pass `--execute` to move the robot.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_THIS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(_THIS_DIR))

from build_policy import (  # noqa: E402
    add_diffusion_config_args,
    build_diffusion_config,
    build_diffusion_policy,
    diffusion_kwargs_from_args,
)
from dataset import HF_DATASET_ACCESS_HINT, build_dataset, load_metadata  # noqa: E402
from relative_actions import compute_relative_action_stats  # noqa: E402
from rollout import (  # noqa: E402
    _build_observation,
    _format_action,
    _load_state_dict,
    _read_saved_config,
    _resolve_native_pretrained_dir,
)
from script_utils import (  # noqa: E402
    DEFAULT_HOME_POSE_PATH,
    DEFAULT_PORTS_PATH,
    follower_config_kwargs,
    load_home_pose,
    load_ports,
    move_robot_to_pose,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a custom relative-action DiffusionPolicy on the SO-101 follower arm."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Custom relative policy_diffusion.pt or a directory containing it.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="robot-learning-team43/so101_teleop_private",
        help="Dataset used to rebuild model features and relative-action normalization stats.",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--dataset-video-backend",
        "--dataset.video_backend",
        dest="dataset_video_backend",
        default="pyav",
        choices=["pyav", "torchcodec"],
        help=(
            "Video decoder backend used while rebuilding relative-action stats. "
            "Defaults to pyav to avoid torchcodec shared-library issues."
        ),
    )
    add_diffusion_config_args(parser)
    parser.add_argument("--relative-stats-workers", type=int, default=0)
    parser.add_argument("--ports-config", type=Path, default=DEFAULT_PORTS_PATH)
    parser.add_argument("--home-pose", type=Path, default=DEFAULT_HOME_POSE_PATH)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compatibility flag; dry-run is already the default unless --execute is passed.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--no-home-return", action="store_true")
    return parser.parse_args()


def _saved_or_cli_config_kwargs(args: argparse.Namespace, saved: dict[str, Any]) -> dict[str, Any]:
    kwargs = diffusion_kwargs_from_args(args)
    for key, saved_value in saved.items():
        if key not in kwargs or kwargs[key] is not None:
            continue
        if key == "resize_shape" and saved_value is not None:
            saved_value = tuple(saved_value)
        kwargs[key] = saved_value
    return kwargs


def _load_relative_policy(args: argparse.Namespace, device: torch.device):
    if _resolve_native_pretrained_dir(args.checkpoint) is not None:
        raise ValueError(
            "rollout_relative.py supports only custom relative checkpoints, not native "
            "LeRobot pretrained_model directories."
        )

    saved_cfg = _read_saved_config(args.checkpoint)
    if not saved_cfg.get("use_relative_actions", False):
        raise ValueError(
            "Checkpoint config does not declare use_relative_actions=true. "
            "Use diffusion_model/rollout.py for absolute-action checkpoints."
        )

    relative_exclude_joints = list(saved_cfg.get("relative_exclude_joints") or ["gripper"])

    try:
        meta = load_metadata(args.dataset_repo_id, args.dataset_root)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load dataset metadata for {args.dataset_repo_id!r}.\n"
            f"{HF_DATASET_ACCESS_HINT}"
        ) from exc

    config_kwargs = _saved_or_cli_config_kwargs(args, saved_cfg)
    tmp_cfg = build_diffusion_config(meta, device=str(device), **config_kwargs)
    dataset, _ = build_dataset(
        args.dataset_repo_id,
        args.dataset_root,
        tmp_cfg,
        episode_filter_mode=saved_cfg.get("training_episode_filter"),
        episodes_whitelist=saved_cfg.get("training_episodes"),
        video_backend=args.dataset_video_backend,
    )
    print(
        "[init] recomputing relative-action stats for rollout "
        f"(exclude_joints={relative_exclude_joints})"
    )
    dataset_stats = dict(meta.stats)
    dataset_stats["action"] = compute_relative_action_stats(
        dataset,
        action_delta_indices=list(tmp_cfg.action_delta_indices),
        exclude_joints=relative_exclude_joints,
        num_workers=args.relative_stats_workers,
    )

    policy, preprocessor, postprocessor, cfg = build_diffusion_policy(
        meta,
        dataset_stats=dataset_stats,
        use_relative_actions=True,
        relative_exclude_joints=relative_exclude_joints,
        device=str(device),
        **config_kwargs,
    )
    state_dict = _load_state_dict(args.checkpoint, device)
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict missing={missing[:5]} unexpected={unexpected[:5]}")
    policy.to(device)
    policy.eval()
    policy.reset()
    return policy, preprocessor, postprocessor, cfg, meta


def main() -> None:
    args = parse_args()
    if args.execute and args.dry_run:
        raise ValueError("Pass either --execute or --dry-run, not both.")

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    if device.type != args.device:
        print(f"[init] requested device={args.device} unavailable; falling back to {device}")

    policy, preprocessor, postprocessor, cfg, meta = _load_relative_policy(args, device)
    print(
        f"[init] relative DiffusionPolicy n_obs_steps={cfg.n_obs_steps} "
        f"horizon={cfg.horizon} n_action_steps={cfg.n_action_steps} "
        f"exclude_joints={cfg.relative_exclude_joints}"
    )

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.utils.robot_utils import precise_sleep

    ports = load_ports(args.ports_config)
    camera_config = {
        args.camera_name: OpenCVCameraConfig(
            index_or_path=args.camera_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.fps,
        )
    }
    robot_cfg = SO101FollowerConfig(
        **follower_config_kwargs(ports["follower"]),
        cameras=camera_config,
    )
    robot = SO101Follower(robot_cfg)

    state_names = list(meta.features["observation.state"]["names"])
    action_names = list(meta.features["action"]["names"])
    home_pose = load_home_pose(args.home_pose) if args.home_pose.exists() else None

    robot.connect()
    sent_any_action = False
    try:
        if args.execute:
            print("[execute] sending relative-policy absolute actions to the SO-101 follower.")
        else:
            print("[dry-run] predicting actions WITHOUT sending them to the robot.")

        for step in range(args.max_steps):
            t0 = time.perf_counter()
            raw_obs = robot.get_observation()
            obs_batch = _build_observation(
                raw_obs,
                camera_name=args.camera_name,
                state_names=state_names,
                device=device,
            )
            obs_batch = preprocessor(obs_batch)
            with torch.inference_mode():
                action = policy.select_action(obs_batch)
            action = postprocessor(action)
            action_dict = _format_action(action, action_names)

            if args.execute:
                robot.send_action(action_dict)
                sent_any_action = True
            elif step % max(args.print_every, 1) == 0:
                pretty = " ".join(f"{key}={value:+.3f}" for key, value in action_dict.items())
                print(f"[dry-run step {step:4d}] {pretty}")

            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\n[abort] KeyboardInterrupt — finishing rollout.")
    finally:
        if sent_any_action and not args.no_home_return and home_pose is not None:
            try:
                print("[exit] returning to home pose")
                move_robot_to_pose(robot=robot, target_pose=home_pose, duration_s=4.0, fps=args.fps)
            except Exception as exc:  # noqa: BLE001
                print(f"[exit] home-pose return failed: {exc!r}")
        robot.disconnect()
        print("[exit] disconnected.")


if __name__ == "__main__":
    main()
