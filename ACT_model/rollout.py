"""Run an ACT policy on the SO-101 follower arm.

Mirrors `scripts/replay_episode.py` but feeds live camera+state into a
trained ACT policy instead of replaying a recorded episode.

Usage:
    bash shell/rollout_act.sh ACT_model/outputs/<run>/policy_act.pt
or:
    python ACT_model/rollout.py --checkpoint <path> [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Reuse robot helpers from scripts/.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from script_utils import (  # noqa: E402
    DEFAULT_FINAL_POSE_PATH,
    DEFAULT_HOME_POSE_PATH,
    DEFAULT_PORTS_PATH,
    extract_joint_pose,
    follower_config_kwargs,
    load_home_pose,
    load_ports,
    move_robot_to_pose,
)
from build_policy import build_act_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained ACT policy on the SO-101 follower arm."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to policy_act.pt (a state_dict produced by ACT_model/train.py).",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="robot-learning-team43/so101_teleop_private",
        help="Dataset used to rebuild the model + normalization statistics.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "so101_teleop",
    )
    parser.add_argument("--ports-config", type=Path, default=DEFAULT_PORTS_PATH)
    parser.add_argument("--home-pose", type=Path, default=DEFAULT_HOME_POSE_PATH)
    parser.add_argument("--final-pose", type=Path, default=DEFAULT_FINAL_POSE_PATH)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Predict and print actions but do NOT send them to the robot.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="In --dry-run mode, only print every Nth predicted action.",
    )
    parser.add_argument(
        "--no-home-return",
        action="store_true",
        help="Skip the home-pose return on exit (always skipped under --dry-run).",
    )
    return parser.parse_args()


def _build_observation(
    raw_obs: dict,
    camera_name: str,
    state_names: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert SO-101 raw observation -> ACT-style batched tensor dict.

    The dataset stores `observation.state` (6-vec) and `observation.images.front`
    (HWC uint8). The robot returns `{f"{motor}.pos": float, "front": ndarray}`.
    """
    state_vec = np.array(
        [float(raw_obs[name]) for name in state_names], dtype=np.float32
    )
    state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)

    image = raw_obs[camera_name]
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image_t = torch.from_numpy(image).to(device)
    image_t = image_t.float() / 255.0
    image_t = image_t.permute(2, 0, 1).contiguous().unsqueeze(0)

    return {
        "observation.state": state_t,
        f"observation.images.{camera_name}": image_t,
        "task": "",
        "robot_type": "so_follower",
    }


def _format_action(action_tensor: torch.Tensor, names: list[str]) -> dict[str, float]:
    arr = action_tensor.detach().cpu().float().numpy().reshape(-1)
    return {name: float(arr[i]) for i, name in enumerate(names)}


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type != args.device:
        print(f"[init] requested device={args.device} unavailable; falling back to {device}")

    # 1. Rebuild the policy + processors from dataset metadata, then load weights.
    from lerobot.datasets import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    policy, preprocessor, postprocessor, cfg = build_act_policy(
        meta,
        chunk_size=args.chunk_size,
        device=str(device),
    )

    state_dict = torch.load(args.checkpoint, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict and not any(
        k.startswith("model") for k in state_dict if k != "model"
    ):
        # Unwrap legacy {"model": state_dict} containers if needed.
        state_dict = state_dict["model"]
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict missing={missing[:5]}... unexpected={unexpected[:5]}...")
    policy.to(device)
    policy.eval()
    policy.reset()
    print(f"[init] loaded ACT policy from {args.checkpoint}")

    # 2. Build the SO-101 follower with one front camera.
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

    state_feature = meta.features["observation.state"]
    state_names = list(state_feature["names"])
    action_feature = meta.features["action"]
    action_names = list(action_feature["names"])

    home_pose = None
    if args.home_pose.exists():
        home_pose = load_home_pose(args.home_pose)

    robot.connect()
    sent_any_action = False
    try:
        if args.dry_run:
            print("[dry-run] predicting actions WITHOUT sending them to the robot.")
        else:
            print("[run] sending predicted actions to the SO-101 follower.")

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

            if args.dry_run:
                if step % max(args.print_every, 1) == 0:
                    pretty = " ".join(f"{k}={v:+.3f}" for k, v in action_dict.items())
                    print(f"[dry-run step {step:4d}] {pretty}")
            else:
                robot.send_action(action_dict)
                sent_any_action = True

            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\n[abort] KeyboardInterrupt — finishing rollout.")
    finally:
        if sent_any_action and not args.no_home_return and home_pose is not None:
            try:
                print("[exit] returning to home pose")
                move_robot_to_pose(
                    robot=robot,
                    target_pose=home_pose,
                    duration_s=4.0,
                    fps=args.fps,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[exit] home-pose return failed: {exc!r}")
        robot.disconnect()
        print("[exit] disconnected.")


if __name__ == "__main__":
    main()
