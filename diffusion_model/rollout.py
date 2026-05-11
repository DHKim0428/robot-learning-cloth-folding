"""Run a trained LeRobot DiffusionPolicy on the SO-101 follower arm.

Default behavior is safe dry-run mode: the robot is connected and observations
are read, but predicted actions are only printed. Pass `--execute` to send
actions to the robot.

Supports both:
- native LeRobot checkpoint directories, e.g.
  `outputs/train/<run>/checkpoints/last/pretrained_model`
- legacy/custom `policy_diffusion.pt` state dicts from `train_custom.py`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_THIS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(_THIS_DIR))

from build_policy import (  # noqa: E402
    add_diffusion_config_args,
    build_diffusion_policy,
    diffusion_kwargs_from_args,
)
from dataset import HF_DATASET_ACCESS_HINT, load_metadata  # noqa: E402
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
        description="Run a trained DiffusionPolicy on the SO-101 follower arm."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Native LeRobot pretrained_model directory, checkpoint directory, "
            "or legacy policy_diffusion.pt from train_custom.py."
        ),
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="robot-learning-team43/so101_teleop_private",
        help="Dataset used to rebuild the model + normalization statistics.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional local dataset root override. Omit to use HF cache/download.",
    )
    add_diffusion_config_args(parser)
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
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send predicted actions to the robot.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Only print every Nth predicted action in dry-run mode.",
    )
    parser.add_argument(
        "--no-home-return",
        action="store_true",
        help="Skip home-pose return on exit.",
    )
    return parser.parse_args()


def _build_observation(
    raw_obs: dict,
    camera_name: str,
    state_names: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
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
    }


def _format_action(action_tensor: torch.Tensor, names: list[str]) -> dict[str, float]:
    arr = action_tensor.detach().cpu().float().numpy().reshape(-1)
    return {name: float(arr[idx]) for idx, name in enumerate(names)}


def _read_saved_config(checkpoint: Path) -> dict[str, Any]:
    candidates = []
    if checkpoint.is_file():
        candidates.extend([checkpoint.parent / "config.json", checkpoint.parent.parent / "config.json"])
    elif checkpoint.is_dir():
        candidates.extend([checkpoint / "config.json", checkpoint.parent / "config.json"])

    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload.get("diffusion_config", {})
    return {}


def _resolve_native_pretrained_dir(checkpoint: Path) -> Path | None:
    """Return a LeRobot `pretrained_model` directory if `checkpoint` points to one."""
    if checkpoint.is_dir():
        candidates = [
            checkpoint,
            checkpoint / "pretrained_model",
            checkpoint / "checkpoints" / "last" / "pretrained_model",
        ]
    else:
        candidates = [checkpoint.parent]

    for candidate in candidates:
        if (candidate / "model.safetensors").exists() and (candidate / "config.json").exists():
            return candidate
    return None


def _merge_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = diffusion_kwargs_from_args(args)
    saved = _read_saved_config(args.checkpoint)
    for key, saved_value in saved.items():
        if key not in kwargs or kwargs[key] is not None:
            continue
        if key == "resize_shape" and saved_value is not None:
            saved_value = tuple(saved_value)
        kwargs[key] = saved_value
    return kwargs


def _load_state_dict(checkpoint: Path, device: torch.device):
    if checkpoint.is_dir():
        pt_path = checkpoint / "policy_diffusion.pt"
        if pt_path.exists():
            checkpoint = pt_path
        else:
            raise FileNotFoundError(
                f"Directory checkpoint must contain policy_diffusion.pt: {checkpoint}"
            )

    if str(checkpoint).endswith(".safetensors"):
        from safetensors.torch import load_file as _load_safetensors

        return _load_safetensors(checkpoint, device=str(device))

    state_dict = torch.load(checkpoint, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict and not any(
        key.startswith("diffusion") for key in state_dict if key != "model"
    ):
        state_dict = state_dict["model"]
    return state_dict


def _load_policy_and_processors(args: argparse.Namespace, device: torch.device):
    """Load native LeRobot checkpoints when possible; fall back to custom state dicts."""
    native_dir = _resolve_native_pretrained_dir(args.checkpoint)
    if native_dir is not None:
        from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors

        cfg = DiffusionConfig.from_pretrained(native_dir)
        cfg.device = str(device)
        policy = DiffusionPolicy.from_pretrained(
            native_dir,
            config=cfg,
        )
        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=str(native_dir),
        )
        policy.to(device)
        policy.eval()
        policy.reset()
        print(f"[init] loaded native LeRobot DiffusionPolicy from {native_dir}")
        return policy, preprocessor, postprocessor, cfg

    try:
        meta = load_metadata(args.dataset_repo_id, args.dataset_root)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load dataset metadata for {args.dataset_repo_id!r}.\n"
            f"{HF_DATASET_ACCESS_HINT}"
        ) from exc

    policy, preprocessor, postprocessor, cfg = build_diffusion_policy(
        meta,
        device=str(device),
        **_merge_config_overrides(args),
    )
    state_dict = _load_state_dict(args.checkpoint, device)
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict missing={missing[:5]} unexpected={unexpected[:5]}")
    policy.to(device)
    policy.eval()
    policy.reset()
    print(f"[init] loaded legacy/custom DiffusionPolicy from {args.checkpoint}")
    return policy, preprocessor, postprocessor, cfg


def main() -> None:
    args = parse_args()
    if args.execute and args.dry_run:
        raise ValueError("Pass either --execute or --dry-run, not both.")

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

    policy, preprocessor, postprocessor, cfg = _load_policy_and_processors(args, device)
    print(
        f"[init] n_obs_steps={cfg.n_obs_steps} horizon={cfg.horizon} "
        f"n_action_steps={cfg.n_action_steps}"
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
            print("[execute] sending predicted actions to the SO-101 follower.")
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
