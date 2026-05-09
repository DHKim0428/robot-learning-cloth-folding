import argparse
import os
from pathlib import Path

import cv2
from datasets import load_dataset
import numpy as np
import pandas as pd
import torch

from features import (
    DATASET_ROOT,
    build_policy_model,
    dataset_shards,
    detect_corners,
    get_fold_target,
    order_corners,
    video_for_opencv,
)


FALLBACK_SCALE = float(os.environ.get("FALLBACK_SCALE", "0.95"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "10"))
CORNER_ALPHA = float(os.environ.get("CORNER_ALPHA", "0.8"))
PHASE0_ALPHA = float(os.environ.get("PHASE0_ALPHA", "0.55"))
PHASE0_MAX_DELTA = float(os.environ.get("PHASE0_MAX_DELTA", "3.5"))
PHASE1_ALPHA = float(os.environ.get("PHASE1_ALPHA", "0.70"))
PHASE1_MAX_DELTA = float(os.environ.get("PHASE1_MAX_DELTA", "2.5"))
STATE_UPDATE_SCALE = float(os.environ.get("STATE_UPDATE_SCALE", "0.001"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("policy.pt"))
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    return parser.parse_args()


def load_episode_corners(dataset_root, episode_id, needed_indices):
    corners_map = {}
    needed_indices = set(needed_indices)

    for video_path, data_path in dataset_shards(dataset_root):
        data = pd.read_parquet(data_path, columns=["index", "episode_index"])
        episode_rows = data[data["episode_index"] == episode_id]
        file_needed = set(episode_rows["index"].astype(int).tolist()) & needed_indices

        if len(file_needed) == 0:
            continue

        start = int(episode_rows.index.min())
        end = int(episode_rows.index.max())
        indices = data.iloc[start:end + 1]["index"].astype(int).tolist()
        cap = cv2.VideoCapture(str(video_for_opencv(video_path, start, end)))
        frame_i = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_i >= len(indices):
                break

            idx = indices[frame_i]
            if idx in file_needed:
                corners = detect_corners(frame)
                if corners is not None:
                    corners_map[idx] = order_corners(corners)

            frame_i += 1

        cap.release()

    return corners_map


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset(
        "robot-learning-team43/so101_teleop_private",
        split="train",
    )

    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location=device)

    state_mean = checkpoint["state_mean"].to(device).float()
    state_std = checkpoint["state_std"].to(device).float()
    action_mean = checkpoint["action_mean"].to(device).float()
    action_std = checkpoint["action_std"].to(device).float()
    action_low = action_mean - 3 * action_std
    action_high = action_mean + 3 * action_std

    action_dim = len(ds[0]["action"])
    input_dim = int(checkpoint.get("input_dim", len(state_mean) + 8 + 2 + 1))

    model = build_policy_model(input_dim, action_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    trajectory = [s for s in ds if int(s["episode_index"]) == args.episode]
    if len(trajectory) == 0:
        raise RuntimeError(f"Episode {args.episode} not found in dataset.")

    needed_indices = [int(s["index"]) for s in trajectory]
    corners_map = load_episode_corners(args.dataset_root, args.episode, needed_indices)

    motions = []
    jumps = []
    skipped = 0
    prev_corners = None
    prev_action = None
    max_jump_info = None
    rollout_state = np.array(trajectory[0]["observation.state"], dtype=np.float32)
    initial_state = rollout_state.copy()

    print("\n--- ROLLOUT START ---")
    print(f"Model  : {args.checkpoint}")
    print(f"Input  : {input_dim}")
    print(f"Episode: {args.episode}")
    print(f"Frames : {len(trajectory)}")
    print(f"Corners: {len(corners_map)} usable frames")
    print(f"Fallback scale: {FALLBACK_SCALE}")
    print(f"Corner alpha: {CORNER_ALPHA}")
    print(f"State update scale: {STATE_UPDATE_SCALE}\n")

    with torch.no_grad():
        for t, sample in enumerate(trajectory):
            idx = int(sample["index"])

            corners = corners_map.get(idx)
            used_fallback_corners = False
            if corners is None:
                if prev_corners is not None:
                    corners = prev_corners
                    used_fallback_corners = True
                else:
                    skipped += 1
                    continue
            else:
                if prev_corners is not None:
                    corners = CORNER_ALPHA * prev_corners + (1 - CORNER_ALPHA) * corners
                prev_corners = corners

            frame_id = int(sample["frame_index"])
            phase = 0.0 if frame_id < len(trajectory) / 2 else 1.0
            alpha = PHASE0_ALPHA if phase < 0.5 else PHASE1_ALPHA
            max_delta = PHASE0_MAX_DELTA if phase < 0.5 else PHASE1_MAX_DELTA

            if used_fallback_corners:
                max_delta *= FALLBACK_SCALE

            normalized_state = (
                torch.tensor(rollout_state, dtype=torch.float32, device=device)
                - state_mean
            ) / state_std

            goal_vec = get_fold_target(corners, phase)
            input_vec = np.concatenate([
                normalized_state.cpu().numpy(),
                corners.flatten(),
                goal_vec,
                [phase],
            ]).astype(np.float32)

            x = torch.tensor(input_vec, dtype=torch.float32, device=device)
            pred = model(x.unsqueeze(0))[0]
            pred_action = pred * action_std + action_mean
            pred_action = torch.clamp(pred_action, action_low, action_high)

            if prev_action is not None:
                pred_action = alpha * prev_action + (1 - alpha) * pred_action

                speed_scale = 0.5 + 0.5 * min(
                    1.0,
                    torch.norm(pred_action).item() / 100,
                )
                max_delta_dynamic = max_delta * speed_scale

                delta = pred_action - prev_action
                delta = torch.clamp(delta, -max_delta_dynamic, max_delta_dynamic)
                pred_action = prev_action + delta

            motion = torch.norm(pred_action).item()
            motions.append(motion)

            if prev_action is not None:
                previous_action = prev_action.clone()
                jump = torch.norm(pred_action - previous_action).item()
                jumps.append(jump)

                if max_jump_info is None or jump > max_jump_info["jump"]:
                    max_jump_info = {
                        "t": t,
                        "phase": phase,
                        "jump": jump,
                        "used_fallback_corners": used_fallback_corners,
                        "prev_action": previous_action.cpu().numpy(),
                        "pred_action": pred_action.cpu().numpy(),
                    }
            else:
                jump = 0.0

            prev_action = pred_action.clone()
            rollout_state = (
                rollout_state + pred_action.cpu().numpy() * STATE_UPDATE_SCALE
            ).astype(np.float32)

            if t % LOG_EVERY == 0:
                print(
                    f"t={t:04d} phase={phase} "
                    f"motion={motion:.4f} jump={jump:.4f} "
                    f"pred={np.round(pred_action.cpu().numpy(), 4)}"
                )

    print("\n--- SUMMARY ---")
    print(f"Skipped frames: {skipped}")
    print(f"Mean motion   : {np.mean(motions):.4f}")
    print(f"Max motion    : {np.max(motions):.4f}")
    print(f"Mean jump     : {np.mean(jumps):.4f}" if jumps else "Mean jump     : n/a")
    print(f"Max jump      : {np.max(jumps):.4f}" if jumps else "Max jump      : n/a")
    print(f"State drift   : {np.linalg.norm(rollout_state - initial_state):.4f}")
    if max_jump_info is not None:
        print(f"Max jump t    : {max_jump_info['t']}")
        print(f"Max jump phase: {max_jump_info['phase']}")
        print(f"Max jump fallback corners: {max_jump_info['used_fallback_corners']}")
        print(f"Max jump prev : {np.round(max_jump_info['prev_action'], 4)}")
        print(f"Max jump pred : {np.round(max_jump_info['pred_action'], 4)}")
    print("--- ROLLOUT END ---\n")


if __name__ == "__main__":
    main()
