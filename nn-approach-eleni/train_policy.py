from datasets import load_dataset
import torch
import numpy as np
import random
import cv2
import pandas as pd
import argparse
from pathlib import Path

from features import (
    DATASET_ROOT,
    build_policy_model,
    dataset_shards,
    detect_corners,
    get_fold_target,
    load_episode_filter,
    order_corners,
    video_for_opencv,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-filter",
        nargs="?",
        const="bad",
        choices=["bad", "meh"],
        default=None,
        help="Enable episode filtering. Use without a value to ignore bad episodes; use 'meh' to ignore bad and meh episodes.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Local LeRobot dataset root used for videos and parquet shards.",
    )
    parser.add_argument(
        "--output-policy",
        type=Path,
        default=Path("policy.pt"),
        help="Output checkpoint path.",
    )
    return parser.parse_args()


# =========================
# LOAD DATASET
# =========================
args = parse_args()
ignored_episodes = load_episode_filter(args.episode_filter)

ds = load_dataset(
    "robot-learning-team43/so101_teleop_private",
    split="train"
)

print("Dataset size:", len(ds))
print("Ignored episodes:", sorted(ignored_episodes))


# =========================
# DIMENSIONS
# =========================
state_dim = len(ds[0]["observation.state"])
action_dim = len(ds[0]["action"])

input_dim = state_dim + 8 + 2 + 1  # state + corners + goal + phase

print("Input dim:", input_dim)


# =========================
# NORMALIZATION
# =========================
sample_size = min(5000, len(ds))

normalization_indices = [
    i
    for i in range(len(ds))
    if int(ds[i]["episode_index"]) not in ignored_episodes
]
normalization_indices = normalization_indices[:sample_size]

if len(normalization_indices) == 0:
    raise RuntimeError("No samples left after applying episode filter.")

states = np.array([ds[i]["observation.state"] for i in normalization_indices])
actions = np.array([ds[i]["action"] for i in normalization_indices])

state_mean = torch.tensor(states.mean(axis=0), dtype=torch.float32)
state_std = torch.tensor(states.std(axis=0) + 1e-6, dtype=torch.float32)

action_mean = torch.tensor(actions.mean(axis=0), dtype=torch.float32)
action_std = torch.tensor(actions.std(axis=0) + 1e-6, dtype=torch.float32)


# =========================
# MODEL
# =========================
model = build_policy_model(input_dim, action_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

# =========================
# PRECOMPUTE VIDEO CORNERS
# =========================
print("Precomputing video corners...")

shards = dataset_shards(args.dataset_root)

if len(shards) == 0:
    raise RuntimeError(f"No dataset shards found under {args.dataset_root}.")

corners_by_index = {}
episode_lengths = {}

for vid_id, (video_path, data_path) in enumerate(shards):
    data = pd.read_parquet(
        data_path,
        columns=["index", "episode_index", "frame_index"]
    )
    data_indices = data["index"].tolist()

    for episode_id, frame_index in zip(data["episode_index"], data["frame_index"]):
        episode_id = int(episode_id)
        frame_index = int(frame_index)
        episode_lengths[episode_id] = max(
            episode_lengths.get(episode_id, 0),
            frame_index + 1
        )

    cap = cv2.VideoCapture(str(video_for_opencv(video_path)))

    frame_count = 0
    usable_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count >= len(data_indices):
            break

        corners = detect_corners(frame)
        if corners is not None:
            corners_by_index[int(data_indices[frame_count])] = order_corners(corners)
            usable_count += 1

        frame_count += 1

    cap.release()

    if frame_count != len(data_indices):
        print(
            f"Warning: video {vid_id} has {frame_count} frames, "
            f"data file has {len(data_indices)} rows"
        )

    print(f"Video {vid_id}: {frame_count} frames, {usable_count} usable corners")

print(f"Precomputed corners for {len(corners_by_index)} frames.")

# =========================
# TRAINING
# =========================
print("Training...")

num_steps = 20000
BATCH_SIZE = 32

for step in range(num_steps):

    inputs = []
    targets = []

    for _ in range(BATCH_SIZE):

        idx = random.randint(0, len(ds) - 1)
        sample = ds[idx]
        if int(sample["episode_index"]) in ignored_episodes:
            continue

        corners_np = corners_by_index.get(int(sample["index"]))
        if corners_np is None:
            continue

        episode_id = int(sample["episode_index"])
        frame_id = int(sample["frame_index"])
        episode_length = episode_lengths.get(episode_id)
        if episode_length is None:
            continue

        fold_phase = 0 if frame_id < episode_length / 2 else 1
        goal_vec_np = get_fold_target(corners_np, fold_phase)

        corners = torch.tensor(corners_np, dtype=torch.float32)

        corners_flat = corners.flatten()
        goal_vec = torch.tensor(goal_vec_np, dtype=torch.float32)

        # STATE / ACTION
        state = torch.tensor(sample["observation.state"], dtype=torch.float32)
        action = torch.tensor(sample["action"], dtype=torch.float32)

        state = (state - state_mean) / state_std
        action = (action - action_mean) / action_std

        # FOLD PHASE: 0 = back-left to front-right, 1 = back-right to front-left
        phase = torch.tensor([fold_phase], dtype=torch.float32)

        # FINAL INPUT
        input_tensor = torch.cat([state, corners_flat, goal_vec, phase])

        inputs.append(input_tensor)
        targets.append(action)

    if len(inputs) == 0:
        continue

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    pred = model(inputs)
    loss = loss_fn(pred, targets)

    optimizer.zero_grad()
    loss.backward()

    # stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step} | Loss {loss.item():.4f}")


# =========================
# SAVE
# =========================
torch.save({
    "model": model.state_dict(),
    "input_dim": input_dim,
    "state_mean": state_mean,
    "state_std": state_std,
    "action_mean": action_mean,
    "action_std": action_std
}, args.output_policy)

print(f"Model saved to {args.output_policy}.")
