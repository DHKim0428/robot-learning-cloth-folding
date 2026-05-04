from datasets import load_dataset
import torch
from torch import nn
import numpy as np
import random
import cv2
import glob
import pandas as pd
import argparse
import tomllib
from pathlib import Path


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
    return parser.parse_args()


def load_episode_filter(mode, path=Path("config/episode_filter.toml")):
    if mode is None:
        return set()

    if not path.exists():
        return set()

    with path.open("rb") as f:
        config = tomllib.load(f)

    episodes = config.get("episodes", {})
    ignored = set(int(ep) for ep in episodes.get("bad", []))

    if mode == "meh":
        ignored.update(int(ep) for ep in episodes.get("meh", []))

    return ignored


# =========================
# CORNER DETECTION
# =========================
def detect_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    pts = approx.reshape(-1, 2)

    if len(pts) < 4:
        return None

    pts = pts[:4]

    # normalize
    pts = pts / np.array([[frame.shape[1], frame.shape[0]]])

    return pts  # (4,2)


def order_corners(corners):
    # Image coordinates: x grows left-to-right, y grows back-to-front.
    corners = np.asarray(corners, dtype=np.float32)
    y_sorted = corners[np.argsort(corners[:, 1])]

    back = y_sorted[:2]
    front = y_sorted[2:]

    back = back[np.argsort(back[:, 0])]
    front = front[np.argsort(front[:, 0])]

    back_left, back_right = back
    front_left, front_right = front

    return np.stack(
        [back_left, back_right, front_left, front_right]
    ).astype(np.float32)


def get_fold_target(corners, phase):
    back_left, back_right, front_left, front_right = corners

    if phase == 0:
        pick_corner = back_left
        target_corner = front_right
    else:
        pick_corner = back_right
        target_corner = front_left

    return target_corner - pick_corner


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

input_dim = 6 + 8 + 2 + 1  # state + corners + goal + phase

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
model = nn.Sequential(
    nn.Linear(input_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, action_dim)
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# =========================
# PRECOMPUTE VIDEO CORNERS
# =========================
print("Precomputing video corners...")

video_files = sorted(glob.glob(
    "data/so101_teleop/videos/observation.images.front/chunk-000/file-*.mp4"
))
data_files = sorted(glob.glob(
    "data/so101_teleop/data/chunk-000/file-*.parquet"
))

if len(video_files) != len(data_files):
    raise RuntimeError(
        f"Found {len(video_files)} video files but {len(data_files)} data files."
    )

corners_by_index = {}
episode_lengths = {}

for vid_id, (video_path, data_path) in enumerate(zip(video_files, data_files)):
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

    cap = cv2.VideoCapture(video_path)

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
    "state_mean": state_mean,
    "state_std": state_std,
    "action_mean": action_mean,
    "action_std": action_std
}, "policy.pt")

print("Model saved.")
