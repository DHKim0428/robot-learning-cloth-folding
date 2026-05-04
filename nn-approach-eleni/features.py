import glob
import subprocess
import tempfile
import tomllib
from pathlib import Path

import cv2
import numpy as np
from torch import nn


DATASET_ROOT = Path("../data/so101_teleop")
EPISODE_FILTER_PATH = Path("../config/episode_filter.toml")


def load_episode_filter(mode, path=EPISODE_FILTER_PATH):
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


def load_all_filtered_episodes(path=EPISODE_FILTER_PATH):
    if not path.exists():
        return set()

    with path.open("rb") as f:
        config = tomllib.load(f)

    episodes = config.get("episodes", {})
    filtered = set(int(ep) for ep in episodes.get("bad", []))
    filtered.update(int(ep) for ep in episodes.get("meh", []))
    return filtered


def dataset_shards(dataset_root=DATASET_ROOT):
    video_files = sorted(glob.glob(
        str(dataset_root / "videos/observation.images.front/chunk-000/file-*.mp4")
    ))
    data_files = sorted(glob.glob(str(dataset_root / "data/chunk-000/file-*.parquet")))
    return list(zip(video_files, data_files))


def video_for_opencv(video_path, start=None, end=None):
    video_path = Path(video_path)
    temp_dir = Path(tempfile.mkdtemp(prefix="eleni_opencv_video_"))
    suffix = f"_{start}_{end}" if start is not None else ""
    output = temp_dir / f"{video_path.stem}{suffix}.mp4"

    command = ["ffmpeg", "-y", "-v", "error", "-i", str(video_path)]
    if start is not None:
        command += [
            "-vf",
            f"select='between(n,{start},{end})',setpts=PTS-STARTPTS",
        ]
    command += ["-an", "-r", "30", str(output)]
    subprocess.run(command, check=True)
    return output


def detect_corners(frame, normalize=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    points = approx.reshape(-1, 2)
    if len(points) < 4:
        return None

    points = points[:4].astype(np.float32)
    if normalize:
        points = points / np.array([[frame.shape[1], frame.shape[0]]])

    return points


def order_corners(corners):
    corners = np.asarray(corners, dtype=np.float32)
    y_sorted = corners[np.argsort(corners[:, 1])]

    back = y_sorted[:2]
    front = y_sorted[2:]

    back = back[np.argsort(back[:, 0])]
    front = front[np.argsort(front[:, 0])]

    return np.stack([back[0], back[1], front[0], front[1]]).astype(np.float32)


def get_pick_and_target(corners, phase):
    back_left, back_right, front_left, front_right = corners
    if phase < 0.5:
        return back_left, front_right
    return back_right, front_left


def get_fold_target(corners, phase):
    pick, target = get_pick_and_target(corners, phase)
    return target - pick


def build_policy_model(input_dim, action_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, action_dim),
    )
