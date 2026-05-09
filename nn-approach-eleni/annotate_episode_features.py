import argparse
from pathlib import Path

import cv2
import pandas as pd

from features import (
    DATASET_ROOT,
    dataset_shards,
    detect_corners,
    get_pick_and_target,
    load_all_filtered_episodes,
    order_corners,
    video_for_opencv,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def episode_locations(dataset_root):
    locations = []
    for video_path, data_path in dataset_shards(dataset_root):
        df = pd.read_parquet(
            data_path,
            columns=["episode_index", "frame_index", "index"],
        )
        locations.append((data_path, video_path, df))

    return locations


def choose_episode(locations, excluded):
    episode_ids = set()
    for _, _, df in locations:
        episode_ids.update(int(ep) for ep in df["episode_index"].unique())

    candidates = sorted(ep for ep in episode_ids if ep not in excluded)
    if not candidates:
        raise RuntimeError("No episode candidates left after applying filter config.")

    return candidates[0]


def draw_overlay(frame, corners, phase, frame_idx, episode_id, missing_corners=False):
    height, width = frame.shape[:2]
    overlay = frame.copy()

    phase_label = "0: back-left -> front-right" if phase == 0 else "1: back-right -> front-left"
    color = (0, 180, 255) if phase == 0 else (255, 120, 0)

    cv2.rectangle(overlay, (0, 0), (width, 92), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    cv2.putText(
        frame,
        f"episode={episode_id} frame={frame_idx} phase={phase_label}",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    phase_x = 14
    phase_y = 54
    phase_w = 220
    cv2.rectangle(frame, (phase_x, phase_y), (phase_x + phase_w, phase_y + 14), (220, 220, 220), 1)
    cv2.rectangle(
        frame,
        (phase_x, phase_y),
        (phase_x + (0 if phase == 0 else phase_w), phase_y + 14),
        color,
        -1,
    )
    cv2.putText(
        frame,
        "fold phase",
        (phase_x + phase_w + 12, phase_y + 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if corners is None:
        text = "corners: not detected"
        if missing_corners:
            text += " (using no feature for this frame)"
        cv2.putText(
            frame,
            text,
            (14, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    labels = ["back_left", "back_right", "front_left", "front_right"]
    colors = [(0, 255, 255), (0, 220, 0), (255, 0, 255), (255, 255, 0)]

    for point, label, point_color in zip(corners, labels, colors):
        x, y = point.astype(int)
        cv2.circle(frame, (x, y), 7, point_color, -1)
        cv2.circle(frame, (x, y), 10, (0, 0, 0), 2)
        cv2.putText(
            frame,
            label,
            (x + 10, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            point_color,
            2,
            cv2.LINE_AA,
        )

    polygon = corners.astype(int).reshape(-1, 1, 2)
    cv2.polylines(frame, [polygon], isClosed=True, color=(255, 255, 255), thickness=2)

    pick, target = get_pick_and_target(corners, phase)
    pick_xy = tuple(pick.astype(int))
    target_xy = tuple(target.astype(int))
    cv2.arrowedLine(frame, pick_xy, target_xy, color, 4, tipLength=0.08)
    cv2.circle(frame, pick_xy, 11, (0, 0, 255), 3)
    cv2.circle(frame, target_xy, 11, (0, 255, 0), 3)
    cv2.putText(frame, "pick", (pick_xy[0] + 12, pick_xy[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(frame, "target", (target_xy[0] + 12, target_xy[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    goal = target - pick
    cv2.putText(
        frame,
        f"goal vector px=({goal[0]:.1f}, {goal[1]:.1f})",
        (14, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )

    return frame


def main():
    args = parse_args()
    locations = episode_locations(args.dataset_root)
    if not locations:
        raise RuntimeError(f"No local dataset shards found under {args.dataset_root}")

    excluded = load_all_filtered_episodes()
    episode_id = args.episode if args.episode is not None else choose_episode(locations, excluded)

    output = args.output
    if output is None:
        output = Path("outputs") / f"episode_{episode_id:03d}_feature_overlay.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_frames = 0
    detected_frames = 0

    for _, video_path, df in locations:
        rows = df[df["episode_index"].astype(int) == episode_id]
        if rows.empty:
            continue

        episode_length = int(rows["frame_index"].max()) + 1
        start = int(rows.index.min())
        end = int(rows.index.max())
        opencv_video_path = video_for_opencv(video_path, start, end)
        cap = cv2.VideoCapture(str(opencv_video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {opencv_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))

        for _, row in rows.iterrows():
            ok, frame = cap.read()
            if not ok:
                continue

            frame_index = int(row["frame_index"])
            phase = 0 if frame_index < episode_length / 2 else 1

            raw_corners = detect_corners(frame, normalize=False)
            corners = order_corners(raw_corners) if raw_corners is not None else None
            if corners is not None:
                detected_frames += 1

            writer.write(draw_overlay(frame, corners, phase, frame_index, episode_id))
            total_frames += 1

        cap.release()

    if writer is None:
        raise RuntimeError(f"Episode {episode_id} not found in local dataset.")

    writer.release()
    print(f"episode={episode_id}")
    print(f"output={output}")
    print(f"frames={total_frames}")
    print(f"detected_corner_frames={detected_frames}")
    print(f"detection_rate={detected_frames / total_frames:.3f}")


if __name__ == "__main__":
    main()
