import argparse
import time
from pathlib import Path

from script_utils import DEFAULT_DATASET_ROOT, DEFAULT_PORTS_PATH, follower_config_kwargs, load_ports


DEFAULT_DATASET_REPO_ID = "local/so101_teleop"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a recorded LeRobot episode on the SO-101 follower arm."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PORTS_PATH,
        help="Path to the JSON file containing leader/follower ports.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=DEFAULT_DATASET_REPO_ID,
        help="Dataset repo identifier used for local dataset lookup.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Base directory where the dataset is stored locally.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to replay.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Optional FPS override for replay. Defaults to the dataset FPS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor import make_default_robot_action_processor
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.utils.constants import ACTION
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import log_say

    ports = load_ports(args.config)
    dataset_root = args.dataset_root / args.dataset_repo_id

    robot = SO101Follower(SO101FollowerConfig(**follower_config_kwargs(ports["follower"])))

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=dataset_root,
        episodes=[args.episode],
    )
    actions = dataset.select_columns(ACTION)
    robot_action_processor = make_default_robot_action_processor()
    replay_fps = args.fps if args.fps is not None else dataset.fps

    robot.connect()

    try:
        log_say(f"Replaying episode {args.episode}")
        for idx in range(dataset.num_frames):
            t0 = time.perf_counter()

            action_array = actions[idx][ACTION]
            action = {
                name: action_array[i] for i, name in enumerate(dataset.features[ACTION]["names"])
            }

            robot_obs = robot.get_observation()
            processed_action = robot_action_processor((action, robot_obs))
            robot.send_action(processed_action)

            precise_sleep(max(1.0 / replay_fps - (time.perf_counter() - t0), 0.0))
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
