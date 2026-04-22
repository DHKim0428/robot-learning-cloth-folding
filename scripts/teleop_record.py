import argparse
import os
from pathlib import Path

from script_utils import DEFAULT_DATASET_ROOT, DEFAULT_PORTS_PATH, load_ports


DEFAULT_DATASET_REPO_ID = "local/so101_teleop"


def build_robot_config(args: argparse.Namespace, follower_port: str):
    from lerobot.robots.so_follower import SO101FollowerConfig

    if not args.camera:
        return SO101FollowerConfig(port=follower_port, id="follower")

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    camera_config = {
        args.camera_name: OpenCVCameraConfig(
            index_or_path=args.camera_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.fps,
        )
    }
    return SO101FollowerConfig(
        port=follower_port,
        id="follower",
        cameras=camera_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PORTS_PATH,
        help="Path to the JSON file containing leader/follower ports.",
    )
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-time-sec", type=int, default=60)
    parser.add_argument("--reset-time-sec", type=int, default=10)
    parser.add_argument("--task", default="SO101 teleoperation task")
    parser.add_argument(
        "--dataset-repo-id",
        default=DEFAULT_DATASET_REPO_ID,
        help="Dataset repo identifier used for local dataset creation metadata.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Base directory where the dataset will be stored locally.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append new episodes to an existing dataset instead of creating a new one.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the recorded dataset to the Hugging Face Hub after recording finishes.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the Hugging Face dataset repo as private when pushing.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Optional Hugging Face dataset card tags to attach when pushing.",
    )
    parser.add_argument(
        "--license",
        default="apache-2.0",
        help="Dataset license metadata used when pushing to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Optional Hugging Face branch to push the dataset to.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable name that stores the Hugging Face token for upload.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Enable the follower camera and record observations with images.",
    )
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--camera-index", default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    return parser.parse_args()


def maybe_login_to_huggingface(token_env_var: str) -> None:
    token = os.getenv(token_env_var) or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return

    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)


def validate_push_to_hub_args(args: argparse.Namespace) -> None:
    if not args.push_to_hub:
        return

    if "/" not in args.dataset_repo_id:
        raise ValueError(
            "--push-to-hub requires --dataset-repo-id to look like '<hf_username>/<dataset_name>'."
        )

    namespace, _ = args.dataset_repo_id.split("/", 1)
    if namespace == "local":
        raise ValueError(
            "--push-to-hub cannot be used with the default local repo id. "
            "Please set --dataset-repo-id to your real Hugging Face namespace, e.g. '<hf_username>/so101_teleop'."
        )


def main() -> None:
    args = parse_args()
    validate_push_to_hub_args(args)

    from lerobot.datasets.feature_utils import hw_to_dataset_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor import make_default_processors
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.scripts.lerobot_record import record_loop
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.utils.control_utils import init_keyboard_listener
    from lerobot.utils.utils import log_say
    from lerobot.utils.visualization_utils import init_rerun

    ports = load_ports(args.config)

    robot_config = build_robot_config(args, ports["follower"])
    teleop_config = SO101LeaderConfig(port=ports["leader"], id="leader")

    robot = SO101Follower(robot_config)
    teleop = SO101Leader(teleop_config)

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    dataset_root = args.dataset_root / args.dataset_repo_id

    if args.resume:
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"Cannot resume because the dataset directory does not exist: {dataset_root}"
            )
        dataset = LeRobotDataset.resume(
            repo_id=args.dataset_repo_id,
            root=dataset_root,
            image_writer_threads=args.image_writer_threads,
        )
    else:
        try:
            dataset = LeRobotDataset.create(
                repo_id=args.dataset_repo_id,
                fps=args.fps,
                root=dataset_root,
                features=dataset_features,
                robot_type=robot.name,
                use_videos=args.camera,
                image_writer_threads=args.image_writer_threads,
            )
        except FileExistsError as e:
            raise FileExistsError(
                f"Dataset directory already exists: {dataset_root}. "
                "Use --resume to append episodes, or choose a different --dataset-repo-id / --dataset-root."
            ) from e

    _, events = init_keyboard_listener()
    init_rerun(session_name="recording")

    robot.connect()
    teleop.connect()

    try:
        teleop_action_processor, robot_action_processor, robot_observation_processor = (
            make_default_processors()
        )

        episode_idx = 0
        while episode_idx < args.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {episode_idx + 1} of {args.num_episodes}")

            record_loop(
                robot=robot,
                events=events,
                fps=args.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                dataset=dataset,
                control_time_s=args.episode_time_sec,
                single_task=args.task,
                display_data=True,
            )

            if not events["stop_recording"] and (
                episode_idx < args.num_episodes - 1 or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=args.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    control_time_s=args.reset_time_sec,
                    single_task=args.task,
                    display_data=True,
                )

            if events["rerecord_episode"]:
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            episode_idx += 1
    finally:
        log_say("Stop recording")
        dataset.finalize()
        robot.disconnect()
        teleop.disconnect()

    if args.push_to_hub:
        maybe_login_to_huggingface(args.hf_token_env)
        dataset.push_to_hub(
            branch=args.branch,
            tags=args.tags,
            license=args.license,
            private=args.private,
        )


if __name__ == "__main__":
    main()
