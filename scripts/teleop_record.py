import argparse
import os
import time
from pathlib import Path

from script_utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_FINAL_POSE_PATH,
    DEFAULT_HOME_POSE_PATH,
    DEFAULT_PORTS_PATH,
    follower_config_kwargs,
    leader_config_kwargs,
    load_final_pose,
    load_home_pose,
    load_ports,
)


DEFAULT_DATASET_REPO_ID = "local/so101_teleop"


def build_robot_config(args: argparse.Namespace, follower_port: str):
    from lerobot.robots.so_follower import SO101FollowerConfig

    config_kwargs = follower_config_kwargs(follower_port)

    if not args.camera:
        return SO101FollowerConfig(**config_kwargs)

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
        **config_kwargs,
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
    parser.add_argument(
        "--home-pose-path",
        type=Path,
        default=DEFAULT_HOME_POSE_PATH,
        help="Path to a saved home pose JSON file.",
    )
    parser.add_argument(
        "--final-pose-path",
        type=Path,
        default=DEFAULT_FINAL_POSE_PATH,
        help="Path to a saved final pose JSON file used when exiting recording.",
    )
    parser.add_argument(
        "--return-pose-source",
        choices=("initial", "home"),
        default="home",
        help="Which pose to return to after each saved episode.",
    )
    parser.add_argument(
        "--return-to-initial-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After each saved episode, move the follower back to the selected return pose.",
    )
    parser.add_argument(
        "--return-move-time-sec",
        type=float,
        default=1.5,
        help="Duration of the smooth motion used when returning to the selected pose.",
    )
    return parser.parse_args()


def maybe_login_to_huggingface(token_env_var: str) -> None:
    token = os.getenv(token_env_var) or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return

    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)


def is_local_repo_id(repo_id: str) -> bool:
    return repo_id.split("/", 1)[0] == "local" if "/" in repo_id else True


def has_local_dataset_metadata(dataset_root: Path) -> bool:
    return (dataset_root / "meta" / "info.json").exists()


def has_local_episode_metadata(dataset_root: Path) -> bool:
    episodes_dir = dataset_root / "meta" / "episodes"
    return episodes_dir.exists() and any(episodes_dir.rglob("*.parquet"))


def has_finalized_local_dataset(dataset_root: Path) -> bool:
    return has_local_dataset_metadata(dataset_root) and has_local_episode_metadata(dataset_root)


def has_partial_local_dataset(dataset_root: Path) -> bool:
    return dataset_root.exists() and any(dataset_root.iterdir()) and not has_finalized_local_dataset(dataset_root)


def prepare_dataset_root_for_create(dataset_root: Path) -> None:
    if dataset_root.exists() and not any(dataset_root.iterdir()):
        dataset_root.rmdir()


def extract_joint_pose(observation: dict[str, object]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in observation.items()
        if key.endswith(".pos")
    }


def move_robot_to_pose(
    robot,
    target_pose: dict[str, float],
    duration_s: float,
    fps: int,
) -> None:
    from lerobot.utils.robot_utils import precise_sleep

    current_pose = extract_joint_pose(robot.get_observation())
    common_keys = [key for key in target_pose if key in current_pose]
    if not common_keys:
        return

    steps = max(int(duration_s * fps), 1)
    for step_idx in range(1, steps + 1):
        t0 = time.perf_counter()
        alpha = step_idx / steps
        action = {
            key: (1.0 - alpha) * current_pose[key] + alpha * target_pose[key]
            for key in common_keys
        }
        robot.send_action(action)
        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))


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

    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        from lerobot.datasets.feature_utils import hw_to_dataset_features
    except ImportError:
        from lerobot.utils.feature_utils import hw_to_dataset_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    try:
        from lerobot.processor import make_default_processors
    except ImportError:
        from lerobot.processor.factory import make_default_processors
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.scripts.lerobot_record import record_loop
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    try:
        from lerobot.utils.control_utils import init_keyboard_listener
    except ImportError:
        from lerobot.common.control_utils import init_keyboard_listener
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import log_say
    from lerobot.utils.visualization_utils import init_rerun

    ports = load_ports(args.config)

    robot_config = build_robot_config(args, ports["follower"])
    teleop_config = SO101LeaderConfig(**leader_config_kwargs(ports["leader"]))

    robot = SO101Follower(robot_config)
    teleop = SO101Leader(teleop_config)

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    dataset_root = args.dataset_root / args.dataset_repo_id

    if args.resume:
        if is_local_repo_id(args.dataset_repo_id) and not has_finalized_local_dataset(dataset_root):
            raise FileNotFoundError(
                f"Cannot resume local dataset because no finalized dataset metadata was found at: {dataset_root}"
            )

        try:
            if not has_finalized_local_dataset(dataset_root) and not is_local_repo_id(args.dataset_repo_id):
                maybe_login_to_huggingface(args.hf_token_env)

            dataset = LeRobotDataset.resume(
                repo_id=args.dataset_repo_id,
                root=dataset_root,
                image_writer_threads=args.image_writer_threads,
            )
        except RepositoryNotFoundError:
            if has_partial_local_dataset(dataset_root):
                raise FileNotFoundError(
                    f"A partial local dataset directory exists at {dataset_root}, but no matching Hub dataset was found. "
                    "This usually happens when a previous run was interrupted before the dataset was pushed. "
                    "Delete or rename the local directory and try again, or rerun without --resume."
                )

            if dataset_root.exists() and any(dataset_root.iterdir()):
                raise FileNotFoundError(
                    f"Could not resume from the Hub, and the local dataset directory is not a finalized dataset: {dataset_root}. "
                    "Delete or rename the directory and try again, or rerun without --resume."
                )

            prepare_dataset_root_for_create(dataset_root)
            log_say(
                "No existing Hub dataset was found. Starting a new local dataset instead."
            )
            dataset = LeRobotDataset.create(
                repo_id=args.dataset_repo_id,
                fps=args.fps,
                root=dataset_root,
                features=dataset_features,
                robot_type=robot.name,
                use_videos=args.camera,
                image_writer_threads=args.image_writer_threads,
            )
        except FileNotFoundError:
            if has_partial_local_dataset(dataset_root) and not is_local_repo_id(args.dataset_repo_id):
                maybe_login_to_huggingface(args.hf_token_env)
                dataset = LeRobotDataset.resume(
                    repo_id=args.dataset_repo_id,
                    root=dataset_root,
                    image_writer_threads=args.image_writer_threads,
                )
            else:
                raise
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
    initial_pose = extract_joint_pose(robot.get_observation())
    if args.return_pose_source == "home":
        try:
            return_pose = load_home_pose(args.home_pose_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Home pose file not found: {args.home_pose_path}. "
                "Run `python scripts/save_home_pose.py` first, or use `--return-pose-source initial`."
            ) from e
    else:
        return_pose = initial_pose
    final_pose = load_final_pose(args.final_pose_path) if args.final_pose_path.exists() else None

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

            if args.return_to_initial_pose:
                log_say(
                    "Returning robot to home pose"
                    if args.return_pose_source == "home"
                    else "Returning robot to initial pose"
                )
                move_robot_to_pose(
                    robot=robot,
                    target_pose=return_pose,
                    duration_s=args.return_move_time_sec,
                    fps=args.fps,
                )
                precise_sleep(0.2)

            dataset.save_episode()
            episode_idx += 1
    finally:
        log_say("Stop recording")
        dataset.finalize()
        if final_pose is not None:
            log_say("Returning robot to final pose")
            move_robot_to_pose(
                robot=robot,
                target_pose=final_pose,
                duration_s=args.return_move_time_sec,
                fps=args.fps,
            )
            precise_sleep(0.2)
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
