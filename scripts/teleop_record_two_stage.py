import argparse
import sys
import threading
import time
from pathlib import Path

from script_utils import (
    DEFAULT_PORTS_PATH,
    DEFAULT_TWO_STAGE_DATASET_ROOT,
    DEFAULT_TWO_STAGE_HOME_POSE_PATH,
    leader_config_kwargs,
    load_home_pose,
    load_ports,
    move_leader_to_pose,
    move_robot_to_pose,
)
from teleop_record import (
    build_robot_config,
    has_finalized_local_dataset,
    has_partial_local_dataset,
    init_episode_keyboard_listener,
    is_local_repo_id,
    maybe_login_to_huggingface,
    prepare_dataset_root_for_create,
    print_recording_status,
    validate_push_to_hub_args,
    wait_for_episode_decision,
)


DEFAULT_DATASET_REPO_ID = "local/so101_two_stage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record SO-101 two-stage folding demonstrations into a separate LeRobot dataset. "
            "Each saved episode appends a recorded return to the two-stage home pose."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PORTS_PATH,
        help="Path to the JSON file containing leader/follower ports.",
    )
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-time-sec", type=int, default=60)
    parser.add_argument(
        "--stage",
        choices=("1", "2"),
        required=True,
        help="Stage label for this recording session.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task string stored in the dataset. Defaults to a stage-specific folding task.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=DEFAULT_DATASET_REPO_ID,
        help="Dataset repo identifier used for local dataset creation metadata.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_TWO_STAGE_DATASET_ROOT,
        help="Base directory where the two-stage dataset will be stored locally.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--license", default="apache-2.0")
    parser.add_argument("--branch", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--camera-index", default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument(
        "--home-pose-path",
        type=Path,
        default=DEFAULT_TWO_STAGE_HOME_POSE_PATH,
        help="Two-stage home pose used at episode start and appended episode end.",
    )
    parser.add_argument(
        "--home-move-time-sec",
        type=float,
        default=2.0,
        help="Duration for moving follower/leader to the two-stage home pose.",
    )
    parser.add_argument(
        "--home-return-time-sec",
        type=float,
        default=1.0,
        help="Duration of the recorded follower return-to-home segment appended to each episode.",
    )
    parser.add_argument(
        "--home-hold-time-sec",
        type=float,
        default=0.5,
        help="Recorded hold duration at home before saving the episode.",
    )
    return parser.parse_args()


def print_two_stage_recording_controls() -> None:
    print("\nKeyboard controls:")
    print("  Space       start recording the current episode")
    print("  Right arrow finish episode, append home return, save")
    print("  Left arrow  finish episode, append home return, discard")
    print("  Esc         abort the recording session without saving the current episode")
    print()


def print_two_stage_elapsed_recording_status(
    events: dict,
    episode_idx: int,
    num_episodes: int,
    episode_time_sec: int,
) -> tuple[threading.Event, threading.Thread]:
    done = threading.Event()

    def update_status() -> None:
        start_t = time.perf_counter()
        while not done.is_set() and not events["exit_early"] and not events["stop_recording"]:
            elapsed_s = time.perf_counter() - start_t
            remaining_s = max(episode_time_sec - elapsed_s, 0.0)
            sys.stdout.write(
                "\r"
                f"[episode {episode_idx + 1}/{num_episodes}] recording in progress "
                f"- elapsed {elapsed_s:05.1f}s / {episode_time_sec}s "
                f"- remaining {remaining_s:05.1f}s "
                "- Right saves with home return, Left discards with home return, Esc aborts"
            )
            sys.stdout.flush()
            done.wait(0.2)

        elapsed_s = time.perf_counter() - start_t
        sys.stdout.write(
            "\r"
            f"[episode {episode_idx + 1}/{num_episodes}] recording paused "
            f"- elapsed {elapsed_s:05.1f}s / {episode_time_sec}s"
            + " " * 40
            + "\n"
        )
        sys.stdout.flush()

    thread = threading.Thread(target=update_status, daemon=True)
    thread.start()
    return done, thread


def wait_at_home_for_episode_start(
    *,
    events: dict,
    episode_idx: int,
    num_episodes: int,
) -> bool:
    from lerobot.utils.robot_utils import precise_sleep

    events["start_episode"] = False
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["save_episode"] = False
    events["episode_decision_enabled"] = False
    print_recording_status(
        episode_idx=episode_idx,
        num_episodes=num_episodes,
        recording=False,
        detail="both arms at home; press Space to release leader and start, or Esc to stop",
    )

    while not events["stop_recording"]:
        if events["start_episode"]:
            events["start_episode"] = False
            sys.stdout.write("\n")
            sys.stdout.flush()
            return True
        precise_sleep(0.1)

    sys.stdout.write("\n")
    sys.stdout.flush()
    return False


def add_home_frame(
    *,
    robot,
    dataset,
    robot_observation_processor,
    target_pose: dict[str, float],
    task: str,
) -> None:
    from lerobot.scripts.lerobot_record import build_dataset_frame
    from lerobot.utils.constants import ACTION, OBS_STR

    obs = robot.get_observation()
    obs_processed = robot_observation_processor(obs)
    observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
    sent_action = robot.send_action(target_pose)
    action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
    dataset.add_frame({**observation_frame, **action_frame, "task": task})


def append_recorded_return_home(
    *,
    robot,
    dataset,
    robot_observation_processor,
    target_pose: dict[str, float],
    fps: int,
    move_time_s: float,
    hold_time_s: float,
    task: str,
) -> None:
    from lerobot.utils.robot_utils import precise_sleep

    current_pose = {
        key: float(value)
        for key, value in robot.get_observation().items()
        if key in target_pose and key.endswith(".pos")
    }
    common_keys = [key for key in target_pose if key in current_pose]
    if not common_keys:
        return

    print("Appending recorded return to two-stage home pose...")
    steps = max(int(move_time_s * fps), 1)
    for step_idx in range(1, steps + 1):
        t0 = time.perf_counter()
        alpha = step_idx / steps
        action = {
            key: (1.0 - alpha) * current_pose[key] + alpha * target_pose[key]
            for key in common_keys
        }
        add_home_frame(
            robot=robot,
            dataset=dataset,
            robot_observation_processor=robot_observation_processor,
            target_pose=action,
            task=task,
        )
        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))

    hold_steps = max(int(hold_time_s * fps), 1)
    hold_action = {key: target_pose[key] for key in common_keys}
    for _ in range(hold_steps):
        t0 = time.perf_counter()
        add_home_frame(
            robot=robot,
            dataset=dataset,
            robot_observation_processor=robot_observation_processor,
            target_pose=hold_action,
            task=task,
        )
        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))


def create_or_resume_dataset(args: argparse.Namespace, dataset_features: dict, robot_name: str):
    from huggingface_hub.errors import RepositoryNotFoundError
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = args.dataset_root / args.dataset_repo_id

    if args.resume:
        if is_local_repo_id(args.dataset_repo_id) and not has_finalized_local_dataset(dataset_root):
            raise FileNotFoundError(
                f"Cannot resume local dataset because no finalized dataset metadata was found at: {dataset_root}"
            )

        try:
            if not has_finalized_local_dataset(dataset_root) and not is_local_repo_id(args.dataset_repo_id):
                maybe_login_to_huggingface(args.hf_token_env)
            return LeRobotDataset.resume(
                repo_id=args.dataset_repo_id,
                root=dataset_root,
                image_writer_threads=args.image_writer_threads,
            )
        except RepositoryNotFoundError:
            if has_partial_local_dataset(dataset_root):
                raise FileNotFoundError(
                    f"A partial local dataset directory exists at {dataset_root}, but no matching Hub dataset was found. "
                    "Delete or rename the local directory and try again, or rerun without --resume."
                )
            prepare_dataset_root_for_create(dataset_root)
            print("No existing Hub dataset was found. Starting a new local dataset instead.")

    try:
        return LeRobotDataset.create(
            repo_id=args.dataset_repo_id,
            fps=args.fps,
            root=dataset_root,
            features=dataset_features,
            robot_type=robot_name,
            use_videos=args.camera,
            image_writer_threads=args.image_writer_threads,
        )
    except FileExistsError as e:
        raise FileExistsError(
            f"Dataset directory already exists: {dataset_root}. "
            "Use --resume to append episodes, or choose a different --dataset-repo-id / --dataset-root."
        ) from e


def main() -> None:
    args = parse_args()
    if args.task is None:
        args.task = f"SO101 two-stage towel folding stage {args.stage}"
    validate_push_to_hub_args(args)
    if not args.home_pose_path.exists():
        raise FileNotFoundError(
            f"Two-stage home pose was not found at {args.home_pose_path}. "
            "Create it first with scripts/capture_two_stage_home_pose.py."
        )

    try:
        from lerobot.datasets.feature_utils import hw_to_dataset_features
    except ImportError:
        from lerobot.utils.feature_utils import hw_to_dataset_features
    try:
        from lerobot.processor import make_default_processors
    except ImportError:
        from lerobot.processor.factory import make_default_processors
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.scripts.lerobot_record import record_loop
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import init_rerun

    ports = load_ports(args.config)
    home_pose = load_home_pose(args.home_pose_path)

    robot_config = build_robot_config(args, ports["follower"])
    teleop_config = SO101LeaderConfig(**leader_config_kwargs(ports["leader"]))

    robot = SO101Follower(robot_config)
    teleop = SO101Leader(teleop_config)

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    dataset = create_or_resume_dataset(args, dataset_features, robot.name)

    listener = None
    robot_connected = False
    teleop_connected = False

    try:
        listener, events = init_episode_keyboard_listener()
        init_rerun(session_name="two_stage_recording")

        robot.connect()
        robot_connected = True
        teleop.connect()
        teleop_connected = True

        teleop_action_processor, robot_action_processor, robot_observation_processor = (
            make_default_processors()
        )

        episode_idx = 0
        while episode_idx < args.num_episodes and not events["stop_recording"]:
            print_two_stage_recording_controls()
            print("Moving follower and leader to two-stage home pose...")
            move_robot_to_pose(
                robot=robot,
                target_pose=home_pose,
                duration_s=args.home_move_time_sec,
                fps=args.fps,
            )
            move_leader_to_pose(
                teleop,
                target_pose=home_pose,
                duration_s=args.home_move_time_sec,
                fps=args.fps,
                release_after=False,
            )

            if not wait_at_home_for_episode_start(
                events=events,
                episode_idx=episode_idx,
                num_episodes=args.num_episodes,
            ):
                break

            teleop.bus.disable_torque()
            precise_sleep(0.1)

            events["exit_early"] = False
            events["rerecord_episode"] = False
            events["save_episode"] = False
            events["episode_decision_enabled"] = True
            status_done, status_thread = print_two_stage_elapsed_recording_status(
                events=events,
                episode_idx=episode_idx,
                num_episodes=args.num_episodes,
                episode_time_sec=args.episode_time_sec,
            )
            try:
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
            finally:
                status_done.set()
                status_thread.join()

            if not events["stop_recording"]:
                append_recorded_return_home(
                    robot=robot,
                    dataset=dataset,
                    robot_observation_processor=robot_observation_processor,
                    target_pose=home_pose,
                    fps=args.fps,
                    move_time_s=args.home_return_time_sec,
                    hold_time_s=args.home_hold_time_sec,
                    task=args.task,
                )

            decision = wait_for_episode_decision(events, episode_idx, args.num_episodes)

            if decision == "discard":
                events["rerecord_episode"] = False
                events["exit_early"] = False
                events["save_episode"] = False
                events["episode_decision_enabled"] = False
                dataset.clear_episode_buffer()
                print_recording_status(
                    episode_idx=episode_idx,
                    num_episodes=args.num_episodes,
                    recording=False,
                    detail="episode discarded; reset the towel, then press Space",
                )
                continue

            if decision == "stop":
                dataset.clear_episode_buffer()
                print_recording_status(
                    episode_idx=episode_idx,
                    num_episodes=args.num_episodes,
                    recording=False,
                    detail="session stopping; unsaved episode discarded",
                )
                break

            events["save_episode"] = False
            events["exit_early"] = False
            events["rerecord_episode"] = False
            events["episode_decision_enabled"] = False
            print_recording_status(
                episode_idx=episode_idx,
                num_episodes=args.num_episodes,
                recording=False,
                detail="saving episode",
            )
            dataset.save_episode()

            episode_idx += 1
            if episode_idx < args.num_episodes:
                print_recording_status(
                    episode_idx=episode_idx,
                    num_episodes=args.num_episodes,
                    recording=False,
                    detail="saved; reset the towel, then press Space",
                )
    finally:
        print("Stop two-stage recording")
        if listener is not None:
            listener.stop()
        try:
            dataset.finalize()
        finally:
            if teleop_connected:
                try:
                    teleop.bus.disable_torque()
                except Exception:
                    pass
                teleop.disconnect()
            if robot_connected:
                robot.disconnect()

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
