import argparse
import sys
import time
from pathlib import Path

from script_utils import (
    DEFAULT_PORTS_PATH,
    DEFAULT_TWO_STAGE_HOME_POSE_PATH,
    follower_config_kwargs,
    extract_joint_pose,
    leader_config_kwargs,
    load_ports,
    save_home_pose,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teleoperate the SO-101 to a camera-friendly two-stage folding home pose, "
            "then press Space to save the follower joint pose."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PORTS_PATH,
        help="Path to the JSON file containing leader/follower ports.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TWO_STAGE_HOME_POSE_PATH,
        help="Path to write the saved two-stage home pose JSON.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--camera-index", default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--no-camera-preview",
        dest="camera_preview",
        action="store_false",
        default=True,
        help="Disable the live OpenCV preview window.",
    )
    parser.add_argument(
        "--camera-preview-window",
        default="SO-101 two-stage home camera",
        help="Window title for the live camera preview.",
    )
    return parser.parse_args()


def init_keyboard_listener() -> tuple[object, dict[str, bool]]:
    try:
        from lerobot.utils.control_utils import is_headless
    except ImportError:
        from lerobot.common.control_utils import is_headless

    if is_headless():
        raise RuntimeError("Keyboard controls are required, but a headless environment was detected.")

    from pynput import keyboard

    events = {"save": False, "exit": False}

    def on_press(key):
        if key == keyboard.Key.space:
            events["save"] = True
        elif key == keyboard.Key.esc:
            events["exit"] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def validate_camera_preview(args: argparse.Namespace) -> None:
    if not args.camera_preview:
        return

    try:
        import cv2
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "OpenCV camera preview is enabled, but the `cv2` Python package is not installed "
            "in the active environment. Install an OpenCV build with HighGUI support in the `lerobot` "
            "environment, or run this script with --no-camera-preview."
        ) from e

    try:
        cv2.namedWindow(args.camera_preview_window, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(args.camera_preview_window)
    except cv2.error as e:
        raise RuntimeError(
            "OpenCV camera preview is enabled, but this Python environment's cv2 build "
            "does not include HighGUI window support (`cv2.getBuildInformation()` reports "
            "`GUI: NONE`). Install an OpenCV build with GUI/HighGUI support in the `lerobot` "
            "environment, for example:\n\n"
            "  pip uninstall -y opencv-python-headless opencv-python\n"
            "  pip install opencv-python\n\n"
            "Alternatively run this script with --no-camera-preview to capture the pose "
            "without opening a window."
        ) from e


def update_camera_preview(observation: dict[str, object], args: argparse.Namespace) -> int | None:
    if not args.camera_preview:
        return None

    import cv2

    image = observation.get(args.camera_name)
    if image is None:
        return None

    # LeRobot OpenCV cameras return RGB by default, while cv2.imshow expects BGR.
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(args.camera_preview_window, frame)
    key = cv2.waitKey(1) & 0xFF
    return key if key != 255 else None


def build_camera_follower_config(args: argparse.Namespace, follower_port: str):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101FollowerConfig

    return SO101FollowerConfig(
        **follower_config_kwargs(follower_port),
        cameras={
            args.camera_name: OpenCVCameraConfig(
                index_or_path=args.camera_index,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
            )
        },
    )


def main() -> None:
    args = parse_args()
    validate_camera_preview(args)

    from lerobot.robots.so_follower import SO101Follower
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.utils.robot_utils import precise_sleep

    ports = load_ports(args.config)
    robot_config = build_camera_follower_config(args, ports["follower"])
    teleop_config = SO101LeaderConfig(**leader_config_kwargs(ports["leader"]))

    robot = SO101Follower(robot_config)
    teleop = SO101Leader(teleop_config)
    listener = None
    robot_connected = False
    teleop_connected = False

    try:
        robot.connect()
        robot_connected = True
        teleop.connect()
        teleop_connected = True

        listener, events = init_keyboard_listener()
        print("Teleoperate to the desired two-stage home pose.")
        print("Camera preview is enabled by default. Use --no-camera-preview to disable it.")
        print("Press Space to save the follower pose. Press Esc or q in the preview to exit.")

        last_status_t = 0.0
        while not events["exit"] and not events["save"]:
            t0 = time.perf_counter()
            obs = robot.get_observation()
            preview_key = update_camera_preview(obs, args)
            if preview_key == ord(" "):
                events["save"] = True
            elif preview_key in (27, ord("q")):
                events["exit"] = True

            action = teleop.get_action()
            robot.send_action(action)

            now = time.perf_counter()
            if now - last_status_t >= 0.5:
                last_status_t = now
                sys.stdout.write("\rPositioning home pose - Space saves, Esc exits")
                sys.stdout.flush()

            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))

        sys.stdout.write("\n")
        sys.stdout.flush()

        if events["save"]:
            pose = extract_joint_pose(robot.get_observation())
            save_home_pose(args.output, pose)
            print(f"Saved two-stage home pose to: {args.output}")
        else:
            print("Exited without saving.")
    finally:
        if args.camera_preview:
            import cv2

            try:
                cv2.destroyWindow(args.camera_preview_window)
            except cv2.error:
                pass
        if listener is not None:
            listener.stop()
        if teleop_connected:
            teleop.disconnect()
        if robot_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
