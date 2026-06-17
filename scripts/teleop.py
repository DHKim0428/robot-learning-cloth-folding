import argparse
from pathlib import Path

from script_utils import (
    DEFAULT_PORTS_PATH,
    follower_config_kwargs,
    leader_config_kwargs,
    load_ports,
    lock_camera_exposure,
)


def show_camera_frame(window_name: str, frame) -> bool:
    import cv2

    if frame is None:
        return True

    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, display_frame)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def build_follower_config(args: argparse.Namespace, follower_port: str):
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
            fps=args.camera_fps,
        )
    }
    return SO101FollowerConfig(
        **config_kwargs,
        cameras=camera_config,
    )


def run_teleop(args: argparse.Namespace) -> None:
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    ports = load_ports(args.config)
    robot_config = build_follower_config(args, ports["follower"])
    teleop_config = SO101LeaderConfig(**leader_config_kwargs(ports["leader"]))

    robot = SO101Follower(robot_config)
    teleop_device = SO101Leader(teleop_config)

    robot.connect()
    teleop_device.connect()
    if args.camera:
        lock_camera_exposure(args.camera_device, args.exposure)

    try:
        while True:
            if args.camera:
                observation = robot.get_observation()
                if not show_camera_frame(args.camera_name, observation.get(args.camera_name)):
                    break
            action = teleop_device.get_action()
            robot.send_action(action)
    finally:
        teleop_device.disconnect()
        robot.disconnect()
        if args.camera:
            import cv2

            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PORTS_PATH,
        help="Path to the JSON file containing leader/follower ports.",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Enable the follower camera and fetch observations during teleoperation.",
    )
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--camera-index", default=2)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--camera-device",
        default="/dev/video2",
        help="V4L2 device node to lock exposure on after the camera opens.",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=30,
        help="Manual exposure_time_absolute set after the camera opens (auto-exposure disabled).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_teleop(args)


if __name__ == "__main__":
    main()
