import argparse
from pathlib import Path

from script_utils import DEFAULT_PORTS_PATH, load_ports


def calibrate_motor(role: str, port: str) -> None:
    if role == "follower":
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        config = SO101FollowerConfig(port=port, id="follower")
        motor = SO101Follower(config)
    else:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        config = SO101LeaderConfig(port=port, id="leader")
        motor = SO101Leader(config)

    motor.connect(calibrate=False)
    try:
        motor.calibrate()
    finally:
        motor.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("leader", "follower"))
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PORTS_PATH,
        help="Path to the JSON file containing leader/follower ports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ports = load_ports(args.config)
    calibrate_motor(args.role, ports[args.role])


if __name__ == "__main__":
    main()
