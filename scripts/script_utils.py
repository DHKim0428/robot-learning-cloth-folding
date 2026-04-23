import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORTS_PATH = PROJECT_ROOT / "config" / "so101_ports.json"
DEFAULT_HOME_POSE_PATH = PROJECT_ROOT / "config" / "so101_home_pose.json"
DEFAULT_FINAL_POSE_PATH = PROJECT_ROOT / "config" / "so101_final_pose.json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "lerobot"


def load_ports(config_path: Path) -> dict[str, str]:
    with config_path.open("r", encoding="utf-8") as f:
        ports = json.load(f)

    missing_roles = {"leader", "follower"} - ports.keys()
    if missing_roles:
        missing = ", ".join(sorted(missing_roles))
        raise ValueError(f"Missing port entries in {config_path}: {missing}")

    return {
        "leader": str(ports["leader"]),
        "follower": str(ports["follower"]),
    }


def load_home_pose(home_pose_path: Path) -> dict[str, float]:
    with home_pose_path.open("r", encoding="utf-8") as f:
        pose = json.load(f)

    if not isinstance(pose, dict) or not pose:
        raise ValueError(f"Home pose file is empty or invalid: {home_pose_path}")

    return {str(key): float(value) for key, value in pose.items()}


def save_home_pose(home_pose_path: Path, pose: dict[str, float]) -> None:
    home_pose_path.parent.mkdir(parents=True, exist_ok=True)
    home_pose_path.write_text(json.dumps(pose, indent=2) + "\n", encoding="utf-8")


def load_final_pose(final_pose_path: Path) -> dict[str, float]:
    return load_home_pose(final_pose_path)


def save_final_pose(final_pose_path: Path, pose: dict[str, float]) -> None:
    save_home_pose(final_pose_path, pose)
