import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORTS_PATH = PROJECT_ROOT / "config" / "so101_ports.json"
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
