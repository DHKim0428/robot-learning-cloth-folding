import argparse
import json
import platform
import time
from pathlib import Path

from script_utils import DEFAULT_PORTS_PATH


DEFAULT_OUTPUT_PATH = DEFAULT_PORTS_PATH


def find_available_ports() -> list[str]:
    if platform.system() == "Windows":
        from serial.tools import list_ports

        return sorted(port.device for port in list_ports.comports())

    return sorted(str(path) for path in Path("/dev").glob("tty*"))


def detect_removed_port(role: str) -> str:
    print(f"\n[{role}] Detecting port...")
    ports_before = find_available_ports()
    print("Ports before disconnecting:")
    for port in ports_before:
        print(f"  - {port}")

    input(f"Disconnect the {role} arm USB cable, then press Enter. ")
    time.sleep(0.5)

    ports_after = find_available_ports()
    removed_ports = sorted(set(ports_before) - set(ports_after))

    if len(removed_ports) != 1:
        raise RuntimeError(
            f"Could not uniquely detect the {role} port. Removed ports: {removed_ports}\n"
            "Make sure only that arm was disconnected, then try again."
        )

    detected_port = removed_ports[0]
    print(f"Detected {role} port: {detected_port}")
    input(f"Reconnect the {role} arm USB cable, then press Enter to continue. ")
    time.sleep(0.5)
    return detected_port


def write_ports(output_path: Path, ports: dict[str, str], force: bool) -> None:
    if output_path.exists() and not force:
        answer = input(f"{output_path} already exists. Overwrite it? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted without modifying the existing file.")
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ports, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved port config to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively detect SO-101 leader/follower USB ports and save them as JSON."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the generated JSON config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file without asking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Make sure both leader and follower USB cables are connected before starting.")
    input("Press Enter when ready. ")

    ports = {
        "leader": detect_removed_port("leader"),
        "follower": detect_removed_port("follower"),
    }
    write_ports(args.output, ports, force=args.force)


if __name__ == "__main__":
    main()
