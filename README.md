# Team 43 — ETH Robot Learning Cloth Folding

Repository for Team 43's ETH Robot Learning project on cloth folding.

## Current goal
Our near-term goal is to collect an initial dataset this week and get multiple first training attempts running as early as possible.

## This week's priorities
- finalize the data collection workflow
- collect an initial dataset
- have different team members try different model/training setups
- get the first end-to-end training runs working
- compare early results and decide what to scale up

## Important docs
- `docs/project_info.md` — project rules summary
- `docs/so101_config.md` — SO-101 setup notes
- `docs/PROJECT_QUESTION_LIST.md` — open questions for TAs / team
- `papers/README.md` — paper notes and references

## SO-101 quickstart
For day-to-day robot setup and data collection, see `docs/so101_config.md`.

Common commands:
- detect ports and write `config/so101_ports.json`
- setup / calibrate leader and follower motors
- teleoperate with or without camera
- record a local dataset under `data/lerobot/`
- optionally push a recorded dataset to Hugging Face

Main scripts:
- `scripts/detect_ports.py`
- `scripts/setup_motor.py`
- `scripts/calibrate_motor.py`
- `scripts/teleop.py`
- `scripts/teleop_record.py`

## Repository structure
- `docs/` — project notes, setup notes, and decisions
- `papers/` — optional paper notes and references
- `scripts/` — robot setup, teleoperation, and data collection scripts
- `config/` — local configuration templates and port files
- `data/lerobot/` — local LeRobot-format recordings (gitignored)

## Working conventions
- keep project notes and decisions in `docs/`
- keep large datasets, checkpoints, and logs out of git
- keep local LeRobot recordings under `data/lerobot/`
- keep machine-specific secrets and tokens out of git; use env files or shell exports instead
- prefer small, concrete updates over large undocumented changes

## Success for the first phase
A good first milestone is **not** a polished final method.
A good first milestone is:
- a small initial dataset
- multiple first training runs from different model choices
- one or more early test results
- one clear list of failure modes
