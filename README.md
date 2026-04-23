# Team 43 — ETH Robot Learning Cloth Folding

Repository for Team 43's ETH Robot Learning project on cloth folding.

## Current milestone
Our current milestone is to complete the course-required **sanity-check pipeline** and verify that the full data-to-deployment loop works end to end before investing heavily in the final method.

Current milestone checklist:
- collect **20 consistent demonstrations** of the task or a simplified version
- record everything in **LeRobot dataset format v3**
- replay demonstrations to verify recording correctness
- upload the dataset to **Brev**
- train a **simple behavior cloning** model to overfit the demonstrations
- deploy the trained policy in a matching scene and verify motion reproduction
- prepare a short **Thursday Slack update** with progress and blockers

## This week's priorities
- finalize the recording and replay workflow
- collect the first **20-demo** sanity-check dataset
- confirm recordings are valid through replay
- get one simple BC overfit run working on Brev
- test deployment with matching object/background setup
- summarize progress and blockers before the Thursday session

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
- send a short update in the team Slack channel before each Thursday session

## Success for the current milestone
A good current milestone is **not** a polished final folding policy.
A good current milestone is:
- a **20-demo** LeRobot v3 dataset collected successfully
- replay verification showing the demonstrations were recorded correctly
- one simple BC training run that clearly overfits or nearly overfits
- one deployment test in a matching scene
- one short list of concrete failure modes / blockers to discuss with the TAs

## After this milestone
Once the sanity-check pipeline works, the next step is to shift back to the actual project objective:
- focus first on **Eval 1 (grasping)**
- expand the dataset and scene diversity
- compare candidate policy families / pretrained starting points
- iterate toward **single fold** and then **double fold**
