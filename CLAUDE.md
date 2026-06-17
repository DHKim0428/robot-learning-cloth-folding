# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Team 43's cloth-folding project for the SO-101 robot arm (ETH Robot Learning). The repo is a thin orchestration layer — Python helpers in `scripts/` and launcher shells in `shell/` — wrapping LeRobot's `lerobot-train` / `lerobot-rollout` and custom policy server/client processes. Two policy stacks are shipped: **SmolVLA** (basic eval) and **MolmoAct2** (bonus eval). `ACT_model/` and `diffusion-policy/` are standalone training baselines, not part of the final eval path.

## Two environments — pick the right one per policy

This is the single most important thing to get right. The two policies use different LeRobot builds and different Python environments:

- **conda env `lerobot`** (Python 3.12) → SmolVLA / basic eval / most teleop & recording. LeRobot is installed editable from a sibling `lerobot/` checkout.
- **uv `.venv`** (created by `uv sync`) → MolmoAct2 / bonus eval. Here LeRobot comes from the `molmoact2/` git submodule, wired in `pyproject.toml` via `[tool.uv.sources] lerobot = { path = "molmoact2", editable = true }`. Activate with `source .venv/bin/activate` and prefix commands with `uv run` where the README does.

Cloning must use `--recursive` (or `git submodule update --init --recursive`) because `molmoact2/` is a submodule.

## Common commands

```bash
# Port detection (writes config/so101_ports.json) — do this per machine, never hardcode ports
python scripts/detect_ports.py

# Calibrate motors once before any eval/record (use `uv run python ...` in the bonus env)
python scripts/calibrate_motor.py

# Basic eval (SmolVLA, conda env)
conda activate lerobot
bash run_eval_basic.sh

# Bonus eval (MolmoAct2, uv env)
source .venv/bin/activate
bash run_eval_bonus.sh

# Teleoperation (leader drives follower)
python scripts/teleop.py [--camera]

# Record a dataset (see docs/so101_config.md for the full push-to-hub workflow)
python scripts/teleop_record.py --camera --num-episodes 2 --dataset-repo-id <repo> --push-to-hub --resume

# Training launchers (each pins the policy/dataset/hyperparams)
bash shell/train_smolvla_HQ.sh    # conda env
bash shell/train_molmoac2.sh      # uv env
bash shell/train_act.sh           # ACT baseline; supports --num-steps / --episodes / --batch-size
```

There is no test suite; do not invent test commands.

### Env-var overrides

Eval and client/server shells read overridable env vars rather than flags. Common ones: `ROBOT_PORT`, `ROBOT_ID`, `CAMERA_INDEX`, `CAMERA_NAME`, `FPS`, `POLICY_PATH`, and for remote inference `SERVER_ADDRESS`, `HOST`, `PORT`. Example: `ROBOT_PORT=/dev/ttyACM0 bash run_eval_basic.sh`. Defaults are inconsistent across files (e.g. README/eval shells default `ROBOT_PORT=/dev/ttyACM1`, but `config/so101_ports.json` lists follower as `ttyACM0`) — always confirm the actual port.

### Serial access & Hugging Face

- Serial ports need the `dialout` group: run `newgrp dialout` once per terminal before any robot script.
- Pushing datasets/checkpoints needs `export HF_TOKEN=<token>` and membership in the `robot-learning-team43` HF org.

## Architecture notes

- **`scripts/script_utils.py` is the shared hub.** It defines every canonical path (`DEFAULT_PORTS_PATH`, follower/leader calibration dirs, home/final pose files, dataset root) and the pose-interpolation helpers (`move_robot_to_pose`, `return_to_pose_if_enabled`). Other scripts import from it rather than hardcoding paths — extend it rather than duplicating path logic.
- **`config/` holds all machine state**: `so101_ports.json` (serial ports), `calibration/robots/so_follower/follower.json` + `calibration/teleoperators/so_leader/leader.json` (per-motor calibration), `so101_home_pose.json` / `so101_final_pose.json`. The SO-101 gripper (motor id 6) uses `RANGE_0_100` normalization keyed off each side's `range_min`/`range_max` — a too-low follower `range_min` over-drives the gripper into a stall/Overload fault.
- **Eval flow**: every `run_eval_*.sh` first runs a `goto_start_pose*.py` to move the arm to a known pose, then `exec`s `lerobot-rollout` with a JSON camera config. `POLICY_PATH` prefers a local `checkpoints/<name>` dir and falls back to the `robot-learning-team43/...` HF repo.
- **Remote inference** (when the policy is too big for the robot machine): a policy server runs on the GPU/Brev machine and a robot client runs on the robot machine, connected via `SERVER_ADDRESS`. Two transport styles exist — **RTC** (real-time chunking; SmolVLA, `shell/rtc_*`) and **async/chunked** (MolmoAct2, `scripts/molmoact_*` + `shell/molmoact_*` and `shell/async_client_molmoact2.*`). See README for the Brev port-forward recipe.
- **ACT baseline** (`ACT_model/`) is intentionally non-compliant with the course's allowed policy family (DDIM/DDPM/Flow Matching) and exists only to validate the data/hardware pipeline; runs write to `ACT_model/outputs/act_<timestamp>/`.

## Coding guidelines (from AGENT.md)

`AGENT.md` holds the team's behavioral rules — follow them; the essentials:

- **Think before coding.** State assumptions explicitly; if multiple interpretations exist, surface them instead of silently picking one. If something is unclear, stop and ask. If a simpler approach exists, say so.
- **Simplicity first.** Write the minimum code that solves the problem — no speculative features, abstractions for single-use code, unrequested configurability, or error handling for impossible cases. If 200 lines could be 50, rewrite.
- **Surgical changes.** Touch only what the task requires. Don't "improve", refactor, or reformat adjacent code, and match existing style. Remove only the imports/vars/functions *your* change orphaned; flag pre-existing dead code rather than deleting it. Every changed line should trace to the request.
- **Goal-driven execution.** Turn tasks into verifiable success criteria and loop until they're met (e.g. "fix the bug" → write a reproducing check, then make it pass).
