# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

ETH Robot Learning course "Project 5: Cloth Folding" (Team 43). Goal: build a diffusion-based policy (DDIM/DDPM/Flow Matching) for vertex-to-vertex towel folding on the SO-101 arm. Demonstrations are recorded in **LeRobot dataset format v3** via teleoperation. See `docs/project_info.md` for evaluation rules and `README.md` for current milestone state.

The current branch (`shak-diff-model`) begins the **diffusion-policy implementation track**. Treat `nn-approach-eleni/` as a sibling baseline and build the new diffusion pipeline alongside it (do not modify it unless the task says so).

### Hard constraints (from `docs/project_info.md` + `docs/BRAINSTORMING_NOTES_2026-04-20.md`)
- Policy noise schedule **must** be one of: **DDIM**, **DDPM**, or **Flow Matching**. SmolVLA / π₀.₅ remain candidates because their action heads are flow-matching.
- Backbone, architecture, scratch-vs-pretrained, and post-training are unrestricted; pretrained components are allowed as long as the deployed policy stays in the diffusion family.
- Hardware: one SO-101 follower + teleop pair. Compute budget: ~**200 H100 GPU-hours** total — be deliberate about ablations.
- Towel: **20×20 cm non-white**, used across all three main milestones; main eval is on a standardized white table in the ETH HG Foyer. Bonus uses a TA-chosen towel/position.
- Eval milestones (5 attempts total, scored by furthest-reached): **grasping** → **single fold** (vertices < 2 cm apart) → **double fold**.
- Initial towel position is team-chosen in the main eval (gripper tip ≥ 6 cm from towel) — exploit this to make corner detection / grasp approach easier.
- Final course-compliant policy must be one model that completes the whole task. Do not use or propose separate grasp/fold policies or a handoff-based final method.

### Open questions (from `docs/PROJECT_QUESTION_LIST.md`) — do not assume answers
External-camera permission, runtime/latency limits, and whether a single fixed checkpoint is required across the 5 main-eval runs are still unresolved with the TAs. Avoid designs that hard-depend on any one of these resolutions.

## Diffusion policy implementation guidance

### Recommended LeRobot baseline config (from `docs/FUTURE_DIFFUSION_NOTES.md`, tracking PR #3202)
Start the first LeRobot diffusion baseline with these non-default values:
- `pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1"` (ImageNet ResNet18)
- `use_group_norm=False` — required when using pretrained ResNet weights (BatchNorm must stay intact)
- `horizon=64` (~2.13 s @ 30 fps; old default 16 ≈ 0.53 s is too short for folding)
- `n_action_steps=32` (~1.07 s)
- `use_separate_rgb_encoder_per_camera=True` — low impact with the current single-camera setup, but keep on by default for when wrist/scene cameras get added
- Load via `LeRobotDataset` / `LeRobotDatasetMetadata` (v3 format)

Caveat: the original Diffusion Policy paper found pretrained vision helped in sim but the strongest real-world results trained the encoder end-to-end. Treat the above as a strong starting baseline, not a settled answer — run the controlled comparisons listed in `FUTURE_DIFFUSION_NOTES.md` (pretrained vs scratch, long vs short horizon, frozen vs finetuned backbone).

### Dataset keys currently present
- Observations: `observation.state`, `observation.images.front`
- Action: `action`

### Sanity-check workflow before any hardware rollout
TA-mandated onboarding (`docs/project_info.md` §"Required sanity-check steps") plus diffusion-specific checks:
1. Confirm ≥ 20 demos exist with consistent motion; replay a few episodes in the matching scene to verify capture.
2. Single-episode overfit first — must drive train loss near zero and reproduce the demo on replay.
3. Inspect predicted action ranges **after unnormalization** before sending anything to the arm.
4. Dry-run / simulated rollout for action smoothness; only then a hardware rollout, grasping milestone first.

### Suggested starting references
- Original Diffusion Policy: <https://arxiv.org/pdf/2303.04137>
- Improved DiT-block Policy: <https://arxiv.org/pdf/2410.10088>
- Folding-specific ideas: <https://arxiv.org/pdf/2505.09109>
- Bi-manual cloth folding (LeRobot space): <https://huggingface.co/spaces/lerobot/robot-folding#the-bigger-picture>
- LeRobot itself ships Diffusion Policy and DiT-Policy implementations under `lerobot.policies.*` — prefer extending those over reimplementing.

## Environment

- Python is run inside the `lerobot` conda env (`conda activate lerobot`).
- The `huggingface/lerobot` repo must be cloned **next to** this repo at `../lerobot` (pinned to commit `fc6c94c82a4624bdfeffffc7a30dd00c67b2065c`) and installed with `pip install -e ../lerobot[...]`. Most scripts here import from `lerobot.*`.
- USB-serial access on the Spark machine requires `newgrp dialout` once per terminal before running robot scripts.
- Hugging Face uploads need `export HF_TOKEN=...` (also accepts `HUGGINGFACE_HUB_TOKEN`). Account must be in the `robot-learning-team43` org for the shared dataset.

## Common commands

Robot setup / data collection (full reference in `docs/so101_config.md`):

```bash
python scripts/detect_ports.py            # writes config/so101_ports.json
python scripts/setup_motor.py {leader|follower}
python scripts/calibrate_motor.py {leader|follower}
python scripts/save_home_pose.py          # writes config/so101_home_pose.json
python scripts/save_final_pose.py         # writes config/so101_final_pose.json
python scripts/teleop.py [--camera]
python scripts/replay_episode.py --episode 0
```

Recording (canonical Spark-machine command for shared team dataset):

```bash
DISPLAY=$DISPLAY python scripts/teleop_record.py \
    --camera --episode-time-sec 120 --return-move-time-sec 4 \
    --num-episodes 2 \
    --dataset-repo-id robot-learning-team43/so101_teleop_private \
    --push-to-hub --resume
```

Recording keys: `Space` start, `→` save, `←` discard, `Esc` stop session (saved episodes are kept and pushed).

Policy training/rollout (Eleni's approach, run from inside `nn-approach-eleni/`):

```bash
cd nn-approach-eleni
python train_policy.py [--episode-filter [meh]] [--output-policy policy.pt]
python rollout.py --episode 0 --checkpoint policy.pt
python annotate_episode_features.py
```

There is no test suite, linter config, or build step in this repo.

## Architecture

### Two parallel code tracks
- `scripts/` — robot I/O (calibration, teleop, recording, replay). Thin wrappers over `lerobot` that read project-local config from `config/`.
- `nn-approach-eleni/` — a non-LeRobot training/rollout pipeline that consumes the recorded LeRobot dataset directly via `datasets.load_dataset` and OpenCV. Treat it as a sibling experiment, not the canonical training entry point. Its paths (`DATASET_ROOT = ../data/so101_teleop`, `EPISODE_FILTER_PATH = ../config/episode_filter.toml`) are relative to that directory, so it must be run from inside `nn-approach-eleni/`.
- `ACT-model/`, `shell/`, `.agents/`, `.codex/` — currently empty placeholder directories.

### Config-driven scripts
All `scripts/*.py` go through `scripts/script_utils.py`, which centralizes project-local paths:
- `config/so101_ports.json` — leader/follower USB ports (machine-specific; do not hardcode `/dev/tty...`).
- `config/calibration/{robots/so_follower,teleoperators/so_leader}/{follower,leader}.json` — calibration JSONs read instead of the default HF cache path.
- `config/so101_home_pose.json`, `config/so101_final_pose.json` — joint poses used by `teleop_record.py` for between-episode reset and end-of-session move.
- `data/lerobot/<dataset-repo-id>` — default local LeRobot dataset root.

`build_robot_config` in `scripts/teleop_record.py` constructs the SO-101 follower with `OpenCVCameraConfig` only when `--camera` is set; without it, no camera stream is recorded.

### Dataset filtering
`config/episode_filter.toml` lists known-bad and "meh" episodes from the shared `robot-learning-team43/so101_teleop_private` dataset. The `--episode-filter` flag controls whether they are skipped at training time (no flag = keep all; flag alone = drop `bad`; `--episode-filter meh` = drop `bad` + `meh`). When adding training scripts, route filtering through `nn-approach-eleni/features.py:load_episode_filter` rather than re-parsing the TOML.

### `nn-approach-eleni` pipeline shape
`features.py` provides Canny-edge corner detection on the front camera, fold-target computation, and a small MLP (`build_policy_model`). Training (`train_policy.py`) precomputes per-frame towel corners from each video shard, builds inputs as `[normalized_state || corners(8) || goal(2) || phase(1)]`, and learns to regress normalized actions. Rollout (`rollout.py`) replays an episode from the dataset and feeds the same features into the policy. Inference uses ffmpeg via `video_for_opencv` to re-encode video shards at 30 fps before OpenCV reads them — ffmpeg is a hard dependency.

## Conventions and gotchas

- The `lerobot` Python package is **not** vendored — broken imports usually mean `../lerobot` is missing or on the wrong commit, not a bug here.
- `config/so101_ports.json` is intentionally tracked but is machine-specific; treat local edits as expected dirty state, don't commit them as feature changes.
- Datasets, checkpoints, and weights (`*.pt`, `*.ckpt`, `wandb/`, `outputs/`, `data/lerobot/*`, `data/so101_teleop/`) are gitignored — never add them.
- Default local dataset id `local/so101_teleop` cannot be combined with `--push-to-hub`; pushes need a real `<org>/<name>` repo id.
- `--resume` against a Hub repo id that doesn't exist yet falls back to creating a new local dataset rather than erroring out.
