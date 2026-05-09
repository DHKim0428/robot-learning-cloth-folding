# AGENT.md

Guidance for AI coding agents working in this repository.

## First Rule

Read `docs/` before designing or changing policy code. The docs define the course rules, evaluation setup, SO-101 workflow, open constraints, and project strategy. If behavior, structure, commands, or model assumptions change, update this file in the same change so future agents inherit the current understanding.

## Project Goal

This is Team 43's ETH Robot Learning cloth-folding project. The target is a diffusion-based policy that folds a thin 20 cm x 20 cm non-white towel vertex-to-vertex with one SO-101 follower arm.

Final-policy methods must be diffusion-policy compliant:

- DDIM
- DDPM
- Flow Matching

Non-diffusion baselines are allowed only as sanity checks or pipeline validation. Do not present ACT, plain BC, or a non-diffusion VLA action head as the final course-compliant method unless it is wrapped/replaced by an allowed diffusion-policy family.

## Evaluation Facts

Main evaluation uses 5 total attempts, scored by the best run's furthest milestone:

- Grasping: grab a towel corner and lift it.
- Single fold: one vertex-to-vertex fold forming a triangle, aligned vertices under 2 cm apart.
- Double fold: two consecutive vertex-to-vertex folds forming a triangle.

Main setup:

- Robot mounted on white tables in the ETH HG Foyer.
- Team chooses the initial towel position.
- Towel must be flat, not touching the robot, and not on the table edge.
- Gripper tip starts at least 6 cm away from the towel.
- The main towel is the same course-provided towel across all main milestones.

Bonus setup may use a different TA-selected towel and TA-selected towel position. Bonus allows policy switching.

## Strategy

The current project stance from `docs/` is:

- Validate the full data-to-deployment loop before optimizing final performance.
- Collect at least 20 consistent SO-101 teleoperation demonstrations as a sanity check.
- Replay demonstrations to verify data quality.
- Train a simple behavior cloning or ACT baseline to overfit data only as pipeline validation.
- Then shift to a compliant diffusion policy for grasping, single fold, and double fold.
- Exploit initial-state freedom in the main evaluation: choose a start pose and towel placement that make corner detection, grasp approach, and fold repeatability easier.
- Prioritize data collected on the provided towel and, if possible, some data in the HG evaluation setup.
- Use lighting/table/towel variation and augmentation for robustness and bonus generalization.

Open questions that still affect design:

- Whether external cameras are allowed during final evaluation.
- Whether hybrid or stage-wise systems are allowed if learned manipulation is diffusion-based.
- Runtime, latency, rollout-duration, and test-time compute constraints.
- Whether main-evaluation policy/checkpoint switching is allowed.

## Environment

Run Python inside the `lerobot` conda environment:

```bash
conda activate lerobot
```

On the Spark machine, before robot scripts:

```bash
cd ~/projects/robot-learning-cloth-folding
newgrp dialout
conda activate lerobot
```

The `huggingface/lerobot` repo is expected next to this repository at `../lerobot`, pinned in the README to commit `fc6c94c82a4624bdfeffffc7a30dd00c67b2065c`, and installed editable with the relevant extras.

Hugging Face upload uses `HF_TOKEN`, falling back to `HUGGINGFACE_HUB_TOKEN`. Shared-team pushes require access to the `robot-learning-team43` org.

## Repository Map

- `docs/` - authoritative project rules, SO-101 instructions, planning notes, and open questions.
- `TWO_STAGE_DIFFUSION_PLAN.md` - root-level implementation plan for a CV-supervised two-stage diffusion system with separate first-fold and second-fold policies.
- `scripts/` - SO-101 setup, calibration, teleoperation, recording, and replay scripts.
- `config/` - project-local robot ports, calibration JSON, home/final poses, and episode filtering.
- `data/lerobot/` - default local LeRobot dataset root for project scripts.
- `data/so101_teleop/` - local dataset root used by the current ACT and `nn-approach-eleni` training tracks.
- `ACT_model/` - ACT sanity baseline using LeRobot policy APIs. This is not a final allowed policy family.
- `nn-approach-eleni/` - older feature-based MLP experiment using Canny towel-corner detection and direct action regression.
- `shell/` - helper shell entry points for ACT train/rollout.
- `papers/` - paper references and notes.

Keep datasets, checkpoints, logs, generated videos, and weights out of git.

## Robot And Data Workflow

Robot scripts centralize project paths in `scripts/script_utils.py`.

Important config files:

- `config/so101_ports.json` - machine-specific leader/follower USB serial ports.
- `config/calibration/robots/so_follower/follower.json` - follower calibration.
- `config/calibration/teleoperators/so_leader/leader.json` - leader calibration.
- `config/so101_home_pose.json` - reusable home pose.
- `config/so101_two_stage_home_pose.json` - two-stage recording home pose with a useful towel camera view; generated by `scripts/capture_two_stage_home_pose.py`.
- `config/so101_final_pose.json` - final pose used at recording session exit.
- `config/episode_filter.toml` - known bad and optional meh episodes for training filters.

Do not hardcode `/dev/tty...` ports. Use:

```bash
python scripts/detect_ports.py
```

Common robot commands:

```bash
python scripts/setup_motor.py follower
python scripts/setup_motor.py leader
python scripts/calibrate_motor.py follower
python scripts/calibrate_motor.py leader
python scripts/save_home_pose.py
python scripts/save_final_pose.py
python scripts/capture_two_stage_home_pose.py --camera
python scripts/teleop.py --camera
python scripts/replay_episode.py --episode 0
```

Canonical shared-dataset recording command on Spark:

```bash
DISPLAY=$DISPLAY python scripts/teleop_record.py \
    --camera \
    --episode-time-sec 120 \
    --return-move-time-sec 4 \
    --num-episodes 2 \
    --dataset-repo-id robot-learning-team43/so101_teleop_private \
    --push-to-hub \
    --resume
```

Recording controls:

- `Space` starts the current episode.
- `Right arrow` saves the current episode.
- `Left arrow` discards the current episode.
- `Esc` stops the session; saved episodes are kept and pushed.

During paused recording, `teleop_record.py` keeps the follower tracking the leader without writing dataset frames. This is intentional and helps reset the towel/leader pose between attempts.

All course demonstrations should be stored in LeRobot dataset format v3.

Two-stage diffusion data collection uses a separate local dataset location by default:

```bash
python scripts/capture_two_stage_home_pose.py --camera
python scripts/teleop_record_two_stage.py --camera --stage 1 --num-episodes 10
python scripts/teleop_record_two_stage.py --camera --stage 2 --num-episodes 10 --resume
```

Defaults:

- Dataset repo id: `local/so101_two_stage`
- Dataset root: `data/lerobot_two_stage`
- Home pose path: `config/so101_two_stage_home_pose.json`

`teleop_record_two_stage.py` moves both follower and leader to the two-stage home pose before each episode. The leader is held at home while waiting, then released when `Space` starts recording. Normal episode termination through `Right`, `Left`, or the episode timer appends a recorded follower return-to-home segment plus a short hold at home before saving/discarding. `Esc` is a hard session abort and discards the current unsaved episode.

## Training Tracks

### ACT_model

`ACT_model/` is the current internal sanity baseline. It constructs `ACTConfig`/`ACTPolicy` from LeRobot dataset metadata, trains with TensorBoard logging, and can roll out on the SO-101.

Useful commands:

```bash
bash shell/train_act.sh
bash shell/train_act.sh --num-steps 5000
bash shell/train_act.sh --episodes 0 --num-steps 2000 --batch-size 4
bash shell/rollout_act.sh ACT_model/outputs/act_<timestamp>/policy_act.pt --dry-run
bash shell/rollout_act.sh ACT_model/outputs/act_<timestamp>/policy_act.pt
```

Defaults:

- Dataset repo id: `robot-learning-team43/so101_teleop_private`
- Dataset root: `data/so101_teleop`
- Chunk size: 100
- FPS assumption: dataset metadata, usually 30 fps
- Input: `observation.state` plus `observation.images.front`
- Output: `action`
- Vision backbone default: `resnet18`
- `use_vae`: true
- Episode filtering reuses `config/episode_filter.toml`

Use ACT to validate data, normalization, action dimensions, image pipelines, and robot rollout plumbing. Do not treat ACT as the final compliant diffusion model.

### nn-approach-eleni

This is an older feature-based MLP experiment. It:

- Loads the shared dataset via `datasets.load_dataset`.
- Reads local videos/parquet shards from `../data/so101_teleop`.
- Uses OpenCV Canny contours for towel corner detection.
- Builds inputs as normalized robot state, 4 towel corners, fold target vector, and phase.
- Regresses normalized actions with a small MLP.
- Uses ffmpeg through `video_for_opencv`; ffmpeg is a hard dependency.

Run this track from inside `nn-approach-eleni/` because several default paths are relative to that directory.

Prefer reusing `features.load_episode_filter` rather than adding another TOML parser.

## Diffusion Model Implementation Guidance

When adding the diffusion-policy branch work:

- Keep the implementation compatible with LeRobot v3 datasets and current observation/action feature names.
- Build dataset loading around `LeRobotDataset`/`LeRobotDatasetMetadata` where possible instead of custom shard parsing.
- Reuse `config/episode_filter.toml` filtering semantics:
  - no flag: keep all episodes
  - `--episode-filter`: drop `bad`
  - `--episode-filter meh`: drop `bad` and `meh`
- Preserve action/state normalization from dataset metadata or save explicit normalization stats with checkpoints.
- Save enough checkpoint metadata to reconstruct model, observation features, action features, horizon/chunk size, normalization, and camera assumptions.
- Include a dry-run rollout mode before sending actions to hardware.
- Keep inference rate, action smoothing, action clipping, home/final-pose behavior, and emergency interruption explicit in rollout code.
- Make camera assumptions configurable. Current recording uses `observation.images.front` from an OpenCV camera named `front`.
- Prefer staged validation: synthetic/smoke batch, one-episode overfit, held-out episodes, dry-run live observations, then hardware rollout.
- Any final policy should clearly state whether it is DDIM, DDPM, or Flow Matching.

Potential compliant directions from the docs:

- Diffusion Policy with a pretrained visual backbone.
- Flow Matching policy for action generation.
- LeRobot's diffusion or DiT-style policy implementations.
- Pretrained components feeding an allowed diffusion/flow action head.

## Data Quality And Filtering

`config/episode_filter.toml` is the project-wide source for episode quality filtering. Current categories:

- `bad` - always drop when filtering is enabled.
- `meh` - optionally drop for stricter training.

Before scaling training:

- Replay representative episodes.
- Verify video/action alignment.
- Check that corner/grasp/fold phases are consistent.
- Overfit a single episode.
- Inspect action ranges and normalize correctly.
- Keep a short list of failure modes in `docs/`.

## Coding Conventions

- Prefer small, concrete changes that fit the existing script style.
- Keep robot-specific path constants centralized in `scripts/script_utils.py`.
- Keep generated artifacts, local datasets, logs, outputs, and checkpoints out of git.
- Treat `config/so101_ports.json` as machine-specific even if tracked.
- Do not commit secrets or Hugging Face tokens.
- Be careful around hardware scripts: add `--dry-run` or explicit confirmation paths for new rollout behavior.
- If adding a new model family, include a README or docs note with train, overfit, dry-run, and rollout commands.

## Maintenance Rule

Update `AGENT.md` whenever you add or change:

- repository structure
- training or rollout entry points
- dataset locations or feature names
- policy architecture assumptions
- robot/camera setup
- evaluation constraints learned from TAs
- filtering semantics
- checkpoint format
- hardware safety behavior

If a detail is only experimental, label it as such. If it is a course rule or documented team decision, cite the relevant file in `docs/`.
