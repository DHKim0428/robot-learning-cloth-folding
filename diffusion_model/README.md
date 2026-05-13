# Diffusion Policy MVP — SO-101 cloth folding

Single end-to-end diffusion-policy baseline for the course-compliant cloth
folding method. Normal training uses LeRobot's native `lerobot-train`
workflow with `--policy.type=diffusion`; the local custom trainer is kept only
for debugging.

## Layout

- `rollout.py` — safe SO-101 rollout; default is dry-run and `--execute` is required to move the robot.
- `train_custom.py` — experimental ACT-style custom trainer, kept for debugging.
- `build_policy.py` / `dataset.py` — helpers used by the custom trainer and legacy checkpoint rollout.

## Dataset access

The default dataset is:

```text
robot-learning-team43/so101_teleop_private
```

By default the scripts let LeRobot download/use the Hugging Face cache. They do
not depend on this repository's `data/` directory. If the dataset is private,
authenticate first:

```bash
hf auth login
# or export HF_TOKEN=...
```

## Train

Recommended native workflow:

```bash
conda activate lerobot
bash shell/train_diffusion.sh --steps=50000 --batch_size=8

# quick smoke / overfit checks through native LeRobot
bash shell/train_diffusion.sh --dataset.episodes='[0]' --steps=10 --batch_size=1 --policy.device=cpu
bash shell/train_diffusion.sh --dataset.episodes='[0]' --steps=5000 --batch_size=4
```

`shell/train_diffusion.sh` expands to `lerobot-train --policy.type=diffusion`
with the default HF dataset and a timestamped output directory under
`outputs/train/`. It explicitly pins the local LeRobot diffusion defaults we
validated in our environment: `n_obs_steps=2`, `horizon=64`, and
`n_action_steps=32`.

Native LeRobot training gives us the official diffusion sampler,
optimizer/scheduler presets, checkpoint format, resume support, and optional
W&B/Hub integration.

To resume a native run:

```bash
lerobot-train \
  --config_path=outputs/train/<run>/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

### Config logging

Native LeRobot saves the full run config automatically in each checkpoint:

```text
outputs/train/<run>/checkpoints/last/pretrained_model/train_config.json
```

That JSON is the source of truth for exact dataset, policy, optimizer,
scheduler, W&B, and checkpoint settings. If `--wandb.enable=true` is passed,
LeRobot's W&B integration also logs the config and training metrics to the W&B
run.

### Episode filtering

Native LeRobot does not read `config/episode_filter.toml` directly. There are
two safe options.

Option A: pass the allowed episode list explicitly:

```bash
bash shell/train_diffusion.sh --dataset.episodes='[0,2,5,6,9]'
```

Or have the wrapper compute the keep-list from `config/episode_filter.toml`:

```bash
bash shell/train_diffusion_filtered.sh bad --steps=50000 --batch_size=8
bash shell/train_diffusion_filtered.sh meh --steps=50000 --batch_size=8
```

Option B: create a filtered dataset with `lerobot-edit-dataset`, then train on
that repo id. This preserves the original dataset because the wrapper always
uses `--new_repo_id`.

```bash
# Drops episodes listed under [episodes].bad in config/episode_filter.toml
bash shell/filter_diffusion_dataset.sh bad

# Drops bad + meh episodes
bash shell/filter_diffusion_dataset.sh meh

# Preview the exact edit command without running it
DRY_RUN=true bash shell/filter_diffusion_dataset.sh bad

# Train on the local/cache filtered dataset
DATASET_REPO_ID=robot-learning-team43/so101_teleop_private_filtered_bad \
  bash shell/train_diffusion.sh --steps=50000 --batch_size=8
```

By default the filtered dataset is local/cache only. To also upload it to the
Hub, run:

```bash
PUSH_TO_HUB=true bash shell/filter_diffusion_dataset.sh bad
```

If `lerobot-edit-dataset` fails with an episode/video length mismatch, use
Option A. That error means the source dataset metadata and video frame ranges
are inconsistent enough that LeRobot's re-encoding path refuses to materialize
a new dataset, while native training with `--dataset.episodes=...` can still
avoid the bad episodes without rewriting videos.

The experimental custom trainer still supports `--episode-filter` directly:

```bash
bash shell/train_diffusion_custom.sh --episode-filter --num-steps 50000
```

### Experimental relative-action diffusion

The native diffusion wrapper remains the absolute-action baseline. For a
separate custom-only experiment, use arm-joint relative actions while keeping
the gripper absolute:

```bash
bash shell/train_diffusion_relative_custom.sh \
  --dataset-root data/so101_teleop_private \
  --num-steps 50000 \
  --batch-size 8 \
  --dataset.video_backend=pyav \
  --wandb.enable \
  --wandb.project folding_43
```

This path does not rewrite dataset files. At startup it computes action
normalization stats in relative space from the loaded dataset view, then trains
the custom diffusion policy on:

```text
arm action = absolute action - current observation.state
gripper action = original absolute gripper command
```

Relative rollout is intentionally a separate entrypoint:

```bash
bash shell/rollout_diffusion_relative.sh \
  diffusion_model/outputs/diffusion_relative_<ts>/policy_diffusion.pt
```

It fails fast if the checkpoint was not trained with `--relative-actions`, and
still requires `--execute` before sending actions to the robot.
For debugging, print the live state, absolute target action, and
`action-state` delta before executing:

```bash
bash shell/rollout_diffusion_relative.sh \
  diffusion_model/outputs/diffusion_relative_<ts>/policy_diffusion.pt \
  --dry-run \
  --debug-actions \
  --abort-on-delta 30
```

The custom trainer supports both TensorBoard and optional W&B. If `wandb` is not
installed, it prints a warning and continues with local logging only.
It defaults to `--dataset.video_backend=pyav`, matching the native CLI workaround
we use on servers where TorchCodec cannot load FFmpeg/CUDA shared libraries.
Its optimizer/scheduler path is also aligned with LeRobot diffusion presets:
Adam with DiffusionConfig betas/eps/weight decay, plus a Diffusers cosine LR
scheduler with 500 warmup steps unless overridden with `--scheduler-name` or
`--scheduler-warmup-steps`.

Custom runs write to `diffusion_model/outputs/diffusion_<timestamp>/`:

- `policy_diffusion.pt` — latest plain state dict used by rollout.
- `step_<N>/` — periodic LeRobot bundle + plain state dict.
- `tb/` — TensorBoard event files.
- `args.json`, `config.json` — run and model config.

```bash
tensorboard --logdir diffusion_model/outputs --port 6006
```

## Rollout

Dry-run is the default; it reads live observations and prints predicted actions
without moving the robot.

```bash
newgrp dialout
conda activate lerobot

bash shell/rollout_diffusion.sh diffusion_model/outputs/diffusion_<ts>/policy_diffusion.pt
bash shell/rollout_diffusion.sh outputs/train/<run>/checkpoints/last/pretrained_model
bash shell/rollout_diffusion.sh diffusion_model/outputs/diffusion_<ts>/policy_diffusion.pt --execute --max-steps 300
```

## Defaults

We keep LeRobot's diffusion defaults unless CLI flags explicitly override them.

| Knob | Default |
|---|---|
| policy | single end-to-end `DiffusionPolicy` |
| dataset | `robot-learning-team43/so101_teleop_private` from HF/cache |
| scheduler | DDPM |
| n_obs_steps | 2 |
| horizon | 64 |
| n_action_steps | 32 |
| vision backbone | resnet18 |
| resize_shape | None |
| use_group_norm | True |
| episode filter | pass `--dataset.episodes=...` for native training; `--episode-filter` only in custom trainer |

No stage-1/stage-2 split is implemented; `episode_filter.toml` is only for
quality filtering.
