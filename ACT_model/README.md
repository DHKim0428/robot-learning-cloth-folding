# ACT MVP — SO-101 cloth folding

Internal sanity baseline for the cloth-folding task. ACT (Action Chunking
with Transformers) is **not** in the allowed final-policy family
(DDIM/DDPM/Flow Matching) — this code exists to validate the data and
hardware pipeline before we commit to a course-compliant diffusion policy.

## Layout
- `build_policy.py` — constructs `ACTConfig` + `ACTPolicy` + pre/post processors from dataset metadata.
- `dataset.py` — wraps `LeRobotDataset` with action-chunking `delta_timestamps` and applies `config/episode_filter.toml`.
- `train.py` — TensorBoard-logged trainer, saves `policy_act.pt` + lerobot bundle.
- `rollout.py` — runs the trained policy on the SO-101 (with `--dry-run` to print actions instead of sending them).

## Train
```bash
conda activate lerobot
bash shell/train_act.sh                        # 100k steps, drops `bad` episodes
bash shell/train_act.sh --num-steps 5000       # smoke test
bash shell/train_act.sh --episodes 0 --num-steps 2000 --batch-size 4   # overfit one episode
```

Each run writes to `ACT_model/outputs/act_<timestamp>/`:
- `policy_act.pt` — latest plain state dict (used by rollout).
- `step_<N>/` — periodic checkpoints (lerobot bundle + state dict).
- `tb/` — TensorBoard event files.
- `args.json`, `config.json` — run config.

```bash
tensorboard --logdir ACT_model/outputs --port 6006
```

## Rollout
```bash
newgrp dialout
conda activate lerobot
# 1) verify predicted actions look sane without moving the arm:
bash shell/rollout_act.sh ACT_model/outputs/act_<ts>/policy_act.pt --dry-run
# 2) actually drive the robot:
bash shell/rollout_act.sh ACT_model/outputs/act_<ts>/policy_act.pt
```

`--dry-run` keeps the camera + observation read loop running so the
policy sees realistic inputs, but skips `robot.send_action(...)` and
prints each predicted action instead.

## Defaults
| Knob | Value |
|---|---|
| chunk size | 100 (~3.3 s @ 30 fps) |
| vision backbone | resnet18 (ImageNet) |
| use_vae | True |
| episode filter | `bad` (drops 10 known-bad episodes) |
| input | `observation.state` (6) + `observation.images.front` (480×640) |
| temporal ensembling | off |
