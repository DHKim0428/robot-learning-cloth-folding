# Two-Stage Diffusion Folding Plan

This note describes a staged system for the SO-101 towel-folding task. It is intentionally kept at the repository root because it is an implementation plan for the branch, while `docs/` remains the authoritative place for course rules, setup notes, and shared project context.

## Goal

Build a more robust and repeatable folding system by decomposing the task into two learned manipulation stages:

1. Stage 1 policy: pick the initial towel corner and execute the first vertex-to-vertex fold.
2. Stage 2 policy: pick the new folded-cloth corner/edge state and execute the second vertex-to-vertex fold.

A simple computer-vision supervisor decides which stage should run from the current cloth state:

- `READY_FOR_STAGE_1`
- `READY_FOR_STAGE_2`
- `TASK_FINISHED`
- `RECOVERY_OR_RESET`

This aligns with the evaluation structure, which separately rewards progress through grasping, first fold, and second fold. The system should be able to stop after a successful first fold rather than blindly attempting the second fold from a bad state.

## System Architecture

The runtime system should be a small state machine:

```text
camera observation
      |
      v
CV cloth-state supervisor
      |
      +--> READY_FOR_STAGE_1 -> diffusion_stage1_policy -> observe again
      |
      +--> READY_FOR_STAGE_2 -> diffusion_stage2_policy -> observe again
      |
      +--> TASK_FINISHED     -> stop / hold / move to final pose
      |
      +--> RECOVERY_OR_RESET -> stop safely, request reset, or run a scripted reset if allowed
```

The diffusion policies should not decide the high-level phase themselves at first. Their job is to produce low-level actions for a specific manipulation phase, while the CV supervisor owns phase selection and termination.

## Learned Policies

### Stage 1 Diffusion Policy

Objective:

- Start from the evaluation initial condition: flat towel, known approximate placement, robot at a consistent home pose.
- Grasp the intended towel corner.
- Lift and place it onto the opposite target corner to form the first triangle.

Inputs:

- `observation.state`
- `observation.images.front`
- Optional CV features later, if helpful:
  - detected towel corners
  - target corner location
  - phase/state label from the supervisor

Output:

- SO-101 action sequence using the current LeRobot `action` feature.

Initial configuration to test:

- LeRobot diffusion policy.
- ImageNet-pretrained ResNet18 visual backbone.
- `use_group_norm=False`.
- `horizon=64`.
- `n_action_steps=32`.
- Dataset format: LeRobot v3.

### Stage 2 Diffusion Policy

Objective:

- Start from a completed first-fold state.
- Grasp the correct corner or folded edge for the second fold.
- Execute the second vertex-to-vertex fold and leave the cloth in the final triangle.

Inputs and output should match Stage 1 where possible, so rollout code and normalization are reusable.

Stage 2 should be trained on demonstrations that start after the first fold is already complete. Do not rely on Stage 1 producing a perfect first fold during early data collection. First collect clean Stage 2 demonstrations from manually arranged or teleoperated first-fold states.

## CV Supervisor

The first supervisor should be deliberately simple and inspectable. It only needs to classify broad cloth states and reject obviously bad conditions.

Candidate inputs:

- Front RGB camera image.
- Known table crop / workspace mask.
- Non-white towel color segmentation, since the target towel is non-white.
- Contours, polygon approximation, convex hull, or keypoint/corner estimates.

Suggested state checks:

- `READY_FOR_STAGE_1`
  - Cloth is flat enough.
  - Four-corner or roughly square/rectangular contour is visible.
  - No fold triangle is detected yet.
  - Target grasp corner and opposite corner are visible with enough confidence.

- `READY_FOR_STAGE_2`
  - Cloth looks like a first-fold triangle or near-triangle.
  - Folded vertices are close enough for the first-fold milestone.
  - A valid second-stage grasp point and target point can be estimated.

- `TASK_FINISHED`
  - Cloth looks like the expected final folded triangle.
  - Final visible contour is stable for several frames.
  - No obvious large unfolded flap remains.

- `RECOVERY_OR_RESET`
  - Towel is out of workspace.
  - Grasp/fold confidence is low.
  - Cloth is bunched, occluded, or ambiguous.
  - Robot state is outside expected bounds.

The supervisor should use temporal smoothing before switching states. For example, require the same state to be detected for 5-10 consecutive frames before launching a policy or declaring completion.

## Data Collection Plan

Collect separate datasets or separate episode labels for the two stages.

Stage 1 demonstrations:

- Begin from the chosen evaluation initial layout.
- End immediately after the first fold has settled.
- Keep the towel placement and home pose consistent at first.
- Add moderate placement and lighting variation only after a clean overfit works.

Stage 2 demonstrations:

- Begin from a completed first-fold triangle.
- Include manually arranged first-fold states and successful Stage 1 outputs.
- End once the second fold has settled.
- Capture variations in first-fold quality, because Stage 2 must tolerate imperfect Stage 1 outcomes.

Metadata to store per episode:

- `stage=1` or `stage=2`.
- initial cloth state quality.
- final cloth state quality.
- whether the episode is clean, meh, or bad.
- any notes about corner choice, failed grasp, slip, or occlusion.

If the LeRobot dataset schema is not extended immediately, keep a sidecar TOML/JSON episode manifest in `config/` and reuse the existing `episode_filter.toml` style.

## Training Plan

Train and validate each policy independently before connecting them.

1. Stage 1 one-episode overfit.
2. Stage 1 small clean-dataset overfit.
3. Stage 1 held-out validation and dry rollout.
4. Stage 2 one-episode overfit from a clean first-fold start.
5. Stage 2 clean-dataset overfit.
6. Stage 2 validation with imperfect first-fold starts.
7. Integrated state-machine dry run.
8. Integrated hardware rollout with stop-after-stage option.

Important comparisons:

- Single end-to-end diffusion policy vs two-stage policies.
- Stage-specific policies with and without CV feature inputs.
- Pretrained ResNet18 vs randomly initialized visual encoder.
- `horizon=64`, `n_action_steps=32` vs shorter horizons if latency or smoothness is poor.

## Rollout Behavior

The initial integrated rollout should be conservative:

- Start at home pose.
- Observe and classify cloth state.
- Run only the policy selected by the supervisor.
- After each policy finishes, stop robot actions and re-observe.
- If Stage 1 succeeds, either stop for scoring or continue to Stage 2 depending on a command-line flag.
- If confidence is low, stop instead of forcing the next stage.

Useful rollout flags:

```bash
--stage auto
--stage 1
--stage 2
--stop-after-stage-1
--dry-run
--cv-debug
--max-policy-seconds
--min-cv-confidence
```

The `--stop-after-stage-1` option is important because it lets the team demonstrate a reliable first-fold milestone while Stage 2 is still improving.

## Evaluation Strategy

This staged approach should exploit the evaluation structure:

- If the system only trusts Stage 1, use it to maximize the single-fold milestone.
- Once Stage 2 is reliable, enable automatic continuation.
- If the CV supervisor detects a poor first fold, stop rather than damaging the score with a bad second-fold attempt.
- During practice runs, log the supervisor state, policy selected, confidence, and final milestone reached.

Success metrics:

- Stage 1 grasp success rate.
- Stage 1 first-fold success rate under the 2 cm vertex criterion.
- Stage 2 success rate from clean first-fold states.
- Stage 2 success rate from real Stage 1 outputs.
- Integrated best-of-5 milestone score.

## Implementation Notes

Likely new modules/scripts:

- `scripts/capture_two_stage_home_pose.py` for teleoperating to a camera-friendly home pose and saving it to `config/so101_two_stage_home_pose.json`.
- `scripts/teleop_record_two_stage.py` for stage-labeled recording into `data/lerobot_two_stage/local/so101_two_stage`; it moves follower and leader to home before each episode and appends a recorded follower return-home plus hold segment before saving.
- `scripts/cv_cloth_state.py` for segmentation, contour/keypoint extraction, and state classification.
- `scripts/rollout_two_stage.py` for the integrated state machine.
- `config/two_stage_folding.toml` for thresholds, workspace crop, policy checkpoint paths, and runtime flags.
- A training entry point or config convention that filters stage-specific episodes.

Keep the first version boring and testable:

- Use explicit thresholds before adding a learned classifier.
- Save CV debug images for failed classifications.
- Make all policy checkpoint paths configurable.
- Keep state transitions logged with timestamps.
- Preserve dry-run mode before hardware execution.

Update this file when the staged system changes, especially if new states, scripts, dataset labels, or policy interfaces are added.
