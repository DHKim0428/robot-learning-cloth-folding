#!/usr/bin/env bash
# DAgger (Human-in-the-Loop) rollout with SmolVLA
#
# Two recording modes — uncomment ONE block below:
#
#   CORRECTIONS-ONLY (default, recommended for targeted interventions)
#     record_autonomous=false (default)
#     Only human-correction windows are recorded; autonomous frames are discarded.
#     Each Tab-start → Tab-stop window becomes one episode.
#     Use Enter to push the dataset to the Hub on demand.
#     Good for: collecting a small set of clean correction demos.
#
#   CONTINUOUS (sentry-style, records everything)
#     record_autonomous=true
#     Both autonomous and correction frames are recorded.
#     Frames are tagged with intervention=true/false.
#     Episodes rotate automatically based on video file size.
#     Good for: building a large mixed dataset for DAgger fine-tuning.
#
# Keyboard controls:
#   Space       pause policy (AUTONOMOUS → PAUSED) or resume (PAUSED → AUTONOMOUS)
#   Tab         start correction (PAUSED → CORRECTING) or stop+save (CORRECTING → PAUSED)
#   ← left      cancel correction — discard frames, return to PAUSED (no save)
#   → right     save correction immediately — same as Tab-stop (CORRECTING → PAUSED)
#   Enter       push dataset to Hub on demand  [corrections-only mode]
#   Esc         stop session
#
# Setup (pre-write calibration to avoid interactive prompts):
#   python scripts/goto_start_pose.py --port /dev/ttyACM1

python scripts/goto_start_pose.py --port /dev/ttyACM1 && \

# =============================================================================
# MODE 1: CORRECTIONS-ONLY  (only human-correction windows are saved)
# =============================================================================
python scripts/rollout_arrows.py \
    --policy.path=robot-learning-team43/molmo_b16_lora_reward_8000 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower \
    --robot.calibration_dir=config/calibration/robots/so_follower \
    --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
    --fps=30 \
    --task="Fold the towel diagonally twice" \
    --policy.inference_action_mode=continuous \
    --policy.model_dtype=bfloat16 \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.num_inference_steps=4 \
    --policy.enable_inference_cuda_graph=true  \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader \
    --teleop.calibration_dir=config/calibration/teleoperators/so_leader \
    --strategy.type=dagger \
    --strategy.record_autonomous=false \
    --strategy.num_episodes=20 \
    --strategy.keyboard.pause_resume=space \
    --strategy.keyboard.correction=tab \
    --strategy.keyboard.upload=enter \
    --inference.type=rtc \
    --inference.rtc.execution_horizon=10 \
    --dataset.repo_id=robot-learning-team43/rollout_dagger \
    --dataset.single_task="Fold the towel diagonally twice" \
    --dataset.push_to_hub=true \
    --dataset.private=false \
    --fps=30 \
    --task="Fold the towel diagonally twice"

# =============================================================================
# MODE 2: CONTINUOUS  (all frames recorded, tagged intervention=true/false)
# =============================================================================
# python scripts/rollout_arrows.py \
#     --policy.path=robot-learning-team43/smoll_vla_b32_80000_reward \
#     --robot.type=so101_follower \
#     --robot.port=/dev/ttyACM1 \
#     --robot.id=follower \
#     --robot.calibration_dir=config/calibration/robots/so_follower \
#     --robot.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
#     --teleop.type=so101_leader \
#     --teleop.port=/dev/ttyACM0 \
#     --teleop.id=leader \
#     --teleop.calibration_dir=config/calibration/teleoperators/so_leader \
#     --strategy.type=dagger \
#     --strategy.record_autonomous=true \
#     --strategy.num_episodes=50 \
#     --strategy.upload_every_n_episodes=5 \
#     --inference.type=rtc \
#     --inference.rtc.execution_horizon=10 \
#     --dataset.repo_id=robot-learning-team43/rollout_dagger_smolvla \
#     --dataset.single_task="Fold the towel diagonally twice" \
#     --dataset.push_to_hub=true \
#     --dataset.private=false \
#     --fps=30 \
#     --task="Fold the towel diagonally twice"
