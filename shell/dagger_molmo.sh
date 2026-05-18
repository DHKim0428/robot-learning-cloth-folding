#!/usr/bin/env bash
# DAgger (Human-in-the-Loop) rollout with MolmoAct2
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
#   FULL-EPISODE ON INTERVENTION
#     record_autonomous=true
#     Both autonomous and correction frames are buffered from the start.
#     Frames are tagged with intervention=true/false.
#     Press → after a rollout: if there was any intervention, save the whole
#     episode; otherwise discard it as repetition.
#     Good for: full-trajectory DAgger data without saving easy successes.
#
# Keyboard controls:
#   Space       start/resume policy from PAUSED
#   Tab         take over directly (AUTONOMOUS → CORRECTING) or stop correction
#               (CORRECTING → PAUSED)
#   ← left      discard the current rollout, reset, and wait at start pose
#   → right     save the whole rollout if it had any intervention, then reset
#               and wait at start pose
#   Enter       push dataset to Hub on demand (episodes also push immediately on save)
#   Esc         stop session
#
# Setup (pre-write calibration to avoid interactive prompts):
#   python scripts/goto_start_pose.py --port /dev/ttyACM1

python scripts/goto_start_pose.py --port /dev/ttyACM1 && \

# =============================================================================
# MODE 1: FULL-EPISODE ON INTERVENTION
# =============================================================================
python scripts/rollout_arrows.py \
    --policy.path=robot-learning-team43/molmo_b16_lora_reward_10000 \
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
    --strategy.record_autonomous=true \
    --strategy.num_episodes=20 \
    --strategy.keyboard.pause_resume=space \
    --strategy.keyboard.correction=tab \
    --strategy.keyboard.upload=enter \
    --inference.type=rtc \
    --inference.rtc.execution_horizon=10 \
    --dataset.repo_id=robot-learning-team43/rollout_molmo_dagger_full \
    --dataset.single_task="Fold the towel diagonally twice" \
    --dataset.push_to_hub=true \
    --dataset.private=false

# =============================================================================
# MODE 2: LEGACY CORRECTIONS-ONLY EXAMPLE
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
