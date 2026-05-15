python scripts/goto_start_pose.py --port /dev/ttyACM1 && \
lerobot-rollout \
    --policy.path=robot-learning-team43/pi05_b32_reward_30000 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower \
    --robot.calibration_dir=config/calibration/robots/so_follower \
    --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
    --fps=30 \
    --task="Fold the towel diagonally twice" \
    --inference.type=rtc \
    --inference.rtc.execution_horizon=10 \
    --interpolation_multiplier=2