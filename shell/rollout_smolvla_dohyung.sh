python scripts/goto_start_pose2.py --port /dev/ttyACM1 && \
lerobot-rollout \
    --policy.path=robot-learning-team43/smolvla_HQ \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower \
    --robot.calibration_dir=config/calibration/robots/so_follower \
    --robot.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
    --fps=30 \
    --task="SO101 teleoperation task" \
    --inference.type=rtc \
    --inference.rtc.execution_horizon=10