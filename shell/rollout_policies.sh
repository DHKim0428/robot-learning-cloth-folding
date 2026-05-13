python scripts/goto_start_pose.py --port /dev/ttyACM0 && \
lerobot-rollout \
      --policy.path=robot-learning-team43/act_cloth_folding_05_11 \
      --robot.type=so101_follower \
      --robot.port=/dev/ttyACM0 \
      --robot.id=follower \
      --robot.calibration_dir=config/calibration/robots/so_follower \
      --robot.cameras='{"front": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' \
      --fps=30

lerobot-rollout \
      --policy.path=robot-learning-team43/smoll_vla_b32_60000_reward \
      --robot.type=so101_follower \
      --robot.port=/dev/ttyACM0 \
      --robot.id=follower \
      --robot.calibration_dir=config/calibration/robots/so_follower \
      --robot.cameras='{"camera1": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' \
      --fps=30 \
      --task="SO101 teleoperation task"

# shorter chunk: predicts 50 actions but only executes 25 before re-querying
python scripts/goto_start_pose.py --port /dev/ttyACM0 && \
lerobot-rollout \
      --policy.path=robot-learning-team43/smolvla-cloth-folding-100000 \
      --robot.type=so101_follower \
      --robot.port=/dev/ttyACM0 \
      --robot.id=follower \
      --robot.calibration_dir=config/calibration/robots/so_follower \
      --robot.cameras='{"front": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' \
      --fps=30 \
      --task="SO101 teleoperation task" \
      --policy.n_action_steps=40

# best-guess: reactive re-query + smooth interpolation + faster denoising

# HIL data collection: DAgger mode with so101 leader (ttyACM1) correcting ACT policy
python scripts/goto_start_pose.py --port /dev/ttyACM0 && \
lerobot-rollout \
    --strategy.type=dagger \
    --strategy.num_episodes=50 \
    --policy.path=robot-learning-team43/act_cloth_folding_05_11 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower \
    --robot.calibration_dir=config/calibration/robots/so_follower \
    --robot.cameras='{"front": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.calibration_dir=/home/mira/Projects/robot-learning-cloth-folding/config/calibration/teleoperators/so_leader \
    --dataset.repo_id=robot-learning-team43/hil-dataset \
    --dataset.single_task="Fold the T-shirt properly"
