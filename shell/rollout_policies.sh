python scripts/goto_start_pose.py --port /dev/ttyACM0 && \
lerobot-rollout \
      --policy.path=robot-learning-team43/act_cloth_folding_05_11 \
      --robot.type=so101_follower \
      --robot.port=/dev/ttyACM0 \
      --robot.id=follower \
      --robot.calibration_dir=config/calibration/robots/so_follower \
      --robot.cameras='{"front": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' \
      --fps=30

python scripts/goto_start_pose.py --port /dev/ttyACM1 && \
lerobot-rollout \
    --policy.path=robot-learning-team43/smoll_vla_b32_80000_reward \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower \
    --robot.calibration_dir=config/calibration/robots/so_follower \
    --robot.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
    --fps=30 \
    --task="SO101 teleoperation task" \
    --inference.type=rtc \
    --inference.rtc.execution_horizon=10




# best-guess: reactive re-query + smooth interpolation + faster denoising

# HIL data collection: DAgger mode with so101 leader (ttyACM1) correcting ACT policy
python scripts/goto_start_pose.py --port /dev/ttyACM0 && \
lerobot-rollout \
      --strategy.type=dagger \
      --strategy.num_episodes=50 \
      --policy.path=robot-learning-team43/smoll_vla_b32_60000_reward \
      --robot.type=so101_follower \
      --robot.port=/dev/ttyACM1 \
      --robot.id=follower \
      --robot.calibration_dir=/home/robot-learning2/project/robot-learning-cloth-folding/config/calibration/robots/so_follower \
      --robot.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
      --teleop.type=so101_leader \
      --teleop.port=/dev/ttyACM0 \
      --teleop.id=leader \
      --teleop.calibration_dir=/home/robot-learning2/project/robot-learning-cloth-folding/config/calibration/teleoperators/so_leader \
      --dataset.repo_id=robot-learning-team43/rollout_hil-dataset \
      --inference.type=rtc \
      --inference.rtc.execution_horizon=10 \
      --dataset.single_task="SO101 teleoperation task" 

# optionla
--dataset.private=true \
# optional
--strategy.record_autonomous=true

