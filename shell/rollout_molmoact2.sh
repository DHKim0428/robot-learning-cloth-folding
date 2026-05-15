python scripts/goto_start_pose.py --port /dev/ttyACM1 && \
lerobot-rollout \
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
    --policy.chunk_size=20 \
    --policy.n_action_steps=20 \
    --policy.num_inference_steps=4 \
    --policy.enable_inference_cuda_graph=true
