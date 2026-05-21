python scripts\goto_start_pose.py --port COM5 && ^
python -m lerobot.async_inference.robot_client ^
    --server_address=<SERVER_IP>:8080 ^
    --robot.type=so101_follower ^
    --robot.port=COM5 ^
    --robot.id=follower ^
    --robot.calibration_dir=config/calibration/robots/so_follower ^
    --robot.cameras="{\"front\": {\"type\": \"opencv\", \"index_or_path\": 0, \"width\": 640, \"height\": 480, \"fps\": 30}}" ^
    --task="Fold the towel diagonally twice" ^
    --policy_type=molmoact2 ^
    --pretrained_name_or_path=robot-learning-team43/molmoact2_HQ_extended_020000 ^
    --policy_device=cuda ^
    --actions_per_chunk=30 ^
    --chunk_size_threshold=0.5 ^
    --aggregate_fn_name=weighted_average ^
    --debug_visualize_queue_size=True ^
    --policy_overrides="{\"inference_action_mode\": \"continuous\", \"model_dtype\": \"bfloat16\", \"chunk_size\": 30, \"n_action_steps\": 30, \"num_inference_steps\": 4, \"enable_inference_cuda_graph\": true}"