## only for training both on one machine
#
rm -rf outputs/pi05_training
lerobot-train \
    --dataset.repo_id=robot-learning-team43/filtered_relative_action \
    --dataset.revision=main \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training \
    --job_name=pi05_training \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper"]' \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --steps=30000 \
    --policy.device=cuda \
    --rename_map='{"observation.images.front": "observation.images.base_0_rgb"}' \
    --sample_weighting.type=rabc \
    --sample_weighting.kappa=0.06 \
    --wandb.project=folding_43 \
    --wandb.entity=folding_43 \
    --batch_size=32 \
    --save_checkpoint=true \
    --save_freq=5000 \
    --policy.push_to_hub=false



rm -rf outputs/molmoact2_cloth_rabc
lerobot-train \
    --dataset.repo_id=robot-learning-team43/molmoact_filtered \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --dataset.revision=main \
    --policy.type=molmoact2 \
    --policy.checkpoint_path=allenai/MolmoAct2-SO100_101 \
    --policy.device=cuda \
    --policy.action_mode=both \
    --policy.enable_lora_vlm=true \
    --wandb.disable_artifact=true \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --policy.setup_type="single so100/so101 robotic arm in molmoact2" \
    --policy.control_mode="absolute joint pose" \
    --policy.image_keys='["observation.images.front"]' \
    --policy.model_dtype=bfloat16 \
    --policy.num_flow_timesteps=8 \
    --policy.gradient_checkpointing=true \
    --policy.freeze_embedding=true \
    --policy.normalize_gripper=true \
    --policy.enable_knowledge_insulation=false \
    --policy.push_to_hub=false \
    --sample_weighting.type=rabc \
    --sample_weighting.kappa=0.06 \
    --wandb.enable=true \
    --wandb.entity=folding_43 \
    --wandb.project=folding_43 \
    --job_name=molmoact2_cloth_rabc \
    --output_dir=outputs/molmoact2_cloth_rabc \
    --steps=20000 \
    --batch_size=16 \
    --num_workers=4 \
    --log_freq=20 \
    --eval_freq=-1 \
    --save_checkpoint=true \
    --save_freq=2000