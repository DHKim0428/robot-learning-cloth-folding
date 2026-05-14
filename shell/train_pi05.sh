# create dataset with relative actions

# Add relative actions
lerobot-edit-dataset \
    --repo_id robot-learning-team43/so101_teleop_private_filtered \
    --new_repo_id robot-learning-team43/filtered_relative_action \
    --operation.type recompute_stats \
    --operation.relative_action true \
    --operation.chunk_size 50 \
    --operation.relative_exclude_joints "['gripper']" \
    --push_to_hub true

# train
lerobot-train \
    --dataset.repo_id=robot-learning-team43/filtered_relative_action \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training \
    --job_name=pi05_training \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper"]' \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --steps=3000 \
    --policy.device=cuda \
    --use_rabc=true \
    --rabc_kappa=0.06 \
    --batch_size=32