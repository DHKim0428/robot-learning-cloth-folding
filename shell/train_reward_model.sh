GIT_LFS_SKIP_SMUDGE=1 uv sync
# Filetered episodes
# filter [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 30, 31, 33, 36, 37]
# according to claude out of sync  [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 30, 31, 32, 33, 36, 37, 50, 58, 65, 66, 69, 72, 121]
lerobot-edit-dataset \
    --repo_id robot-learning-team43/so101_teleop_private \
    --new_repo_id robot-learning-team43/so101_teleop_private_filtered \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 30, 31, 33, 36, 37]"

# set private
hf upload robot-learning-team43/so101_teleop_private_filtered \
    /home/mira/.cache/huggingface/lerobot/robot-learning-team43/so101_teleop_private_filtered \
    --repo-type dataset \
    --private

 lerobot-train   --dataset.repo_id=robot-learning-team43/so101_teleop_private_filtered     --dataset.revision=main  --reward_model.type=sarm   --reward_model.annotation_mode=single_stage   --reward_model.image_key=observation.images.front   --reward_model.push_to_hub=false   --output_dir=outputs/train/sarm_single   --batch_size=32   --steps=5000   --wandb.enable=true   --wandb.project=folding_43
