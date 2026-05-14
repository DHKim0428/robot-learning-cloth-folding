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
python -c "from huggingface_hub import HfApi; HfApi().update_repo_settings('robot-learning-team43/so101_teleop_private_filtered', repo_type='dataset', private=True)"
# unset private
python -c "from huggingface_hub import HfApi; HfApi().update_repo_settings('robot-learning-team43/so101_teleop_private_filtered', repo_type='dataset', private=False)"

# clone dataset
python -c "from huggingface_hub import HfApi; HfApi().duplicate_repo(from_id='robot-learning-team43/so101_teleop_private_filtered', to_id='robot-learning-team43/so101_teleop_private_filtered_v2', repo_type='dataset',
  private=True)"

# rename dataset
● python -c "from huggingface_hub import HfApi; HfApi().move_repo(from_id='robot-learning-team43/old-name', to_id='robot-learning-team43/new-name', repo_type='dataset')"

# wandb api key

export WANDB_API_KEY='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'
export HF_TOKEN='xx'

# Run only this
sudo apt-get install -y ffmpeg

# Training docs https://huggingface.co/docs/lerobot/sarm

#Train
lerobot-train   --dataset.repo_id=robot-learning-team43/so101_teleop_private_filtered     --dataset.revision=main  --reward_model.type=sarm   --reward_model.annotation_mode=single_stage   --reward_model.image_key=observation.images.front   --reward_model.push_to_hub=false   --output_dir=outputs/train/sarm_single   --batch_size=32   --steps=5000   --wandb.enable=true   --wandb.project=folding_43

# updated weights command
python .venv/lib/python3.12/site-packages/lerobot/rewards/sarm/compute_rabc_weights.py     --dataset-repo-id robot-learning-team43/so101_teleop_private_filtered     --reward-model-path outputs/train/sarm_single/checkpoints/last/pretrained_model     --dataset-revision main     --head-mode sparse     --num-visualizations 5     --push-to-hub

