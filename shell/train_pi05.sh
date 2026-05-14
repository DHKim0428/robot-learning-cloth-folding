# create dataset with relative actions
export WANDB_API_KEY='xx'
export HF_TOKEN='xx'
# Add relative actions
lerobot-edit-dataset \
    --repo_id robot-learning-team43/so101_teleop_private_filtered \
    --new_repo_id robot-learning-team43/filtered_relative_action \
    --operation.type recompute_stats \
    --operation.relative_action true \
    --operation.chunk_size 50 \
    --operation.relative_exclude_joints "['gripper']" \
    --push_to_hub true
# add progress since it seems not to be copied
python3 -c "from huggingface_hub import HfApi, hf_hub_download; api = HfApi(); api.upload_file(path_or_fileobj=hf_hub_download('robot-learning-team43/so101_teleop_private_filtered', 'sarm_progress.parquet', repo_type='dataset'), path_in_repo='sarm_progress.parquet', repo_id='robot-learning-team43/filtered_relative_action', repo_type='dataset')"
# Rename task to Fold the towel diagonally twice
python3 -c "import pandas as pd; from huggingface_hub import HfApi, hf_hub_download; f = hf_hub_download('robot-learning-team43/filtered_relative_action', 'meta/tasks.parquet', repo_type='dataset'); df = pd.read_parquet(f); df.index = df.index.str.replace('SO101 teleoperation task', 'Fold the towel diagonally twice'); df.to_parquet('/tmp/tasks.parquet'); HfApi().upload_file(path_or_fileobj='/tmp/tasks.parquet', path_in_repo='meta/tasks.parquet', repo_id='robot-learning-team43/filtered_relative_action', repo_type='dataset'); print('Done:', df)"

# train


lerobot-train \
    --dataset.repo_id=robot-learning-team43/so101_teleop_private_filtered \
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
    --batch_size=64 \
    --policy.push_to_hub=false