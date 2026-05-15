################# docs ########
# https://github.com/allenai/lerobot/blob/molmoact2-policy/docs/source/molmoact2.mdx
################################

export WANDB_API_KEY='xx'
export HF_TOKEN='xx'

################################
# Preparing DATASET (run once)
################################
# copy dataset
python -c "from huggingface_hub import HfApi; HfApi().duplicate_repo(from_id='robot-learning-team43/so101_teleop_private_filtered', to_id='robot-learning-team43/molmoact_filtered', repo_type='dataset', private=True)"

python -c "from huggingface_hub import HfApi; HfApi().update_repo_settings('robot-learning-team43/molmoact_filtered', repo_type='dataset', private=True)"
# unset private
python -c "from huggingface_hub import HfApi; HfApi().update_repo_settings('robot-learning-team43/molmoact_filtered', repo_type='dataset', private=False)"


# Rename task to Fold the towel diagonally twice
python3 -c "import pandas as pd; from huggingface_hub import HfApi, hf_hub_download; f = hf_hub_download('robot-learning-team43/molmoact_filtered', 'meta/tasks.parquet', repo_type='dataset'); df = pd.read_parquet(f); df.index = df.index.str.replace('SO101 teleoperation task', 'Fold the towel diagonally twice'); df.to_parquet('/tmp/tasks.parquet'); HfApi().upload_file(path_or_fileobj='/tmp/tasks.parquet', path_in_repo='meta/tasks.parquet', repo_id='robot-learning-team43/molmoact_filtered', repo_type='dataset'); print('Done:', df)"

# add quantile stats required by MolmoAct2 (run once)
python molmoact2/src/lerobot/scripts/augment_dataset_quantile_stats.py --repo-id=robot-learning-team43/molmoact_filtered

################################
# Train
################################

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