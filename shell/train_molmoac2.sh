################# docs ########
# https://github.com/allenai/lerobot/blob/molmoact2-policy/docs/source/molmoact2.mdx
################################

export WANDB_API_KEY='xx'
export HF_TOKEN='xx'

################################
# Preparing DATASET (run once)
################################
# copy dataset
python -c "from huggingface_hub import HfApi; HfApi().duplicate_repo(from_id='robot-learning-team43/so101_filtered_dohyung_HQ', to_id='robot-learning-team43/so101_filtered_dohyung_HQ_molmoact', repo_type='dataset', private=False)"

# copy version tag from source (duplicate_repo doesn't copy tags)
python -c "from huggingface_hub import HfApi; HfApi().create_tag('robot-learning-team43/so101_filtered_dohyung_HQ_molmoact', tag='v3.0', repo_type='dataset')"

# Rename task to Fold the towel diagonally twice
python3 -c "import pandas as pd; from huggingface_hub import HfApi, hf_hub_download; f = hf_hub_download('robot-learning-team43/so101_filtered_dohyung_HQ_molmoact', 'meta/tasks.parquet', repo_type='dataset'); df = pd.read_parquet(f); df.index = df.index.str.replace('SO101 teleoperation task', 'Fold the towel diagonally twice'); df.to_parquet('/tmp/tasks.parquet'); HfApi().upload_file(path_or_fileobj='/tmp/tasks.parquet', path_in_repo='meta/tasks.parquet', repo_id='robot-learning-team43/so101_filtered_dohyung_HQ_molmoact', repo_type='dataset'); print('Done:', df)"

# add quantile stats required by MolmoAct2 (run once)
python molmoact2/src/lerobot/scripts/augment_dataset_quantile_stats.py --repo-id=robot-learning-team43/so101_filtered_dohyung_HQ_molmoact

# compute RABC weights (requires a trained SARM reward model — run train_reward_model.sh first)
python scripts/compute_rabc_weights.py \
    --dataset-repo-id=robot-learning-team43/so101_filtered_dohyung_HQ_molmoact \
    --reward-model-path=outputs/train/sarm_single \
    --push-to-hub

################################
# Train
################################

lerobot-train \
    --dataset.repo_id=robot-learning-team43/so101_filtered_dohyung_HQ_molmoact \
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
    --wandb.enable=true \
    --wandb.entity=joaquin-gajardo \
    --wandb.project=folding_43 \
    --job_name=molmoact2_cloth_rabc \
    --output_dir=outputs/molmoact2_cloth_uniform \
    --steps=20000 \
    --batch_size=16 \
    --num_workers=4 \
    --log_freq=20 \
    --eval_freq=-1 \
    --save_checkpoint=true \
    --save_freq=2000
    #--sample_weighting.type=rabc \
    #--sample_weighting.kappa=0.06 \