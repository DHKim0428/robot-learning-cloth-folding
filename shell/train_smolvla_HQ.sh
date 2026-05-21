export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

lerobot-train \
    --dataset.repo_id=robot-learning-team43/so101_filtered_dohyung_HQ \
    --dataset.revision=main \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda \
    --policy.repo_id=robot-learning-team43/smolvla_HQ \
    --policy.private=true \
    --policy.empty_cameras=2 \
    --batch_size=64 \
    --steps=50000 \
    --output_dir=outputs/train/smolvla_HQ \
    --wandb.enable=true \
    --wandb.project=folding_43 \
    --rename_map='{"observation.images.front": "observation.images.camera1"}'