export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

lerobot-train \
    --dataset.repo_id=klrshak/cloth_folding_two_cam \
    --dataset.revision=main \
    --policy.path=robot-learning-team43/smolvla_HQ \
    --policy.device=cuda \
    --policy.repo_id=klrshak/smolvla_HQ_wrist \
    --policy.private=true \
    --policy.empty_cameras=2 \
    --batch_size=64 \
    --steps=50000 \
    --output_dir=outputs/train/smolvla_HQ_wrist \
    --wandb.enable=true \
    --wandb.project=folding_43 \
    --rename_map='{"observation.images.wrist": "observation.images.camera1"}'