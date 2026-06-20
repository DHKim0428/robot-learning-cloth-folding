export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

lerobot-train \
    --dataset.repo_id=klrshak/cloth_folding_two_cam \
    --dataset.revision=cb5116155ade19b9a118abff292a72d415c3bed6 \
    --policy.path=robot-learning-team43/smolvla_HQ \
    --policy.device=cuda \
    --policy.repo_id=klrshak/smolvla_HQ_two_cam \
    --policy.private=true \
    --policy.empty_cameras=1 \
    --batch_size=64 \
    --steps=50000 \
    --output_dir=outputs/train/smolvla_HQ_two_cam \
    --wandb.enable=true \
    --wandb.project=folding_43 \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
