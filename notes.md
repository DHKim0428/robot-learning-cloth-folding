Before training data generation or inference run "bash shell/lock_camera.sh"


===============================
For SmolVLA training, keep the camera name as front, because the training script uses:

  --rename_map='{"observation.images.front": "observation.images.camera1"}'

  After recording, verify by replaying an episode:

  python scripts/replay_episode.py \
    --dataset-repo-id local/so101_teleop \
    --episode 0

  For more advanced data generation, shell/dagger_smolvla.sh records human corrections while a policy runs, but I’d start with clean teleop demos from teleop_record.py.

================================
How did they reduce VLM layers?

=======================================================
Main Script
  shell/train_smolvla_HQ.sh:1

  It trains/fine-tunes from:

  --policy.path=lerobot/smolvla_base

  on dataset:

  --dataset.repo_id=robot-learning-team43/so101_filtered_dohyung_HQ

  and saves/pushes as:

  --output_dir=outputs/train/smolvla_HQ
  --policy.repo_id=robot-learning-team43/smolvla_HQ
  --policy.private=true

  Run it with:

  conda activate lerobot
  bash shell/train_smolvla_HQ.sh

  Workflow

  1. Uses the lerobot conda environment.
  2. Loads the Hugging Face dataset robot-learning-team43/so101_filtered_dohyung_HQ.
  3. Starts from the pretrained base checkpoint lerobot/smolvla_base.
  4. Renames camera key:

  --rename_map='{"observation.images.front": "observation.images.camera1"}'

  This adapts the dataset image key to what SmolVLA expects.

  5. Trains on CUDA:

  --policy.device=cuda
  --batch_size=64
  --steps=50000

  6. Logs to W&B project folding_43.
  7. Writes local checkpoints under:

  outputs/train/smolvla_HQ

  8. Pushes the trained policy to:

  robot-learning-team43/smolvla_HQ

  That is the same policy path used by run_eval_basic.sh unless a local checkpoints/smolvla_HQ exists.

  Experimental Script
  There is also shell/train_smolvla_rewards_and_normal.sh:1.

  It contains two SmolVLA training runs:

  1. RABC/reward-weighted training:

  --policy.repo_id=robot-learning-team43/smolvla_rabc
  --sample_weighting.type=rabc
  --sample_weighting.head_mode=sparse
  --sample_weighting.kappa=0.01

  2. Normal fine-tuning:

  --policy.repo_id=robot-learning-team43/smolvla_finetuned

  Both use:

  --dataset.repo_id=robot-learning-team43/so101_teleop_private_filtered
  --policy.path=lerobot/smolvla_base
  --steps=100000
  --batch_size=32

  Caveat: both commands currently use the same --output_dir=outputs/train/policy_rabc, so if run sequentially, the second run may overwrite or mix with the first output directory. I would change one output
  dir before using that script seriously.

  Related
  shell/dagger_smolvla.sh:1 is not a training script. It runs a SmolVLA policy with human correction to collect more data for later fine-tuning.

  ==============================================================