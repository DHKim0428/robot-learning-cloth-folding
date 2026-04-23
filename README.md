# Team 43 — ETH Robot Learning Cloth Folding

Repository for Team 43's ETH Robot Learning project on cloth folding.

## Current milestone
Our current milestone is to complete the course-required **sanity-check pipeline** and verify that the full data-to-deployment loop works end to end before investing heavily in the final method.

Current milestone checklist:
- collect **20 consistent demonstrations** of the task or a simplified version
- record everything in **LeRobot dataset format v3**
- replay demonstrations to verify recording correctness
- upload the dataset to **Brev**
- train a **simple behavior cloning** model to overfit the demonstrations
- deploy the trained policy in a matching scene and verify motion reproduction
- prepare a short **Thursday Slack update** with progress and blockers

## This week's priorities
- [x] finalize the recording and replay workflow
- [x] collect the first **20-demo** sanity-check dataset
- [x] confirm 1 recording is valid through replay
- [ ] quality checking of dataset
- [ ] get one simple BC overfit run working on Brev
- [ ] collect minimum 50 high-quality demos to train first VLA  
- [ ] summarize progress and blockers before the Thursday session and write to TA in Slack

## In ~2-3 weeks
- [ ] test policy in HG and collect some data there too 

## Setup

```bash
# Install conda (aarch64 / Jetson)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
# restart your shell, then:

# Clone lerobot version to install locally to make sure blackwell GPU works
git clone https://github.com/huggingface/lerobot.git
git -C lerobot checkout fc6c94c82a4624bdfeffffc7a30dd00c67b2065c

git clone https://github.com/DHKim0428/robot-learning-cloth-folding.git
cd robot-learning-cloth-folding
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install ffmpeg -c conda-forge
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -e "../lerobot[feetech,training,viz,datasets,diffusion]" \
    --extra-index-url https://download.pytorch.org/whl/cu128
pip install pynput
```

> The `lerobot` source directory must sit **next to** this repo (i.e. `../lerobot`).

## SO-101 quickstart
For day-to-day robot setup and data collection, see **[docs/so101_config.md](docs/so101_config.md)**.

To start recording immediately, jump to [Recording on the Spark machine](docs/so101_config.md#recording-on-the-spark-machine).

## Repository structure
- `docs/` — project notes, setup notes, and decisions
- `papers/` — optional paper notes and references
- `scripts/` — robot setup, teleoperation, and data collection scripts
- `config/` — local configuration templates and port files
- `data/lerobot/` — local LeRobot-format recordings (gitignored)

## Main scripts
- `scripts/detect_ports.py` — detect USB ports for leader/follower and write `config/so101_ports.json`
- `scripts/setup_motor.py` — initialise motor IDs on a new arm
- `scripts/calibrate_motor.py` — run the calibration routine for leader or follower
- `scripts/teleop.py` — bare teleoperation loop with no recording; useful for checking the robot works after calibration or practising the task without cluttering the dataset
- `scripts/teleop_record.py` — teleoperation with full dataset recording (episode management, home-pose return, HF push)

## Important docs
- `docs/project_info.md` — project rules summary
- `docs/so101_config.md` — SO-101 setup notes
- `docs/PROJECT_QUESTION_LIST.md` — open questions for TAs / team
- `papers/README.md` — paper notes and references

## Working conventions
- keep project notes and decisions in `docs/`
- keep large datasets, checkpoints, and logs out of git
- keep local LeRobot recordings under `data/lerobot/`
- keep machine-specific secrets and tokens out of git; use env files or shell exports instead
- prefer small, concrete updates over large undocumented changes
- send a short update in the team Slack channel before each Thursday session

## Success for the current milestone
A good current milestone is **not** a polished final folding policy.
A good current milestone is:
- a **20-demo** LeRobot v3 dataset collected successfully
- replay verification showing the demonstrations were recorded correctly
- one simple BC training run that clearly overfits or nearly overfits
- one deployment test in a matching scene
- one short list of concrete failure modes / blockers to discuss with the TAs

## After this milestone
Once the sanity-check pipeline works, the next step is to shift back to the actual project objective:
- focus first on **Eval 1 (grasping)**
- expand the dataset and scene diversity
- compare candidate policy families / pretrained starting points
- iterate toward **single fold** and then **double fold**
