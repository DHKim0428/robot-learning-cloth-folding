# Project 5: Cloth Folding - Updated Project Info

## Project goal
Build a **diffusion-based policy** that folds a **thin towel vertex-to-vertex**.

Main milestones:
- grasping
- single fold
- double fold

## Logistics
- Join the course Slack channel **`folding_{your team ID}`**. Future clarifications will be shared there.
- If you did not receive the Slack invitation, contact the course staff / TAs.
- Before each **Thursday session**, send a short progress update in your group's Slack channel summarizing:
  - current progress
  - issues / blockers
- Teams will receive a **20 cm × 20 cm non-white towel** for the project.
- The **same towel** will be used across all three main-evaluation milestones.
- You may use other towels, home setups, and vision/data augmentation for generalization work.
- Do **not** use custom hardware beyond what was provided in the course box.
- Treat the hardware carefully; it must be returned fully functional.

## Allowed methods and data
Your policy must use one of these noise schedules:
- DDIM
- DDPM
- Flow Matching

There are **no restrictions** on:
- feature backbone
- policy architecture
- training from scratch vs. pretrained / out-of-the-box policy
- post-training methods

Training data may come from:
- publicly available data
- teleoperation
- synthetic generation

All demonstrations recorded for the course should use **LeRobot dataset format v3**:
- https://huggingface.co/docs/lerobot/lerobot-dataset-v3

## Main evaluation (150 pts)
All three milestones are evaluated together over **5 total attempts**.  
Scoring is based on the **run that reaches the furthest milestone**.

### Environment setup
- The robot will be mounted on the edge of the **white tables in the ETH HG Foyer**.
- The team may choose the initial towel position, but the towel:
  - must **not touch the robot**
  - must lie **fully flat** on the table
  - must **not** be on the edge of the table
- The tip of the gripper must start at least **6 cm away** from the towel.

### Milestones
- **Eval 1 — Grasping (50 pts):** grab a corner of the towel and lift it into the air.
- **Eval 2 — Single Fold (50 pts):** achieve one vertex-to-vertex fold to form a triangle. Success requires the aligned vertices to be **less than 2 cm apart**.
- **Eval 3 — Double Fold (50 pts):** achieve two consecutive vertex-to-vertex folds forming a triangle.

## Bonus evaluation: generalization (+50 pts)
- You may attempt the bonus even if you do **not** achieve double folding in the main evaluation.
- You may **switch policies** for the bonus task.

### Bonus environment setup
- The TAs will choose a **different towel** in color, dimensions, and/or material.
- The towel location on the table will also be chosen by the TAs.
- The setup will be standardized across teams and kept within the robot's reachable range.
- The towel will have enough contrast with the table.
- The team may choose the arm starting position, but the top of the gripper must remain at least **6 cm away** from the towel.

### Bonus trials and scoring
- Each team gets **5 tries**.
- Each try is scored using the same milestones:
  - **Grasping:** 10 pts
  - **Single Fold:** 15 pts
  - **Double Fold:** 25 pts
- The team's **final bonus score** is the **average over all 5 tries**.
- Bonus leaderboard points are assigned only to the **top 5 teams** with score **> 0**:
  - 1st: 50
  - 2nd: 40
  - 3rd: 30
  - 4th: 20
  - 5th: 10
- Teams outside the top 5 receive **no bonus points**.
- If two teams tie, both receive the higher point level and the next level is skipped.

## Restrictions
- Do **not** modify the table.
- Do **not** modify the robot.
- Do **not** introduce other objects that affect the evaluation.

## Recommended workflow
1. Set up and calibrate the robot.
2. Get teleoperation working and record a few episodes.
3. Replay a recorded episode to validate the recording pipeline and LeRobot-format data.
4. Run an overfit test on a single episode.
5. First focus on **Eval 1 (grasping)**, then iterate toward **Eval 2** and **Eval 3**.

## Required sanity-check steps from the course team
These were requested as common onboarding steps regardless of project-specific plans:

1. Collect **20 demonstrations** of the task or a simplified version of it with the SO-101 arms.
2. Keep the motion as simple and consistent as possible across demonstrations.
3. Replay some demonstrations with the objects in the **same positions/background** used during recording to verify the data was captured correctly.
4. Upload the demonstrations to **Brev**.
5. Train a **simple behavior cloning** policy to overfit those demonstrations.
6. Deploy the trained policy in a matching scene setup and check whether the robot can reproduce the recorded motion.

Notes:
- This sanity check is meant as an initial pipeline validation, not necessarily the final project method.
- If nobody in the team has a GPU available for deployment, ask other groups or contact the TAs on Slack.

## Suggested starting resources
- **LeRobot:** https://github.com/huggingface/lerobot
  - Includes Diffusion Policy and DiT-Policy implementations, plus training/deployment/visualization tooling.
- **Original Diffusion Policy:** https://arxiv.org/pdf/2303.04137
- **Improved DiT-block Policy:** https://arxiv.org/pdf/2410.10088
- **Potentially useful ideas for folding:** https://arxiv.org/pdf/2505.09109
- **Bi-manual cloth folding reference:** https://huggingface.co/spaces/lerobot/robot-folding#the-bigger-picture

## Practical tips from the TAs
- Budget compute carefully.
- Validate teleop data before training.
- Debug early with a single episode before scaling up.
- Using the **LeRobot format** can save substantial setup time.
- Collect at least some data in the **HG evaluation setup** if possible.
- Use varied lighting conditions and tables to improve robustness.
