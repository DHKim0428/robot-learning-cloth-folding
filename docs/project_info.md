# Project 5: Cloth Folding - Final Rules

## Task
Fold a **thin 20cm x 20cm towel** (provided by the TAs) using a **diffusion policy**.

Allowed diffusion-policy families:
- DDIM
- DDPM
- Flow Matching

There are **no restrictions** on:
- backbone used for feature extraction
- policy architecture
- training from scratch vs. pretrained / out-of-the-box policy
- post-training methods
- training data source

Allowed training data sources include:
- publicly available resources
- teleoperation data
- synthetic data

Teams may also use additional resources at home for generalization, such as:
- a different towel
- vision-based data augmentation

## Main evaluation setup
- The environment will be **standardized**.
- Evaluation will use:
  - a **white surface**
  - the **same non-white towel used for home preparation**
- The three main milestones are assessed together by **running the policy five times**.
- Points are awarded based on the **single run that reaches the furthest milestone**.

### Start-of-run constraints
At the beginning of each run:
- the **team chooses the initial towel position**
- the towel must **not touch the robot**
- the towel must be **fully flat on the table**
- the towel must **not be on the edge of the table**
- the robot hand must start with the **tip of the gripper at least 6 cm away from the towel**

## Milestones (main evaluation, max 150/150)

### Eval 1 — 50 pts
**Goal:** grab a corner of the towel and lift it into the air.

- 5 tries are allowed.
- Full points are awarded if **at least one attempt** is successful.

### Eval 2 — 50 pts
**Goal:** achieve a **single vertex-to-vertex fold** to form a triangle, starting from any vertex.

Success condition:
- the aligned vertices are **less than 2 cm apart**

Scoring:
- 5 tries are allowed.
- Full points are awarded if **at least one run** is successful.

### Eval 3 — 50 pts
**Goal:** starting from a team-chosen initial towel position, achieve **two vertex-to-vertex folds** to form a triangle, starting from any vertex.

Scoring:
- 5 tries are allowed.
- Full points are awarded if **at least one run** is successful.

## Bonus: generalization (50 pts)
- Teams are allowed to **switch policies** for the bonus stage.
- 5 tries are allowed.
- For the bonus:
  - the **towel and its position are chosen by the TAs**
  - the setup is **standardized for all teams**
  - the towel remains **within the gripper's range**
  - the student chooses the **starting gripper position**, but it must remain **at least 6 cm away from the towel**

### Bonus success definition
- Success is achieving **two folds**, where only **vertex-to-vertex triangular folds** count.
- Teams may attempt the bonus even if they did **not** achieve two folds in the main evaluation.

### Bonus scoring
- If multiple teams succeed, the 50 bonus points are split **proportionally to two-fold success rates** among the winning teams.
- If **no** team achieves at least one successful two-fold run, then bonus scores are calculated from **single-fold success rates**.
- If exactly **one** team achieves two folds in the bonus, that team receives the **full 50 points**.

## Restrictions
- Modifying the **table** is not allowed.
- Modifying the **robot** is not allowed.
- Introducing **other objects that affect the evaluation** is not allowed.
