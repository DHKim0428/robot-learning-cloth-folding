# ETH Robot Learning Project — Working Summary

_Date updated: 2026-04-20_

## Context
This summary captures the current understanding after the TAs shared the **final rules and evaluation details** for Project 5.

## Confirmed project formulation
- The task is now explicitly a **diffusion-policy project**.
- Allowed policy families:
  - **DDIM**
  - **DDPM**
  - **Flow Matching**
- There are **no restrictions** on:
  - feature backbone
  - policy architecture
  - training from scratch vs. pretrained policy
  - post-training method
- Allowed training data sources include:
  - public datasets / publicly available resources
  - teleoperation data
  - synthetic data
- This means using a pretrained policy is allowed, but the final policy must still be in the allowed diffusion-policy family.

## Hardware / resources on our side
- Available hardware:
  - **one SO-101 follower arm**
  - **teleoperation pair** for data collection
- Available compute:
  - approximately **200 H100 GPU-hours**

## Towel / environment details from the TAs
- The towel is a **thin 20 cm × 20 cm towel** provided by the course staff.
- The same towel will be used for the **three main evaluation milestones**.
- For the main evaluation:
  - the environment is **standardized**
  - the surface is **white**
  - the towel is the **same non-white towel used during home preparation**

## Main evaluation protocol
- The three milestones are assessed together by **running the policy 5 times**.
- The score is determined by the **furthest milestone reached by the best run**.
- At the start of each run:
  - the team chooses the **initial towel position**
  - the towel must **not touch the robot**
  - the towel must be **fully flat on the table**
  - the towel must **not be on the edge of the table**
  - the gripper tip must start at least **6 cm away** from the towel

## Milestones
### Eval 1 — 50 pts
- Grab a **corner** of the towel and lift it into the air.
- 5 tries are allowed.
- Full points if at least one attempt succeeds.

### Eval 2 — 50 pts
- Achieve **one vertex-to-vertex fold** to form a triangle.
- Can start from **any vertex**.
- Success requires the aligned vertices to be **< 2 cm apart**.
- 5 tries are allowed.
- Full points if at least one run succeeds.

### Eval 3 — 50 pts
- Starting from a team-chosen initial position, achieve **two vertex-to-vertex folds** to form a triangle.
- Can start from **any vertex**.
- 5 tries are allowed.
- Full points if at least one run succeeds.

## Bonus: generalization
- Teams may **switch policies** for the bonus stage.
- 5 tries are allowed.
- For the bonus:
  - the **TAs choose the towel and its position**
  - the setup is standardized across teams
  - the towel will remain within the robot's reachable range
  - the student chooses the gripper start pose, but it must remain at least **6 cm away** from the towel
- Bonus success is defined by achieving **two vertex-to-vertex triangular folds**.
- If multiple teams succeed, the 50 bonus points are split **proportionally by success rate**.
- If no team achieves a successful two-fold bonus run, bonus scoring falls back to **single-fold success rates**.
- If exactly one team achieves two folds, that team gets the **full 50 points**.

## What changed relative to our earlier assumptions
- Earlier we thought the task might not be restricted to diffusion policies. That is now **incorrect**.
- The project is now clearly restricted to **diffusion-policy methods**, though the backbone / architecture / pretraining choices remain flexible.
- Earlier we were concerned that the main evaluation might vary cloth and desk setup substantially. The new rules indicate that the **main evaluation is standardized**.
- However, **generalization still matters** for the bonus, where the TAs choose the towel and towel position.

## Strategic implications
### 1) Model choice must respect the diffusion-policy constraint
- A non-diffusion VLA fine-tuning plan is no longer sufficient on its own.
- If we use pretrained components, they should feed into an allowed **diffusion policy**.
- Practical directions now include:
  - diffusion policy with a pretrained visual backbone
  - flow-matching policy for action generation
  - diffusion-policy-style post-training on teleoperation data
- In particular, **SmolVLA** and **π₀.₅** appear to remain viable candidates because their action heads are based on **Flow Matching**, which is explicitly allowed by the TAs.

### 2) Teleoperation data remains valuable
- Since the evaluation towel is known and the main setup is standardized, targeted teleoperation data on the provided towel should be high-value.
- Additional towels / augmentations can still help with robustness and bonus generalization.

### 3) The start-state freedom is important
- Because the team can choose the initial towel position in the main evaluation, we should exploit this strategically.
- We should search for an initial placement that makes:
  - corner detection easier
  - grasp approach safer
  - the first fold more stable
  - the second fold more repeatable

## Open questions that still matter
Some earlier questions are now answered, but a few still remain important:
- What camera setup is allowed during evaluation?
- Are external cameras allowed, or only the onboard wrist camera?
- Are hybrid / stage-wise systems acceptable as long as the learned policy itself is diffusion-based where needed?
- Are there any runtime or deployment constraints during evaluation?

## Recommended current stance
- Update the method search space from “general VLA fine-tuning” to “**diffusion-policy-compliant methods with optional pretrained components**.”
- Keep teleoperation data collection as a core part of the plan.
- Prioritize clarifying the **camera/sensor setup** before finalizing the perception stack.
- Use the freedom in initial towel placement as part of the evaluation strategy.

## Suggested immediate next steps
1. Update team planning based on the **diffusion-policy-only** rule.
2. Revise the question list to remove answered items and keep only unresolved constraints.
3. Start planning data collection on the provided towel.
4. In the next discussion, split responsibilities across the 5 team members.
