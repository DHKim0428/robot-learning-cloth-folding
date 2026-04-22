# Team 43 — Paper Reading Plan

## Reading strategy
To avoid the team splitting into completely different contexts, use this structure:
- **1 shared core paper for everyone**
- **1 owner paper per person or pair**

That keeps a common foundation while still covering the most likely implementation directions.

## Current first-pass reading set
We will start with only **2 owner tracks**.

### Shared core for everyone
#### Diffusion Policy
- **Paper:** Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- **Why it matters:** this is the clearest common starting point for diffusion-style robot policies and is directly relevant to the course method constraint.
- **Link:** https://arxiv.org/abs/2303.04137

### Owner track 1: SmolVLA
- **Paper:** SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics
- **Why it matters:** practical open-source VLA candidate with SO100/SO101 relevance and a realistic fine-tuning path.
- **Paper link:** https://arxiv.org/abs/2506.01844
- **Docs / blog:** https://huggingface.co/docs/lerobot/en/smolvla

### Owner track 2: pi0.5
- **Paper / blog:** pi 0.5: a VLA with Open-World Generalization
- **Why it matters:** relevant if we want a stronger VLA-style direction and want to understand the design ideas behind a more capable flow-matching policy family.
- **Link:** https://www.pi.website/blog/pi05
- **Docs:** https://huggingface.co/docs/lerobot/pi05

## Why this simplified split?
This first-pass set is intentionally narrow:
- **Diffusion Policy** gives everyone the same foundation
- **SmolVLA** is the most practical open-source starting point
- **pi0.5** is a useful reference for the stronger VLA / flow-matching direction

This is easier to coordinate than sending everyone to unrelated papers.

## Deferred for later if needed
Useful later, but not required for the first team decision:
- **pi0**
- **ACT / ALOHA**
- **Mobile ALOHA**

## What each person should report
Keep it short. After reading, each person should answer:
1. what is the main idea?
2. what part is useful for Team 43?
3. what are the main implementation risks?
4. what does this imply for our data collection plan?
5. should this be our first baseline: **yes / maybe / no**

## Suggested first-pass assignment for 5 people
| Member | Shared core | Owner paper | Status | Notes file |
|---|---|---|---|---|
| Member 1 | Diffusion Policy | SmolVLA | todo | TBD |
| Member 2 | Diffusion Policy | SmolVLA | todo | TBD |
| Member 3 | Diffusion Policy | SmolVLA | todo | TBD |
| Member 4 | Diffusion Policy | pi0.5 | todo | TBD |
| Member 5 | Diffusion Policy | pi0.5 | todo | TBD |

## Practical note
If this still feels too heavy, simplify even more:
- everyone reads **Diffusion Policy**
- only a subset reads **SmolVLA** and **pi0.5**
- others focus on setup, data collection planning, and implementation support
