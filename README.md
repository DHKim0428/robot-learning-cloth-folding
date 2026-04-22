# Team 43 — ETH Robot Learning Cloth Folding

Repository for Team 43's ETH Robot Learning project on cloth folding.

## Current goal
Our near-term goal is to choose a practical first method, collect an initial dataset, and get a first training/evaluation loop running as early as possible.

## Current working assumption
A reasonable near-term workflow is:
1. clarify team ownership and project scope
2. read a small set of key papers
3. choose the first baseline / starting policy
4. prepare data collection
5. run first training
6. test, debug, and tune

## This week's priorities
- align on ambition level and ownership
- assign paper reading
- decide the first baseline candidate
- prepare environment and data collection plan
- identify the fastest path to a first end-to-end run

## Important docs
- `docs/project_info.md` — project rules summary
- `docs/so101_config.md` — SO-101 setup notes
- `docs/Robot_Learning_FS26_Brev_Instruction.pdf` — instructions for using the course computing resources
- `docs/BRAINSTORMING_NOTES_2026-04-20.md` — early personal brainstorming notes
- `docs/PROJECT_QUESTION_LIST.md` — open questions for TAs / team
- `papers/README.md` — paper list, reading plan, and assignments

## Paper reading plan
We do **not** want everyone reading completely different things with no shared context.

Recommended structure:
- **shared core paper**: everyone reads 1 common paper
- **owner paper**: each member reads 1 additional paper and reports back

Current recommendation:
- shared core: **Diffusion Policy**
- owner papers: **SmolVLA, pi0.5**

See `papers/README.md` for the current two-track reading plan and suggested assignments.

## Team workflow
- keep project notes and decisions in `docs/`
- keep paper notes and reading assignments in `papers/`
- keep large datasets, checkpoints, and logs out of git
- prefer small, concrete next steps over broad vague plans

## Initial ownership ideas
To be decided as a team, but likely buckets are:
- environment / setup
- data collection
- first baseline training
- evaluation / testing
- experiment tracking / notes

## Success for the first phase
A good first milestone is **not** a polished final method.
A good first milestone is:
- one chosen starting policy
- a small initial dataset
- one training run
- one test result
- one clear list of failure modes
