# ETH Robot Learning Project — Remaining Clarification Questions

_Prepared on 2026-04-20 after the TAs shared the final project rules._

## Questions that still seem worth asking

### 1) Allowed camera / sensor setup
**What visual sensing setup is permitted during the final evaluation?**  
Our current setup uses a single wrist-mounted camera on the SO-101, and we are concerned that this may be insufficient for reliably perceiving the full towel state during folding. Are we allowed to use one or more external cameras during evaluation, or must the system operate strictly with the onboard single-camera setup?

### 2) Hybrid or stage-wise systems under the diffusion-policy rule
**Is it acceptable to use a hybrid or stage-wise system rather than a single end-to-end policy, as long as the final learned manipulation policy is diffusion-based?**  
For example, can we use separate modules for perception and control, or separate learned stages for corner grasping and folding?

### 3) Runtime / inference constraints
**Are there runtime or latency constraints during evaluation?**  
For example, is there a limit on inference latency, rollout duration, or available test-time compute?

### 4) Policy switching in the main evaluation
**During the main evaluation, do we need to use a single policy across all 5 runs, or can we switch between different policies or checkpoints across runs?**  
(The bonus explicitly allows policy switching, so we would like to confirm whether the main evaluation assumes a single fixed policy.)

