# DAgger Recording — Keyboard Controls

All DAgger shell scripts (`dagger_smolvla.sh`, `dagger_molmo.sh`) use
`python scripts/rollout_arrows.py` instead of `lerobot-rollout` directly.
This wrapper adds two extra arrow-key controls on top of the standard bindings.

---

## State machine

```
AUTONOMOUS  ──[Space]──►  PAUSED  ──[Tab]──►  CORRECTING
    ▲                        ▲                     │
    └────────[Space]──────────┘        [Tab]  save episode
                                       [→]   save episode
                                       [←]   discard episode
                                             └──────► PAUSED
```

---

## Full keyboard reference

| Key | Valid from | Effect |
|-----|-----------|--------|
| `Space` | AUTONOMOUS or PAUSED | Toggle policy on/off |
| `Tab` | PAUSED → CORRECTING | Start recording a correction |
| `Tab` | CORRECTING → PAUSED | **Stop and save** the correction episode |
| `→` right arrow | CORRECTING | **Stop and save** the correction immediately (same as Tab-stop) |
| `←` left arrow | CORRECTING | **Cancel** — discard all frames, return to PAUSED without saving |
| `Enter` | any | Push dataset to Hub on demand *(corrections-only mode)* |
| `Esc` | any | Stop the session |

---

## Corrections-only mode (`record_autonomous=false`)

Each Tab/`→` stop creates one episode. `←` throws it away cleanly.

```
1. Space          → PAUSED        (policy stops)
2. Tab            → CORRECTING    (recording starts)
3a. Tab or →      → PAUSED        (episode saved, counter +1)
3b. ←             → PAUSED        (episode buffer cleared, counter unchanged)
4. Space          → AUTONOMOUS    (policy resumes)
```

**Tip:** Use `←` whenever a correction goes wrong mid-take — you land back in
PAUSED and can Tab-start a fresh correction without the bad frames counting
towards your target episode total.

---

## Continuous mode (`record_autonomous=true`)

Autonomous and correction frames share one rolling episode buffer.

| Key | Effect |
|-----|--------|
| `←` | Exits correction phase (same as Tab-stop). Frames already recorded **stay** in the buffer — partial discard is not supported in this mode. |
| `→` | Forces an immediate episode rotation: saves the current episode (regardless of phase) and starts a fresh one. |

---

## How it works

`scripts/rollout_arrows.py` subclasses `DAggerStrategy` as
`ArrowKeyDAggerStrategy` and patches the strategy factory at import time.
No lerobot source files are modified.

- A second `pynput.Listener` runs alongside the existing one, handling only
  `Key.left` and `Key.right`.
- **Cancel (`←`)**: calls `dataset.clear_episode_buffer()`, re-enables teleop
  torque, and sets the DAgger phase to PAUSED directly — bypassing the
  transition that would otherwise trigger `save_episode()`.
- **Save (`→`)**: calls `events.request_transition("correction")`, which
  injects the normal CORRECTING→PAUSED transition so the main loop's save
  path runs unchanged (episode lock respected, `recorded` counter incremented).
