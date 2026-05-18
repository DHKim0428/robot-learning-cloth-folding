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
| `Tab` | CORRECTING → PAUSED | Stop the correction and return to PAUSED |
| `→` right arrow | not CORRECTING | Save the full rollout only if it had an intervention, reset, and start the next rollout |
| `←` left arrow | any | Discard the current rollout, reset, and start the next rollout |
| `Enter` | any | Push dataset to Hub on demand |
| `Esc` | any | Stop the session |

---

## Full-episode-on-intervention mode (`record_autonomous=true`)

The rollout is buffered from the beginning. If you intervene at least once,
press `→` after the rollout finishes to save the **whole episode**. If there
was no intervention, `→` discards the rollout instead of saving repetition.

```
1. AUTONOMOUS     → whole rollout is buffered from the start
2. Space          → PAUSED        (policy stops)
3. Tab            → CORRECTING    (human takes over; intervention=True)
4. Tab            → PAUSED        (correction ends; rollout continues)
5. Space          → AUTONOMOUS    (policy resumes)
6a. →             → save full episode if any intervention occurred, reset, start next rollout
6b. ←             → discard the current rollout, reset, start next rollout
```

**Tip:** Use `←` if a rollout or correction goes bad and should not enter the
dataset at all.

---

## Corrections-only mode (`record_autonomous=false`)

Only the human-correction window is recorded. This is useful for targeted
snippets, but it does **not** preserve the full trajectory from the start.

Each Tab-start → Tab-stop window becomes one short episode.

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
