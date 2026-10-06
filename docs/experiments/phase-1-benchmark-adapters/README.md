# Phase 1 benchmark adapter experiments

**Plan:** [../../phases/phase-1-benchmark-adapters-plan.md](../../phases/phase-1-benchmark-adapters-plan.md)
**Status (2026-10-05):** step 1 scaffolding in progress; no option has produced measured numbers yet.

## Options under experiment

| Dir | Option | Status |
|---|---|---|
| [option-alpha/](option-alpha/) | Minimal direct — `recall.content` as answer, deterministic scoring where possible | **pending** — no numbers yet |
| [option-gamma/](option-gamma/) | Dual-mode — direct + RAG side-by-side, judge-cached | **pending** — no numbers yet |

**Option β** (RAG-only parity) was dropped at plan time (`chosen: γ dominates on same axes`). No experiment directory until a backtrack revisits that drop.

## Comparison

[comparison.md](comparison.md) is the side-by-side table the phase plan cites when recording the final choice. It stays **pending** until step 2 (EpBench direct-mode) produces the first comparable numbers.

## What goes in each option directory

Per [feedback_experiment_multiple_candidates_keep_results.md](../../../../../../.claude/projects/c--Users-nihan-Desktop-my-ai-projects-psychology-base/memory/feedback_experiment_multiple_candidates_keep_results.md):

- `README.md` — what this option is + headline numbers + verdict
- Raw per-run outputs (JSON / CSV / markdown tables)
- The script or test that produced each number
- Seeds, scenario config, SUT config, git SHA, timestamp

Dropped-option directories are never deleted; if a Phase 2 result invalidates a Phase 1 number, add an "INVALIDATED YYYY-MM-DD by <change>" line at the top of the affected README rather than silently re-running.
