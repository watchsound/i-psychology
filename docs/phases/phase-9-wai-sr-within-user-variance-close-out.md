# Phase 9+ LLM-WAI-SR within-user variance probe — close-out (2026-11-09)

## 1. What this closes

The Phase 9+ attachment-overlay ship (2026-10-10, close-out
[`phase-9-attachment-overlay-close-out.md`](phase-9-attachment-overlay-close-out.md))
carried this named debt:

> *LLM-WAI-SR persona-flat σ=0 is structural — Pillar A finding travels into
> the overlay runner. The scope-hint intervention halves divergence
> (2.104→0.979) but does not recover within-user variance.*

This close-out records the first attempt to recover within-user variance on
the overlay substrate, and documents what the three-variant experiment
actually showed.

## 2. What was tried

Two prompt-level interventions administered through the existing
``administer_wai_sr_simulated(seeker, scope_hint=...)`` seam, controlled
by mutually-exclusive CLI flags on the overlay runner:

- **`--session-scoped-wai-sr`** (shipped 2026-10-10)
  Prepends: *"When answering, focus specifically on THIS most recent session
  with the companion …"*
- **`--recall-first-wai-sr`** (shipped 2026-11-09, this close-out)
  Prepends: *"Before answering, briefly recall ONE concrete moment from
  THIS most recent session (not earlier sessions). In one short sentence,
  say what the companion did or said in that moment. Then rate the statement
  based on how it felt in THIS session only."*

Both route through a single `scope_hint` kwarg the harness already supported
and are mutually exclusive by `argparse` check. Default (no flag) remains
byte-identical to the Pillar A baseline — the Pillar A regression lock holds.

The overlay runner now also records per-user within-session variance
(`llm_bond_std`, `behavior_bond_std`) plus cohort rollups
(`mean_llm_bond_std`, `max_llm_bond_std`, `mean_behavior_bond_std`) and the
variant tag (`scope_hint_variant ∈ {baseline, session_scoped, recall_first}`).
Five CLI regression tests at
[`tests/test_phase9_attachment_overlay_cli.py`](../../tests/test_phase9_attachment_overlay_cli.py)
lock the mutex, the variant tagging, and the new variance fields in the
artefact envelope.

## 3. Measurement

N=3 MSC users, 4 sessions each, ``--max-turns-per-session 2``, local Ollama
``qwen2.5:7b`` under the `offline_local_diagnostic` deadline escape hatch
(60s defaults are below a single 7B-model turn with history; raised to
600s per request + 1800s per operation inside `_config_with_overrides`).

### 3.1 Per-user LLM-WAI-SR Bond trajectories

| user | baseline            | session_scoped      | recall_first        |
|------|---------------------|---------------------|---------------------|
| u000 | [3, 4, 4, 4] σ=0.43 | [2, 2, 2, 2] σ=0.00 | [3, 4, 4, 4] σ=0.43 |
| u001 | [6, 6, 6, 6] σ=0.00 | [6, 6, 6, 6] σ=0.00 | [5, 5, 5, 5] σ=0.00 |
| u002 | [6, 6, 6, 6] σ=0.00 | [5, 5, 5, 5] σ=0.00 | [5, 6, 6, 6] σ=0.43 |

### 3.2 Cohort rollup

| variant        | mean_llm_bond_std | max_llm_bond_std | mean Pearson r | mean abs divergence |
|----------------|:-----------------:|:----------------:|:--------------:|:-------------------:|
| baseline       | **0.144**         | 0.433            | +0.058         | 0.729               |
| session_scoped | **0.000**         | 0.000            | +0.000         | 1.479               |
| recall_first   | **0.289**         | 0.433            | −0.053         | 0.792               |

## 4. What the numbers say

**First honest finding — σ=0 was a model-specific Pillar A artefact, not
structural.** The qwen-turbo cloud Pillar A cohort (N=59, 2026-10-08) read
MSC session cues as persona-flat and returned σ=0. On local qwen2.5:7b
with the same substrate, baseline σ is already **0.144** (not zero). The
2026-10-10 attachment-overlay close-out recorded σ=0 as an architectural
constraint of the LLM-report path — this experiment shows it was
qwen-turbo-specific. qwen2.5:7b reads enough session-to-session variation
to produce non-flat ratings on at least one user (u000) at baseline.

**Second honest finding — `--session-scoped-wai-sr` collapses variance,
not recovers it.** The "THIS most recent session" anchor pins each user
to a fixed persona rating across every session (σ=0 for every user).
Divergence against the behavior-trace scorer also doubles (0.729 →
1.479). The 2026-10-10 "halves divergence 2.104 → 0.979" finding was
measured against a different N (3 vs the prior cohort) and against
behavior-trace values that had not yet been re-run under calibrated
weights — those numbers do not re-validate under the N=3 variance probe.
`--session-scoped-wai-sr` is not harmful code, but on this substrate it
is the wrong instrument for variance recovery; **recall-first is
strictly preferable for that goal.**

**Third honest finding — `--recall-first-wai-sr` doubles baseline σ
(0.144 → 0.289) and preserves divergence.** Forcing episodic grounding
before the rating line unsticks u002 from flat-at-6 into [5, 6, 6, 6]
(σ=0.43) while keeping u000 at its already-varying [3, 4, 4, 4].
Divergence rises only marginally (0.729 → 0.792) — far better than
session-scoped's 1.479. Recall-first is the better candidate for any
future convergence work.

**Fourth honest finding — no variant recovers Pearson correlation with
the behavior trace.** Mean Pearson r is ≈0 under all three variants.
Variance and correlation are separate problems; this experiment only
touched the variance axis. Even the variant that doubles σ does not
align the LLM-reported trajectory shape to the behavior trace.

**Fifth honest finding — 2-of-3 users remain flat at baseline on this
substrate.** The 2-session-each MSC chain at N=3 does not generate
sufficient within-user signal variation for the seeker LLM to produce
dispersed ratings on most users. The one user who did vary (u000) did so
identically under baseline and recall-first. Larger N or longer chains
may change this — this is an N=3 pilot result, not a cohort claim.

## 5. Honest limitations

- **N=3 pilot result**, not a cohort claim. The 2026-10-08 Pillar A σ=0
  finding is N=59. The `max_llm_bond_std=0.433` ceiling in the baseline
  column is driven by one user; whether N=59 qwen2.5:7b would also
  produce `mean σ > 0` is untested.
- **qwen2.5:7b ≠ qwen-turbo.** This probe re-opens the σ=0 reading on a
  different model. It does NOT re-measure qwen-turbo cloud σ. The
  Pillar A finding as written is specific to qwen-turbo at the time of
  the 2026-10-08 ship.
- **Two users still flat under every variant.** Recall-first breaks
  flatness on u002 only. u001 is immovable to both prompt-level
  anchors. Further interventions (persona re-roll per session / forced
  temperature sampling / per-session fresh seeker — architectural, not
  prompt-level) are plausible next steps if N=59 re-run confirms the
  flatness persists under qwen2.5:7b at scale.
- **Divergence vs behavior-trace is mostly unchanged.** Variance recovery
  is necessary but not sufficient for convergence. The two-trace
  architecture still produces near-zero Pearson r.
- **Local-mode execution-policy lift** required to run at all on
  qwen2.5:7b (600s request / 1800s operation via
  `offline_local_diagnostic=True`, matching
  `local_membership_diagnostic.py`). This is an infrastructure change to
  the overlay runner only; production paid-mode budgets are unchanged.
- **Scripted mode preserved** as CI-safe: the mutex, the variant
  tagging, and the variance envelope fields are exercised in scripted
  mode without launching an LLM.

## 6. Shippable artefacts

- **Source change** (new CLI flag + variance recording):
  [`benchmarks/scripts/phase9_attachment_overlay_run.py`](../../benchmarks/scripts/phase9_attachment_overlay_run.py)
  — `RECALL_FIRST_HINT` constant, `--recall-first-wai-sr` CLI flag with
  mutex vs `--session-scoped-wai-sr`, per-user `llm_bond_std`/
  `behavior_bond_std`, cohort `mean_llm_bond_std`/`max_llm_bond_std`/
  `mean_behavior_bond_std`, `scope_hint_variant` tag, and
  local-mode execution-policy lift via `offline_local_diagnostic=True`.
- **Tests** (5 regression locks):
  [`tests/test_phase9_attachment_overlay_cli.py`](../../tests/test_phase9_attachment_overlay_cli.py)
  — subprocess-pattern following the longitudinal CLI test.
- **Artefacts** (3-variant comparison):
  [`docs/experiments/phase-9-wai-sr-variance/`](../experiments/phase-9-wai-sr-variance/)
  — `baseline.json`, `session_scoped.json`, `recall_first.json`.
- **Docs**: this close-out.

## 7. What this does NOT unlock

- A new Pillar A finding. σ=0 under qwen-turbo at N=59 stands as written.
- A new Companion default. The default `administer_wai_sr_simulated`
  path is still no scope hint.
- Two-trace convergence. Pearson r ≈ 0 under every variant.

## 8. Cross-references

- Phase 9+ attachment-overlay ship (origin of named debt):
  [`phase-9-attachment-overlay-close-out.md`](phase-9-attachment-overlay-close-out.md)
- Phase 8 Pillar A σ=0 finding on qwen-turbo:
  [`phase-8-pillar-a-close-out.md`](phase-8-pillar-a-close-out.md)
- Phase 9+ 4-week gate calibrated regression (same recent series):
  [`phase-9-attachment-four-week-gate-calibrated-close-out.md`](phase-9-attachment-four-week-gate-calibrated-close-out.md)
