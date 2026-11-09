# Phase 9+ 绵延 decay study — close-out (2026-11-08)

## 1. What the study was

Phase 6 close-out §5 flagged carried-debt:

> *绵延 metric beyond ep1000 — the plan declared ep1000 as the horizon;
> nothing says the ratio continues to hold at ep5000+. A longer-horizon
> stress is Phase 7-or-later research (not v1 scope).*

This study closes that debt. It extends Phase 6's sparse-recall harness
past ep1000 — ep1000 / ep2000 / ep3000 / ep5000 / ep10000 — under the
same cell contract (same `_build_stream`, same `_run_cell`, same
`scheduled_v1_continuity_ratio` metric) across the same stream × policy
matrix (`{sparse-weekly, sparse-monthly, adversarial-silent-runs} ×
{baseline, S1 Decay, S2 Capacity-only}`).

Per CLAUDE.md and the Phase 6 decision stance, the deliverable is
**honest documentation of where the ratio breaks (if it breaks)**, not a
"fix" to make a failing cell pass.

## 2. Verdict

**PASS at the declared decision gate.** The V1 continuity ratio ≥ 0.80
holds across the full sweep horizon out to ep10000.

- **Min-ratio cell:** baseline / MonthlySchedule / ep5000 → **0.9894**.
- **Max ep_long at or above 0.80:** 10000 (the longest horizon tested).
- **Broken cells below 0.80:** 0 of 33.

Ouroboros's named open problem (mianyan decay past ep20) does not
reproduce on this substrate at the 500× horizon tested. Phase 6
extended it 50×; Phase 9+ extends another 10× without softening the
gate.

## 3. Numbers

Ratio per cell (schedule × policy × horizon). All cells cleared 0.80.

### Sparse-weekly

| policy   | ep1000 | ep2000 | ep3000 | ep5000 | ep10000 |
|----------|--------|--------|--------|--------|---------|
| baseline | 0.992  | 0.991  | 0.990  | 0.992  | 0.993   |
| S1       | 1.010  | 1.010  | 1.010  | 1.010  | 1.010   |
| S2       | 1.010  | 1.010  | 1.010  | 1.010  | 1.010   |

### Sparse-monthly

| policy   | ep1000 | ep2000 | ep3000 | ep5000 | ep10000 |
|----------|--------|--------|--------|--------|---------|
| baseline | 0.993  | 0.991  | 0.991  | 0.989  | 0.992   |
| S1       | 1.010  | 1.010  | 1.010  | 1.010  | 1.010   |
| S2       | 1.010  | 1.010  | 1.010  | 1.010  | 1.010   |

### Adversarial-silent-runs (ticks=(5,50,150))

| policy   | ep5000 |
|----------|--------|
| baseline | 0.994  |
| S1       | 1.010  |
| S2       | 1.010  |

## 4. Honest limitations (what the PASS does NOT claim)

The PASS is a measurement result. It is **not** a mechanism claim. Three
things remain structurally unresolved and must not be inferred from this
result:

- **S1 vs S2 Δ = 0.000** across every one of 30 (weekly + monthly) cells,
  same as Phase 6. The decay-curve shape and the capacity-only shape
  produce byte-identical ratios at every horizon. Recall-boost cadence
  drives retention under this harness; the aging policy choice does not
  show up in the ratio. The stream-shape confound Phase 3 called out is
  **not** resolved by extending the horizon — same documented limitation
  Phase 6 shipped.

- **The substrate is stable-after-observe, not actively-decaying.** Under
  `byte_cap=512`, the stream of 59 observations (6 anchors + 12 affect
  rows + 1 covenant + 40 routine updates) is purged during the initial
  `observe()` pass, not during the schedule-walked ticks between
  windows. After the first window, fragment count sits at 3 or 4 and
  stays there (`fragment_count_after` in the artefacts). The horizon
  walk fires schedule-cadenced probes but does not ingest new content.
  So "ratio ≥ 0.80 at ep10000" means *"what survived initial purge
  stays surviving under repeated recall probes for 10000 ticks"* — a
  real-but-narrow claim.

- **绵延 as application metric remains open.** The CLAUDE.md thesis is
  explicit that *"month-horizon 绵延 is a separate open research
  track"* and *"Ouroboros's mianyan decays past ep20; a long-term
  companion lives for months — honest open problem, not hand-waved."*
  This study tests the V1-continuity analog, not months-of-use
  behavioral drift. Ep10000 ≈ 208 days at 30 min/tick cadence, which is
  encouraging but does not substitute for real-user longitudinal
  evidence. Phase 8 Pillar A at N=59 and Phase 9+ attachment-overlay
  (persona-flat σ=0 on LLM-WAI-SR) already flag that the LLM-reported
  Bond doesn't reflect cross-session memory even when the substrate
  carries it. The ratio-holds result is substrate-level, not
  application-level.

## 5. What this unlocks and what it doesn't

**Unlocks:**
- The v1 roadmap can drop the "ep5000+ open" caveat from Phase 6
  carried-debt. The ratio is stable out to ep10000.
- Future Phase 9+ companion-pilot work (if/when real-user cohort funding
  arrives) can run against a substrate where the state-signature metric
  is known to be ratio-stable at least that long without re-running this
  sweep.

**Does not unlock:**
- A 绵延-based SOTA claim. The CLAUDE.md SOTA thesis is explicit:
  *"SOTA is defined by application outcomes… best in-application result
  for the two surfaces (学习陪伴 + 情感陪伴)."* Substrate-level
  measurements produce *mechanism-viability* evidence, not a SOTA claim.
- Any claim that S1 or S2 is empirically preferred. Phase 6 shipped S1
  as default on stylistic grounds (decay-curve shape nearer the
  reconstructive-memory principle); this study does not re-open that
  choice — Δ is still 0.000.
- A fix for the stream-shape confound. Stronger probes + more diverse
  content + longer stream diversity would be needed — none are in
  scope for a horizon-extension study.

## 6. Decision-gate alignment

Per the roadmap discipline (capture date + honest limitation) and
CLAUDE.md "Discovered Issues" convention:

- **Hard gate (ratio ≥ 0.80 at every cell):** PASS.
- **Soft gate (S1 vs S2 Δ ≥ 0.05):** FAIL, same as Phase 6 (expected).
- **Decision stance declared in the runner's docstring:** *"if the ratio
  falls below 0.80 at any horizon, declare the limitation."* Not
  triggered. Nothing to retrofit.

## 7. Artefacts shipped

- **Runner:** [`benchmarks/scripts/phase9_mianyan_decay_run.py`](../../benchmarks/scripts/phase9_mianyan_decay_run.py) — reuses `_run_cell` and `_run_sweep` from Phase 6 verbatim.
- **Experiment JSONs:** 4 artefacts under [`docs/experiments/phase-9-mianyan-decay/`](../experiments/phase-9-mianyan-decay/):
  - `sparse-weekly-run-01.json`
  - `sparse-monthly-run-01.json`
  - `adversarial-silent-runs-run-01.json`
  - `rollup.json`
- **Docs:** this close-out.

No source changes; no test changes; no new primitive. The sweep is pure
re-exercise of the Phase 6 cell contract over a longer horizon ladder.

## 8. Cross-references

- Phase 6 plan: [phase-6-month-horizon-plan.md](phase-6-month-horizon-plan.md)
- Phase 6 close-out: [phase-6-close-out.md](phase-6-close-out.md)
- Project SOTA thesis + 绵延 track framing: [../../CLAUDE.md](../../CLAUDE.md) §"Memory thesis"
- Ouroboros prior-art mianyan-decay-past-ep20 open problem: [../../CLAUDE.md](../../CLAUDE.md) §"Memory thesis"
