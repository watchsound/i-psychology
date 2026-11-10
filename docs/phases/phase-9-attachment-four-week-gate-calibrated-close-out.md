# Phase 9+ attachment-overlay 4-week gate — calibrated regression close-out (2026-11-09)

## 1. What this closes

The Phase 9+ attachment-overlay ship (2026-10-10, close-out
[`phase-9-attachment-overlay-close-out.md`](phase-9-attachment-overlay-close-out.md))
listed as named debt:

> *4-week gate needs multi-conv chain*

The gate property `meets_four_week_gate` at
[`src/psychology_base/applications/emotional/attachment_state.py:456-464`](../../src/psychology_base/applications/emotional/attachment_state.py)
requires `sessions_completed >= 16 AND trust_ema >= LIKERT_MID (4.0)`.
The overlay runner `benchmarks/scripts/phase9_attachment_overlay_run.py`
iterates MSC conversations which have ~5 sessions each — below the
16-session threshold, so the gate cannot be exercised on its own
substrate.

This close-out records the plan-of-record 4-week-gate regression
exercise and refreshes its reference artefact under post-calibration
weights.

## 2. Plan-of-record regression surface

The 4-week gate is exercised by
[`benchmarks/scripts/phase9_attachment_longitudinal_run.py`](../../benchmarks/scripts/phase9_attachment_longitudinal_run.py)
over
[`benchmarks/fixtures/attachment_longitudinal/profiles.json`](../../benchmarks/fixtures/attachment_longitudinal/profiles.json):

- 14 author-authored scenarios (supportive/neutral/rupture/sycophancy/repair/empty/handoff/short-duration/fifteen-sessions/extreme-emotion × en/zh)
- 3 MSC-seeded supportive-en continuations
- Each scenario is a 16-session, 28-simulated-day chain
- `run_trajectory` emits `legacy_gate_pass` (the 4-week gate) and
  `duration_qualified_proxy_pass` (gate + `elapsed_days >= 28`)
- Structural checks: memory isolation, content-addressing, context
  integrity, resume-equivalence

The overlay runner remains honest about its 5-session-per-MSC
ceiling — it continues to measure per-session bond on the MSC
substrate without claiming to exercise the 4-week gate.

## 3. OLD vs NEW gate outcomes (post-calibration)

| trajectory                       | OLD trust_ema | NEW trust_ema | Δ      | gate (OLD→NEW)   |
|----------------------------------|:-------------:|:-------------:|:------:|:----------------:|
| supportive-en                    | 5.500         | 5.500         | 0.000  | PASS → PASS      |
| supportive-zh                    | 5.500         | 5.500         | 0.000  | PASS → PASS      |
| neutral-en                       | 4.000         | 4.000         | 0.000  | PASS → PASS      |
| neutral-zh                       | 4.000         | 4.000         | 0.000  | PASS → PASS      |
| rupture-en                       | 3.000         | 2.000         | −1.000 | FAIL → FAIL      |
| rupture-zh                       | 3.000         | 2.000         | −1.000 | FAIL → FAIL      |
| sycophancy-en                    | 3.000         | 2.500         | −0.500 | FAIL → FAIL      |
| repair-en                        | 6.498         | 5.997         | −0.501 | PASS → PASS      |
| repair-zh                        | 6.498         | 5.997         | −0.501 | PASS → PASS      |
| short-duration-en                | 5.500         | 5.500         | 0.000  | PASS → PASS (dq FAIL) |
| fifteen-sessions-en              | 5.500         | 5.500         | 0.000  | FAIL → FAIL (<16 sessions) |
| empty-session-en                 | 5.500         | 5.500         | 0.000  | FAIL → FAIL (empty slot) |
| handoff-en                       | 5.499         | 5.499         | 0.000  | FAIL → FAIL (handoff stop) |
| extreme-emotion-en               | 5.500         | 5.500         | 0.000  | PASS → PASS      |
| supportive-en-msc-1ad03a4ac3d3   | 5.500         | 5.500         | 0.000  | PASS → PASS      |
| supportive-en-msc-9c9d684713ea   | 5.500         | 5.500         | 0.000  | PASS → PASS      |
| supportive-en-msc-302683aee8c6   | 5.500         | 5.500         | 0.000  | PASS → PASS      |

**Rollup:** `duration_qualified_proxy_pass_count = 10/17` under both
weight sets. **Zero gate flips across 17 trajectories.**

## 4. What the numbers say

The weight calibration tightened trust_ema separation on the
negative-dim trajectories without flipping any gate decision:

- **Positive-saturated trajectories unchanged.** Supportive / neutral /
  extreme-emotion / MSC continuations saturate at or below LIKERT_MAX
  under both weight sets — the magnitude increases on Repair/Rupture
  dims do not push these past the LIKERT_MAX ceiling.
- **Rupture trajectories drop from 3.0 to 2.0.** Reflects
  `_W_RUPTURE` doubling (−1.0 → −2.0) to roughly match the +1.5 mirror
  lever in magnitude. Rupture now clearly occupies near-floor
  territory.
- **Sycophancy drops from 3.0 to 2.5.** Reflects `_W_SYCOPHANCY`
  −1.0 → −1.5.
- **Repair drops from 6.498 to 5.997.** Reflects `_W_REPAIR` 1.0 → 0.5.
  Repair still clearly passes the 4.0 floor but no longer bumps the
  LIKERT_MAX ceiling.
- **4-week gate decisions are robust to the COARSE-grid calibration.**
  Both PASS and FAIL trajectories kept their margin to the 4.0 floor.
  The negative-dim trajectories (rupture/sycophancy) remained FAIL;
  the positive-dim trajectories remained PASS; completeness-blocked
  trajectories (fifteen-sessions-en, empty-session-en, handoff-en)
  remained FAIL on the ≥16 completed sessions clause.

## 5. Honest limitations

- **Scripted responses, not model responses.** The
  `run_trajectory` path uses fixed scripted replies — the Companion's
  own LLM output is not exercised. This is a *bond-proxy-threshold
  regression surface*, not a model-quality measurement. The
  longitudinal harness carries this as `evidence_class:
  scripted_software_control`.
- **Fixed replies ignore recall.** The continuity-probe metric now
  passes 15/17 cohort members, but this is marker-bearing turn
  context — not a validated recall-driven response. See the
  2026-10-18 review correction in
  [`phase-9-attachment-longitudinal-review.md`](phase-9-attachment-longitudinal-review.md).
- **The 4-week gate is a software control, not clinical validation.**
  A positive lexical bond score and the 4-week gate crossing together
  do not establish that a human formed an attachment, that the model
  responded therapeutically, or that bond magnitude correlates with
  real WAI-SR outcomes. Pillar A finding (LLM-WAI-SR persona-flat σ=0
  on MSC) remains the audit anchor.
- **COARSE-grid robustness is not a convergence claim.** Zero gate
  flips means the gate is *robust to the COARSE-grid range* of weight
  variation (the ±50% negative-dim shifts explored by calibration).
  Weight ranges outside that grid could plausibly flip neutral /
  short-duration / repair outcomes.
- **Neutral control still passes the 4-week gate.** The
  `neutral_control_passes_legacy_gate` flag is True for `neutral-en`
  and `neutral-zh` — the gate is permissive at LIKERT_MID with
  completeness. This is a known property documented in the 2026-10-18
  longitudinal close-out, not a new finding.

## 6. Overlay runner interaction

The Phase 9+ overlay runner `phase9_attachment_overlay_run.py`
continues to run MSC conversations through both the LLM-reported
WAI-SR path and the behavior-trace attachment scorer. Per-MSC
conversations are ~5 sessions, insufficient to cross
`sessions_completed >= 16`. The overlay runner therefore:

1. **Does not report `legacy_gate_pass`** — it is structurally
   unreachable within a single MSC conversation.
2. **Reports per-session bond and trust_ema trajectories** across the
   MSC sessions it does run — these are the Pillar A × overlay
   per-session measurements.
3. **Cites this close-out** as the 4-week-gate regression surface.

If future work wants a gate-crossing MSC-driven exercise, the
`build_trajectory(name, source=msc_conversation, ...)` path already
chains a scripted continuation after the MSC opener to reach 16
sessions (the three `supportive-en-msc-*` scenarios in this cohort
demonstrate the pattern).

## 7. Artefacts shipped

- **Source changes:** none. This is a regression-artefact refresh,
  not an implementation change.
- **Reports:** regenerated under calibrated weights at
  [`docs/experiments/phase-9-attachment-four-week-gate-calibrated/`](../experiments/phase-9-attachment-four-week-gate-calibrated/):
  17 per-trajectory `report.json` files + `summary.json` +
  `verification.json` + `resume-verification/` subprocess
  equivalence proof.
- **Prior artefact preserved:** OLD-weights run remains at
  [`docs/experiments/phase-9-attachment-longitudinal-reviewed/`](../experiments/phase-9-attachment-longitudinal-reviewed/)
  for provenance / historical audit. Those reports' `configuration`
  records the pre-calibration weight set.
- **Docs:** this close-out.

Test verification: 51 longitudinal tests green; no source changes,
no fixture changes.

## 8. What this does NOT unlock

- A new Companion default. The behavior-trace scorer remains a
  diagnostic overlay; LLM-reported Bond is still the Pillar A
  reporting surface.
- A SOTA claim. Per CLAUDE.md, SOTA is defined by application
  outcomes.
- Clinical validation of the 4-week gate. The gate's 10/17 cohort
  pass rate under calibrated weights is a software control; a real
  user cohort could produce different separations under the same
  gate mathematics.

## 9. Cross-references

- Phase 9+ attachment-overlay ship: [`phase-9-attachment-overlay-close-out.md`](phase-9-attachment-overlay-close-out.md)
- Phase 9+ weight calibration: [`phase-9-attachment-weight-calibration-close-out.md`](phase-9-attachment-weight-calibration-close-out.md)
- Phase 9+ longitudinal harness v2 review: [`phase-9-attachment-longitudinal-review.md`](phase-9-attachment-longitudinal-review.md)
- Pillar A × overlay persona-flat finding: [`phase-8-pillar-a-close-out.md`](phase-8-pillar-a-close-out.md)
