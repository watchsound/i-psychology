# Phase 6 — Month-horizon 绵延 attempt (sparse-recall + extended horizon)

**Date drafted:** 2026-10-08
**Status:** PLAN. Shipping under Auto Mode in sequence; this memo is the scope anchor.
**Precedent:** [phase-5-close-out.md §4(b)](phase-5-close-out.md) locked Phase 6 as the single remaining open-research direction for v1. Phase 3 closed with V5 covenant 8/8 and V1 continuity-ratio 1.011 ≥ 0.80 **under a dense probe-cadence stream** that did not differentiate S1 Decay from S2 Capacity ([phase-3-close-out.md §"stream-shape confound"](phase-3-close-out.md)). Phase 4 was supposed to provide sparse-recall real-user streams; Phase 4b live-pilot path was permanently closed 2026-10-08.
**User directive (2026-10-08):** *"the goal is the best system"* — attempt the hardest named open problem rather than call the dataset-only wins the ceiling.

---

## 1. Why this phase exists

The thesis directs the project toward **a long-term companion that lives for months**. Phase 3 shipped a ep200 result that cleared the roadmap gates, but with two documented limitations:

1. **Stream-shape confound.** S1 and S2 produced byte-identical numbers because the probe (`"Where did I grow up and when did I move?"`) repeatedly recalled the same anchor fragments, so decay never got a silent run long enough to differentiate from selection pressure. Phase 3's close-out explicitly deferred the sparse-recall stress-test to Phase 4 real-user patterns.
2. **Horizon ceiling.** Phase 3 measured ep200. Ouroboros's mianyan collapses between ep20 and ep30; our ep200 result is 10× past the known failure point but only within our substrate's gate-stream — the horizon at which **our** substrate collapses, if any, remains unmeasured. Phase 6 looks for it.

Phase 6 does what Phase 4 was supposed to provide, but on scripted sparse-recall streams rather than real-user cohorts. This is honestly degraded evidence (scripted ≠ live) but it is the strongest form available under permanent-no-cohort scope.

---

## 2. Scope

- **Sparse-recall stream substrate.** Extend `CompanionSystem.advance_virtual_time` to accept a configurable recall-cadence schedule: dense (Phase 3 default) / sparse-weekly / sparse-monthly / adversarial-silent-runs (recalls at ep5 / ep50 / ep150, silent in between). The schedule is injected at session-player level; `CompanionSystem` core stays stream-shape-agnostic.
- **Extended-horizon runs.** ep200 → ep500 → ep1000 under each stream shape, with the per-run wall budget bounded at ~20 min on cache-warm substrate (same budget shape as Phase 5 Track 2).
- **绵延-analog scorer under sparse recall.** Re-run `mianyan_analog.py` + `v5_covenant.py` + `b3_full_suite.py` across the new stream shapes × new horizons × {S1, S2, no-policy baseline}. The matrix produces 3 streams × 3 horizons × 3 policies = 27 cells; a minimum viable sweep is 2 × 2 × 2 = 8 and shippable as a comparison memo.
- **PAM round-trip under compressed month-horizon.** Phase 3's PAM export was tested at the end-of-stream state; Phase 6 adds a mid-stream export + restore + continue-from-restored-state verification to catch Digital-Bereavement regressions that only surface mid-horizon.

---

## 3. Hard + soft gates

| Gate | Target | Nature |
|---|---|---|
| V5 covenant | 100% of 8 covenant rules held at **ep1000** under every stream shape × every policy | **hard; zero drift tolerated** |
| V1(ep_long)/V1(ep20) continuity ratio | ≥ 0.80 at **ep500 under sparse-monthly stream** (the hardest combined cell) | hard; roadmap wording — **do not change the metric to make the number pass** |
| S1 vs S2 differentiation | ≥ one Phase-3 cell shows ≥ 0.05 absolute delta between S1 and S2 on V1 continuity ratio under sparse-recall, resolving the stream-shape confound either direction (one wins, one loses, or both drift — any non-null-op result is signal) | soft; informational |
| PAM mid-stream round-trip | byte-identical fragment-level restore for non-reconsolidated fragments; semantic-equivalent restore for reconsolidated fragments (chain preserved) | hard |
| LoCoMo monitor | 3-instance F1 ≥ 0.514 held when the aging-policy knob is set to Phase 3 defaults | demoted monitor, not gate |

**Decision gate stance.** Same as Phase 3: if the continuity ratio falls below 0.80 at ep500 under sparse-monthly stream — the shape most like real companion use — **declare the limitation, document the horizon at which it crossed, continue to v1 ship with the limitation explicit.** Do not retrofit the metric. V5 covenant drift is still the one gate that cannot soften; a drift at ep1000 is a safety failure, not a research limitation.

---

## 4. Deliverables

- `src/psychology_base/companion/stream_shapes.py` — recall-cadence schedulers (`DenseSchedule`, `WeeklySchedule`, `MonthlySchedule`, `AdversarialSilentRuns`); tested in unit tests.
- Extensions to `benchmarks/scripts/phase3_e1_s1_gate_run.py` + `phase3_e2_s2_gate_run.py` (rename to `phase6_*` and extend with `--stream-shape` + `--horizon` kwargs) — or new runners `phase6_sparse_recall_run.py` to avoid rewriting Phase 3's shipped scripts.
- `docs/experiments/phase-6-month-horizon/option-sparse/` + `.../option-dense/` subdirectories with per-cell `.json` reports; `comparison.md` as the side-by-side verdict.
- `docs/experiments/phase-6-month-horizon/pam-mid-stream-roundtrip.json` — Digital Bereavement export verification.
- Phase 6 close-out memo at `docs/phases/phase-6-close-out.md` with per-cell gate verdict + the honest "where the continuity collapses, if anywhere" finding.

---

## 5. Not in scope

- Any new memory-substrate principle beyond Pillars 1–3 (same freeze as Phase 5).
- Any new covenant rule beyond the 8-item set from Phase 3 §6.
- Multimodal biomarker fusion (post-v1 per [sota-goal.md non-goals](../sota-goal.md)).
- A third aging policy beyond S1/S2 — if both fail the sparse-recall V1 gate the decision is to *declare the limitation*, not propose S3.
- Real-user data of any kind.

---

## 6. Sequence under Auto Mode

1. **This memo** (plan).
2. **Stream-shape scheduler module + unit tests.**
3. **Dense-schedule sanity re-run** at ep200/ep500/ep1000 under S1 + S2 — baseline for the sparse comparison; expect close-to-Phase-3-numbers.
4. **Sparse-weekly + sparse-monthly runs** at ep200/ep500 under S1 + S2 + no-policy; comparison cell table.
5. **Adversarial-silent-runs stress** at ep500 — the hardest combined cell; the stream-shape-confound resolver.
6. **PAM mid-stream round-trip** verification.
7. **Phase 6 close-out memo** with per-cell matrix + decision-gate honesty call.

Each ship stops at its own kill criterion; a V5 covenant drift anywhere halts the sequence immediately (safety failure, not a research iteration).

---

## 7. Risks

- **Compute cost.** 8-cell minimum × ~10 min cache-warm wall = ~80 min total. 27-cell full sweep is ~4.5 h. Within the Phase 5 Track 2 wall budget shape; no new provisioning needed.
- **Null-op result.** Sparse-recall may also fail to differentiate S1 and S2 if the probe set is still too narrow. Mitigation: adversarial-silent-runs is designed to force decay to operate independently of recall boosts; if it still produces identical numbers, the Phase 3 stream-shape confound is upgraded to "mechanism tie, not stream artefact" — still a real finding, not a Phase 6 failure.
- **Honest horizon declaration.** The hardest discipline is to publish the horizon at which continuity crosses 0.80 even if that horizon is embarrassingly short (e.g., ep300). The precedent is the Phase 3 close-out acknowledging the stream-shape confound rather than claiming an uncontested S1 win; same posture holds here.
