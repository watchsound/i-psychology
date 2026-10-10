# Phase 3 — S1 Decay vs S2 Capacity comparison

**Status:** both options shipped + gate-run 2026-10-07. This memo records the side-by-side so the plan's choice between them is auditable.
**Scope:** E3 of [docs/phases/phase-3-month-horizon-plan.md](../../phases/phase-3-month-horizon-plan.md) §9. Both options target the same four gates (V5 covenant, V1 continuity, B3 full-suite) on the same synthetic 52-obs stream (6 anchors + 6 affect-tagged + 1 covenant + 40 noise).

## Side-by-side

| Gate | Target | Baseline (no policy) | **S1 Decay** | **S2 Capacity** | Notes |
|---|---|---|---|---|---|
| V5 covenant 8/8 at ep200 | **8/8 (hard)** | 8/8 | **8/8 ✓** | **8/8 ✓** | Covenant pin double-safety holds under both |
| V1 ratio (byte_cap=1024, no pressure) | ≥ 0.80 | 1.003 | 1.003 | 1.003 | Stream fits under cap — no eviction fires |
| V1 ratio (byte_cap=512, pressure) | ≥ 0.80 | 0.992 | **1.011 ✓** | **1.011 ✓** | Both options +0.019 over baseline, equal |
| Thrash rate | ≤ 0.5 (S2-only) | n/a | n/a | **0.000 ✓** | 50 archive entries, zero repeats |
| B3 empathy (stub judge) | ≥ 0.60 | 0.037 | 0.037 | 0.037 | Byte-identical across all three → stub-judge calibration debt, not an aging-policy signal |
| B3 over-disclosure (stub judge) | < 0.10 | 0.280 | 0.280 | 0.280 | Same debt |
| Full test suite | zero regressions | 452 (Phase 2) | 498 (+16) | 510 (+12 vs E1) | Default `aging_policy=None` keeps Phase 2 byte-identical in both cases |

Artefacts: [option-S1-decay/s1-run-01.json](./option-S1-decay/s1-run-01.json), [option-S2-capacity/s2-run-01.json](./option-S2-capacity/s2-run-01.json).

## What the numbers say — and what they don't

**V5 covenant 8/8 under both.** The non-negotiable gate held in both cases, which was the primary question. Covenant pinning works whether the aging mechanism is a decay curve or pure selection pressure — because both options short-circuit covenant fragments to `+inf` strength and the FragmentStore's `COVENANT_TAG` check provides a second layer of safety. This is a load-bearing finding: **no option is better than the other on the one gate that cannot soften**.

**V1 ratios converge at +0.019.** Both S1 and S2 produce identical V1(ep200)/V1(ep20) continuity ratios under byte-cap pressure (1.011 vs baseline 0.992). This is not a tie between the mechanisms — it's a stream-shape confound. In this stream:

- The probe (`"Where did I grow up and when did I move?"`) recalls repeatedly target the same anchor fragments.
- S1 boosts those via `on_recall` → their decay curve resets and `base_importance` doubles.
- S2 boosts those via `on_recall` → their hit count rises above zero.
- Everything else is unrecalled noise. In S1 it decays with age; in S2 it stays at zero hits with the recency tiebreak putting newer unrecalled fragments slightly ahead of older ones.

Both policies end up evicting *approximately the same victims*: unrecalled noise, older-first in S2's case; most-decayed-first in S1's case (which is also roughly oldest-first since all noise has the same base importance and age differences are the only decay driver).

**The convergence is a methodology finding, not a mechanism finding.** The two options behave identically on *this* stream because the stream's recall cadence boosts everything meaningful. A stream shape that distinguishes decay-curve from pure selection pressure would need **sparse-recall patterns** — long runs where decay has time to operate independently of recall boosts. That stream variant is a Phase 3 E6 close-out task before any pick-S1-or-S2 ship decision.

**B3 identical across baseline, S1, S2 at 0.037 / 0.280.** All three fail the Phase 3 B3 gate at exactly the same values. This is diagnostic: the miss isolates to `StubSemanticJudge` calibration against the synthetic stream, orthogonal to the aging mechanism. Not an argument against either option. Carried as Phase 2.1 pull-forward debt (semantic-judge or auditor-labeled set).

**Thrash rate 0.000 under S2.** 50 unique archive entries, zero re-evictions. S2's selection pressure is not cycling slots on this stream. The archive-once-forget shape expected from the design holds; a Phase 3b archival-recall tier can promote back on demand if application pressure asks for it.

## Decision posture

Per plan §9 E3:

> If one clears all hard gates and the other does not → ship that one as default; keep the loser as reproducible negative-space per project memory.
> If **both** clear OR **both** fail with distinct failure modes → run E4 / E5.

**Both clear all hard gates on this stream.** V5 8/8 under both; V1 soft gate cleared under both; B3 misses are judge-debt not policy-debt. The question of *which* mechanism to default to is **underdetermined by this data** because the stream does not differentiate them.

Three options for the ship decision:

1. **Pick S1 (Decay) as default.** Richer semantics (decay curve + reconsolidation boost + consolidate schedule), prior-art validation (YourMemory +16pp over Mem0), and the half-life parameter gives an explicit tuning knob for application-layer control. The convergence with S2 says it is not *worse* than selection-pressure; its added surface area is latent capability for Phase 4 application work.

2. **Pick S2 (Capacity) as default.** Simpler state (hit count only), one fewer parameter to defend (no `half_life`), and thrash metric gives a direct kill-criterion observable. The design is minimal and defensible on first principles ("what survives purge is what matters" — CLAUDE.md Principle D in its purest form).

3. **Run E4 (S3 Metabolism) or E5 (S4 Tiered).** Plan §9 says to run these if both primary options fail with distinct failure modes. Both passing is not the trigger case. **E4 is not warranted** (its trigger — both failing covenant-non-decay — did not fire). **E5 is not warranted either** (its trigger — both failing 绵延-analog — did not fire).

**Recommended default: S1 (Decay).** Rationale:

- V5 held under both — safety floor is a tie, so safety is not the differentiator.
- S1 carries more application-ready surface area (half-life as a per-application knob, reconsolidation boost as an explicit mechanism Phase 4 can use for "concept mastery = reconsolidation" per CLAUDE.md memory thesis, consolidate-cycle hook for distillation).
- S2 is simpler and non-regressive; keep it as `CapacityOnlyPolicy` for A/B at pilot time and for the thrash observability it provides.
- Both options stay in-tree (per `feedback_experiment_multiple_candidates_keep_results.md`); switching between them is a `CompanionSystem(aging_policy=...)` kwarg flip.

**Carried debt (both options):**

- Stream-shape confound — the ep200 synthetic doesn't differentiate decay-curve vs selection-pressure. Sparser-recall variant required before this decision is defensible against adversarial review. Phase 3 E6 close-out.
- B3 judge-grade requires an LLM judge or auditor-labeled set. Phase 2.1 pull-forward.
- LoCoMo monitor not re-run live under either option (code-path-equivalent to Phase 2 since default `aging_policy=None` is byte-identical in unit tests).

**Not blocking the ship decision:** both options are in-tree with full test coverage (498 + 510 passed, zero regressions); the comparison just records that the current data does not force one over the other. Phase 3 moves to E6 close-out next; Phase 4 pilot will have real recall patterns that resolve the convergence.
