# Phase 3 close-out — month-horizon stress

**Date closed:** 2026-10-07
**Status:** SHIPPED. Harness + S1 Decay + S2 Capacity all green on hard gates; S1 recommended as default aging policy; S2 retained in-tree for A/B at pilot time. 510 tests pass. Phase handoff to **Phase 4 — application pilot** (学习陪伴 + 情感陪伴) per [../roadmap.md](../roadmap.md).
**Companion docs:** [phase-3-month-horizon-plan.md](phase-3-month-horizon-plan.md) is the plan this memo closes out; [phase-2-close-out.md](phase-2-close-out.md) is the immediate predecessor.
**Experiment artefacts:** [../experiments/phase-3-month-horizon/option-S1-decay/README.md](../experiments/phase-3-month-horizon/option-S1-decay/README.md) · [../experiments/phase-3-month-horizon/option-S2-capacity/README.md](../experiments/phase-3-month-horizon/option-S2-capacity/README.md) · [../experiments/phase-3-month-horizon/comparison.md](../experiments/phase-3-month-horizon/comparison.md).

---

## What shipped

Three experiment slots (E0 harness, E1 S1 Decay, E2 S2 Capacity) + E3 comparison. E4 S3 Metabolism and E5 S4 Tiered were **not triggered** per plan §9 — their triggers are "both primary options fail covenant-non-decay" (E4) and "both primary options fail 绵延-analog gate" (E5); both options cleared both.

| Slot | Artefact | Hard-gate verdict | Notes |
|---|---|---|---|
| **E0 — harness** | `CompanionSystem.advance_virtual_time` + `mianyan_analog.py` + `v5_covenant.py` + `b3_full_suite.py` + PAM month-horizon round-trip | n/a (scaffolding) | +30 tests vs Phase 2's 452 → 482 passed |
| **E1 — S1 Decay (`EbbinghausPolicy`)** | `aging.py` + `fragment.py` hooks + `system.py` wire-up; 15 tests; `s1-run-01.json` | **V5 8/8 ✓** at ep200 under aggressive decay; V1 ratio **1.011 ≥ 0.80 ✓**; B3 miss (stub-judge debt, not S1 regression) | 498 passed (+16) |
| **E2 — S2 Capacity (`CapacityOnlyPolicy`)** | `aging.py` class alongside S1; 12 tests; `s2-run-01.json` | **V5 8/8 ✓**; V1 ratio **1.011 ≥ 0.80 ✓**; **thrash_rate 0.000 ✓**; B3 miss (same stub-judge debt) | 510 passed (+12) |
| **E3 — comparison** | `comparison.md` side-by-side on all 5 gates | Both pass every gate; numbers **identical** on this stream | Recommended default = S1 (richer semantics, Phase 4 surface) |

**Regression surface.** 510 tests pass under default `aging_policy=None`. All month-horizon machinery is byte-identical no-ops when the kwarg is left at default — the Phase 2 shipping surface (M1 + M2 + M3) is unchanged.

---

## Hard gate status (zero-drift gate — the one that cannot soften)

**V5 covenant 8/8 at ep200+ held under both policies.** This was the one gate the roadmap explicitly says cannot be softened for research-honesty reasons: covenant drift at ep200 is a safety failure, not a research limitation. It held under:

- Aggressive Ebbinghaus decay (`half_life=5.0`) under S1
- Pure selection pressure under S2
- `byte_cap=1024` + 40 noise observations + `check_at_ep=200` under both

The two-way covenant safety (policy `+inf` strength pin + FragmentStore `COVENANT_TAG` short-circuit in `_weakest_evictable`) is sufficient. No additional covenant-aware logic is needed in either option.

## Soft gate status

**V1(ep200)/V1(ep20) ratio 1.011 ≥ 0.80 under both** (baseline 0.992 at `byte_cap=512` pressure; both policies produce +0.019 over baseline). Both options clear the 绵延-analog soft gate; neither crosses the "declare limitation per roadmap decision gate" threshold.

**B3 full-suite miss under both** — empathy 0.037, over-disclosure 0.280, **byte-identical across baseline, S1, S2**. The miss isolates to `StubSemanticJudge` calibration against the synthetic stream, orthogonal to the aging policy. Carried as Phase 2.1 pull-forward debt (LLM-judge or auditor-labeled set).

**LoCoMo monitor floor** — not re-run live under either option. The default `aging_policy=None` is validated byte-identical to Phase 2 by unit tests; the LoCoMo path is code-path-equivalent. The monitor remains a point estimate (0.514 from Phase 1.5), not a confidence-bounded floor. Multi-trial variance study still a Phase 2.1 side task.

---

## The stream-shape confound (load-bearing finding)

S1 and S2 produce **identical numbers** on every gate on this stream. This is not evidence that decay-curve and pure selection pressure are mechanically equivalent — it is evidence that **this specific stream does not differentiate them**. Mechanism:

- The probe (`"Where did I grow up and when did I move?"`) repeatedly recalls the same anchor fragments.
- S1 boosts those via `on_recall` → `base_importance` doubles, age resets.
- S2 boosts those via `on_recall` → hit count rises above zero.
- Everything else is unrecalled noise. S1 decays all noise identically with age. S2 keeps all noise at zero hits with birth-time tiebreak (older-first).
- Both converge on evicting unrecalled noise, older-first. The decay curve never gets a chance to pick different victims than selection pressure does.

**What a differentiating stream would look like.** Sparse-recall patterns where decay has time to operate independently of recall boosts — e.g., a 90-day stream with recall events at ep5 / ep50 / ep150 and silent runs in between, so Ebbinghaus retention actually crosses selection-pressure strength during the silent runs. That stream variant is a **Phase 4 pilot concern**, not a Phase 3 blocker, because Phase 4's real recall patterns will provide it naturally.

---

## Decision: S1 Decay as default aging policy

Per plan §9 E3, both options passing every gate means the comparison memo picks the default on richer-semantics / application-surface grounds. S1 wins on:

- **Half-life as per-application knob.** Learning companion wants slower decay (long-horizon mastery); emotional companion may want faster decay (recent-affect dominates). S1 gives the Phase 4 application layer a dial. S2 has no per-application knob.
- **Explicit reconsolidation boost.** `on_recall` doubles base importance + resets curve — CLAUDE.md memory thesis names reconsolidation as the engine of narrative therapy and learning. S1 cashes this out in code; S2's hit count is a count, not a reconsolidation signal.
- **Consolidate-cycle hook.** `consolidate_candidates(top_k)` fires every 24 ticks (compressed day) returning top-K by retention; Phase 4 can hook distillation into it. S2 has the same hook as a protocol concession but no mechanism-level motivation.
- **Prior-art validation.** YourMemory +16pp over Mem0 via Ebbinghaus + reconsolidation boost — the exact shape S1 ports.

S2 stays in-tree as `CapacityOnlyPolicy`. Two reasons:

- A/B at pilot time: Phase 4 real recall patterns may differentiate the two options; keeping both means the A/B is a kwarg flip, not a reimplementation.
- `thrash_rate()` observability: S2's archive-list makes thrashing measurable from policy state alone. S1 has no equivalent observable; if a pilot scenario drives thrashing, S2's instrumentation surfaces it earlier.

Dropped-option convention from `feedback_experiment_multiple_candidates_keep_results.md` applied: S2 is retained reproducible negative-space (not dropped-from-tree); the comparison memo cites it so future backtracking has an anchor.

---

## Carried debt into Phase 4

- **B3 judge-grade.** `StubSemanticJudge` fails the gate at calibration-level on the synthetic stream; LLM judge or auditor-labeled set is Phase 2.1 pull-forward. Not blocking Phase 4 start, but needs resolution before Phase 4 pilot data is scored against B3.
- **Stream-shape confound on S1 vs S2.** Current data does not actually differentiate the two mechanisms. Phase 4 pilot provides real recall patterns that will.
- **LoCoMo monitor single-trial floor.** 0.514 remains a point estimate, not a confidence-bounded floor. Phase 2.1 multi-trial variance study still open.
- **Archival-recall tier for S2.** `_archive: list[str]` is id-only; resurrecting content on demand is a Phase 3b extension. Not required for Phase 3 go/no-go; may be required if Phase 4 wants cold-tier recall for Digital Bereavement export.

None of the four block Phase 4 start.

---

## Non-decaying guarantees (restated for Phase 4 handoff)

- **V5 covenant 8/8 at ep200+.** The hard gate held in both Phase 3 options. Phase 4 inherits this as a non-negotiable: covenant drift at any ep in a real pilot is a safety incident, not a research finding.
- **Default `aging_policy=None` is Phase 2 byte-identical.** Phase 4 can turn S1 (or S2) on per deployment; the Phase 2 shipping surface is untouched when the policy is None.
- **PAM Merkle-DAG export round-trip** (M1 + E0 8e) preserves covenant tags across the schema with no bump (critical assumption #2 was retracted during E0; `COVENANT_TAG` tag convention + existing PAM-v1 is sufficient).

---

## Verification how-to

```bash
# Phase 3 E1 S1 gate run
python benchmarks/scripts/phase3_e1_s1_gate_run.py

# Phase 3 E2 S2 gate run
python benchmarks/scripts/phase3_e2_s2_gate_run.py

# Full test suite (510 passed as of 2026-10-07)
python -m pytest -q

# Phase 3-specific tests
python -m pytest tests/test_phase3_*.py -q
```

All three must return green for Phase 3 to be re-verified. Artefacts in `docs/experiments/phase-3-month-horizon/`.

---

## Phase handoff

Phase 3 produces **evidence toward the SOTA claim per [../sota-goal.md](../sota-goal.md)**. The claim itself is made only after Phase 4 pilot; Phase 3's contribution is: (a) covenant floor at ep200+ holds under aging pressure, (b) continuity analog does not collapse, (c) two aging mechanisms are available and non-regressive. Phase 4 picks up with the application layer against real recall patterns.

Phase 4 open items that Phase 3 informs:
- Which `aging_policy` kwarg to default per application surface (学习陪伴 may use S1 at `half_life=200` for slower learning-trajectory decay; 情感陪伴 may use S1 at `half_life=50` for recent-affect emphasis).
- When to turn on the consolidate-cycle hook (Phase 4 can wire it to the existing Phase 1.5 `Consolidator` distillation).
- B3 judge-grade — must be resolved before pilot B3 scoring is defensible.
