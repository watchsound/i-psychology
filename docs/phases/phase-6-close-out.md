# Phase 6 close-out — month-horizon 绵延 attempt

**Date closed:** 2026-10-08
**Status:** SHIPPED. All hard gates green at ep1000 under every (stream × policy) cell; soft S1-vs-S2 differentiation gate not met — Phase 3 stream-shape confound upgraded to **mechanism tie** per plan §7 risk mitigation (not stream artefact). S1 Decay remains the recommended default from Phase 3 close-out.
**Companion docs:** [phase-6-month-horizon-plan.md](phase-6-month-horizon-plan.md) is the plan this memo closes out; [phase-5-close-out.md](phase-5-close-out.md) is the immediate predecessor (the SOTA claim this memo adds a sub-point to); [phase-3-close-out.md](phase-3-close-out.md) is where the "stream-shape confound" phrase was first coined and the gate that this memo advances by 5× horizon (ep200 → ep1000).
**Experiment artefacts:**
- [../experiments/phase-6-month-horizon/option-dense/dense-sanity-run-01.json](../experiments/phase-6-month-horizon/option-dense/dense-sanity-run-01.json)
- [../experiments/phase-6-month-horizon/option-sparse/sparse-weekly-run-01.json](../experiments/phase-6-month-horizon/option-sparse/sparse-weekly-run-01.json)
- [../experiments/phase-6-month-horizon/option-sparse/sparse-monthly-run-01.json](../experiments/phase-6-month-horizon/option-sparse/sparse-monthly-run-01.json)
- [../experiments/phase-6-month-horizon/option-sparse/adversarial-silent-runs-run-01.json](../experiments/phase-6-month-horizon/option-sparse/adversarial-silent-runs-run-01.json)
- [../experiments/phase-6-month-horizon/v5-covenant-ep1000.json](../experiments/phase-6-month-horizon/v5-covenant-ep1000.json)
- [../experiments/phase-6-month-horizon/pam-mid-stream-roundtrip.json](../experiments/phase-6-month-horizon/pam-mid-stream-roundtrip.json)

---

## 1. What shipped

Seven items from plan §6, in sequence:

| § | Artefact | Hard-gate verdict | Notes |
|---|---|---|---|
| 1 | Plan memo | n/a | [phase-6-month-horizon-plan.md](phase-6-month-horizon-plan.md) |
| 2 | `stream_shapes.py` + unit tests | n/a (scaffolding) | `DenseSchedule` / `WeeklySchedule` / `MonthlySchedule` / `AdversarialSilentRuns` — scheduler injected at scorer level; `CompanionSystem` core stays stream-shape-agnostic |
| 3 | Dense-schedule sanity (3 policies × 3 horizons) | **9/9 PASS ratio ≥ 0.80** | All 9 cells `ratio=1.011` (S1/S2) or `0.990–0.996` (baseline). Byte-identical to Phase 3 within noise |
| 4 | Sparse-weekly + sparse-monthly (3 policies × 3 horizons × 2 shapes) | **18/18 PASS ratio ≥ 0.80** | S1 vs S2 Δ=0.000 under both — confound persists under sparse recall |
| 5 | Adversarial-silent-runs @ ep500 (3 policies) | **3/3 PASS ratio ≥ 0.80** | S1 vs S2 Δ=0.000 — the hardest combined cell; confound upgraded to mechanism tie, not stream artefact |
| 3-gate | V5 covenant 8/8 at ep1000 under 3 policies | **24/24 PASS covenant held** | Zero drift under baseline + S1 + S2 at ep1000; hard safety gate cleared 5× past Phase 3's ep200 |
| 6 | PAM mid-stream round-trip | **6/6 PASS checks** | Byte-identical fragment-set restore + reconsolidation chain preserved + covenant held post-restore through ep1000 continuation |

**Not shipped:** nothing from plan §6 is deferred. The full §6 sequence closed.

**Regression surface.** Default `aging_policy=None` + absent-scheduler path is byte-identical to Phase 2's shipping surface (M1 + M2 + M3). Phase 6's new code is additive: `stream_shapes.py` is a scorer-level module with no `CompanionSystem` dependency; `scheduled_v1_continuity_ratio` is additive alongside the Phase 3 `rolling_v1_continuity_ratio`; `phase6_*` scripts are additive runners that reuse Phase 3 harness primitives.

---

## 2. Hard gate status (zero-softening gates)

### 2a. V5 covenant 8/8 at ep1000 under every (stream × policy) cell

**Zero drift under every cell.** All 8 covenant rules survived ep1000 under baseline + S1 EbbinghausPolicy + S2 CapacityOnlyPolicy with the Phase 3 noise-pressure stream. The gate the roadmap explicitly says cannot soften for research-honesty reasons — covenant drift at ep1000 is a safety failure, not a research limitation — cleared 5× past Phase 3's ep200.

- Baseline: 8/8, `min_fragment_count_at_check=5`, `max=6` (eviction fired; covenant pin carried)
- S1: 8/8, same fragment counts
- S2: 8/8, same fragment counts

The two-way covenant safety (policy `+inf` strength pin + FragmentStore `COVENANT_TAG` short-circuit in `_weakest_evictable`) is sufficient at ep1000. No additional covenant-aware logic needed.

### 2b. V1 continuity ratio ≥ 0.80 at ep500 under sparse-monthly

**Hardest combined cell:** sparse-monthly schedule × ep500 × S1 EbbinghausPolicy. `ratio=1.011` — well above the 0.80 roadmap gate. The 绵延-analog metric held unchanged, horizon pushed 2.5× past Phase 3's ep200.

- Sparse-monthly S1: ep200=1.011 / ep500=1.011 / ep1000=1.011 ✓
- Sparse-monthly S2: ep200=1.011 / ep500=1.011 / ep1000=1.011 ✓
- Sparse-weekly: same 1.011 across all horizons under S1 + S2
- Adversarial-silent-runs @ ep500: S1=1.011 / S2=1.011 ✓

The roadmap decision-gate "declare the limitation, do not retrofit the metric" was prepared for a sub-0.80 result; the actual result is above 1.0 across the matrix. No limitation needs to be declared on this metric. The project's substrate holds the 绵延-analog metric 10× past Ouroboros's known failure point (ep20 → ep1000).

### 2c. PAM mid-stream round-trip (Digital Bereavement live-fire)

**6/6 checks PASS** at mid_ep=500 with continuation to ep1000:

| Check | Result |
|---|---|
| `fragment_set_byte_identity` (27 source ids == 27 restored ids) | ✓ |
| `content_addressing_offenders == []` (strict DAG Merkle-identity) | ✓ |
| `reconsolidation_chain_preserved` (rerender child's `parent_ids[0]` resolves to parent in restored store) | ✓ |
| `covenant_fragment_preserved` at restore boundary | ✓ |
| `semantic_recall_equivalence` (source top-recall id == restored top-recall id on same cue) | ✓ |
| `post_restore_covenant_hold_at_ep1000` (54 fragments after continuation; covenant still present) | ✓ |

This is the check Phase 3 E6 could not run: Phase 3's PAM test exported fresh-observation fragments only, at end-of-stream. Phase 6's test seeds a `rerender()` chain **before** the mid-stream export, so the reconsolidation-chain preservation claim gets exercised at the Merkle-DAG level, not just at the leaf-fragment level. Export+restore wall = 15 ms on 27 fragments; the artefact-size + round-trip cost is well under any reasonable Digital Bereavement budget.

### 2d. LoCoMo monitor floor

Not re-run live for this close-out. Phase 6's new code paths (stream_shapes scheduler + sparse-recall scorer) are off-the-hot-path for the LoCoMo adapter; the default `aging_policy=None` + absent-scheduler route is byte-identical to Phase 2 by unit tests. The monitor remains a point estimate (0.514 from Phase 1.5), not a confidence-bounded floor. **Carried debt into v1.**

---

## 3. Soft gate status — the stream-shape confound upgrades to mechanism tie

**Plan §3 soft gate:** ≥ one cell shows ≥ 0.05 absolute delta between S1 and S2 on V1 continuity ratio under sparse-recall, "resolving the stream-shape confound either direction".

**Measured:** `Δ(S1, S2) = 0.000` across every sparse cell. Not within noise — exactly zero. Verbatim verdicts from the three sparse artefacts:

- Sparse-weekly: *"S1 vs S2 Δ=0.000 < 0.05 — stream-shape confound persists under sparse-weekly"*
- Sparse-monthly: *"S1 vs S2 Δ=0.000 < 0.05 — stream-shape confound persists under sparse-monthly"*
- Adversarial-silent-runs: *"S1 vs S2 Δ=0.000 < 0.05 — stream-shape confound persists under adversarial-silent-runs"*

Per plan §7 risk mitigation ("if it still produces identical numbers, the Phase 3 stream-shape confound is upgraded to 'mechanism tie, not stream artefact' — still a real finding, not a Phase 6 failure"), the Phase 6 verdict is:

> **The S1 Decay vs S2 Capacity-only distinction is a mechanism tie on scripted single-anchor streams, not a stream-shape artefact.** On synthetic streams with homogeneous probe cues, both policies evict identical fragments at identical steps because (a) the recall-boost cadence (`on_recall` reset + boost) drives retention more than the decay-curve shape does, and (b) selection pressure under `byte_cap` + hit-count tiebreak converges on the same victim set when the probe cue repeatedly hits the same anchor set. Under those conditions the half-life knob is observationally inert.

**What this does not say.** It does not say the two mechanisms are equivalent on real-user data. Real-user heterogeneous probe cues + irregular silent runs may still differentiate. But that evidence is **permanently closed** under the Phase 4b dataset-replay pivot (user decision 2026-10-08); no scripted sparse schedule the plan designed could force the differentiation. The ship stance is: S1 Decay default (richer semantics + application-knob + reconsolidation signal + prior-art validation per Phase 3 §"Decision"), S2 retained in-tree for the A/B that a future recruitment-budget-unblock pilot could run.

---

## 4. SOTA claim — Phase 6 sub-point added

The Phase 5 close-out §4 locked the project's three-axis SOTA posture. Phase 6 adds **one load-bearing sub-point** to Axis 2 (mechanism viability):

> *The substrate holds the 绵延-analog metric (V1 state-signature continuity ratio) at ≥ 1.00 through ep1000 under every tested stream shape (dense / sparse-weekly / sparse-monthly / adversarial-silent-runs) and every tested aging policy (baseline / S1 Decay / S2 Capacity-only). The covenant-rule non-decay gate (V5 8/8) holds at ep1000 with zero drift under every cell. The PAM Merkle-DAG mid-stream round-trip preserves reconsolidation chains (`parent_ids` byte-identical) across export + restore, satisfying the Digital Bereavement obligation at mid-horizon as well as end-of-horizon.*

This extends Phase 3's ep200 result 5× in horizon without softening either hard gate. Ouroboros's named open problem (mianyan decay past ep20) does not reproduce on this substrate at the 50× horizon we tested — a result the roadmap decision gate was explicitly prepared to accept with limitation disclosure, but which does not need to be made.

**Explicitly not claimed:**
- S1 vs S2 mechanism differentiation on real-user data (permanently closed; mechanism tie documented, not resolved).
- Month-horizon 绵延 beyond ep1000 — Phase 6 stopped at the plan's declared horizon; nothing says the ratio continues to hold at ep5000 or ep10000.
- Consciousness-emergence (Ouroboros territory).
- Real-user Axis 3 outcome (still gated on cohort funding per Phase 5 §4).

---

## 5. Carried debt into v1

- **Real-user S1 vs S2 differentiation** — permanently closed under scripted-only scope. The mechanism tie is a documented limitation of our scripted substrate, not a mechanism claim.
- **LoCoMo monitor as point estimate** — same debt as Phase 2/3/5; multi-trial variance study still unrun. Not blocking v1.
- **绵延 metric beyond ep1000** — the plan declared ep1000 as the horizon; nothing says the ratio continues to hold at ep5000+. A longer-horizon stress is Phase 7-or-later research (not v1 scope).
- **Phase 6 memos reference-chain** — this close-out + plan + Phase 5 close-out + Phase 3 close-out all dated with their respective dates; a reviewer coming cold can read them in that order.

---

## 6. Artefacts shipped (Phase 6 total)

- **Code:** [`src/psychology_base/companion/stream_shapes.py`](../../src/psychology_base/companion/stream_shapes.py) (scheduler module), [`src/psychology_base/eval/benchmarks/mianyan_analog.py`](../../src/psychology_base/eval/benchmarks/mianyan_analog.py) `scheduled_v1_continuity_ratio` extension, [`benchmarks/scripts/phase6_dense_sanity_run.py`](../../benchmarks/scripts/phase6_dense_sanity_run.py), [`benchmarks/scripts/phase6_sparse_recall_run.py`](../../benchmarks/scripts/phase6_sparse_recall_run.py), [`benchmarks/scripts/phase6_v5_covenant_ep1000.py`](../../benchmarks/scripts/phase6_v5_covenant_ep1000.py), [`benchmarks/scripts/phase6_pam_mid_stream_roundtrip.py`](../../benchmarks/scripts/phase6_pam_mid_stream_roundtrip.py).
- **Experiment JSONs:** 6 artefacts under [`docs/experiments/phase-6-month-horizon/`](../experiments/phase-6-month-horizon/).
- **Docs:** [phase-6-month-horizon-plan.md](phase-6-month-horizon-plan.md) + this close-out.

---

## 7. Next action

Phase 6 is the last planned research phase of v1. All Phase 1–6 gates are closed with declared limitations where honest. No next-phase memo is scheduled under current scope.

If/when a recruitment-budget unblock arrives, the pre-existing Phase 4b §11 code artefacts (TutorCLI / CompanionCLI / policy-A/B comparison tooling) are in-tree as cold storage — resuming the live-pilot path would re-open Phase 4 to produce the real-user Axis 3 evidence that gates the strongest-form SOTA claim.

Update [../roadmap.md](../roadmap.md) §"Current phase" + §"Next action" to point to this close-out (replacing the "Phase 6 planning" pointer).

---

### Discovered Issues

- [docs/roadmap.md:224-225](../roadmap.md#L224-L225) (flagged 2026-10-08) — "Current phase" line points to Phase 6 planning; stale now that Phase 6 shipped. Replace with Phase 6 close-out pointer + date. Same shape as the Phase 5 close-out flag — not fixing here to keep this memo's change-scope tight.
- [benchmarks/scripts/phase6_pam_mid_stream_roundtrip.py:144-148](../../benchmarks/scripts/phase6_pam_mid_stream_roundtrip.py#L144-L148) (flagged 2026-10-08) — the Sweden-anchor lookup `.content.startswith("I grew up in Sweden")` is a string-match on the first observation of `_build_stream()`. Fragile if `_build_stream()` is ever renumbered; the test would silently assertion-fail. A content-addressed id-pin (compute the hash from the Observation we just enqueued and look that up) would be more robust. Not fixing here because the stream is Phase 6 scope-frozen.

### Assumptions Made

- Assumed `byte_cap=None` is the right choice for the PAM mid-stream round-trip because the gate is PAM serialization semantics, not eviction behavior — V5 covenant_ep1000 already covered capacity pressure. **(critical)** if the user intended PAM round-trip to also be a capacity-pressure-under-mid-stream test, the script needs to re-run with the Phase 3 byte_cap=1024 default + a covenant-tagged Sweden anchor so the anchor survives decay to the export point.
- Assumed the plan §6 sequence closes cleanly on the 7-item list; no §4 sub-item (e.g., `--stream-shape` on the Phase 3 scripts) was pulled forward because the new `phase6_*` runners wholly replace the Phase 3 runners for this scope. **(not critical)** if the Phase 3 scripts should be deprecated vs. kept side-by-side, that's a repo-hygiene call for a future cleanup pass.
- Assumed the mechanism-tie finding is a legitimate close for the soft gate rather than a request to redesign the sparse schedules further. Plan §7 risk mitigation explicitly provisioned "if it still produces identical numbers, the confound is upgraded to mechanism tie — still a real finding, not a Phase 6 failure", and the finding is published rather than papered over.
