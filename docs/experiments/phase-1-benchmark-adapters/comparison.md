# Comparison — Phase 1 benchmark adapter options

**Status (2026-10-06):** real-data gate FAILS on LoCoMo single-hop. Fixture-scale primitives passed (see fixture rows); real-scale LoCoMo pilot (2 of 10 instances) measures γ-RAG single-hop at 0.004 vs the 0.90 gate — well below the floor. Root cause is substrate scale ceiling, documented below.

This document will become the side-by-side artifact [phase-1-benchmark-adapters-plan.md](../../phases/phase-1-benchmark-adapters-plan.md) cites once step 2 (EpBench direct-mode) lands.

## Target table (populated as numbers arrive)

| Benchmark | Floor | Option α (direct) | Option γ direct | Option γ RAG | Winner |
|---|---|---|---|---|---|
| LongMemEval | ≥ 90% | **0.667** (fixture-mini, 3 Q, 2026-10-05) | **0.667** (same run) | **1.000** (same run) | **γ-RAG leading** (fixture scale, not gate) |
| LoCoMo (single-hop) **fixture** | ≥ 90% | **0.123** (fixture-mini, 4 Q, 2026-10-05) | **0.123** (same run) | **1.000** (same run) | **γ-RAG leading** (fixture scale, not gate) |
| **LoCoMo (single-hop) real** | ≥ 90% | **0.016** (real, 2 instances / 43 Q, 2026-10-06) | **0.016** (same run) | **0.004** (same run) | **BOTH FAIL — Phase 1 exit blocked** |
| LoCoMo (all categories) | — | **0.099** (fixture-mini, 5 Q, 2026-10-05) | **0.099** (same run) | **1.000** (same run) | γ-RAG leading (fixture scale) |
| EpBench | F1 ≥ 0.70 | **0.0** (fixture-mini, 2026-10-05) — floor **unreachable in direct mode by construction** | pending | pending | **γ-RAG required** |

## LoCoMo real-data gate FAIL — root cause (2026-10-06)

**Finding.** On the real snap-research/locomo locomo10.json shim, the first 2 instances (43 single-hop questions) produce α-direct=0.016 and γ-RAG=0.004. Fixture-mini scored γ-RAG=1.000 on 4 Q from the same adapter — the 250× fixture-to-real drop is not an adapter bug; it is substrate scale ceiling. Artifact: [option-gamma/locomo-real-pilot.json](option-gamma/locomo-real-pilot.json).

**Mechanism (verified by introspection on `conv-26`, 419 history turns).**
1. **Byte-cap under-sizes.** `CompanionSystem.inspect()` reports `fragment_count=14, byte_cap=4096` after ingesting all 419 turns. Selection pressure (`S1`/`S7`/`S5`) compressed 419 observations into 14 fragments at 4 KB — 13 are session_1, 1 is session_17. Every gold-bearing fact in sessions 2–19 was purged. This is the Phase 1 Option C design: `byte_cap=4096` was tuned for ep20-horizon experiments (per [baselines.md](../../baselines.md)), not for 500+-turn LoCoMo dialogues.
2. **Recall is cue-insensitive.** The 14 surviving fragments return in the same order regardless of question. Measured on 4 single-hop questions with distinct golds (`Adoption agencies`, `Transgender woman`, `Single`, `Sweden`), Q0-vs-Q1 fragment-set overlap = **6/6 = 1.000**. The reconstruction string is 95%+ identical across questions; the reader correctly refuses (`"not stated"`) on 19/20 because the surfaced fragments contain zero gold tokens.

**This is not fixable at the adapter / reader / RAG-wrapper layer.** Enlarging `byte_cap` lets more fragments survive but does not add cue-sensitivity to selection at recall time — the symptom becomes "larger uniform retrieval" rather than "right retrieval". Adding cue-sensitive recall (semantic index, re-rank by cue embedding) is a substrate architectural change — a Phase 2 scope move.

**Decision per plan §4.** The decision rule reads: *"α and γ both fail → re-open brainstorm; the plan's option set did not cover the problem."* Both do fail (α=0.016, γ-RAG=0.004 vs 0.90 floor). Phase 1 does **not** exit on the floor-regression gate; roadmap requires pause-and-redesign. Specifically:
- The Phase 1 primitives (fragment+heartbeat, R1/R2/R7 reconstruction, S1/S7/S5 selection pressure, W1/W4/W7/W3 Witness) are empirically validated at ep20 scale ([baselines.md](../../baselines.md) step 4 numbers all in-band).
- The Phase 1 scope assumption — *"ep20 primitives suffice to clear LongMemEval / LoCoMo single-hop / EpBench floors"* — was wrong. The substrate's two scale-ceiling mechanisms (byte-cap purge + cue-insensitive recall) make long-dialogue retrieval structurally impossible.
- Phase 2 as written ([phase-2-plan.md](../../phases/phase-2-plan.md) not yet drafted) must include **scale-sensitive recall** as a precondition, not as a Phase 2 feature.

The full 10-instance real gate was **not** completed because the 2-instance pilot is already a decisive signal — scaling to 10 would multiply API cost without changing the finding. If the user authorizes it after reviewing this decision, the full run would take ~2–3 h and ~$10–20 in reader calls.

**EpBench α-direct result note.** The 0.0 F1 is on a synthetic 2-instance fixture, not real EpBench. But the finding — direct-mode `recall.content` is a token bag, not a formatted item list — is substrate-structural and will reproduce on real data regardless of universe. See [option-alpha/epbench-fixture-smoke.json](option-alpha/epbench-fixture-smoke.json) + [option-alpha/README.md](option-alpha/README.md). The step-2 gate wording needs revision; flagged under "Discovered Issues" in the step-2 completion report.

**LongMemEval step-3 smoke result note.** On the 3-instance test fixture: direct = 2/3, RAG = 3/3 (judge = Tongyi qwen-max; see [option-gamma/longmemeval-fixture-smoke.json](option-gamma/longmemeval-fixture-smoke.json)). Per-question diagnostic:
- `lm-mini-0` (color?): with the fixture now carrying the identity link (user introduces herself as Mira in turn 0 of session 0), the reader has sufficient grounding to answer "red" from fragment [0] verbatim; direct's token-bag reconstruction also contains "red" and the judge accepts it.
- `lm-mini-1` (two flowers): direct token-bag passes the lenient judge; RAG cleanly composes "tulips, daffodils" → correct both ways.
- `lm-mini-2` (abstention): direct asserts unrelated session content → incorrect; RAG reader correctly abstains → correct. **The abstention case is the clean RAG-mode win — direct can't abstain by construction, only RAG can.**
Fixture scale (n=3) is far below the step-3 gate (50 instances on real HuggingFace data). The pipeline is proven end-to-end; the gate check awaits the dataset shim converter.

**Earlier (pre-fixture-fix) run.** An initial 2026-10-05 smoke recorded direct=0.333, RAG=0.667 — the RAG mis-score on lm-mini-0 was traced to a fixture defect: the question asked about "Mira's bicycle" but no turn in the history linked `user = Mira`. The reader correctly abstained under the step-3 locked prompt; the diagnostic flipped from "reader too conservative" to "fixture missing identity link". Fixture patched; result above is the post-patch number. Investigation preserved here because the triage pattern (clean fragments + conservative reader → fixture-vs-reader attribution) will recur as real data lands.

**LoCoMo step-4 smoke result note.** On the 2-instance test fixture (5 Q = 4 single-hop + 1 adversarial): direct = 0.099 aggregate / 0.123 single-hop; **RAG = 1.000 aggregate / 1.000 single-hop + 1.0 adversarial**. Judge = Tongyi qwen-max; see [option-gamma/locomo-fixture-smoke.json](option-gamma/locomo-fixture-smoke.json). Per-question diagnostic:
- Direct-mode on all four single-hop (gold = "Westerly 28", "Hull", "June 14", "Jordan Hall") lands F1 ≤ 0.27 — the raw `recall.content` is a sentence-fragment token bag that contains the gold tokens *amid dozens of other tokens*, so precision tanks (overlap ≈ 2 / len(candidate) ≈ 25). This is the same substrate-structural pattern EpBench exposed at step 2: direct-mode can't produce short-form answers from a reconstruction stream by construction.
- RAG-mode lifts all four single-hop to F1 = 1.0 — the reader cleanly extracts the short-form answer from the fragments.
- Adversarial RAG passed: reader output `"not stated"`, judge rated 1.0 as equivalent-to-refusal. **This is the clean RAG-over-direct win: direct-mode's token-bag can't abstain by construction; only RAG + a tolerant judge can.**
Direct→RAG delta on this fixture is **+0.901**, the sharpest signal in Phase 1 so far. Fixture scale (n=2, 5 Q) is far below the step-4 gate (full snap-research/locomo10.json). The pipeline is proven end-to-end on the two-speaker turn shape; the gate check (LoCoMo single-hop F1 ≥ 90% on real data) awaits `python -m psychology_base.eval.scripts.convert_locomo --input locomo10.json --output ...` + a runner invocation.

**Earlier (pre-fixture-fix) LoCoMo run.** An initial 2026-10-05 smoke recorded direct=0.101, RAG=0.800 — the one RAG single-hop miss was `locomo-mini-0-q0`: question `"What kind of boat did Jennifer buy?"`, gold `"Westerly 28"`, reader answer `"sailboat"`. Traced to fixture-question ambiguity: `"what kind of"` invites a category answer and the locked reader prompt rule `"match the specificity of the question"` picks the hypernym `"sailboat"` present in fragment [3] over the specific model `"Westerly 28"` in fragment [2]. Mem0 achieves 92.5% single-hop F1 on real LoCoMo, meaning real questions don't carry this ambiguity. Fix was at the fixture level (question → `"What model is Jennifer's new boat?"`), not the reader level. Investigation preserved here because the attribution pattern (ambiguous question + specificity-matching reader → fixture-shape issue, not reader deficiency) will recur as real data lands.

## Dropped options (recorded, not deleted)

- **Option β — RAG-mode parity only** (dropped at plan time). Reason: floor-guard-only, no application-signal; γ dominates since γ-RAG is a strict superset of β. No experiment directory until a backtrack revisits that drop.

## Decision rules once numbers land

- **α passes the floor on all three** → α is chosen (lower complexity).
- **α fails but γ-RAG passes, γ-direct within 2pt of γ-RAG** → reader is a no-op, use α-direct even where it under-shoots, flag the gap as Phase 2 attention item (per plan §3 Option γ "pass evidence" clause).
- **α fails and γ-RAG passes with large direct/RAG gap** → γ is chosen but the gap is logged as a Phase 2 substrate-quality concern.
- **α and γ both fail** → re-open brainstorm; the plan's option set did not cover the problem.
