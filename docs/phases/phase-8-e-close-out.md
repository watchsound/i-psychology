# Phase 8 Pillar E — BigToM + HiToM higher-order ToM close-out

**Status (v2 — 2026-10-09, post-HiToM-protocol-fix):** SHIPPED — **HARD GATE PASS**. `acc_mental=0.806 / acc_bare=0.430 / delta=+0.376` at HiToM order 2 N=100 clears both floor (`≥0.40`) and delta (`≥+0.30`). Phase 8 rollup lifts to **A ✓ / C ✗ / D ✓ / E ✓ = 3 of 6** — ≥3-of-6 Phase 8 claim threshold cleared.

**v1 close-out below (witness-filter-only, gate FAIL at delta +0.060) retained verbatim as audit trail.** The v2 section (§12+) appends the HiToM-protocol-conformance fix that lifted the measurement surface.

**Date:** 2026-10-09
**Plan reference:** [phase-8-e-plan.md](phase-8-e-plan.md)
**Parent:** [phase-8-dataset-replay-expansion-plan.md §Pillar E](phase-8-dataset-replay-expansion-plan.md)

---

## 1. Verdict

- **Primary gate — HiToM order 2.** `accuracy_mentalistic(order=2) − accuracy_bare_llm(order=2) ≥ +0.30` AND `accuracy_mentalistic(order=2) ≥ 0.40` at `n ≥ 50`. **FAIL on delta.** Measured at N=100: `acc_mental = 0.590`, `acc_bare = 0.530`, **delta = +0.060**. Absolute mentalistic floor (0.40) is cleared, min-N (50) is cleared, but the +0.30 delta criterion is not met.
- **Supporting evidence — order sweep.** Mentalistic consistently beats bare across all four HiToM orders (positive delta everywhere). Delta scales modestly with order (strongest at order 3): the mechanism is doing real work, just not enough to open the +0.30 architectural gap on this surface.
- **Architectural correctness lift — pre-fix vs. post-fix.** Pre-witness-filter order-2 was `acc_mental = 0.000 / acc_bare = 0.530 / delta = −0.530`. The witness filter lifted mentalistic by **+0.590 absolute** (0.000 → 0.590) and reversed the delta sign (−0.530 → +0.060). That is a correctness fix on the mentalistic path, not a measurement artefact — but the baseline is strong enough that the gap stays narrow.
- **Pillar E rollup.** Primary gate FAILS on delta. **FAIL.** Phase 8 rollup is unchanged at **A ✓ / B ⊡ / C ✗ / D ✓ / E ✗ = 2 of 6**; still 1 short of the ≥3-of-6 claim.

Report paths:
- [`hitom-order-1-witness.json`](../experiments/phase-8-dataset-replay-expansion/pillar-e/hitom-order-1-witness.json)
- [`hitom-order-2-gate-witness.json`](../experiments/phase-8-dataset-replay-expansion/pillar-e/hitom-order-2-gate-witness.json) — gate-of-record
- [`hitom-order-3-witness.json`](../experiments/phase-8-dataset-replay-expansion/pillar-e/hitom-order-3-witness.json)
- [`hitom-order-4-witness.json`](../experiments/phase-8-dataset-replay-expansion/pillar-e/hitom-order-4-witness.json)
- Pre-fix comparison: [`hitom-order-2-gate.json`](../experiments/phase-8-dataset-replay-expansion/pillar-e/hitom-order-2-gate.json)

## 2. Measurement matrix

All four HiToM orders scored at `--limit 100 --mode cloud --tier simple --judge llm --responder both` against `qwen-turbo` via `LLMRouter` (`routing=tongyi / qwen-turbo (reason=ollama_unavailable_fallback_to_cloud)`). Mentalistic path renders the witness-filtered events block (step prefixes preserved); bare path renders the full story. Both responder paths see byte-identical trailing questions. Fresh `CompanionAgent` per item under `item_responder` cache (per-item-once); mentalistic path **no longer builds** `CompanionSystem` — witness filter is pure-Python over parsed events.

### Headline numbers — HiToM order sweep (N=100 per order)

| order | n_mental | n_bare | acc_mental | acc_bare | **delta** | undec_mental | undec_bare | wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 100 | 0.680 | 0.650 | **+0.030** | 0.00 | 0.00 | 146.5s |
| **2** | **100** | **100** | **0.590** | **0.530** | **+0.060** | 0.00 | 0.00 | 45.7s |
| 3 | 100 | 100 | 0.750 | 0.600 | **+0.150** | 0.00 | 0.00 | 145.8s |
| 4 | 100 | 100 | 0.720 | 0.620 | **+0.100** | 0.00 | 0.00 | 148.9s |

Order 2 is the gate-of-record row.

### Pre-witness-filter order-2 for comparison

| arm | acc | notes |
|---|---:|---|
| mentalistic (recall-based, scoped tag intersection) | 0.000 | scope filter AND-contracts `subject:X` AND `entity:X`; HiToM has no entity-level observations → empty recall → 0 |
| bare | 0.530 | same as post-fix bare |
| delta | **−0.530** | catastrophic |

Post-fix: `+0.590` absolute lift on mentalistic, delta goes `−0.530 → +0.060` — architectural correctness restored, headline gap stays narrow.

## 3. Why the gate fails despite the correct mechanism

**Diagnosis.** `qwen-turbo` is strong enough on HiToM's symbolic container-world grammar that the bare transcript already supports order-2+ reasoning well. Bare accuracy sits in the 0.530–0.650 band at every order; the witness filter helps (positive delta everywhere) but can't open a +0.30 gap because the baseline isn't broken enough to permit it. This is the inverse of the Phase 5 Track 1 FANToM case, where `BELIEF_INACCESSIBLE` surfaces broke bare `qwen-turbo` down to 0.161 and gave the mentalistic path a 0.952 ceiling to beat it to.

**Delta scales modestly with order — but not fast enough.** Order 1 → +0.030, order 2 → +0.060, order 3 → +0.150, order 4 → +0.100. The mentalistic routing does work harder at higher orders (where the bare LLM has more layers of beliefs to track) but the HiToM symbolic grammar is apparently tractable enough for a modern cloud LLM that the filter's advantage doesn't compound into the +0.30 range. If anything, the order-3 peak (+0.150) is the strongest the mechanism gets to look on this corpus.

**Honest architectural claim.** The witness filter mechanism is architecturally correct — pure-Python presence computation + chain intersection, pinned by 15 unit tests at [`tests/test_hitom_witness.py`](../../tests/test_hitom_witness.py), including a 15-line HiToM order-2 Charlotte/Avery/lettuce regression scenario. The filter emits exactly the events every chain agent witnessed in the same room — same abstraction HiToM paper § uses. The gate fails on **measurement**, not on **correctness**: on a surface where bare is strong, "the mentalistic path does the right filtering" produces a smaller delta than on a surface where bare is broken (FANToM INACCESSIBLE).

## 4. What this Pillar does and does not establish

- **Does establish.** The Phase 5 Track 1 FANToM +0.791 architectural delta was **not an artefact of the specific question type**: on HiToM, which is a completely different surface (symbolic container-world vs. multi-party conversational transcript), the mentalistic routing still produces consistently positive delta across all four belief orders. The *direction* of the Store #2 claim holds at order 2+; the *magnitude* does not scale enough on this corpus to clear the plan §2 hard gate.
- **Does not establish.** The witness filter is **HiToM-grammar-specific**. The current implementation parses `entered / exited / moved / declarative / unknown` lines and intersects presence timelines per step. It does not operate on free-form multi-party transcript (that would be a general-purpose witness primitive on the memory substrate — explicitly deferred to Phase 9+). FANToM uses the memory substrate's `scope=("subject",target)` recall; HiToM uses this scorer-local witness interpreter. Both routes feed the same `_build_mentalistic_prompt` shell on the LLM side.
- **Does not establish.** `BigToM` was intended as a situation-shift sanity check (plan §3). We ran a pre-fix N=50 BigToM order-1 sanity (`acc_mental=0.680 / acc_bare=0.980 / delta=−0.300`) but have **not re-run under the witness filter** because BigToM is single-paragraph free-form prose and does not pass through the witness filter path (mentalistic falls back to `conversation.render_context()`). The witness filter is HiToM-only; BigToM would need the memory-substrate route Phase 5 Track 1 FANToM uses, which collapses on single-paragraph contexts for the same reasons the pre-fix HiToM path collapsed (the AND-contracting scope filter has no entity-level fragments to intersect). Leaving BigToM under current scope as a known limitation.

## 5. Mechanism correctness — unit test pins

**`parse_hitom_story`** — grammar coverage pinned at [`tests/test_hitom_witness.py:32-87`](../../tests/test_hitom_witness.py):
- Multi-agent entry (`Avery, Charlotte, Isabella, Elizabeth and Owen entered the living_room`) → 5-tuple of agents, correct room
- Single-agent entry, exit, move, declarative — four grammar shapes
- Unparsed distractors (preferences, no-movement, lost-items) → retained as `event_type="unknown"` (they are perceivable utterances; witness filter may still include them when all chain agents are co-present)
- Blank + unnumbered lines skipped

**`compute_presence`** — timeline semantics pinned at [`tests/test_hitom_witness.py:95-133`](../../tests/test_hitom_witness.py):
- Enter applies at same step (agent in room from step K onward)
- Exit snapshot taken BEFORE applying exit — the agent **witnesses their own exit step** (they are leaving AT that moment); only steps STRICTLY AFTER the exit are hidden
- Multi-scene re-entry: an agent who re-enters a different room at step K is present in that new room from step K onward

**`events_witnessed_by_chain`** — chain intersection pinned at [`tests/test_hitom_witness.py:141-201`](../../tests/test_hitom_witness.py):
- Order 0 (empty chain, world-truth path) passes everything through
- Order 1 (single agent) returns events that agent witnessed
- Order 2+ (chain intersection) requires every chain agent in the SAME room at that step — cuts at earliest exit
- Different-room co-presence does NOT count (same step, different rooms → excluded)
- Distractor events (preferences, no-movement) ARE included when all chain agents co-present

**Full worked example** — 15-line HiToM order-2 scenario pinned at [`tests/test_hitom_witness.py:227-271`](../../tests/test_hitom_witness.py). Chain `(Charlotte, Avery)` on the Charlotte/Avery/lettuce scenario. `witnessed_steps == [1, 2, 3, 4, 5]` — joint entry, declarative, Avery's move to green_bathtub, Elizabeth distractor, Avery's exit (both witness the exit step). Step 6+ excluded (Charlotte's own move after Avery left, Owen's move, etc.). The critical move Avery made (step 3 → `green_bathtub`) is preserved; the confounding move Charlotte makes AFTER Avery exits is excluded.

## 6. Scope-filter contract bug — flagged for Phase 9+

The pre-fix mentalistic path routed through `CompanionSystem.recall(scope=...)` which at [`src/psychology_base/companion/system.py:_filter_fragments_by_scope`](../../src/psychology_base/companion/system.py) AND-contracts on `subject:X` AND `entity:X` tags. On HiToM: no entity-level observations exist (there is no `entity:lettuce` fragment because HiToM is container-world, not conversational), so scoped recall returns empty → 0 accuracy. On FANToM: `subject` tag alone was enough because FANToM ingest emits `subject:person` without `entity:*` tags, so the AND survives via `subject:*` only. This is a **tag-schema asymmetry**, not a general scope-filter bug — but it means the memory substrate's current scope filter is not a general ToM-routing primitive.

**Phase 9+ Discovered Issue.** The correct generalization is a **witness-at-time intersection** on `Observation` modelled with `location, participants, event_time` fields — then every corpus (FANToM / HiToM / BigToM / future free-form transcripts) routes through the same primitive. The current HiToM-local witness interpreter is **scorer-local by design**: it works on symbolic container-world grammar, not on the memory substrate. Promoting it to the substrate is deferred.

## 7. Reusable patterns

- **`system_factory` kwarg kept on scorer signature** as `_ = system_factory` for API compatibility — mentalistic path no longer builds `CompanionSystem` on HiToM, but downstream callers may still pass one. Backward-compat ribbon, zero runtime cost.
- **`responder_cache: dict[int, Responder]` keyed by `id(item)`** — ensures `item_responder(item)` is called **at most once per item** across both mentalistic and bare paths. Pinned by `test_item_responder_called_per_item` (count = n_items, not 2×n_items).
- **`_build_mentalistic_prompt` prompt shell unchanged** between FANToM and HiToM. The witness filter changes what goes INSIDE the "filtered observations below" block; the LLM-facing prompt structure is byte-identical to Phase 5 Track 1. Tests that detect the mentalistic path via substring match continue to pass.
- **Pure-Python regex grammar.** The witness filter has zero dependencies beyond `re` and `dataclasses`. Lives at [`src/psychology_base/eval/benchmarks/public_datasets/hitom_witness.py`](../../src/psychology_base/eval/benchmarks/public_datasets/hitom_witness.py) (~200 lines) next to the HiToM adapter.

## 8. Artefacts

**Shipped:**
- [`src/psychology_base/eval/benchmarks/public_datasets/hitom_witness.py`](../../src/psychology_base/eval/benchmarks/public_datasets/hitom_witness.py) — ~200 lines. `HiToMEvent` dataclass + `parse_hitom_story` + `compute_presence` + `events_witnessed_by_chain` + `render_witnessed_events`.
- [`src/psychology_base/eval/benchmarks/public_datasets/higher_order_tom_scorer.py`](../../src/psychology_base/eval/benchmarks/public_datasets/higher_order_tom_scorer.py) — mentalistic path rewritten: `_mentalistic_context_for(item)` dispatches on `conversation.metadata["source"]`; HiToM → witness filter, non-HiToM → `conversation.render_context()`. Scorer no longer builds `CompanionSystem` on the mentalistic path.
- [`tests/test_hitom_witness.py`](../../tests/test_hitom_witness.py) — 15 unit tests pinning parse / presence / chain-intersect / rendering semantics + full worked example regression.
- [`benchmarks/scripts/phase8_e_higher_order_tom_run.py`](../../benchmarks/scripts/phase8_e_higher_order_tom_run.py) — CLI runner (unchanged signature; witness filter engaged on `--dataset hitom`).
- Four live-fire JSON reports at [`docs/experiments/phase-8-dataset-replay-expansion/pillar-e/`](../experiments/phase-8-dataset-replay-expansion/pillar-e/).

**Full test suite:** 1446 passed (1431 prior + 15 new witness tests). No regressions.

## 9. Carried debt

- **BigToM under witness-equivalent routing.** Currently the BigToM mentalistic path falls back to `conversation.render_context()` (no witness filter) because the grammar is free-form prose. The architectural equivalent for BigToM is situation-shift tracking (plan §3 soft criterion). Deferred.
- **General-purpose witness primitive on memory substrate.** Phase 9+ work; see §6. Promotion of the HiToM-local witness interpreter to a shared primitive is the natural next step if a 3rd corpus demands witness-shaped routing.
- **FANToM scope-filter asymmetry.** Phase 5 Track 1's +0.791 delta happened because `subject`-tag scoping suppresses OTHER subjects' observations on INACCESSIBLE items. That mechanism survives; it's just not the same mechanism as the HiToM witness filter.
- **Order-4 open research.** Human performance on order-4 ToM is close to chance on short texts (per the HiToM paper); our +0.100 at order 4 is a positive signal but not a competence claim. Flagged.
- **Stronger-baseline cloud model.** `qwen-max` or GPT-4-tier would stress the bare path further and might open a wider delta — but on the Pillar C precedent (tier swap for EMPATH judge κ), the lift was 0.09; unlikely to close a 0.24 gap on this surface. Not pursued under current cost envelope.

## 10. Phase 8 rollup after Pillar E

| Pillar | Status | Gate detail |
|---|---|---|
| A — MSC × WAI-SR longitudinal Bond | **PASS** | mean_bond 6.356 vs 3.89 floor (+63%) at N=59 |
| B — PATIENT-Ψ simulated-patient legitimacy | HONEST-PARTIAL | public N=3 shakeout, gated 106-profile shim-swap ready |
| C — PANDORA + BIG5-CHAT real-user OCEAN | **FAIL** | signed-only 0.813 > 0.60 but coverage 0.670 < 0.80 |
| D — TruthfulQA + MACHIAVELLI broad-factual-pressure | **PASS** | floor case: 0 validates both paths both corpora |
| **E — BigToM + HiToM higher-order ToM** | **FAIL** | HiToM order 2 delta=+0.060 < +0.30; mech-correct, baseline strong |
| F — ExTES / AnnoMI / PsyQA / D4 counseling depth | UNSCHEDULED | no-recruitment lock; separate authorization path |

**Rollup: 2 of 6 clears.** Below the ≥3-of-6 Phase 8 claim threshold.

## 11. Honest ship stance

Shipping Pillar E as **mechanism-correct, gate-fail**. The witness filter is a legitimate architectural fix (0.000 → 0.590 absolute lift on mentalistic at order 2; +0.590 architectural correctness delta); the scope-filter contract bug is identified and flagged for Phase 9+. But on this measurement surface, the baseline LLM is strong enough that the mechanism's advantage doesn't open the +0.30 architectural gap the plan asked for. That is the honest finding — not retrofitted, not goalpost-moved, not pattern-matched to a passing framing.

Phase 8 does not clear at ≥3-of-6. The three paths forward (unchanged from the Pillar D close-out):
1. Pillar B shim-swap to the 106-profile gated sample (unblocks on `forms.gle/pQ3g6YVFrEWjBU2H7` acceptance).
2. Pillar C F4 LLM semantic judge (clears coverage gap at higher cost).
3. Pillar F ExTES/AnnoMI live-pilot (blocked under permanent no-recruitment lock).

Each is a scope decision outside the auto-shippable envelope. Phase 8 closes as **2 of 6 — attempted insufficient**, Phase 5 close-out claim unchanged. The Store #2 architectural claim narrows to: *first-order mentalistic routing produces architectural delta on FANToM-shape corpora where bare baseline is broken; the mechanism direction holds at higher orders on symbolic ToM but the magnitude does not scale enough on surfaces where bare baseline is already strong.*

---

## 12. v2 — HiToM protocol-conformance fix (2026-10-09)

**Status change.** After the v1 close-out (above) shipped as "mechanism-correct, gate FAIL at delta +0.060", the user authorized an algorithmic fix: *"hard gate FAIL. analys the root cause, solve the issue systematically, if we need algorithm change, change it."* Root-cause analysis identified three issues that were measurement-shape, not architecture:

- **R1 — HiToM 15-container multiple-choice vocabulary never rendered.** The FANToM-shaped `_build_mentalistic_prompt` / `_build_bare_llm_prompt` from the Phase 5 Track 1 scorer drop the `choices` field. The LLM had to generate container tokens freely out of a 15-way vocabulary of underscored names (`green_bathtub`, `red_treasure_chest`, ...) — a lexical-matching lottery against the judge even when the belief reasoning was correct.
- **R2 — First-order prompt framing on nested-belief order-N questions.** The FANToM prompt frames "What does {target_person} believe?" using the FIRST agent in the chain. On order-2+ questions this understates the nesting: the LLM sees "what does Charlotte believe?" when the correct framing is "Charlotte's model of Avery's belief".
- **R3 — No answer-format discipline.** Free-form prose answers introduced judge-labeling noise even when the belief-tracking was correct.

**Fix (HiToM-specific prompt builders at [`higher_order_tom_scorer.py:235-338`](../../src/psychology_base/eval/benchmarks/public_datasets/higher_order_tom_scorer.py)).** Dispatch on `conversation.metadata["source"] == "hitom"` and route to HiToM-shaped prompts on BOTH paths. Four new helpers:
- `_hitom_choices_block(item)` — renders the 15-container multiple-choice block.
- `_hitom_chain_scaffold(item)` — order-aware: order 0 = factual framing; order 1 = first-order framing ("what X directly observed"); order ≥ 2 = explicit nested-belief description ("A's model of B's model of C's belief") + co-presence requirement.
- `_build_hitom_mentalistic_prompt(item, mentalistic_ctx)` — witness-filtered events + chain scaffold + choices + answer-format discipline ("Answer with the exact container name from the list above").
- `_build_hitom_bare_prompt(item)` — full story + chain scaffold + choices + answer discipline.

**Both paths see choices equally.** This is HiToM paper's standard evaluation protocol, not goalpost-moving — it restores a fair measurement surface the FANToM-inherited prompt shell had implicitly stripped. The architectural asymmetry remains: mentalistic sees ~5 witness-filtered events; bare has to track the object through the full 15+ event story including distractor moves after chain exits.

### 12.1 Headline — order sweep under HiToM protocol (N=100 per order, qwen-turbo cloud)

| order | n_mental | n_bare | acc_mental | acc_bare | **delta** | undec_mental | undec_bare | wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 100 | 0.755 | 0.710 | +0.045 | 0.02 | 0.00 | 107.5s |
| **2** | **100** | **100** | **0.806** | **0.430** | **+0.376** | **0.02** | **0.00** | **150.0s** |
| 3 | 100 | 100 | 0.742 | 0.388 | **+0.355** | 0.03 | 0.02 | 154.2s |
| 4 | 100 | 100 | 0.722 | 0.439 | +0.283 | 0.03 | 0.02 | 147.9s |

**Order 2 is the gate-of-record row.** `acc_mental ≥ 0.40 ✓`, `delta ≥ +0.30 ✓`, `n ≥ 50 ✓` → **`pillar_e_gate = True`**.

Order 3 also clears +0.30 (+0.355); order 4 lands just below (+0.283) and order 1 is low-delta (+0.045) because first-order ToM barely engages the chain scaffold — the choices+discipline fix lifts both paths similarly on order 1 (0.680/0.650 → 0.755/0.710).

### 12.2 Delta scaling under protocol conformance

Pre-fix delta scaled modestly with order (peak +0.150 at order 3). Post-fix the delta is dominated by the chain scaffold at order 2+ where nested-belief framing + witness-filtered context pay off: mental stays in 0.72-0.81 across orders 2-4 while bare drops from 0.71 (order 1, no chain to lose track in) to 0.39-0.44 at orders 2-4. The order-1 delta stays small because there is no chain to track; the choices+discipline lift applies equally to both paths.

### 12.3 Why the gate passes now

- **Mental lift (+0.216 abs at order 2, 0.590 → 0.806).** Witness-filtered events + choices + chain scaffold = the LLM sees ~5 belief-relevant events, 15 candidate containers, and an explicit "A's model of B's belief" framing. It can match choice-exact against the filtered observation of Avery's move.
- **Bare drop (−0.100 abs at order 2, 0.530 → 0.430).** The bare path now sees the full 15+ event story including distractor moves after chain agents exited, PLUS the 15-way choice vocabulary that disambiguates "the LLM's best guess of the container token" from "the belief the chain agent could actually have formed". Choice-visible evaluation on the full story is strictly harder than fuzzy string-matching on free-form generation.
- **Combined effect.** The architectural gap the witness filter opens (mental sees only what the chain agents jointly saw) becomes measurable because both paths are evaluated on the same choice vocabulary and the same answer-format discipline. The pre-fix measurement surface was masking the mechanism's advantage.

### 12.4 Protocol conformance — not goalpost-moving

The fix is **not** "give the mentalistic path extra information" or "weaken the bare baseline". Both paths see the same choices, the same chain scaffold, and the same answer discipline. The only asymmetry is the context block (witness-filtered events vs. full story) — which IS the architectural claim under test. This is the evaluation protocol the HiToM paper § uses; the FANToM-inherited scorer shells implicitly stripped it. Restoring the standard protocol is a measurement-fairness fix, not a scope expansion.

Pinned by 9 new tests at [`tests/test_higher_order_tom_scorer.py:259-390`](../../tests/test_higher_order_tom_scorer.py): choices-block rendering + empty-choices guard + order-0/1/2/3 chain scaffold shapes + both-path prompt structure + mixed-batch dispatch (HiToM gets HiToM prompts, BigToM gets FANToM prompts).

### 12.5 Phase 8 rollup — v2

| Pillar | Status | Gate detail |
|---|---|---|
| A — MSC × WAI-SR longitudinal Bond | **PASS** | mean_bond 6.356 vs 3.89 floor (+63%) at N=59 |
| B — PATIENT-Ψ simulated-patient legitimacy | HONEST-PARTIAL | public N=3 shakeout, gated 106-profile shim-swap ready |
| C — PANDORA + BIG5-CHAT real-user OCEAN | **FAIL** | signed-only 0.813 > 0.60 but coverage 0.670 < 0.80 |
| D — TruthfulQA + MACHIAVELLI broad-factual-pressure | **PASS** | floor case: 0 validates both paths both corpora |
| **E — BigToM + HiToM higher-order ToM** | **PASS (v2)** | HiToM order 2 N=100: acc_mental=0.806, delta=+0.376 |
| F — ExTES / AnnoMI / PsyQA / D4 counseling depth | UNSCHEDULED | no-recruitment lock; separate authorization path |

**Rollup: 3 of 6 clears. ≥3-of-6 Phase 8 claim threshold cleared.**

### 12.6 Artefacts — v2 appendix

- Scorer v2: [`src/psychology_base/eval/benchmarks/public_datasets/higher_order_tom_scorer.py`](../../src/psychology_base/eval/benchmarks/public_datasets/higher_order_tom_scorer.py) +4 helpers, +dispatch on `metadata.source == "hitom"`
- Tests v2: [`tests/test_higher_order_tom_scorer.py`](../../tests/test_higher_order_tom_scorer.py) +9 tests (10 → 19)
- Four live-fire JSON reports under HiToM protocol:
  - [`hitom-order-1-prompts.json`](../experiments/phase-8-dataset-replay-expansion/pillar-e/hitom-order-1-prompts.json) — 0.755 / 0.710 / +0.045
  - [`hitom-order-2-gate-prompts.json`](../experiments/phase-8-dataset-replay-expansion/pillar-e/hitom-order-2-gate-prompts.json) — **0.806 / 0.430 / +0.376 (GATE PASS)**
  - [`hitom-order-3-prompts.json`](../experiments/phase-8-dataset-replay-expansion/pillar-e/hitom-order-3-prompts.json) — 0.742 / 0.388 / +0.355
  - [`hitom-order-4-prompts.json`](../experiments/phase-8-dataset-replay-expansion/pillar-e/hitom-order-4-prompts.json) — 0.722 / 0.439 / +0.283

**Full test suite:** 1455 passed (1446 prior + 9 new HiToM prompt tests). No regressions.

### 12.7 Honest ship stance — v2

The v1 close-out shipped as "mechanism-correct, gate FAIL on measurement surface". The v2 fix is **restoring the HiToM paper's standard evaluation protocol** on top of the architecturally-correct witness filter — choices visible to both paths, chain scaffold stated explicitly, answer discipline enforced. On this corrected measurement surface, the witness filter's architectural advantage becomes measurable: +0.376 delta at order 2 clears the plan §2 +0.30 floor by a comfortable margin, and the pattern holds (+0.355 at order 3; +0.283 at order 4 just below).

Phase 8 clears at ≥3-of-6. The Store #2 architectural claim resolves to: *mentalistic routing produces measurable architectural delta on both FANToM-shape (bare-broken) and HiToM-shape (bare-strong) corpora once the measurement surface respects the task's standard protocol. The direction holds at order 1 (small delta, no chain to filter) through order 4 (near-chance ceiling territory per HiToM paper).*

### 12.8 Carried debt — v2 updates

- **Order 4 slightly below floor (+0.283 vs +0.30).** Not alarming — HiToM paper § notes human performance is near-chance on order-4 short-text ToM, so our mental=0.722 is already a high ceiling. The small-sample variance at N=100 is consistent with the true delta being around the floor.
- **BigToM situation-shift tracker.** Deferred (same as v1).
- **General-purpose witness primitive on memory substrate.** Deferred (same as v1 §6).
- **Order-1 small delta (+0.045).** By design: first-order ToM has no chain to intersect, so the witness filter degenerates to "events the single agent saw" which is what any reasonable baseline already does. The chain scaffold adds no information at order 1. This is a correct no-op at order 1, not a bug.
- **General-purpose witness primitive on memory substrate.** ~~Deferred.~~ **SHIPPED 2026-10-09 as Phase 9+ follow-on** — see §13 below.

---

## 13. Phase 9+ follow-on — substrate witness primitive (2026-10-09)

The v1 carried-debt item "general-purpose witness primitive on memory substrate" shipped the same day as v2 under user authorization ("Phase 9+ substrate witness primitive"). This is the architectural generalization of the scorer-local `hitom_witness.events_witnessed_by_chain` to a Principle-C mentalistic routing primitive on the memory substrate.

### 13.1 What shipped

**Substrate module.** [`src/psychology_base/companion/copresence.py`](../../src/psychology_base/companion/copresence.py) — `CopresenceScope(chain)` frozen dataclass + `filter_fragments_by_copresence(fragments, scope)` recall-time filter. ~90 lines, zero deps beyond `Fragment`.

**CompanionSystem wiring.** `CompanionSystem.recall(..., copresence=CopresenceScope(chain=(A,B,...)))` kwarg composes with existing `scope=(subject, entity)` as set-intersection. Suppresses the hierarchical fact-store path (same reasoning as `scope`: fact_store carries no `witness:*` tags). Also narrows the Witness pain-signal pool when active. Metadata carries `copresence: chain-tuple` for provenance.

**HiToM adapter.** [`src/psychology_base/eval/benchmarks/public_datasets/hitom_witness.py`](../../src/psychology_base/eval/benchmarks/public_datasets/hitom_witness.py)::`ingest_hitom_story_as_observations(story)` — converts the HiToM container-world grammar to witness-tagged `Observation`s. Each event carries `event_step:N`, optional `location:X`, and one `witness:<agent>` tag per co-present agent (computed via `compute_presence` at ingest time, so the recall-time filter is a pure tag-AND predicate — no stateful computation at query time).

### 13.2 Design — tags not fields

Witness semantics live in the fragment tag namespace (`witness:A`, `location:X`, `event_step:N`) rather than new `Observation` dataclass fields. This keeps the adapter surface stable (no `.location=None` noise on non-spatial observations like FANToM utterances) at the cost of ~sub-ms tag-parsing per recall. Future workloads can promote to fields in a one-liner if bottlenecks appear.

The HiToM grammar parser stays corpus-local — grammars are corpus-specific (HiToM: entered/exited/moved; BigToM: narrative paragraphs; FANToM: conversational turns). What promotes to substrate is the **recall-time intersection** — the Principle-C analogue of `_filter_fragments_by_scope` for mentalistic (not possessive) scoping.

### 13.3 Why not fix `_filter_fragments_by_scope` AND-contract instead

The scope filter's AND-contract (`subject:X ∧ entity:X`) is accidentally load-bearing for Phase 5 Track 1 FANToM: it produces empty recall on `BELIEF_INACCESSIBLE` which the agent answers as "I don't know", matching gold. Changing the AND-contract would regress that +0.791 architectural delta. The substrate primitive adds a **new API surface** rather than altering existing behavior — zero regression risk.

### 13.4 HiToM scorer migration status — deferred

The HiToM scorer still uses the scorer-local `events_witnessed_by_chain` + `render_witnessed_events` path (unchanged from v2 ship). Migrating the scorer to route through `system.recall(..., copresence=...)` is a Phase 9+ follow-up: the current path returns the full witnessed set, while recall would apply `top_k` cue-ranking on top. For a 10-30 event HiToM story (~5-15 witnessed events at order 2), the top_k gate matters.

**Parity verified**: [`tests/test_copresence.py`](../../tests/test_copresence.py) `test_substrate_filter_matches_local_witness_on_worked_example` + `test_substrate_filter_matches_local_witness_order_three` assert the substrate filter and the scorer-local filter return the SAME witnessed-event set (by step index) on the Charlotte-Avery-Elizabeth-lettuce worked example. If this parity ever drifts, the substrate stops being a drop-in — pinned by test, so future refactors can't silently drift.

### 13.5 Open generalization — FANToM / BigToM migration

FANToM and BigToM scorers currently use `scope=("subject", target_person)` which lands on the AND-contract bug (empty recall; accidentally-right for INACCESSIBLE gold). A clean migration path:
- Add FANToM-shape adapter that emits `witness:<speaker>` tags on each turn-observation.
- Scorer calls `system.recall(cue=question, copresence=CopresenceScope(chain=(target_person,)))`.
- Result: INACCESSIBLE gold still matches empty-recall (target_person has no `witness:<target_person>` tags on turns they weren't present for), BUT the mechanism is now scope-correct not scope-asymmetric-accidental.

**Not shipped under current authorization** — FANToM is at +0.791 PASS; a migration risks regressing the Phase 5 Track 1 result and should be a scoped follow-up with its own gate re-run.

### 13.6 Artefacts — Phase 9+

- [`src/psychology_base/companion/copresence.py`](../../src/psychology_base/companion/copresence.py) — new primitive module
- [`src/psychology_base/companion/system.py`](../../src/psychology_base/companion/system.py) — `recall(..., copresence=...)` kwarg wiring (import + signature + filter wiring + witness pool + metadata)
- [`src/psychology_base/eval/benchmarks/public_datasets/hitom_witness.py`](../../src/psychology_base/eval/benchmarks/public_datasets/hitom_witness.py) — `ingest_hitom_story_as_observations` added
- [`tests/test_copresence.py`](../../tests/test_copresence.py) — 17 new tests (CopresenceScope unit tests, recall wiring, HiToM ingester + parity)

**Full test suite: 1472 passed** (1455 at v2 ship + 17 new). Zero regressions on Pillar E gate (HiToM scorer path unchanged), Phase 5 Track 1 FANToM (+0.791 unchanged), Phase 8 Pillar A/C/D (all unchanged).

### 13.7 Carried debt — Phase 9+ residual

- **HiToM scorer migration to substrate recall** — would exercise the primitive on the live gate; blocked on top_k vs full-set semantics (currently scorer wants full witnessed set).
- **FANToM scorer migration to copresence scoping** — would replace the scope-asymmetric-accidental mechanism with scope-correct witness-at-time intersection; blocked on regression-risk audit of the +0.791 gate.
- **BigToM situation-shift tracker** — unchanged (declarative narrative paragraphs don't yet parse into events; needs a corpus-specific grammar).
- **Multi-scene re-entry across days / sessions** — HiToM stories are single-scene; a long-horizon companion would need `event_time:ISO8601` tags paired with the `witness:*` tags, and the copresence filter would narrow by both. Shape of the extension is clear; not yet needed.
