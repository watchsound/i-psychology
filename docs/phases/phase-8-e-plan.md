# Phase 8 Pillar E — BigToM + HiToM (higher-order mentalistic generalization)

**Date drafted:** 2026-10-09
**Status:** PLAN. Shipping in sequence under Auto Mode as the deciding fourth pillar after A ✓ / B ⊡ honest-partial / C ✗ / D ✓ left the Phase 8 rollup at **2 of 6** (needs ≥ 3).
**Parent:** [phase-8-dataset-replay-expansion-plan.md §Pillar E](phase-8-dataset-replay-expansion-plan.md)
**Reuse targets:**
- Phase 5 Track 1 FANToM scorer substrate ([fantom_tom_scorer.py](../../src/psychology_base/eval/benchmarks/public_datasets/fantom_tom_scorer.py), [phase5_fantom_tom_run.py](../../benchmarks/scripts/phase5_fantom_tom_run.py)) — same mentalistic-vs-bare paired path, same `LLMFANToMJudge`, same fresh-`CompanionSystem`-per-conversation discipline.
- Pillar B/D CLI runner shape ([phase8_b_patient_psi_run.py](../../benchmarks/scripts/phase8_b_patient_psi_run.py), [phase8_d_truthfulqa_run.py](../../benchmarks/scripts/phase8_d_truthfulqa_run.py)) — `--mode {local,cloud} --tier {simple,advanced} --responder {bare,mentalistic,both}` surface.
- 3-mirror SHA256-pinned provisioner pattern ([provision_fantom.py](../../benchmarks/scripts/provision_fantom.py), [provision_patient_psi.py](../../benchmarks/scripts/provision_patient_psi.py), [provision_truthful_qa.py](../../benchmarks/scripts/provision_truthful_qa.py)).

---

## 1. Why this pillar

Phase 5 Track 1 FANToM landed `accuracy_mentalistic = 0.952` vs `accuracy_bare_llm = 0.161` → **architectural delta +0.791** on BELIEF_INACCESSIBLE at N=70 ([project_phase_5_fantom_track1_passed.md](../../memory/project_phase_5_fantom_track1_passed.md)). That was a **first-order** result on a specific question type: "person A was absent; what do they believe about X?" — the model demonstrated that `scope=("subject", target_person)` recall at the memory substrate layer produces an accuracy win over feeding the raw transcript to the same LLM.

Pillar E tests whether that architectural delta **generalizes** on two axes neither FANToM nor Phase 5 Track 2 touched:
- **Belief order.** BigToM (Gandhi et al. 2024, MIT) and HiToM (Wu et al. 2023, MIT) include higher-order items — "A believes that B believes that C holds X" at orders 2, 3, and 4. A positive delta at order 2 says the mentalistic routing scales past "who was in the room"; a flat delta says the architecture is first-order-only and the Store #2 claim softens accordingly.
- **Situation model.** BigToM varies the *situation* behind the belief (e.g., same utterance pair with the physical setup swapped), isolating whether the model is tracking *beliefs* or *lexical cues*. If mentalistic beats bare on BigToM, the delta is not an artifact of surface-form routing.

A pass here converts the FANToM result from "architectural delta on one hard first-order case" into "architectural delta that scales with ToM depth." A fail at order 2 is a legitimate honest ship stance: the Store #2 claim gets explicitly scoped to first-order.

**Phase 8 rollup leverage.** With A ✓ and D ✓ already in hand, Pillar E PASS → **A + D + E = 3 / 6 → Phase 8 clears**. Pillar E is the single highest-leverage remaining shot absent user-side action to unlock Pillar B's 106-profile gated sample.

---

## 2. Hard gate

```
accuracy_mentalistic(order=2) − accuracy_bare_llm(order=2) ≥ +0.30
```

Measured on **HiToM order 2** (second-order belief items). Both arms share identical responder (same LLM, same temperature, same prompt shell, same `LLMFANToMJudge`); the only difference is whether the question is preceded by a mentalistic-recall block or by the raw context, exactly as in Phase 5 Track 1.

**Why +0.30 and not +0.79.** Phase 5 Track 1's +0.791 was on the hardest-cued first-order case (BELIEF_INACCESSIBLE), where the mentalistic route literally hides from the model what the target person did not witness. At order 2 the win can only propagate through TWO layers of scoping — "what does A think about what B believes" — and part of the signal is going to come from the bare LLM getting *some* order-2 answers right from context alone. A 2.6× falloff from +0.791 to **+0.30** is the honest expected floor; higher would surprise us upward (celebrate), lower is the measured ceiling.

**Floor on absolute accuracy.** `accuracy_mentalistic(order=2) ≥ 0.40` is additionally required, same floor Phase 5 §2 kill-criterion imposed on FANToM. A +0.30 delta over a 0.10-vs-0.40 pair is architecturally interesting but the raw mentalistic number has to clear 0.40 for the result to be a competence claim rather than a lesser-of-two-floors.

**Minimum N.** `min_items = 50` on BOTH paths at the chosen subset (same Phase 5 §2 convention). A reader cannot sneak a passing delta from a tiny imbalanced bucket.

---

## 3. Soft criteria

- **Order 3 generalization.** `accuracy_mentalistic(order=3) − accuracy_bare_llm(order=3) ≥ +0.10` — nice-to-have; a flat delta here is still an honest ship with the claim scoped to order ≤ 2.
- **Order 4 is open research.** Report the number; no gate. Human performance on order-4 ToM is itself close to chance on short texts (per the HiToM paper); a positive signal would be noteworthy.
- **BigToM situation-shift sanity.** For item pairs where BigToM varies the physical setup, report whether the mentalistic path's answer flips with the situation (correct behavior) vs. stays locked on surface lexical features (bare-pattern behavior).
- **Rule 1a corrects → hedges reshuffle.** Report the pattern (same lens as Pillar D / Pillar B). Not expected to apply strongly here — ToM questions are factual, not sycophancy-pressure — but worth tracking.

---

## 4. Scope — what ships

### 4.1 Provisioning

- `benchmarks/scripts/provision_bigtom.py`
  - Target: `cicl-stanford/procedural-evals-tom` (BigToM official release) GitHub + HF mirror.
  - 3-mirror SHA256-pinned (jsDelivr → ghproxy → raw.github OR HF datasets `cicl-stanford/BigToM`).
  - Shim: `.data/bigtom/items.jsonl` — one row per item with `item_id, order, scenario_type, context, question, correct_answer, incorrect_answer, situation_variant_id, source_split`.
- `benchmarks/scripts/provision_hitom.py`
  - Target: HiToM paper authors' GitHub release OR HF mirror (`ShenzhiWu/hi-tom` if present; else resolve during impl step 1).
  - Same 3-mirror SHA256 pattern.
  - Shim: `.data/hitom/items.jsonl` — one row per item with `item_id, order (1|2|3|4), story, question, options, correct_option, agent_chain`.

Both shims gitignored under `.data/`; MIT license-clean on the derivatives.

### 4.2 Adapters

- `src/psychology_base/eval/benchmarks/public_datasets/bigtom.py`
  - `BigToMItem(item_id, order, scenario_type, context, question, correct_answer, incorrect_answer, situation_variant_id)` frozen dataclass.
  - `BigToMOrder` enum: `ORDER_1 / ORDER_2`.
  - `BigToMScenarioType` enum: `FORWARD_BELIEF / BACKWARD_BELIEF / FORWARD_ACTION / BACKWARD_ACTION` per the paper's four conditions.
  - `iter_bigtom_items(shim_path, *, limit, order=None, scenario_type=None) -> Iterator[BigToMItem]`.
  - `bigtom_item_to_fantom_shape(item) -> FANToMItem`-equivalent — reuses existing `FANToMQuestion` + `FANToMContextTurn` dataclasses by projection, so the Pillar E scorer can call `score_fantom_tom` directly against BigToM items without a parallel verdict pool. (Design decision: adapter projects → FANToM shape, scorer stays shared.)

- `src/psychology_base/eval/benchmarks/public_datasets/hitom.py`
  - `HiToMItem(item_id, order, story, question, options, correct_option, agent_chain)` frozen dataclass.
  - `HiToMOrder` enum: `ORDER_1 / ORDER_2 / ORDER_3 / ORDER_4`.
  - `iter_hitom_items(shim_path, *, limit, order=None) -> Iterator[HiToMItem]`.
  - `hitom_item_to_fantom_shape(item) -> FANToMItem`-equivalent — same projection pattern.
  - `agent_chain` field preserves the nested target structure `[A, B, C]` for order-3 ("A believes that B believes that C holds X"); the scorer's scope selector uses the full chain (see §4.4).

### 4.3 Scope selector — the one architectural decision

Phase 5 Track 1 FANToM used `scope=("subject", target_person)` — a flat single-subject recall. HiToM order 2+ asks "what does A believe about what B believes about X" — the mentalistic path has to filter observations by **both** A's access AND B's access. There are two defensible routings:

- **Option S-flat** (default — minimal change): compose scope as `("subject", A)` and feed B's name into the prompt as the target-of-reasoning, letting the LLM collapse the nested access. Preserves the Store #2 single-subject invariant; delegates nesting to the LLM. Expected: delta attenuates with order but stays positive.
- **Option S-chained** (new — one architectural extension): compose scope as `("subject_chain", (A, B))` where recall filters observations by `witnessed_by_all(A, B)`. Requires one new method `CompanionSystem.recall(scope=("subject_chain", chain))` or an extension of the existing recall path. Preserves architectural purity; higher expected delta at order 2+.

**Default: Option S-flat for Pillar E ship.** Rationale: minimum architectural change, same substrate as Phase 5 Track 1, cleaner cross-pillar comparison. Option S-chained is a Phase 8+ or Pillar E-prime extension if S-flat delta falls below +0.30 at order 2 (named carried-debt item, not retrofit).

**Why not pick S-chained first.** The Phase 5 Track 1 ship stance was "the architectural contribution is the single-subject access filter". Jumping straight to multi-subject chained filtering mixes two claims in one gate — a pass could be the chain filter working OR the single-subject filter being enough at order 2. S-flat first isolates the latter; S-chained if needed makes a cleaner marginal claim.

### 4.4 Scorer

- `src/psychology_base/eval/benchmarks/public_datasets/higher_order_tom_scorer.py`
  - Thin wrapper over the existing [score_fantom_tom](../../src/psychology_base/eval/benchmarks/public_datasets/fantom_tom_scorer.py). Projects `BigToMItem` / `HiToMItem` into `FANToMItem`-shaped records (adapter method from §4.2), then calls `score_fantom_tom`. Report subclass `HigherOrderToMReport(FANToMToMReport)` with order-stratified accessors.
  - `HigherOrderToMReport.accuracy(path, *, order=None, scenario_type=None, drop_undecidable=True) -> float` — per-order and per-scenario subsetting.
  - `HigherOrderToMReport.architectural_delta(order=None) -> float` — `accuracy(mentalistic, order=o) − accuracy(bare, order=o)`.
  - `HigherOrderToMReport.meets_pillar_e_gate(*, delta_floor=0.30, accuracy_floor=0.40, min_items=50, order=HiToMOrder.ORDER_2) -> bool` — encodes §2.
  - **Mutex kwarg contract (same as Pillar D):** `responder` XOR `item_responder` on `score_higher_order_tom`; former applies the same callable to every item, latter supplies per-item responders (used by the paired bare/mentalistic runner).

### 4.5 CLI runner

- `benchmarks/scripts/phase8_e_higher_order_tom_run.py`
  - Args: `--dataset {bigtom,hitom,both}`, `--order {1,2,3,4,all}`, `--limit`, `--mode {local,cloud}`, `--tier {simple,advanced}`, `--responder {bare,mentalistic,both}` (default `both` — paired), `--judge {lexical,llm}` (default `llm` — ToM semantic match is beyond lexical reliability; same stance as Phase 5 Track 1 FANToM live-fire), `--seed 42`.
  - Fresh `CompanionSystem` per conversation (FANToM discipline — same `system_factory` pattern).
  - Routing: default `--mode cloud --tier simple` → tongyi/qwen-turbo with cloud fallback on Ollama unavailability (same reason-tag pattern as Pillar D).
  - Output: `docs/experiments/phase-8-dataset-replay-expansion/pillar-e/higher-order-tom.json` with both-arm reports + per-order accuracy table + delta table.

### 4.6 Tests

- `tests/test_bigtom_adapter.py` — shim round-trip, order/scenario field preservation, projection to `FANToMItem` shape, deterministic iteration.
- `tests/test_hitom_adapter.py` — shim round-trip, order routing, `agent_chain` preservation, projection to `FANToMItem` shape.
- `tests/test_higher_order_tom_scorer.py` — stub-labeler end-to-end on both datasets, per-order delta semantics, `meets_pillar_e_gate` boundary cases (delta exactly at 0.30 / exactly below floor / insufficient N on target order), mutex-kwarg contract, scope-selector option S-flat verified byte-identical to Phase 5 Track 1 single-subject routing on first-order items.
- `tests/test_scope_selector_s_flat.py` — one small regression pin that Pillar E ship uses S-flat, not S-chained (named architectural choice; test fails if S-chained accidentally ships).

### 4.7 Live-fire + close-out

1. `phase8_e_higher_order_tom_run.py --dataset bigtom --order 1 --limit 100 --responder both --judge llm --tier simple` — baseline sanity (should replicate Phase 5 Track 1-ish delta on first-order items).
2. `phase8_e_higher_order_tom_run.py --dataset hitom --order 2 --limit 100 --responder both --judge llm --tier simple` — **the gate run**.
3. `phase8_e_higher_order_tom_run.py --dataset hitom --order 3 --limit 100 --responder both --judge llm --tier simple` — soft criterion.
4. `phase8_e_higher_order_tom_run.py --dataset hitom --order 4 --limit 50 --responder both --judge llm --tier simple` — open-research report.
5. Write `docs/phases/phase-8-e-close-out.md` with per-order breakdown + gate verdict + Phase 8 rollup update.
6. Update memory + MEMORY.md index entry.

Expected total cloud cost: ~400 items × 2 arms = ~800 round-trips at `--tier simple`; comparable to Pillar D's TruthfulQA 817 run (~34 min cloud time cache-cold, ~5 min cache-warm).

---

## 5. Explicit non-scope

- **Option S-chained multi-subject chain recall.** Named extension if S-flat under-delivers at order 2; not shipping in Pillar E v1.
- **Fine-tuning or ToM-specific prompt engineering.** The bare and mentalistic arms share the same prompt shell as Phase 5 Track 1; no per-dataset prompt tuning.
- **Narrative-shift evaluation.** BigToM's forward/backward-action split is reported but not gated; this is Phase 8+ or Pillar E-prime.
- **Bilingual coverage.** Both datasets are EN-only. ZH ToM is Pillar F's territory (D4 counseling has some implicit ToM via family-conflict scenes but is scored on Bond / EMPATH, not ToM).
- **Order-5+ or recursive beliefs.** HiToM caps at order 4; nothing to do here.
- **LLM-judge calibration against human raters.** Reuse `LLMFANToMJudge` as-is; cross-rater κ calibration is Phase 8+ carried debt if the ship stance later needs it.

---

## 6. Risks

- **S-flat scope selector is insufficient at order 2.** The architectural delta attenuates faster than the +0.30 floor allows. Mitigation: honest FAIL ship with S-chained called out as the clean Phase 8+ follow-up; the Store #2 claim scopes to "single-subject access filter at order 1". Zero retrofit.
- **BigToM situation-model items leak into bare LLM via pretraining.** 2026-era `qwen-turbo` has likely seen BigToM items in training; bare accuracy may be unexpectedly high, compressing the measurable delta. Mitigation: report per-item provenance where known; if bare clears 0.80 on order 2, flag as "measured delta may underestimate architectural contribution due to pretraining contamination" — this is the same honest-ship stance as TruthfulQA on Pillar D.
- **LLM-judge mis-labels on second-order answers.** Correct order-2 answers often phrase as "A would think B doesn't know Y" — the judge may collapse this to "wrong" if it fails to parse the nested structure. Mitigation: `--judge llm` default on live-fire; stub lexical judge for CI only; the `max_undecidable_rate` kwarg stays off (1.0) for the same reason FANToM BELIEF_INACCESSIBLE kept it off — undecidable on nested-belief gold is a judge-labeling artifact, not pathology.
- **HiToM pool is small per order.** The HiToM paper reports ~6k items total across orders 1-4; at order 2 the per-order N may run tight to the `min_items=50` floor. Mitigation: provisioner reports per-order counts up front; if order 2 < 50 items after license-clean filtering, surface before live-fire so user calls the shot (ship as "attempted insufficient" vs. go to Pillar F for the deciding slot).
- **Shim-schema drift from original HF release.** HiToM has multiple published variants; the shim should pin to the one the paper reports results on. Mitigation: pin SHA256 at provisioning time; verification test asserts the first 10 item IDs match a known golden set.
- **Phase 8 rollup over-weighting.** A pass here clears Phase 8 trivially (A + D + E = 3 / 6). The rollup clearance should NOT be interpreted as "the architectural claim is validated across all six carried-debt items." Three remaining pillars (B at honest-partial, C at FAIL, F unscheduled) remain carried debt. Close-out doc must state this explicitly to avoid overclaiming.

---

## 7. Execution order

1. **Investigate** BigToM + HiToM release URLs, license verification, HF-mirror availability. Decision point: open MIT release confirmed for both.
2. **Provision** both shims (JSONL) with 3-mirror SHA256 pin.
3. **Build + test** BigToM adapter + FANToM-shape projection.
4. **Build + test** HiToM adapter + FANToM-shape projection.
5. **Build + test** `higher_order_tom_scorer.py` thin-wrapper + `meets_pillar_e_gate`.
6. **Build + test** scope-selector S-flat regression pin.
7. **CLI runner** + wiring.
8. **Full test suite** green (1368 → 1368 + N; expected N ~ 25-35 new tests).
9. **Dry-run** 5-item live-fire for sanity on each dataset.
10. **Full live-fire** paired bare / mentalistic at `--dataset hitom --order 2 --limit 100` (gate run) + supporting runs for order 1 / 3 / 4 + BigToM.
11. **Close-out doc** + memory + MEMORY.md index + Phase 8 rollup update.

---

## 8. Pass criteria interpretation

- `hard_gate_pass = True` → **Pillar E PASS**, Phase 8 rollup = A + D + E = **3 / 6** → **Phase 8 clears**. SOTA claim shape (per [sota-goal.md §Axis 3](../sota-goal.md)) gets one additional supporting line: *higher-order ToM delta +0.30 at order 2 under single-subject scope recall*.
- `delta ≥ +0.30` on order 2 but `accuracy_mentalistic < 0.40` → **Pillar E honest-partial** (relative-win-at-low-floor). Rollup contribution noted with scope caveat; not counted as clean PASS.
- `delta < +0.30` on order 2 with `delta ≥ +0.10` on order 2 → **Pillar E FAIL on hard gate; S-flat ceiling named**. Honest ship: "architectural delta attenuates past first-order under single-subject scope; S-chained is the clean Phase 8+ follow-up." Rollup stays at 2 / 6; next-deciding is Pillar F.
- `delta < +0.10` at order 2 → **Pillar E hard-FAIL**. Store #2 claim rescoped to order 1 only. Rollup stays at 2 / 6; honest ship; no retrofit.
- Order 2 N < 50 after license-clean filtering → **Pillar E pre-fire halt**, same stance as Pillar B §9: no gate verdict, flagged as plumbing debt. Shift deciding-slot to Pillar F.

---

## 9. Why Pillar E over Pillar F for the deciding slot

Both pillars are candidate deciders. The selection logic (reproduced here so a reader doesn't have to reconstruct it):

| | Pillar E (BigToM + HiToM) | Pillar F (ExTES + AnnoMI + PsyQA + D4) |
|---|---|---|
| Dataset count | 2 (both MIT) | 4 (CC BY 4.0 / research-use / CC BY-SA 4.0) |
| Provisioning cost | Low (2 shims, same pattern) | High (4 shims, 2 ZH-specific tokenization quirks) |
| Reuse of existing substrate | Very high (FANToM scorer as-is) | Medium (ESConv WAI-SR substrate + EMPATH judge adapted per-dataset) |
| Expected engineer-time | ~1 day impl + 1 day live-fire | ~3-4 days impl + 1 day live-fire |
| Rollup math | PASS = clear Phase 8 (3/6) | PASS = clear Phase 8 (3/6) |
| Axis 3 signal richness | Mechanism-viability extension (Axis 2) | Application-outcome richness (Axis 3) — bigger SOTA claim |
| Risk of surprise FAIL | Medium (order-2 attenuation unknown) | Medium-low (ESConv Bond 4.160 precedent on EN clinical register) |

**Chose Pillar E because:** (a) lowest provisioning + impl cost per unit of rollup-clearing probability; (b) direct extension of Phase 5 Track 1's biggest architectural win rather than a new application surface; (c) if E passes, F stays as a Phase 8b candidate; if E fails, F has stronger provenance for the deciding slot because the mechanism-viability path has been bounded.

**Deferred for Phase 8b (not scope of this plan):** Pillar F as the Axis 3 counseling-depth richness pillar — contingent on Pillar E outcome.
