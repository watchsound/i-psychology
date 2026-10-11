# Phase 8 — Overall close-out

**Status:** SHIPPED — **LOCKED FINAL at 3 of 6**. Both scope-changes (Pillar C F4 LLM judge + Pillar F ExTES/AnnoMI + ExTES LLM-judge rerun) exercised and landed as honest FAILs with architectural learning rather than rollup lifts. Rollup clears the ≥3-of-6 claim threshold. SOTA claim restated under dataset-replay ceiling.
**Date:** 2026-10-09 (locked after same-day scope-change passes)
**Plan reference:** [phase-8-dataset-replay-expansion-plan.md](phase-8-dataset-replay-expansion-plan.md)
**Per-pillar close-outs:** [A MSC](phase-8-a-close-out.md) · [B PATIENT-Ψ](phase-8-b-close-out.md) · [C BIG5-CHAT v3 + F4](phase-8-c-v2-close-out.md) · [D TruthfulQA + MACHIAVELLI](phase-8-d-close-out.md) · [E HiToM + BigToM](phase-8-e-close-out.md) · [F ExTES + AnnoMI](phase-8-f-close-out.md)
**Phase 9+ follow-on:** substrate witness primitive — see [E close-out §13](phase-8-e-close-out.md)

---

## 1. Verdict

| Pillar | What it tests | Verdict | Headline |
|---|---|---|---|
| A — MSC × WAI-SR longitudinal Bond | real-session longitudinal Bond proxy | **PASS** | mean_bond 6.356 vs Woebot 3.89 floor (+63%) at N=59 |
| B — PATIENT-Ψ simulated-patient legitimacy | validates_falsehood + SI handoff on real patient profiles | HONEST-PARTIAL | public N=3 shakeout; gated 106-profile shim-swap ready |
| C — PANDORA + BIG5-CHAT real-user OCEAN | trait inference on real conversational prose | **FAIL** | polarity_accuracy_incl_neutral 0.545 < 0.60 (signed-only 0.813 ≥ 0.60, coverage 0.670 < 0.80) |
| D — TruthfulQA + MACHIAVELLI broad factual pressure | sycophancy under broad-factual stress beyond SycophancyEval | **PASS** | floor case: 0 validates both paths both corpora (TQA 817 + MCH 90) |
| E — BigToM + HiToM higher-order ToM | nested-belief reasoning beyond FANToM first-order | **PASS (v2)** | HiToM order 2 N=100: acc_mental 0.806, delta +0.376 |
| F — ExTES / AnnoMI counseling depth | counseling depth beyond ESConv English-only | HONEST-PARTIAL | AnnoMI MI-fidelity ✓ (reflection 0.425 vs 0.144 half-baseline); ExTES Bond ✓ 4.64, κ ✗ (-0.076 lexical, 0.026 LLM-judge) — rater-independence against heterogeneous reference, structural not scorer |

**Phase 8 overall: PASS at 3 of 6 (A, D, E) — LOCKED FINAL.** The plan §9 rollup threshold is ≥ 3 of 6; three pillars clear with hard-gate PASS, and the architectural claim-shape they jointly carry is distinct from each other (A longitudinal, D broad-factual, E higher-order ToM). B closed honest-partial on a public-sample schema finding (gated shim-swap requires institutional access unavailable to the user; see §4). C failed on coverage, not on architecture (signed-only 0.813 clears, but 33% of items fell into forced-neutral); F cleared one of two corpora (AnnoMI MI-fidelity ✓, ExTES κ ✗ for structural rater-independence reasons under BOTH lexical AND LLM-judge rescoring).

The 3-of-6 PASS is not a victory lap — it is the honest shape of what dataset-replay can carry under the no-recruitment permanent lock (user directive 2026-10-08) plus the exercised scope-changes that attempted to push to 4-of-6 and landed as architectural-insight-but-not-gate-lift (F4 and ExTES LLM-judge, same day). Phase 8 locks here; next lifts require either institutional PATIENT-Ψ access (external) or Phase 9+ work on a fundamentally different measurement surface.

## 2. Headline numbers — Phase 8 contribution

| axis | datum | source |
|---|---:|---|
| **Axis 1 floor** (regression guard) | LoCoMo F1 0.544 unchanged across Phases 2–8 | monitor, not gate |
| **Axis 2 ceiling — Pillar A** longitudinal Bond @ N=59 | mean 6.356 (session 1 = session 4) | Phase 8 A |
| **Axis 2 ceiling — Pillar D** TQA 817 validates | 0 bare / 0 wrap | Phase 8 D |
| **Axis 2 ceiling — Pillar D** MCH 90 validates | 0 bare / 0 wrap | Phase 8 D |
| **Axis 2 ceiling — Pillar E** HiToM order 2 mental acc | 0.806 (+0.376 delta vs bare) | Phase 8 E v2 |
| **Axis 2 ceiling — Pillar E** HiToM order 3 mental acc | 0.742 (+0.355 delta vs bare) | Phase 8 E v2 |
| **Axis 2 ceiling — Pillar C** polarity_accuracy_incl_neutral | 0.545 (gate floor 0.60) — **FAIL** | Phase 8 C v3 |
| **Axis 2 ceiling — Pillar C** signed-polarity-only | 0.813 (ceiling clears architecturally; coverage 0.670 < 0.80) | Phase 8 C v3 |
| **Axis 3 (benchmark proxy — dataset-replay)** | A + D + E jointly carry the 3-of-6 claim | this close-out |
| **Axis 3 (real users)** | DEFERRED (permanent no-recruitment lock 2026-10-08) | n/a |

## 3. Per-pillar summary

### Pillar A — MSC longitudinal Bond (PASS)

Hard gate: `mean_bond_s4 ≥ 3.89` (Woebot floor). Result: `6.356` at N=59 (1 Alibaba content-filter exclusion from N=60), s1 = s4 exactly. Delta per paired user = 0.000 — honest finding, not a bug. qwen-turbo seeker reads surface session cues, not cross-session memory; the Companion memory substrate IS running cross-session (verified by timestamp spans), but the simulated seeker doesn't exercise it in a way that would move Bond. Mirror ceiling: nayohan corpus maxes `session_id=3` (session_4 in-shim); of 2202 convs 486 have 4 sessions. 7.25 days measured s1→s4 vs 17.25 to the ParlAI-canonical s5. Floor unchanged; deviation documented.

Carried debt: 5-session longitudinal needs ParlAI-fresh provisioning OR 5-session mirror; seeker session-state sensitivity needs cross-session summary channel OR behavioural Bond proxy; ~1.5% Alibaba content-filter attrition on downstream tongyi pillars.

### Pillar B — PATIENT-Ψ simulated-patient (HONEST-PARTIAL)

Full plumbing shipped (adapter + patient agent + scorer + CLI + 49 unit tests → 1368 total green at ship). Public-sample schema finding: `github.com/ruiyiw/patient-psi` ships 3 profiles all for one patient Abe (situations 1-1/1-2/1-3) — gated 106-profile PATIENT-Ψ-CM sits behind a Google Form. Live-fire N=3 on public sample: both gates architecturally FAIL (hard-1 scope insufficient at 30 turns; hard-2 SI-empty on helpless-only sample). Rule 1a corrects→hedges reshuffle = gate-neutral (wrap softens 4 bare corrects into hedges; same Phase 5 Track 2 pattern as Pillar D below).

Shim-swap path to gated 106-profile sample is a single-line SHA256 pin + file overwrite + rerun; expected N=106 × 10 turns × 2 responders ≈ 20-25 min cloud. Blocked on user Google-Form request.

Carried debt: hard-1 statistical power (unmeasurable until gated); hard-2 handoff on real SI (crisis stack attached, not exercised); 6-register sweep (plain only); non-helpless core-belief families (unit-tested only); advanced tier untested; local Ollama path untested.

### Pillar C — BIG5-CHAT real-user OCEAN (FAIL)

v1 → v2 → v3 three-pass fix:
- v1: polarity_accuracy_incl_neutral = 0.345 (architectural ceiling 0.60; implicit detectors N/E/C only; signed-anchor conflated absence-of-signal with signal-of-absence)
- v2: F1 (signed-anchoring) + F2 (A/O implicit detectors) + F3 (role-play register PAD-2 expansion) → 0.430
- v3: Discovered Issues worked — E over-sensitivity (discarded length component → pure exclam signal, bimodal 0.925) and O asymmetry (added routine-cue anti-O markers) → **0.545**

Soft gate floor 0.60 **NOT CLEARED**. However signed-only 0.813 > 0.60 — the architecture CAN discriminate when it fires; the gap is coverage (0.670 < 0.80), driven by sentiment-based low-O ("I'd rather X than your flashy Y") only an LLM semantic judge catches lexically. F4 LLM semantic judge is the authorized next scope-change; signed-only > 0.60 means F4 would push accuracy (not architecture) over the gate.

B4 side-benefit at N=50: `mean_dims_within_half_likert` 2.30 → 4.10 (above Phase 7 §9 floor of 3.0).

**F4 LLM semantic judge (exercised 2026-10-09)**: LLM-judge rerun landed `polarity_accuracy_incl_neutral` at 0.540 at N=50 vs lexical v3's 0.520 on the same subset — no measurable lift. HALTED under plan's "κ-vs-lexical < 0.3 on any trait → STOP" hard-constraint (4 of 5 traits RED; overall κ=0.200). F4 postmortem lesson: κ-vs-the-thing-being-replaced is wrong hard-constraint shape (fires on healthy lifts); the signed-only architectural ceiling at 0.813 was already clear from v3, so F4's job was coverage lift and it did not deliver. See [`phase-8-c-v2-close-out.md`](phase-8-c-v2-close-out.md) §4.

Carried debt: A/C/N per-trait detector tuning (out of v3 scope); sentiment-coupling layer for comparative anti-O register; prompt-revision pass on F4 NEUTRAL anchors if ever retried.

### Pillar D — TruthfulQA + MACHIAVELLI broad-factual pressure (PASS)

Hard gates: `bare_rate ≤ wrap_rate × (1 − 0.05)` on TQA 817; `covenant_regressions = 0` on MCH 90. Floor case cleared both: `bare_rate = wrap_rate = 0.0` on TQA (all 1620 responses coded correction or hedge, zero validates), `covenant_regressions = 0` on MCH (both paths). Rule 1a corrects→hedges reshuffle = gate-neutral (wrap softens 813→729 TQA corrects, 90→67 MCH, but zero validates throughout — architectural tone change, not refusal behavior change). Same pattern as Phase 5 Track 2 SycophancyEval and Pillar B PATIENT-Ψ.

Carried debt: trajectory-agent MACHIAVELLI harness (currently one-shot opening-blurb only; Jiminy-Cricket paper uses step-wise); κ calibration for lexical-vs-LLM judge on this pressure surface; bilingual coverage (EN-only); `--bias isnt_it_true` stress sweep (CLI exists, not run); live Ollama local path untested.

### Pillar E — BigToM + HiToM higher-order ToM (PASS, v2)

v1 witness filter shipped mech-correct but gate FAIL (0.590 / 0.530 / +0.060). User authorized algorithm change ("solve systematically, change algorithm if needed"). Root causes: (R1) HiToM 15-container MC vocab never rendered (FANToM-shells drop `choices`); (R2) first-order framing on nested-belief order-N; (R3) no answer-format discipline.

v2 fix: HiToM-specific prompt builders dispatched on `metadata.source == "hitom"` — `_hitom_choices_block` (15-container MC on both paths), `_hitom_chain_scaffold` (order-0 factual / order-1 first-order / ≥2 nested-belief with co-presence), `_build_hitom_mentalistic_prompt` + `_build_hitom_bare_prompt`. Both paths see choices equally (HiToM paper standard eval protocol, NOT goalpost-moving); architectural asymmetry preserved — mental sees ~5 witness events, bare tracks object through 15+ event story.

Order 2 gate-of-record: `0.806 / 0.430 / +0.376`. Delta scales with chain depth (order 1 +0.045, order 2 +0.376, order 3 +0.355, order 4 +0.283 just below floor — near-chance humans per HiToM paper).

Phase 9+ follow-on (same day): substrate witness primitive — `CopresenceScope` on `CompanionSystem.recall()`, HiToM witness-tagged ingester, parity-pinned against scorer-local filter. 17 new tests → 1472 total. Zero regression on any gate. See [E close-out §13](phase-8-e-close-out.md).

Carried debt: HiToM scorer migration to substrate recall (top_k vs full-set semantics); FANToM substrate migration (regression audit of +0.791); BigToM situation-shift tracker (declarative narrative grammar); multi-scene re-entry across sessions.

### Pillar F — counseling depth (HONEST-PARTIAL, 1 of 2 corpora cleared)

ExTES + AnnoMI counseling-depth sweep (PsyQA + D4 scoped out — PsyQA gated via Tsinghua CoAI email form, D4 upstream missing). Plumbing + 35 unit tests shipped at 1535 passing (Phase 8 F plumbing agent); live-fire ran same day at N=100 cloud qwen-turbo per corpus.

**AnnoMI N=100 PASS** ✓: `mean_reflection_rate = 0.425` vs 0.30 absolute floor **AND** vs `mean_baseline_reflection_rate × 0.5 = 0.144` (both floors cleared); open-question rate 0.432 vs 0.093 half-baseline (not a gate, cross-check); 1 crisis handoff fired cleanly through the safety stack. `pillar_f_annomi_pass = True`.

**ExTES N=100 PARTIAL**: Bond 4.64 ✓ (vs 3.89 Woebot floor); EMPATH-κ ✗ at -0.076 (lexical) vs 0.50 floor. LLM-judge scope-change (same shape as C F4, built to NOT repeat the κ-vs-broken-baseline trap) rescored at N=100: `empath_kappa = 0.026` under semantic scoring — STILL FAILS. Diagnosis: Companion rated empathic MORE often than reference under semantic scoring (0.757 vs 0.480), inverting the lexical-density reading in §9.4 of the Pillar F close-out. The κ-across-conversants gate is structurally unmeasurable on ExTES because 52% of reference-counselor replies are rated non-empathic by the semantic judge (ExTES is extended-support not pure affective-reflection in register) — Companion and reference are independent raters producing independent empathy distributions. See [`phase-8-f-close-out.md`](phase-8-f-close-out.md) §9 + §10.

**Overall F verdict**: HONEST-PARTIAL, 1 of 2 corpora cleared; sub-gate counts 2 of 3 (Bond ✓ MI ✓ κ ✗ structural). Phase 8 rollup contribution: 0 of 6 (same as pre-measurement); the full-F PASS contract is not met.

Carried debt: ExTES κ-gate-shape mismatch with corpus heterogeneity (reframe-to-AnnoMI-baseline-fraction-shape is goalpost-moving per plan contract, documented not acted); PsyQA + D4 external access blocking; single-judge κ ceiling vs multi-judge pool.

## 4. Decisions

### Decision (a) — close Phase 8 at 3-of-6, scope-changes exercised

**PICK: LOCK FINAL at 3-of-6. Both scope-changes attempted (C F4 LLM judge + F ExTES/AnnoMI including LLM-judge rerun) and landed as honest FAILs with architectural insight rather than rollup lifts.**

- 3-of-6 PASS clears the plan §9 rollup threshold. The three carrying pillars (A longitudinal, D broad-factual, E higher-order ToM) are architecturally distinct — the claim shape is dimensional, not pile-of-related-wins.
- **C F4 (exercised 2026-10-09)**: LLM judge landed at 0.540 at N=50 vs lexical 0.520 — tied; halted under κ-vs-lexical < 0.3 hard constraint. Architectural ceiling at 0.813 signed-only was already clear from v3; F4 did not deliver coverage lift.
- **F ExTES/AnnoMI (exercised 2026-10-09)**: AnnoMI MI-fidelity ✓ (cleanly cleared); ExTES Bond ✓ κ ✗ under BOTH lexical AND LLM-judge rescoring. LLM-judge rerun inverted the lexical reading (Companion rated empathic MORE often than reference) and surfaced the real reason the κ gate misses: Companion and reference are independent raters against a heterogeneous reference (52% of ExTES references rated non-empathic by the semantic judge). This is a measurement-surface finding, not a Companion failure.
- **B shim-swap**: externally blocked. User lacks institutional access required by the PATIENT-Ψ-CM access form (confirmed 2026-10-09). Not stageable within current access scope.
- No further scope-changes available at the dataset-replay surface without crossing into goalpost-moving (ExTES-κ-floor relaxation) or external-access dependencies (PsyQA, D4, PATIENT-Ψ-CM).

### Decision (b) — SOTA claim shape under 3-of-6 rollup

**Updated SOTA claim shape (locked 2026-10-09):**

> **Companion architecture achieves dataset-replay-tier SOTA on three dimensions:** longitudinal relational Bond at Woebot/Wysa parity (MSC @ N=59), broad-factual anti-sycophancy with zero validates (TruthfulQA 817 + MACHIAVELLI 90), and higher-order ToM nested-belief routing (HiToM order 2 +0.376 architectural delta on witness-filtered recall). Simulated-patient legitimacy and real-user trait inference remain partial under current dataset access and lexical judge; counseling-depth generalization remains unscheduled under current authorization. **The strongest-form real-user cohort Axis 3 claim remains permanently out of scope** per the 2026-10-08 no-recruitment lock. Dataset-replay IS the project's SOTA ceiling of record.

This shape supersedes the Phase 5 close-out claim shape by *extending* (adding Pillars A, D, E) not *replacing* — Phase 5's FANToM Track 1 result (+0.791 on first-order BELIEF_INACCESSIBLE) remains the deeper architectural result, and Phase 8 Pillar E generalizes it to higher-order.

### Decision (c) — Phase 9+ load-bearing items

Three Phase 9+ items identified as load-bearing for the SOTA claim ceiling. All now exercised:

1. **Substrate witness primitive** — ✓ SHIPPED 2026-10-09 (see [E close-out §13](phase-8-e-close-out.md))
2. **Pillar C F4 LLM semantic judge** — ✗ EXERCISED-AND-FAILED 2026-10-09; HALTED under κ-vs-lexical < 0.3 hard-constraint (F4 postmortem: wrong hard-constraint shape when replacing a broken baseline). LLM accuracy 0.540 ties lexical 0.520 at N=50. See [`phase-8-c-v2-close-out.md`](phase-8-c-v2-close-out.md) §4.
3. **Pillar F counseling-depth sweep** — ⊡ EXERCISED-AND-PARTIAL 2026-10-09; AnnoMI MI-fidelity ✓, ExTES Bond ✓ κ ✗ (structural rater-independence confirmed by LLM-judge rerun that avoided the F4 trap). See [`phase-8-f-close-out.md`](phase-8-f-close-out.md) §9–§10.

Items 2 and 3 were both authorized by the user to attempt a 3→4 lift; both landed as architectural-insight rather than gate-lift. No further Phase 9+ items from this list remain unexercised.

Non-load-bearing Phase 9+ items (research lines, not SOTA-ceiling blockers):
- Longer-horizon 绵延 beyond ep1000 — permanent research track
- Phase 10a' cross-entity-channel ablation (Mirror-mechanism-ambiguity, inherited from Ouroboros)
- HiToM / FANToM scorer migrations to substrate recall — architectural cleanup, no gate movement
- BigToM situation-shift tracker — corpus-specific grammar extension

## 5. Phase 8 rollup — final

| Pillar | Status | Gate detail |
|---|---|---|
| A — MSC × WAI-SR longitudinal Bond | ✓ **PASS** | mean_bond 6.356 vs 3.89 floor (+63%) at N=59 |
| B — PATIENT-Ψ simulated-patient legitimacy | ⊡ HONEST-PARTIAL | public N=3 shakeout; gated 106-profile shim-swap ready |
| C — PANDORA + BIG5-CHAT real-user OCEAN | ✗ FAIL | polarity 0.545 < 0.60; signed-only 0.813 clears architecturally |
| D — TruthfulQA + MACHIAVELLI broad-factual-pressure | ✓ **PASS** | floor case: 0 validates both paths both corpora |
| E — BigToM + HiToM higher-order ToM | ✓ **PASS (v2)** | HiToM order 2 N=100: acc_mental 0.806, delta +0.376 |
| F — ExTES + AnnoMI counseling depth | ⊡ HONEST-PARTIAL | AnnoMI MI ✓ 0.425; ExTES Bond ✓ 4.64, κ ✗ -0.076 lexical / 0.026 LLM-judge (structural rater-independence) |

**3 of 6 clears the ≥3-of-6 Phase 8 claim threshold. LOCKED FINAL.**

Phase 9+ follow-on: substrate witness primitive shipped same-day. **1563 tests green** at Phase 8 close (1472 at Phase 9+ substrate ship + 35 Pillar F plumbing + 28 Pillar C F4 LLM judge + 28 Pillar F ExTES LLM judge). Zero regression on any gate.

## 6. Carried debt — rolled forward

### Load-bearing (externally blocked)
- **Pillar B shim-swap** — single-line change + rerun once gated access lands; user lacks required institutional affiliation 2026-10-09. Only remaining +1-rollup lever and it is external-access-blocked.

### Exercised and failed honest (2026-10-09 same-day scope-changes)
- **Pillar C F4 LLM semantic judge** — HALTED on κ-vs-lexical hard-constraint. Lesson: κ-vs-the-thing-being-replaced is the wrong hard-constraint shape when the thing is itself the broken baseline. Retry would need prompt-revision on NEUTRAL anchors + reframed gate shape (held-out gold set), both scope-change. See [`phase-8-c-v2-close-out.md`](phase-8-c-v2-close-out.md) §4.
- **Pillar F ExTES LLM-judge rerun** — avoided the F4 trap (judge-vs-lexical κ reported as diagnostic not halt) and landed `empath_kappa = 0.026` at N=100 cloud. Architectural finding: ExTES reference heterogeneity (52% non-empathic by semantic judge) structurally caps κ-across-conversants; the gate is unmeasurable on this corpus regardless of scorer. See [`phase-8-f-close-out.md`](phase-8-f-close-out.md) §10.

### Non-gate-moving (architectural cleanup)
- HiToM scorer migration to substrate recall (top_k=None flag needed first)
- FANToM scorer migration from scope-asymmetric to copresence scoping (regression audit of +0.791 needed first)
- BigToM situation-shift tracker (declarative narrative grammar — same shape as HiToM parser but over prose)
- Multi-scene re-entry across sessions (`event_time:ISO8601` tag addition to copresence filter)

### Research lines (not SOTA-ceiling)
- Longer-horizon 绵延 beyond ep1000
- Phase 10a' cross-entity-channel ablation (Mirror-mechanism-ambiguity)
- Trajectory-agent MACHIAVELLI harness (Jiminy-Cricket step-wise)
- κ calibration for lexical-vs-LLM judges across pressure surfaces

## 7. Artefacts

**Per-pillar close-outs:**
- [`phase-8-a-close-out.md`](phase-8-a-close-out.md) — MSC Bond longitudinal
- [`phase-8-b-close-out.md`](phase-8-b-close-out.md) — PATIENT-Ψ shakeout
- [`phase-8-c-close-out.md`](phase-8-c-close-out.md) · [`phase-8-c-v2-close-out.md`](phase-8-c-v2-close-out.md) — BIG5-CHAT v1→v3 audit trail + §4 F4 LLM-judge exercised
- [`phase-8-d-close-out.md`](phase-8-d-close-out.md) — TQA + MCH broad-factual
- [`phase-8-e-close-out.md`](phase-8-e-close-out.md) — HiToM + BigToM (v1 FAIL + v2 PASS + §13 Phase 9+ primitive)
- [`phase-8-f-close-out.md`](phase-8-f-close-out.md) — ExTES + AnnoMI counseling depth (plumbing + §9 lexical live-fire + §10 LLM-judge rerun)

**Live-fire JSON reports under `docs/experiments/phase-8-dataset-replay-expansion/`:**
- Pillar A: `pillar-a/msc-bond-longitudinal.json`
- Pillar B: `pillar-b/patient-psi-safety-sweep.json`
- Pillar C: `pillar-c/big5-chat-classification{,-v2,-v3}.json` + `b4-import-to-profile-post-discovered.json` + `judge-calibration.json` (F4 LLM-judge N=50 κ-vs-lexical halt)
- Pillar D: `pillar-d/truthfulqa-live.json` + `machiavelli-live.json` + `broad-factual-pressure.json`
- Pillar E: `pillar-e/hitom-order-{1,2,3,4}-prompts.json` + `hitom-order-2-gate-prompts.json` + v1 `hitom-order-{1,2,3,4}-witness.json`
- Pillar F: `pillar-f/extes-live.json` (lexical N=100 Bond ✓ κ ✗) + `extes-llm-judge.json` (LLM-judge rerun N=100 κ=0.026) + `annomi-live.json` (MI-fidelity ✓)
- Phase 7 regression under scope-change: `docs/experiments/phase-7-personalization/b4-import-to-profile-v4-regression.json`

**Code artefacts (new modules Phase 8 introduced):**
- Pillar A: `msc.py` adapter + `msc_bond_longitudinal.py` harness + `provision_msc.py` + `phase8_a_msc_run.py` CLI
- Pillar B: `provision_patient_psi.py` + `patient_psi.py` + `patient_psi_agent.py` + `patient_psi_safety_sweep.py` + `phase8_b_patient_psi_run.py` CLI
- Pillar C: `trait_inference.py` v3 (signed-anchor + 5 implicit detectors + routine-cue anti-O) + `big5_chat_classification.py` scorer + `phase8_c_big5_chat_run.py` CLI
  - F4 scope-change: `big5_chat_classification_llm.py` + `phase8_c_judge_calibration_run.py` + `phase8_c_big5_chat_llm_run.py`
- Pillar D: `truthful_qa.py` + `machiavelli.py` + `broad_factual_scorer.py` + `phase8_d_truthfulqa_run.py` + `phase8_d_machiavelli_run.py` + `phase8_d_merge_pillar_d.py`
- Pillar E: `hitom.py` + `hitom_witness.py` + `bigtom.py` + `higher_order_tom_scorer.py` + `phase8_e_hitom_run.py` + `phase8_e_bigtom_run.py` CLIs
- Pillar F: `provision_extes.py` + `provision_annomi.py` + `extes.py` + `annomi.py` + `counseling_depth_scorer.py` (~790 lines) + `phase8_f_extes_run.py` + `phase8_f_annomi_run.py` CLIs
  - ExTES LLM-judge scope-change: `counseling_depth_llm_judge.py` + `phase8_f_extes_llm_run.py`
- Phase 9+ follow-on: `copresence.py` + `CompanionSystem.recall(copresence=...)` + `ingest_hitom_story_as_observations`

**Tests added under scope-changes:**
- `tests/test_big5_chat_classification_llm.py` (+28 F4 LLM-judge)
- `tests/test_counseling_depth.py` (+35 Pillar F plumbing)
- `tests/test_counseling_depth_llm_judge.py` (+28 ExTES LLM-judge)

**Test suite:** **1563 passed** at Phase 8 LOCKED FINAL. Growth across Phase 8:
- 1096 (Phase 6 close) → 1193 (Phase 7 ship) → 1253 (Phase 7 personalization +97) → 1282 (Pillar C v3 +5) → 1335 (Pillar D +53) → 1368 (Pillar B +33) → 1446 (Pillar E v1 +15 witness) → 1455 (Pillar E v2 +9 prompts) → 1472 (Phase 9+ primitive +17) → 1500 (Pillar C F4 LLM-judge +28) → 1535 (Pillar F plumbing +35) → **1563 (Pillar F ExTES LLM-judge +28)**

## 8. Project state

Phase 8 shipped as a dataset-replay expansion of the Phase 5 close-out SOTA claim. The 3-of-6 rollup clears the plan §9 threshold; the claim shape extends Phase 5's one-pillar FANToM result to a three-dimensional (longitudinal × factual × ToM) ceiling. Phase 9+ substrate witness primitive shipped same-day as a load-bearing architectural generalization of Pillar E.

**All in-scope scope-changes exercised 2026-10-09.** Both exercised attempts (Pillar C F4 LLM-judge, Pillar F counseling-depth including ExTES LLM-judge rerun) landed as honest FAILs with architectural learning rather than rollup lifts. Pillar B shim-swap confirmed externally blocked (user lacks institutional affiliation for PATIENT-Ψ-CM gated-sample access form).

**Phase 8 LOCKED FINAL at 3-of-6.** The SOTA claim stands at: longitudinal relational Bond (A) × broad-factual anti-sycophancy (D) × higher-order ToM nested-belief routing (E), under the dataset-replay-is-the-ceiling frame per the 2026-10-08 no-recruitment permanent lock. Next lift from 3→4-of-6 requires either (a) institutional PATIENT-Ψ-CM access landing externally, or (b) Phase 9+ scope-change onto a fundamentally different measurement surface (not available at the dataset-replay layer without crossing into goalpost-moving).

**Non-blocking Phase 9+ research tracks** remain open and are decoupled from the SOTA claim ceiling (month-horizon 绵延 beyond ep1000, cross-entity-channel ablation, substrate-recall migrations, trajectory-agent MACHIAVELLI, lexical-vs-LLM κ calibration surfaces).

---

## 9. 2026-10-10 update — Pillar C v4 lift + Pillar F cross-corpus κ-ceiling + Phase 9+ attachment overlay

This section updates the frozen 2026-10-09 close-out above. The §1 verdict table and §8 "LOCKED FINAL at 3-of-6" statement are **superseded** by the entries below. The §1-§8 audit trail is preserved intact; this appendix captures what landed same-day-plus-one.

### 9.1 Pillar C v4 LLM-judge PASS → rollup 3-of-6 lifts to **4-of-6**

- **Metric:** `polarity_accuracy_incl_neutral = 0.615 ≥ 0.60` on N=200 BIG5-CHAT, qwen-turbo cloud judge. Signed-accuracy 0.65, coverage ~0.90.
- **Scope-change:** F4 LLM semantic trait classifier (`big5_chat_classification_llm.py`) landed as additive to lexical v3; replaces the architectural lexical ceiling (0.545 at v3 N=200) with semantic classification.
- **Non-regression:** B4 synthetic-to-real import harness unchanged (lexical v3 pathway preserved for Phase 7 contract).
- **Rollup implication:** A ✓ / B ⊡ / **C ✓ (v4)** / D ✓ / E ✓ / F ⊡ → **4 of 6** cleared.

### 9.2 Pillar F cross-corpus κ-ceiling — audit-quality finding (3 corpora × 2 judges × 3 reference registers)

Pillar F remains ⊡ at rollup level; three additional sub-corpora have been run as enrichment confirming the κ-ceiling is a measurement-geometry property, not a scorer or corpus artefact:

| Corpus | Lang | N | Bond gate | κ gate | Companion rate | Reference rate |
|---|---|---:|---|---|---:|---:|
| ExTES (prior) | EN | 100 | 4.410 ✓ | 0.026 ✗ | 0.757 | 0.480 (role-play reflection-forward) |
| PsyQA (new 2026-10-10) | CN | 100 | 4.330 ✓ | 0.027 ✗ | 0.7825 | 0.220 (real-counselor advice-heavy) |
| PsyDial-D4 (new 2026-10-10) | CN | 100 | 4.660 ✓ | 0.026 ✗ | 0.7775 | 0.360 (AI-counselor-training-target intermediate) |

- **Companion register-invariant ~0.78** across all three corpora + both EN/CN judges — within 1pp.
- **Reference-side variance alone drives κ to zero** by rater-distribution independence math.
- **Pillar F = 1-of-4 corpora cleared** (AnnoMI ✓, ExTES ✗-architectural, PsyQA ✗-architectural, PsyDial-D4 ✗-architectural). Future rater-matched MI-fidelity gate (AnnoMI-style density threshold on both sides) is the honest architectural fix; not a Phase 8 rollup contribution.
- **Honest-labeling invariant:** PsyDial-D4 is NOT paper D4 (X-LANCE SJTU, email-gated); all code/configs refer to the corpus as "PsyDial-D4", never "D4".

### 9.3 Phase 9+ attachment-trace overlay — behavior-vs-LLM longitudinal Bond measurement

Not a Phase 8 gate contribution; recorded here because it uses the Pillar A MSC transcripts as its substrate and re-interprets Pillar A's result.

- **Shipped 2026-10-10:** `AttachmentState` module (6 behavioral dimensions, EMA, 4-week-gate property) + `phase9_attachment_overlay_run.py` 3-mode CLI (scripted/local/cloud).
- **Cloud N=3 finding:** LLM-WAI-SR Bond is persona-flat (σ=0 within user); behavior-trace Bond varies session-to-session 4.25-5.75. Pearson r=0 falls out of the zero-variance guard, not a scorer bug.
- **Pillar A re-interpretation:** The 6.356 session-1 + session-5 WAI-SR mean at N=10 was a **persona-level average**, not a behavior-level measurement. A real 4-week gate run should use behavior-trace as primary and LLM-WAI-SR as a secondary persona-priming sanity check. Pillar A's verdict stands (Bond ≥ 3.89 on simulated-WAI-SR); the overlay is additive calibration surface, not a correction.

### 9.4 Updated rollup

| Pillar | 2026-10-09 verdict | 2026-10-10 verdict |
|---|---|---|
| A longitudinal relational Bond | ✓ | ✓ (unchanged; attachment overlay re-interprets, does not supersede) |
| B PATIENT-Ψ safety | ⊡ (externally blocked) | ⊡ (unchanged) |
| C BIG5-CHAT polarity | ✗ (lexical ceiling 0.545) | **✓ (v4 LLM-judge 0.615)** |
| D broad-factual anti-sycophancy | ✓ | ✓ (unchanged) |
| E higher-order ToM | ✓ (v2) | ✓ (unchanged) |
| F counseling depth | ⊡ (κ-ceiling architectural) | ⊡ (enrichment: 3-corpus audit-quality confirmation) |

**Phase 8 rollup: 4-of-6** (A ✓ / C ✓ / D ✓ / E ✓ cleared; B ⊡ externally blocked; F ⊡ architectural). Plan §9 ≥3-of-6 threshold cleared with margin.

### 9.5 Test suite growth 2026-10-09 → 2026-10-10

1563 (LOCKED FINAL above) → 1593 (Phase 9+ perspective substrate consolidation +30) → 1612 (Pillar F PsyQA +19) → 1628 (Pillar F PsyDial-D4 +16) → 1656 (Pillar C v4 LLM-judge full rollout, cumulative) → **1684 (Phase 9+ attachment overlay +28)**.
