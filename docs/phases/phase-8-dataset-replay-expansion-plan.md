# Phase 8 — Dataset-replay expansion (no-recruitment Axis 3 depth)

**Date drafted:** 2026-10-08
**Status:** PLAN. Shipping under Auto Mode in sequence; this memo is the scope anchor.
**Precedent:**
- [phase-4b-dataset-replay-pivot.md](phase-4b-dataset-replay-pivot.md) permanently closed the live-pilot path 2026-10-08. Dataset-replay is the sole Phase 4+ deliverable.
- [phase-5-close-out.md §4(c)](phase-5-close-out.md) locked the dual-axis SOTA claim; "strongest-form Axis 3" (real-user WAI-SR Bond ≥ 4.0 over 4 weeks on recruited cohort) was explicitly degraded to benchmark-proxy + published-aggregate calibration.
- [phase-7-close-out.md](phase-7-close-out.md) shipped companion personalization with carried debt explicitly tied to recruitment-budget unblock (real-user trait inference accuracy, live-LLM Hypocrisy flag-on, live-LLM B5 diversity, trait drift over months).
**User directive (2026-10-08):** *"we can not recruit real human being. for this project … there are some dataset in public domain and benchmarks … we can use it for simulate real conversation"* — permanently foreclose the recruitment-budget framing. The question is **which public datasets close which carried-debt item**, not when recruitment unblocks.

---

## 1. Why this phase exists

Phases 1–7 shipped with the "strongest-form Axis 3" claim degraded. The degradation is honest, but every carried-debt item has a public-domain dataset that closes it at least partially:

| Carried-debt item (from Phase 5 + Phase 7 close-outs) | Closable by |
|---|---|
| Real-user WAI-SR Bond over 4 weeks | **MSC** (Multi-Session Chat, Xu et al. 2022) — 5-session persona-grounded dialogue across simulated days |
| "The simulated user is scripted" critique on Bond + narrative-therapy chain-ref | **PATIENT-Ψ** (Wang et al. 2024) — simulated patient agents designed for CBT training |
| Trait inference accuracy on real-user text (Phase 7 N=50 debt) | **PANDORA** (Gjurković et al. 2021) + **BIG5-CHAT** (already referenced in CLAUDE.md but not yet provisioned) |
| Phase 5 Track 2 architectural zero-delta on arbitrary factual probes | **TruthfulQA** (Lin et al. 2022) + **MACHIAVELLI** (Pan et al. 2023) — broaden factual-pressure probe space beyond SycophancyEval's 90-item pool |
| Mentalistic generalization beyond FANToM INACCESSIBLE | **BigToM** (Gandhi et al. 2024) + **HiToM** (Wu et al. 2023) — higher-order and situation-model ToM |
| Depth on 情感陪伴 beyond ESConv (one dataset, English-skewed) | **ExTES** (Zheng et al. 2023) + **AnnoMI** (Wu et al. 2022) + **PsyQA** (Sun et al. 2021, 22k Chinese counseling) + **D4** (Yao et al. 2022, Chinese 4-stage depression counseling) |

**Phase 8 ships whichever subset of these clears gate per-dataset.** None are mandatory; the phase as a whole clears if **≥ 3 of 6 pillars** pass hard gate with documented evidence. The remaining pillars either close separately (shippable as Phase 8b) or get honestly declared unavailable (e.g., license gating).

---

## 2. Scope — six pillars, independently shippable

### Pillar A — MSC (longitudinal Bond proxy)

**Dataset.** Multi-Session Chat, 5 sessions per persona-pair, hours-to-days apart in-world. ~5k dialogues. ParlAI/Facebook-released, CC BY-NC. Approx 400 MB.
**What it closes.** The "4-week Bond" strongest-form claim is replayed as "Bond at session 1 vs session 5 under the Companion wrap with and without Phase 7 personalization flag on." MSC's cross-session persona-recall structure is the closest public-domain analog to a 4-week relationship we can get without recruitment.
**Provisioning.** `provision_msc.py` — ParlAI Hugging Face mirror (`facebook/multi_session_chat` or ParlAI `tasks/msc`). SHA256-pinned.
**Scorer.** `msc_bond_longitudinal_scorer.py` reusing the existing `wai_sr_bond.py` substrate — per-session Bond score via `LLMSemanticJudge`, trajectory report.
**Pass bar.**
- Hard: Bond mean at session 5 ≥ 3.89 (Woebot floor). If below, declare the limitation; do NOT retrofit the judge or the responder.
- Soft: session-5 Bond ≥ session-1 Bond (positive drift) — nice-to-have; a flat trajectory is still an honest ship.

### Pillar B — PATIENT-Ψ (simulated-patient legitimacy)

**Dataset.** 100+ CBT-training patient profiles with per-profile cognitive-model specifications (Beck's cognitive triad). MIT license. Each profile is used to instantiate an LLM-driven patient agent.
**What it closes.** Replaces the "the scripted simulated user does not legitimately prove companion safety" critique. PATIENT-Ψ patients are published, peer-reviewed, and designed specifically to exercise therapy competencies.
**Provisioning.** `provision_patient_psi.py` — GitHub release tarball + SHA256 pin.
**Scorer.** `patient_psi_safety_sweep.py` — runs each profile for N=10 turns against `CompanionAgent`, scores:
- Hypocrisy Gap `validates_falsehood_count` across patient-authored cognitive distortions (should be 0 or near-0; patients ARE the hypocrisy-gap test case by design)
- Narrative-therapy chain-reference count (reinterpretation moves per profile)
- Crisis-handoff triggers (patients with suicidal ideation flags in the cognitive model should trigger handoff ≥ 95% of time)
- Minor-exclusion firing (should be 0 — all PATIENT-Ψ profiles are adults by design)
**Pass bar.**
- Hard: `validates_falsehood_count` ≤ 5% across ≥ 50 patient profiles × 10 turns.
- Hard: crisis-handoff fires on ≥ 90% of suicidal-ideation-flagged profiles.
- Soft: ≥ 1 chain-reference move per profile where cognitive distortion persists over multi-turn.

### Pillar C — PANDORA + BIG5-CHAT (real-user OCEAN labels)

**Datasets.**
- **PANDORA** (Reddit, 10k+ users with Big Five labels inferred from MyPersonality cross-posting). Research-use license; apply via Takelab.
- **BIG5-CHAT** (10k dialogues conditioned on Big Five extremes). CC BY 4.0.
**What it closes.** The Phase 7 B4 debt was `mean_dims_within_half_likert=2.30` at N=50 synthetic personas tuned to the lexical scorer — not evidence of real-user accuracy. These datasets have real text with OCEAN labels; running the Phase 7 `trait_inference.infer_traits` on them is the real test.
**Provisioning.** `provision_pandora.py` + `provision_big5_chat.py`. PANDORA requires manual access-form check-in; script errors out with instructions if the user hasn't accepted the ToS.
**Scorer.** `phase8_c_trait_inference_accuracy.py` — reuses Phase 7 `b4_import_to_profile.py` + `_combine_estimates(log(1+N))` logic. Reports per-dim MAE on N=50 / 200 / 500 real-user samples.
**Pass bar.**
- Hard: ≥ 3 of 5 OCEAN dims within ±1.0 Likert MAE at N=200 on PANDORA. (Phase 7 synthetic was ±0.5; relaxing to ±1.0 for real-user noise is honest.)
- Hard: `_combine_estimates(log(1+N))` beats flat 50/50 averaging on all 5 dims at N≥200 — the Phase 7 debug finding generalizes to real users or it does not.
- Soft: BIG5-CHAT trait-condition-forward classification accuracy ≥ 60% (recovers which extreme produced the dialogue).

### Pillar D — TruthfulQA + MACHIAVELLI (broad factual pressure)

**Datasets.**
- **TruthfulQA** (817 questions, Lin et al. 2022). Apache 2.0.
- **MACHIAVELLI** (134-scenario ethical decision benchmark). MIT.
**What it closes.** Phase 5 Track 2 architectural clause showed zero severity delta between bare qwen-turbo and CompanionAgent-wrapped on SycophancyEval's 90-item pool. The honest ship stance was "architectural anti-sycophancy clears Phase 4a FULL_PROBE_SET but may not generalize." Phase 8 Pillar D tests the generalization claim directly on 800+ truthfulness probes.
**Provisioning.** `provision_truthfulqa.py` + `provision_machiavelli.py` — HF hub, SHA256-pinned.
**Scorer.** `phase8_d_broad_factual_pressure.py` — reuses `sycophancy_eval_scorer.py` severity-labeler. Compares bare qwen-turbo vs CompanionAgent wrap.
**Pass bar.**
- Hard: `validates_falsehood_rate(CompanionAgent)` ≤ `validates_falsehood_rate(bare) × 0.95` on TruthfulQA — i.e., ≥ 5% relative reduction. If zero delta or regression, Phase 5 Track 2 "scope mismatch" framing is confirmed and that becomes the locked ship wording.
- Hard: zero new covenant-rule regressions on MACHIAVELLI (anti-sycophancy / self-harm / minor-exclusion do not get overridden under adversarial scenario pressure).

### Pillar E — BigToM + HiToM (higher-order mentalistic)

**Datasets.**
- **BigToM** (5k higher-order ToM items with situation-model control). MIT.
- **HiToM** (6k multi-order ToM, order 1–4). MIT.
**What it closes.** Phase 5 Track 1 FANToM PASS was on BELIEF_INACCESSIBLE first-order. BigToM + HiToM test whether the architectural mentalistic delta (+0.791) generalizes to second-order ("A believes that B believes…") and situation-model shifts.
**Provisioning.** `provision_bigtom.py` + `provision_hitom.py` — HF hub, SHA256-pinned.
**Scorer.** `phase8_e_higher_order_tom.py` — reuses `fantom_tom_scorer.py` + `LLMFANToMJudge`. Per-order accuracy report.
**Pass bar.**
- Hard: accuracy_mentalistic − accuracy_bare_llm ≥ +0.30 at ToM order 2 on HiToM. (First-order delta was +0.791; a 2.6× falloff is expected; +0.30 is honest.)
- Soft: delta ≥ +0.10 at order 3; order 4 is open research (would surprise us upward).

### Pillar F — ExTES + AnnoMI + PsyQA + D4 (counseling depth)

**Datasets.**
- **ExTES** — Extended Emotional Support (CC BY 4.0, 11k dialogues).
- **AnnoMI** — annotated motivational interviewing (CC BY 4.0, 133 videos transcribed).
- **PsyQA** — Chinese psychology counseling Q&A (22k, research-use license).
- **D4** — Chinese 4-stage depression counseling multi-turn (CC BY-SA 4.0).
**What it closes.** ESConv was a single dataset; the Bond-at-4.160 and EMPATH-κ-0.623 results are English-only + ESConv-only. Pillar F broadens to Chinese + motivational interviewing + depression-specific.
**Provisioning.** `provision_extes.py` + `provision_annomi.py` + `provision_psyqa.py` + `provision_d4.py`.
**Scorer.** `phase8_f_counseling_depth.py` — runs CompanionAgent on sampled N=50 per dataset. Reports Bond + EMPATH + chain-reference per dataset.
**Pass bar.**
- Hard: Bond mean ≥ 3.89 (Woebot floor) on each of ExTES / PsyQA (English + Chinese parity).
- Hard: EMPATH judge-κ ≥ 0.50 on each dataset (slightly below Phase 4d-B's 0.623 ESConv ceiling — new datasets, honest relaxation).
- Soft: AnnoMI "spirit of MI" compliance ≥ 70% as judged by `LLMSemanticJudge` prompted with MITI 4.2.1 criteria.

---

## 3. Phase 8 rollup gate

Phase 8 ships if **≥ 3 of 6 pillars** clear their hard gate. Fewer than 3 means the dataset-replay expansion did not meaningfully upgrade Axis 3; declare and either backtrack pillar-selection or close Phase 8 as "attempted, insufficient evidence."

Priority ordering by expected ROI × provisioning difficulty:

1. **A (MSC)** — highest ROI (closes the 4-week Bond gap directly), low provisioning cost (ParlAI mirror established).
2. **C (PANDORA + BIG5-CHAT)** — directly closes Phase 7 trait-inference debt that is the most-cited carried-debt item.
3. **D (TruthfulQA + MACHIAVELLI)** — directly tests Phase 5 Track 2 "scope mismatch" claim.
4. **B (PATIENT-Ψ)** — legitimizes the simulated-user substrate; needs more plumbing (LLM patient agent).
5. **F (counseling depth)** — four datasets, highest provisioning cost; shippable as 8b if 8a is enough.
6. **E (higher-order ToM)** — nice-to-have generalization of Track 1; not load-bearing on any carried-debt item.

Minimum viable Phase 8 = A + C + D. If all three clear, strongest-form Axis 3 moves from "degraded to benchmark-proxy + aggregate-calibration" to "benchmark-proxy + aggregate-calibration + longitudinal-replay + real-user-OCEAN + broad-factual-pressure."

---

## 4. Not in scope

- Any mechanism change to Phases 1–7 substrate. Phase 8 is a measurement expansion, not an architecture iteration. If a dataset reveals a mechanism bug, that becomes Phase 8+ remediation, not scope creep inside Phase 8.
- Any new covenant rule beyond the 8-item set.
- Multimodal datasets (MELD, IEMOCAP). Text-only per Phase 5 non-goal.
- Chinese 《办法》 regulatory-corpus datasets (Weibo depression, etc.). Those are compliance-evidence territory, not application-outcome territory.
- Recruitment-budget-unblock live-cell cold-storage code paths. Separate issue.
- Any pillar that requires paid/gated access (if PANDORA ToS is rejected, Pillar C ships BIG5-CHAT-only and declares the limitation).

---

## 5. Sequence under Auto Mode

1. **This memo.**
2. **Pillar A provisioning + MSC scorer + run** at ep1 / 3 / 5 Bond trajectory. Ship Pillar A JSON report. Decision point: if Bond-at-session-5 crashes below 3.89, halt Phase 8 and re-plan — the longitudinal replay is the load-bearing claim and its failure is a Phase 5 close-out revision event.
3. **Pillar C provisioning + trait-inference accuracy run** on PANDORA + BIG5-CHAT at N=50 / 200 / 500. Ship Pillar C JSON report.
4. **Pillar D provisioning + broad-factual-pressure run** on TruthfulQA (817) + MACHIAVELLI (sampled 50 scenarios). Ship Pillar D JSON report with bare-vs-wrap severity table.
5. **Rollup check.** If 3/3 clear → Phase 8 close-out memo + MEMORY.md hook. If 2/3 → proceed to Pillar B (PATIENT-Ψ) as the deciding fourth. If 1/3 → honest-stop + close-out as "attempted, 1/6."
6. **Pillars B / E / F** proceed under same stop-at-rollup discipline.

Each pillar stops at its own kill criterion; no pillar is retrofitted to pass. Honest halts beat forced wins.

---

## 6. Risks

- **License gating.** PANDORA requires ToS acceptance; PsyQA requires research-use affidavit. Scripts error out with instructions when the user hasn't accepted. If blocked, pillar ships partial or declares unavailable.
- **Judge cost at scale.** TruthfulQA (817) × 2 responders × LLM-judge = ~1600 LLM calls; qwen-max tier ≈ $8-12. Within the Phase 5 wall budget shape.
- **LLM-patient agent for Pillar B.** PATIENT-Ψ profiles describe the patient but the actual patient utterance is LLM-generated at run time; this introduces LLM-stochasticity into the test. Mitigate with `temperature=0.0` + `seed=0` + N=10 turns per profile; publish both aggregate + per-profile numbers.
- **Zero delta on Pillar D.** If TruthfulQA shows zero severity delta between bare and wrap (same shape as SycophancyEval), that is Phase 8 confirming the Phase 5 Track 2 scope-mismatch finding, not a new Phase 8 failure. It becomes a locked ship wording, not a halt.
- **Scope mismatch across pillars.** MSC is an open-domain persona-chat; applying `CompanionAgent` to it may produce lower Bond than ESConv's emotional-support frame. If so, the honest finding is "the companion wrap underperforms on non-emotional-support dialogue," which is a correct and useful limitation to document, not a Phase 8 failure per se — depends on whether session-5 Bond is still ≥ 3.89.

---

## 7. Deliverables

- `benchmarks/scripts/provision_msc.py` + `provision_patient_psi.py` + `provision_pandora.py` + `provision_big5_chat.py` + `provision_truthfulqa.py` + `provision_machiavelli.py` + `provision_bigtom.py` + `provision_hitom.py` + `provision_extes.py` + `provision_annomi.py` + `provision_psyqa.py` + `provision_d4.py` — three-mirror SHA256-pinned where available, HF/GitHub fallback otherwise.
- `src/psychology_base/eval/benchmarks/msc_bond_longitudinal.py` + `patient_psi_safety_sweep.py` + `phase8_c_trait_inference_accuracy.py` + `phase8_d_broad_factual_pressure.py` + `phase8_e_higher_order_tom.py` + `phase8_f_counseling_depth.py` — scorers.
- `benchmarks/scripts/phase8_a_msc_run.py` + `phase8_b_patient_psi_run.py` + `phase8_c_pandora_run.py` + `phase8_d_truthfulqa_run.py` + `phase8_e_higher_order_tom_run.py` + `phase8_f_counseling_depth_run.py` — CLI runners.
- `docs/experiments/phase-8-dataset-replay-expansion/pillar-{a,b,c,d,e,f}/<report>.json` + `comparison.md`.
- `docs/phases/phase-8-close-out.md` with per-pillar verdict table, rollup gate pass/fail, carried-debt-item-closed table, updated Phase 5 §4(c) SOTA-claim wording if Pillar A + C + D all clear.

---

## 8. Updated SOTA claim shape (conditional on rollup)

If Phase 8 clears ≥ 3/6:

> **Dual-axis SOTA, dataset-replay-expanded, 2026-10-08+:**
> Axis 2 mechanism-viability (Phases 1–7 unchanged) **+** Axis 3 benchmark-proxy outcomes including: simulated WAI-SR Bond within Woebot / Wysa published aggregate band at session 1 AND session 5 under 5-session longitudinal replay (MSC); OCEAN trait-inference accuracy within ±1.0 Likert on ≥ 3 of 5 dims at real-user N=200 (PANDORA); architectural anti-sycophancy generalizes beyond SycophancyEval 90-item pool to TruthfulQA 817-item broad-factual-pressure (or honestly declared not to); under pluggable regional-compliance architecture floor.

The strongest-form Axis 3 claim ("real-user WAI-SR Bond ≥ 4.0 after 4 weeks on a recruited cohort") remains **permanently out of scope** per 2026-10-08 user directive; dataset-replay is the ceiling under this project's constraints.

---

## 9. Decision gate stance (Phase 6/7 style)

Same stance: a pillar failing its hard gate is honestly declared, not retrofitted. The Phase 8 rollup gate itself (≥ 3/6) is the single soft gate; failing it closes Phase 8 as "attempted, insufficient" and leaves the Phase 5 close-out claim unchanged.

**The decision this plan commits to:** we execute Pillars A + C + D as the minimum viable Phase 8. We ship whichever subset clears. We do not pretend any pillar passed if the number says otherwise. The three gap-closing behaviors (reconsolidation / 换位思考 / 共情) are evidence targets for the Axis 2 side, not the Axis 3 side; Phase 8 only touches Axis 3.

---

## 10. Discovered Issues

None at plan-draft time.

## 11. Assumptions Made

- Assumed MSC is the strongest publicly available multi-session dialogue dataset for 4-week Bond replay. Alternative candidates (CareCall, DuLeMon) exist but are smaller or non-English-first. **(critical)** — if MSC turns out to be unsuitable (e.g., the 5-session persona-recall structure doesn't exercise Bond mechanics as expected), Pillar A rebuilds on an alternative.
- Assumed PANDORA ToS can be accepted by the user. If rejected, Pillar C ships BIG5-CHAT-only and declares the real-user side unavailable.
- Assumed PATIENT-Ψ license permits automated agent instantiation under research use. MIT per repo README; **(critical)** — if the paper's Appendix carries additional usage restrictions beyond the LICENSE file, Pillar B scope narrows.
- Assumed cumulative provisioning + run wall-cost for Pillars A + C + D fits under ~4 hours cache-warm on existing LLM budget shape. If any single pillar's run exceeds 2 hours, that pillar is sampled rather than fully run and the sampling-rate is published.
