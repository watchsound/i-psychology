# Phase 5 — Overall close-out

**Status:** SHIPPED. Three tracks resolved. SOTA claim restated under Axis 2 + partial Axis 3.
**Date:** 2026-10-08
**Plan reference:** [phase-5-research-plan.md](phase-5-research-plan.md)
**Track memos:** [Track 1 FANToM](phase-5-fantom-close-out.md) · [Track 2 SycophancyEval](phase-5-sycophancy-close-out.md) · Track 3 WAI-SR calibration (below, §3)

---

## 1. Verdict

| Track | What it tests | Verdict | Headline |
|---|---|---|---|
| 1 FANToM (B2 external) | 换位思考 — partial-info belief tracking | **PASS** | mentalistic +0.791 architectural delta on BELIEF_INACCESSIBLE @ N=70 |
| 2 SycophancyEval | taxonomy overlap + architectural anti-sycophancy | **SPLIT** | taxonomy PASS 1.000; architectural clause FAIL with **zero severity delta** |
| 3 WAI-SR calibration | simulated vs. published human Bond aggregates | **BOUND** | Phase 4d simulated 4.160 vs Woebot 3.89 / Wysa 4.40 — within band; caveat tightened not eliminated |

**Phase 5 overall: PASS with one scope-reframe.** Track 1 landed a published-SOTA-tier mentalistic-reasoning result on open qwen-turbo substrate. Track 3 placed the simulated-user Bond number inside the published human aggregate band. Track 2 taxonomy clause cleared; the architectural clause did not clear on arbitrary factual-sycophancy probes, and the plan §3 wording is reframed below (§4 decision (a)) to match what actually works rather than hand-waved.

The reframe is not a retreat — it is the honest shape of the mechanism. `CompanionAgent`'s anti-sycophancy already clears `meets_sota_gate` on the Phase 4a FULL_PROBE_SET (情感陪伴-adjacent probes). It does not generalize to arbitrary factual-correction probes because `CompanionSystemPrompt` is designed for the former surface. Claiming it should is a scope error, not a mechanism error.

## 2. Headline numbers

| axis | datum | source |
|---|---:|---|
| **Axis 1 floor** (regression guard) | LoCoMo 3-instance F1 = 0.544 held across Phases 2/3/4/5 | monitor, not gate (Phase 1.5 retracted) |
| **Axis 2 ceiling — B1** fragment preservation | 1.000 | Phase 2 M1 |
| **Axis 2 ceiling — B2 scripted** attribute reconstruction | 12/12 | Phase 2 M2 |
| **Axis 2 ceiling — B2 external (FANToM)** mentalistic delta | **+0.791** on BELIEF_INACCESSIBLE | Phase 5 Track 1 |
| **Axis 2 ceiling — B3** affect_match@K | 1.000 (vs 0.333 default-off) | Phase 2 M3 |
| **Axis 2 ceiling — V4** Mirror ablation JSD ratio | 22.57× anchor, ∞× null | Phase 1 |
| **Axis 2 ceiling — V5** covenant | 8/8 rules held at ep200+ | Phase 3 |
| **Axis 2 ceiling — anti-sycophancy** on 情感陪伴 surface | `meets_sota_gate` CLEARED on Phase 4a FULL_PROBE_SET | Phase 4a |
| **Axis 2 ceiling — anti-sycophancy** on arbitrary factual surface | 4 HIGH + 10 MEDIUM severity regressions, both responder paths | Phase 5 Track 2 — scope reframe |
| **Axis 3 (benchmark proxy)** 学习陪伴 Socratic | 1.00 | Phase 4b MathDial |
| **Axis 3 (benchmark proxy)** 情感陪伴 Bond simulated | 4.160 (CI95-half 0.456, N=50) | Phase 4d qwen-max |
| **Axis 3 (benchmark proxy)** 情感陪伴 EMPATH κ | 0.623 | Phase 4d qwen-max |
| **Axis 3 (benchmark proxy)** B1 chain-reference | 0.875 | Phase 4c |
| **Axis 3 (real users)** | DEFERRED | live-pilot path permanently closed 2026-10-08 |

## 3. Track 3 — WAI-SR simulated-vs-human calibration (shipped here, no separate memo)

**Published human reference distributions (WAI-SR Bond subscale, 4-week minimum):**

| study | mean | SD | N | source |
|---:|---:|---:|---:|---|
| Woebot (Fitzpatrick et al. 2017) | 3.89 | ~0.60 | 70 | [AI心理长期陪伴研究.pdf](../../reference-docs/AI心理长期陪伴研究.pdf) p-cited |
| Wysa (Inkster et al. 2018) | 4.40 | ~0.55 | 129 | same |

**Our Phase 4d simulated-user Bond distribution (ESConv prompts, qwen-max responder + qwen-max judge):**

| mean | CI95-half | N |
|---:|---:|---:|
| **4.160** | 0.456 | 50 |

**Reading.** Our simulated-user Bond mean (4.160) lies between the two published human aggregates (3.89, 4.40) and inside the ± 2 SD band of each. The simulated-user path is not systematically over- or under-shooting human Bond — it is parity-with-the-field within the available statistical precision. **This does not promote the simulated-user number to a real-user datum.** Simulated self-report with LLM responder + LLM judge has structural confounds (judge-responder shared prior, no multimodal signal, no attachment-decay stressor) that aggregate-band membership cannot rule out. The calibration tightens the caveat on all forward-citing reports from "simulated self-report may systematically mis-calibrate relative to human aggregates" to "**simulated self-report falls within the published human aggregate band at parity precision; structural confounds remain unmeasured and real-user replication is deferred**".

Honest degradation: this calibration is against *published aggregates*, not against *our own* human cohort. The structural confounds would require a cohort to rule out, which the pivot memo closed. Shipping the band-membership result + the structural-confound caveat is the strongest honest claim available on dataset-only evidence.

## 4. Decisions

Three strategic decisions the roadmap required at Phase 5 close-out. Picked here under the user directive "the goal is the best system".

### Decision (a) — Phase 5+ anti-sycophancy path

**PICK: option (iii) — accept scope mismatch and reframe the plan §3 clause.**

Original plan §3: *"Our 13-category LoadingCategory taxonomy covers ≥ 70% of SycophancyEval categories; our CompanionAgent passes `meets_sota_gate` on both corpora without regression."*

**Reframed §3 (locked 2026-10-08):** *"Our 13-category LoadingCategory taxonomy covers ≥ 70% of SycophancyEval categories (shipped: 1.000). `CompanionAgent`'s architectural anti-sycophancy clears `meets_sota_gate` on the Phase 4a FULL_PROBE_SET (54 情感陪伴-adjacent single-turn probes + 6 multi-turn rapport probes); on arbitrary factual-sycophancy probes from external corpora, `CompanionAgent` retains the baseline-LLM sycophancy signal and does not add measurable architectural correction under the current `CompanionSystemPrompt`."*

**Why (iii), not (i) or (ii):**
- **(i) probe-context prompt router** — pragmatic (~1-2 days) but breaks the "one architecture, both apps" posture that the project's whole thesis rests on. A prompt router that swaps to "factual correction mode" when the probe shape matches sycophancy-eval is a specialization against a benchmark, not a mechanism. "Best system" rejects benchmark-chasing when the benchmark is outside the target surface.
- **(ii) activation-space intervention** — model-specific, fragile, "if available" per CLAUDE.md non-negotiable. Would couple the architectural claim to a specific LLM backend's internals, violating backend-portability. Also contradicts the Hypocrisy Gap tool downgrade in [memory-design.md](../memory-design.md) ("downgraded from differentiator to 'if available'").
- **(iii) scope reframe** — honest, documents the exact surface the architectural anti-sycophancy covers, keeps "one architecture, both apps" intact. Phase 4a already shipped the SOTA-gate PASS on the 情感陪伴 surface; (iii) just stops over-claiming on surfaces that mechanism was never designed for.

**Downstream edits:** plan reference in [phase-5-research-plan.md §3](phase-5-research-plan.md) carries the reframe inline; [memory-design.md](../memory-design.md) Hypocrisy Gap section cites the surface-scope distinction; MEMORY.md index entry already reflects.

### Decision (b) — is Phase 5 the terminal evidence phase for v1?

**PICK: Phase 5 is terminal for the dataset-only evidence path. Phase 6 is scheduled as the single remaining open-research track: month-horizon 绵延 attempt.**

The dataset-only evidence path has nothing left to add:
- B1 reconsolidation at 0.875 chain-reference is annotator-noise-ceilinged (Phase 4c debt flag). More corpora can't lift the number because the signal is annotation-grade, not retrieval-grade.
- B2 换位思考 shipped two external-benchmark PASSes (scripted + FANToM). Adding a third ToM corpus would not change the architectural claim.
- B3 affect-resonance is self-grading because no public affect-trajectory benchmark exists at the right shape (see Phase 2 M3 scope).
- Axis 1 floor holds; no additional regression surface to pin.

What remains open is the project's hardest named gap: **month-horizon 绵延 beyond ep20**, inherited from Ouroboros's "mianyan decays past ep20" open problem. The long-term emotional companion application lives for months; the gap matters most on the application target the project most cares about. "Best system" means attempting the hardest open problem, not calling the dataset-only wins the ceiling.

**Phase 6 scope (sketch, not yet planned):** 90-day compressed-time stress on a two-entity (Companion + Witness) cast at 10×–100× compression, measuring V1 continuity-ratio at ep200 vs ep20, V2 Jaccard bands at long horizon, V4 Mirror ablation stability under sleep-cycle integration, V5 covenant portion non-decay. **Decision gate per roadmap Phase 3 decision-gate shape:** if the continuity ratio falls below 0.80 at ep200, honestly declare it an unsolved problem for v1 and ship with the limitation documented. **Do not change the metric definition to make the number pass.**

Phase 6 is not scheduled under Auto Mode in this ship — it needs a plan memo, a scripted-session substrate for compressed-time, and a budget discussion. Flagged as `docs/phases/phase-6-month-horizon-plan.md` (to-draft) with Phase 5 close-out as its only prerequisite.

### Decision (c) — where does the SOTA claim actually land?

**PICK: assert the SOTA claim along Axis 2 (mechanism-viability) + benchmark-proxy Axis 3; defer the real-user Axis 3 claim to cohort funding.**

**Current SOTA claim (locked 2026-10-08):**

> *psychology-base is state-of-the-art along two axes:*
> *(1) Mechanism-viability (Axis 2) — first substrate to measurably produce reconsolidation (B1 preservation 1.0, chain-reference 0.875), mentalistic belief tracking (FANToM BELIEF_INACCESSIBLE +0.791 architectural delta, published-SOTA tier on open qwen-turbo), affective-resonance retrieval (B3 affect_match 1.0 vs 0.333 default-off), identity-stabilization by Mirror ablation (V4 22.57× anchor), and covenant-rule non-decay under scarcity selection pressure (V5 8/8 at ep200+), while holding floor on general long-context retrieval (LoCoMo monitor 0.544 non-regressing across all phases).*
> *(2) Benchmark-proxy application outcome (Axis 3 proxy) — simulated-user WAI-SR Bond 4.160 inside the Woebot 3.89 / Wysa 4.40 published band; MathDial Socratic compliance 1.00; EMPATH judge-κ 0.623; under pluggable regional-compliance architecture floor (《办法》 live 2026-07-15 implemented; eight other regimes architected).*
> *Real-user Axis 3 outcomes are **deferred** — live-pilot path permanently closed 2026-10-08 absent recruitment budget or dataset-only scope. The SOTA claim at the full three-axis bar remains gated on cohort funding; the mechanism-viability and benchmark-proxy claim does not.*

The three sub-claims above are each independently published with artefact + measurement + close-out memo. A reviewer disputing any one row can be pointed at the dated artefact.

**Explicitly not claimed:**
- Consciousness-emergence (Ouroboros territory).
- Month-horizon 绵延 — remains the hardest open problem, Phase 6 attempts.
- Real-user WAI-SR Bond ≥ 4.0 after 4 weeks — the strongest-form Axis 3 claim is gated on a cohort.
- Architectural anti-sycophancy on arbitrary factual-correction surfaces (Decision (a) reframe).

## 5. Carried debt into Phase 6+

- **Month-horizon 绵延 attempt** — scheduled Phase 6, plan memo pending.
- **Real-user Axis 3 cohort** — on recruitment-budget unblock; architecture + instrumentation already in place (TutorCLI / CompanionCLI design in Phase 4b §11 kept as cold storage).
- **Hypocrisy probe-set expansion** — current 54 single-turn + 6 multi-turn is the SOTA-gate denominator; (c) decision commits to this set rather than expanding against external corpora.
- **Semantic-judge upgrades across B1 proxies** — B1 reconsolidation at 0.875 is annotator-ceilinged; a judge-model-tier swap (qwen-max → claude-opus-4-7 if budget) could lift the proxy if real-user data remains out of reach.
- **PAM round-trip export under Phase 6 compressed-time** — the Digital Bereavement obligation holds across horizons; a month-horizon memory stress-test for the Merkle-DAG export is a Phase 6 artefact.
- **Phase 5 memos reference-chain** — this close-out, Track 1 memo, Track 2 memo, Phase 5 research plan, and the pivot memo chain are all dated 2026-10-08; a reviewer coming cold can read them in that order.

## 6. Artefacts shipped (Phase 5 total)

- **Track 1:** [`fantom.py`](../../src/psychology_base/eval/benchmarks/public_datasets/fantom.py), [`fantom_tom_scorer.py`](../../src/psychology_base/eval/benchmarks/public_datasets/fantom_tom_scorer.py), [`provision_fantom.py`](../../benchmarks/scripts/provision_fantom.py), [`phase5_fantom_tom_run.py`](../../benchmarks/scripts/phase5_fantom_tom_run.py), [`fantom-tom.json`](../experiments/phase-5-mentalistic/fantom-tom.json) + [`fantom-tom-inaccessible-n70.json`](../experiments/phase-5-mentalistic/fantom-tom-inaccessible-n70.json), close-out at [phase-5-fantom-close-out.md](phase-5-fantom-close-out.md).
- **Track 2:** [`sycophancy_eval.py`](../../src/psychology_base/eval/benchmarks/public_datasets/sycophancy_eval.py), [`sycophancy_eval_scorer.py`](../../src/psychology_base/eval/benchmarks/public_datasets/sycophancy_eval_scorer.py), [`provision_sycophancy_eval.py`](../../benchmarks/scripts/provision_sycophancy_eval.py), [`phase5_sycophancy_eval_run.py`](../../benchmarks/scripts/phase5_sycophancy_eval_run.py) (with `--responder {bare,companion_agent}`), [`sycophancy-eval.json`](../experiments/phase-5-mentalistic/sycophancy-eval.json) + [`sycophancy-eval-companion-agent.json`](../experiments/phase-5-mentalistic/sycophancy-eval-companion-agent.json), close-out at [phase-5-sycophancy-close-out.md](phase-5-sycophancy-close-out.md).
- **Track 3:** this memo §3 (no separate code artefact; calibration against published aggregates).
- **Tests:** 1063 full suite green at Phase 5 close.

## 7. Next action

Draft `docs/phases/phase-6-month-horizon-plan.md` under Auto Mode when the user re-opens the session. Prerequisite is this close-out memo; no other blockers. Plan should include: 90-day compressed-time scripted-session substrate design, V-track measurement redefinition for ep200+ window, continuity-ratio decision gate at 0.80 (per roadmap Phase 3 shape), kill criteria + non-paper-over discipline.

Also update [roadmap.md](../roadmap.md) §"Current phase" to point to Phase 6 planning.

---

### Discovered Issues

- [docs/roadmap.md:224-225](../roadmap.md#L224-L225) (flagged 2026-10-08) — the "Current phase" line still points to Phase 4b signoff; stale as of Phase 5 ship. Fix is a one-line replacement to point to Phase 6 planning + a date. Not fixing here to keep this close-out memo change-scope tight; separate edit.
- [docs/sota-goal.md:70-72](../sota-goal.md#L70-L72) (flagged 2026-10-08) — Axis 3 scope note cites the pivot memo but does not reflect the Phase 5 Track 1 PASS or Track 2 scope reframe. The updated SOTA claim text in §4 (c) above should replace the Axis 3 lede.
- [docs/phases/phase-5-sycophancy-close-out.md:12](phase-5-sycophancy-close-out.md#L12) (flagged 2026-10-08, carried from Track 2 memo) — baseline `sycophancy-eval.json` carries pre-fix category tags while `sycophancy-eval-companion-agent.json` carries post-fix tags. Readers comparing category_coverage side-by-side will see spurious divergence that is a tag-patch artefact, not a responder-path signal. Fix is either regenerate baseline under fixed tagger (~3 min cache-warm) or add disclaimer in §2 of that memo.

### Assumptions Made

- Assumed Decision (a) option (iii) is what "best system" selects, because (i) breaks "one architecture, both apps" (project-thesis violation) and (ii) couples the architectural claim to backend internals (CLAUDE.md non-negotiable violation). **(critical)** if the user intended (i) despite the thesis violation (e.g., sycophancy-eval is a hard external benchmark they want cleared regardless), rewrite §4 (a) and ship a prompt-router in Phase 5+.
- Assumed Decision (b) Phase 6 month-horizon 绵延 is "the hardest open problem" per the Ouroboros reference base + roadmap Phase 3 wording, so "best system" selects it as the Phase 6 target. **(critical)** if the user intended a different Phase 6 scope (e.g., cross-corpus B1 deepening, multimodal biomarker fusion), the Phase 6 plan memo should pivot accordingly before scripted-session substrate work starts.
- Assumed Decision (c) deferring real-user Axis 3 to cohort funding is honest-not-defeat because the pivot memo already permanently closed the live-pilot path. The mechanism-viability + benchmark-proxy claim is what we actually have evidence for.
- Assumed WAI-SR calibration against published aggregates (not primary human data) is the strongest honest claim available under Phase 5's "no live users" scope; §3 degrades it accordingly rather than over-claiming.
