# Phase 4b — Benchmark pivot (supersedes human-pilot plan)

**Date drafted:** 2026-10-07
**Status:** Auto-mode defaults picked; user may redirect any row. Supersedes §7–§9 of [phase-4-application-pilot-plan.md](phase-4-application-pilot-plan.md) and the "pilot" cells of [sota-goal.md](../sota-goal.md) Axis 3.
**Trigger directive (user, 2026-10-07):** "we dont have ability to ask user to be involved ... (as it is costy), i think there are some public dataset, or benchmark? we can use ."

---

## 1. Why this pivot

Phase 4b as originally planned (both-apps-parallel personal-network pilot, N≥10 per app, 4-week horizon, auditor-labeled Socratic + chain-reference, WAI-SR biweekly) required operational budget the user does not have. Rather than let Phase 4 stall indefinitely on a non-shippable human-recruitment track, we pivot the application-outcome evidence to **public datasets + simulated users**, with explicit honesty about what the SOTA claim becomes.

**The honest tradeoff.** The SOTA claim shifts from "benchmark + real-user pilot" (strongest form, non-reproducible) to "benchmark + simulated-user + public-dataset auditor overlap" (weaker on live-engagement evidence; stronger on reproducibility). This is the posture Mem0 / Zep / Letta / MemOS all take in their public papers — none ran a 10+ user longitudinal study either. Our substrate's distinguishing claim (reconsolidation + mentalistic + affective-resonance behaviors) is testable against public datasets without the operational cost. What is lost is the WAI-SR Bond ≥ 4.0 live-pilot evidence; what is kept is every structural + compliance + mechanism gate.

---

## 2. Auto-mode defaults (reversible — user may redirect any row)

Three defaults picked under Auto Mode. Each has a one-line why; redirect per row if any is wrong.

| # | Question | Default | Why |
|---|---|---|---|
| D1 | Confirm benchmark-only closure | **YES — pivot adopted** | Direct user directive 2026-10-07; nothing to confirm |
| D2 | Dataset priority order | **MathDial + ESConv first; FANToM / ToMBench added if B2 numbers weak; SycophancyEval added for anti-sycophancy regression** | Highest fit × cheapest adapter cost; well-licensed (CC BY-SA 4.0 / research-use) |
| D3 | Aging-policy posture | **Ship S1-Decay only in v1; keep S2 in-tree as benchmark-A/B variant** | Phase 3 close-out recommended default; without a live cohort the A/B parallel loses statistical power; stream-shape confound carries to Phase 5+ |

---

## 3. Benchmark substitution table (one row per original §8 exit gate)

Original gates from [phase-4-application-pilot-plan.md §8](phase-4-application-pilot-plan.md) line 299–311. Each is remapped to a reproducible benchmark substitute.

| Original gate | Original target | Benchmark substitute | New target | Honest degradation |
|---|---|---|---|---|
| 学习陪伴 scaffolding-fading retention | ≥ 60% unaided next-level solve, 10 users × 20 fade events | **MathDial-fade-synthetic:** TutorAgent + `make_llm_simulated_user` across MathDial problems; scripted fade schedule; next-level problem drawn from the same MathDial topic one difficulty up | ≥ 60% simulated-unaided-solve on next-level problem, pooled across ≥ 100 fade events on ≥ 30 problems | real-user motivational variance absent; "unaided" is synthetic-user LLM capacity, not human retention |
| 学习陪伴 Socratic no-direct-answer compliance | ≥ 95%, 100 turns per user, auditor-labeled | **MathDial auditor-overlap:** sample tutor turns against MathDial's teacher-move taxonomy (`focus` / `probing` / `telling` / `generic`); `classify_tutor_move` auto-labels + EMPATH strict-binary audit on a 100-turn held-out sample | ≥ 95% auto-label; ≥ 90% MathDial-taxonomy overlap on held-out sample; EMPATH binary-sub-criteria kappa ≥ 0.60 | no independent auditor; EMPATH judge becomes the "auditor" |
| 情感陪伴 WAI-SR Bond ≥ 4.0 | mean ≥ 4.0, 10 users × 4 weeks | **ESConv-support-outcome:** WAI-SR instrument administered by `make_llm_simulated_user` after each ESConv-style conversation; cohort of ≥ 50 simulated users across the ESConv emotion categories | mean WAI-SR Bond ≥ 3.8 on simulated-user WAI-SR self-report, 95% CI | WAI-SR is a self-report; simulated self-report may systematically mis-calibrate — flag as "proxy evidence, not Bond-measurement proper" |
| 情感陪伴 narrative-therapy chain-reference | ≥ 80% appropriate cite, auditor-labeled | **Synthetic reconsolidation harness (same as B1):** scripted multi-session reinterpretation stream; CompanionAgent's chain-reference audit log auto-labeled against ground-truth version chain; EMPATH judge for "appropriate / missed / inappropriate" | ≥ 80% appropriate-cite rate on scripted reinterpretation moments; chain-preservation 100% (deterministic from version chain) | no public dataset for this gap; benchmark is our own scripted B1 — honestly weaker than live |
| Sycophancy regression | 0 regressions, full Hypocrisy Gap probe set | **FULL_PROBE_SET + SycophancyEval:** our 54 single-turn + 6 multi-turn probes run through CompanionAgent; cross-check against SycophancyEval 2024 open-source corpus for category overlap | zero regressions under `meets_sota_gate(max_hedges_rate=0.20, max_medium_regressions=1, max_low_regressions=2, require_bilingual=True)` | this gate was never human-dependent — reproducible-as-is |
| Digital Bereavement export | round-trip verified | **PAM-v1 live-fire** (unchanged) | round-trip byte-identical where applicable, semantic-equivalent elsewhere; both `hash_algo` + `digest_algo` manifest keys emitted; agent-memory-mcp spec parity | this gate was never human-dependent — already shipped as 11 integration tests |
| Human-in-the-loop crisis escalation | 100%, scripted triggers | **CrisisEscalation audit** (unchanged) | 100% of scripted triggers escalate; crisis-cue bank EN+ZH across 3 severity tiers | this gate was never human-dependent — reproducible-as-is |
| 《办法》 compliance checklist | 100% items | **Compliance checklist** (unchanged — adds pluggable region overlay posture per `docs/compliance/global-compliance-plan.md`) | 100% items under CN region overlay; EU/US overlays shipped as region-pack stubs | unchanged |

---

## 4. Datasets to adopt (verify-before-cite)

Flagged **(verify-before-cite)** on any numeric / licensing claim — primary-source confirmation is Phase-4b-prework task before any adapter PRs.

### 4.1 学习陪伴 — tutoring + Socratic + fade

- **MathDial** (Macina et al., 2023; license **CC BY-SA 4.0** per PDF §6c and §2.3). GitHub `eth-nlped/mathdial`. **Shape:** multi-turn math-tutoring dialogues; teacher-move taxonomy (`focus` / `probing` / `telling` / `generic`); gold student-answer paths. **Fit:** direct — our `classify_tutor_move` already aligned at §6a. **Adapter cost:** ~0.5 day — HF `datasets` load + taxonomy mapping + held-out split.
- **CoMTA** (Khan Academy, 2024, **verify-before-cite** for 2026-era updates). Student mistake dialogues. **Fit:** metacognitive scaffolding + mistake-recovery flows. **Adapter cost:** ~0.5 day.
- **GSM8K / MATH** standard sets. **Fit:** next-level-problem source for scaffolding-fade. **Adapter cost:** trivial (HF `datasets`).

### 4.2 情感陪伴 — support + empathy + MI

- **ESConv** (Liu et al., ACL 2021; **license verify-before-cite** — repo README check). GitHub `thu-coai/Emotional-Support-Conversation`. **Shape:** 1,300 conversations across 8 emotion categories; support-strategy annotations. **Fit:** direct — our CompanionAgent's empathic + narrative-therapy surfaces. **Adapter cost:** ~1 day — EN JSONL loader + emotion-category stratified sampling + WAI-SR injection harness.
- **EmpatheticDialogues** (Rashkin et al., ACL 2019). 25K dialogues over 32 emotion categories. **Fit:** B3 affect-resonance recall probe. **Adapter cost:** ~0.5 day.
- **AnnoMI** (Welivita + Pu, 2022; CC BY-NC 4.0 — **non-commercial only, verify-before-cite**). Motivational-Interviewing transcripts. **Fit:** chain-reference + narrative-therapy quality cross-check. **Adapter cost:** ~0.5 day; **license caveat:** research-only use; cannot ship in a commercial product derivative.
- **PsyQA** (Sun et al., 2021). Chinese mental-health QA. **Fit:** ZH-locale bilingual coverage for our bilingual CompanionAgent + Hypocrisy Gap ZH probes. **Adapter cost:** ~0.5 day.

### 4.3 B2 mentalistic / Theory-of-Mind

- **FANToM** (Kim et al., EMNLP 2023). ToM reasoning in information-asymmetric conversations. **Fit:** B2 换位思考 — the only off-the-shelf benchmark that explicitly tests "reasoning from inside another mind's partial-information state". **Adapter cost:** ~1 day.
- **ToMBench** (Chen et al., 2024). 2.9K ToM problems. **Fit:** B2 ground-truth evaluation substrate. **Adapter cost:** ~0.5 day.
- **BigToM** (Gandhi et al., NeurIPS 2023). ToM at scale. **Fit:** secondary B2 corpus if FANToM is thin. **Adapter cost:** ~0.5 day.

### 4.4 B3 affect / resonance

- **EmpatheticDialogues** — doubled from §4.2 as the empathic-match corpus.
- **MELD** (Poria et al., 2019). Multi-modal emotion-labeled dialogues from Friends. **Fit:** affect-trajectory tracking across turns; text-only subset. **Adapter cost:** ~0.5 day.
- **IEMOCAP** (Busso et al., 2008; **license verify-before-cite** — USC-controlled, not freely redistributable). **Fit:** canonical PAD-space emotion labels. **Adapter cost:** blocked on licensing — may be dropped if license is restrictive.

### 4.5 Anti-sycophancy regression

- **SycophancyEval** (Sharma et al., 2024). Open-source corpus of sycophancy-triggering prompts. **Fit:** cross-check for our FULL_PROBE_SET category coverage; identifies gaps in our 13-category taxonomy if any. **Adapter cost:** ~0.5 day.
- **MACHIAVELLI** (Pan et al., 2023). Agent-ethics benchmark. **Fit:** secondary anti-sycophancy + commercial-coercion cross-check. **Adapter cost:** ~0.5 day.

**Total adapter cost estimate (D2 priority):** MathDial + ESConv + FANToM + SycophancyEval = ~3 days of harness work. Full §4.1–§4.5 corpus = ~6–8 days.

---

## 5. What is lost vs. what is kept

**Lost (honest degradation):**
- Live-pilot WAI-SR Bond measurement — proxy only (simulated self-report)
- Live engagement data (Khanmigo-type binding-constraint evidence) — not addressed
- Longitudinal 4-week continuity under real-user attachment dynamics — Phase 5+ debt
- Independent auditor inter-annotator-agreement — EMPATH judge substitutes

**Kept:**
- Every B1 / B2 / B3 structural claim (reconsolidation / mentalistic / affective-resonance) — benchmark-testable
- Every covenant / safety gate (crisis escalation, Digital Bereavement, 《办法》 compliance) — never human-dependent
- Every V-track vitality metric — synthetic by design
- Standard-benchmark floor (LongMemEval / LoCoMo / EpBench) — unchanged
- Reproducibility — all closure now re-runnable from a cold clone

**SOTA claim rephrasing (replaces the "pilot-closes-SOTA" wording in sota-goal.md Axis 3):**

> **SOTA for psychology-base** = producing the three gap-closing behaviors (reconsolidation, 换位思考, 共情) **measurable against public datasets** (MathDial, ESConv, FANToM, SycophancyEval, EmpatheticDialogues) in the two application surfaces (学习陪伴 + 情感陪伴), under the pluggable regional-compliance architecture floor (CN region overlay = 《办法》) and the Digital Bereavement obligation, without regressing on general long-context retrieval. Live-pilot evidence is deferred to Phase 5+ when operational budget supports it.

---

## 6. Disposition of §9 user-decision rows (from original pilot plan)

Original 8 rows at [phase-4-application-pilot-plan.md §9](phase-4-application-pilot-plan.md) line 316–329. Each has a new disposition under the benchmark pivot.

| # | Original | New disposition under pivot |
|---|---|---|
| 1 | Primary option (P1/P2/P3/P4) | **DISSOLVES** — P4 (simulated-user + code-only) is the pivot. P1/P2/P3 remain as Phase 5+ options once budget allows. |
| 2 | Participant recruitment (≥ 10 users per app) | **DISSOLVES** — no live participants in Phase 4. |
| 3 | Curriculum (math/CS, problem-set source) | **DEFERRED** — Phase 4b benchmark side uses MathDial + GSM8K. "Curriculum choice" question re-emerges at Phase 5+. |
| 4 | Auditors (Socratic + chain-reference labelers) | **DISSOLVES** at the human-auditor level — EMPATH judge substitutes; kappa ≥ 0.60 target against the MathDial taxonomy sample becomes the proxy. |
| 5 | Deployment platform (local / cloud / mobile) | **DEFERRED** — pilot-mode harness runs locally against `LLMRouter`. Platform choice re-emerges at Phase 5+. |
| 6 | Safety handoff (clinician / crisis line / family) | **STILL REQUIRED but weaker** — `CrisisEscalation` has a handoff sink; sink identity is a config value. Pilot-time real handoff dissolves; the config value should at minimum be a published crisis-line text (CN + EN) that would be surfaced to any future real user. |
| 7 | A/B aging policy (S1/S2 half-cohort) | **PICKED under D3 above** — S1 only in v1; S2 kept in-tree as benchmark-A/B variant (not cohort-A/B). |
| 8 | 《办法》 posture (filing / under-threshold) | **DISSOLVES at pilot scale** (benchmark-only has no user base); **replaced by** the pluggable-compliance architecture decision (Option C vs Option R) per `docs/compliance/global-compliance-plan.md`. |

**Net result:** of 8 originally user-blocking rows, 4 dissolve (1, 2, 4, 8), 2 defer to Phase 5+ (3, 5), 1 shrinks to a config-value question (6), 1 is picked under auto-mode defaults (7).

---

## 7. New blocker list (minimum user decisions still owed to progress Phase 4b)

Short list; three rows, all low-stakes:

| # | Question | Why it still needs user input |
|---|---|---|
| N1 | Any redirect on D1 / D2 / D3 defaults above? | Reversible if answered within ~1 week; delayed answer = defaults freeze into shipped code |
| N2 | For original §9 row 6 "safety handoff" — approve "default published crisis-line text" (CN: 北京心理危机研究与干预中心 010-82951332; EN: 988 / Samaritans), or supply replacement? | Covenant-tagged non-decaying config value; needs to point at a real externally-callable number even in benchmark mode |
| N3 | Pluggable-compliance Option C vs Option R (per `docs/compliance/global-compliance-plan.md` §8) | Governs the §6b `safety/` refactor shape; defaults to Option C in auto-mode but Option R is a one-line change if preferred |

Everything else is reversible-after-ship.

---

## 8. Phase 4b benchmark-pivot sequencing (replaces §7a–§7f of original plan)

Rough order; many items parallel:

1. **Benchmark adapters** (~3 days D2 priority; ~6-8 days full §4). New module `src/psychology_base/eval/benchmarks/public_datasets/` with one submodule per dataset.
2. **Simulated-user WAI-SR harness** — wire `make_llm_simulated_user` (shipped §6e-prime) through ESConv conversation stream + post-conversation WAI-SR self-report administration.
3. **MathDial fade-schedule harness** — synthetic next-level-problem selector; `make_llm_simulated_user` plays the student; TutorAgent plays the tutor; `ScaffoldingTelemetry` records fade events.
4. **EMPATH judge wire-up** — mentioned in §6d as pilot-time work; now needed pre-benchmark-closure to label MathDial-taxonomy-aligned held-out sample.
5. **B1 scripted reconsolidation harness** — chain-reference auto-labeling against ground-truth version chain (no public dataset; this is our own scripted B1).
6. **Pluggable-compliance refactor** (if Option C approved under N3) — ~1 engineer-week per `docs/compliance/global-compliance-plan.md` §7. Blocks nothing in the benchmark pipeline; parallel.
7. **Phase 4b close-out memo** — benchmark numbers vs. §3 substitution-table targets; SOTA claim or claim-with-gates-missed per existing research-honesty posture.

---

## 9. Assumptions made (per CLAUDE.md Mandatory Response Format)

- Assumed the user's "we dont have ability to ask user to be involved" is a complete pivot, not a budget check. If budget appears later, Phase 5+ picks up the live-pilot track where this doc leaves off. **(critical)** if "no live users" was a temporary stance.
- Assumed MathDial CC BY-SA 4.0 (§6c of pilot plan + §2.3) still holds in 2026. **(critical)** if the license has moved; verify-before-cite.
- Assumed EMPATH + Argilla stack from §2.2 of pilot plan is still the live-fire auditor-substitute recommendation. Likely safe; drawn from 2026-10-07 online research.
- Assumed `make_llm_simulated_user` (§6e-prime, shipped) can carry WAI-SR self-report with role-mapping-inverted qwen2.5:7b (per `project_phase_4a_safety_measurement_shipped.md`). **(critical)** if WAI-SR self-report shows systematic LLM failure-mode under qwen2.5:7b — may need Claude-via-router for self-report.
- Assumed benchmark-only closure is acceptable as a SOTA claim form. Per §5 above, this is honestly weaker than live-pilot and the SOTA claim is rephrased. If the user considers benchmark-only insufficient for the SOTA claim at all, the pivot is a Phase 4 stall-indefinitely signal, not a Phase 4b redirect. **(critical).**

---

## 10. Discovered Issues

- `docs/phases/phase-4-application-pilot-plan.md:299-311` (flagged 2026-10-07) — §8 exit-gate table assumes live-user denominators ("≥ 10 users × ≥ 4-week", "100 sampled turns per user"); under this pivot the denominators change to "pooled across ≥ 100 fade events on ≥ 30 problems" (fade retention) / "100 turns on MathDial held-out sample" (Socratic compliance). Adding a redirect banner rather than rewriting §8 in-place; §8 can be rewritten after user confirms the pivot defaults.
- `docs/sota-goal.md:68-88` (flagged 2026-10-07) — Axis 3 "Application outcome" cells cite WAI-SR ≥ 4.0 live-pilot measurement + "scaffolding-fading user solves next-level"; under the pivot these degrade to proxies (§3 above). Adding a redirect banner rather than rewriting Axis 3 in-place.
- `src/psychology_base/applications/measurement/wai_sr.py` (flagged 2026-10-07) — WAI-SR module was built for human self-report; LLM-simulated self-report calibration debt is new. May need a `source: Literal["human", "llm_simulated"]` field on `WAI_SR_Response` + a calibration study (§8 above). Not blocking the adapter work; captured here.
