# Phase 4 — Application pilot plan

**Date drafted:** 2026-10-07
**Status:** Phase 4a EXIT GREEN (shipped 2026-10-07, 693 tests green). Phase 4b blocked on §9 user sign-off — [phase-4b-signoff.md](phase-4b-signoff.md) drafts decisions for all eight items; user signs the approval table to unblock.
**Companion docs:** [../roadmap.md](../roadmap.md) §Phase 4 is the authoritative exit-gate spec; [../sota-goal.md](../sota-goal.md) defines the SOTA claim this phase closes; [phase-3-close-out.md](phase-3-close-out.md) is the handoff memo (what Phase 3 shipped + carried debt into Phase 4); [phase-4b-signoff.md](phase-4b-signoff.md) is the §9 decision memo awaiting user approval.
**Phase planning process:** per `memory/feedback_phase_planning_process.md` — PDF research + online + OSS → brainstorm → compress to options → cross-check against north stars. All three channels run; findings folded below.

---

## 1. What Phase 4 is (and what makes it different)

Phase 4 is where the SOTA claim becomes legitimate. Phases 1–3 produced evidence against scripted/synthetic probes. Phase 4 is **real users, real recall patterns, real 4-week horizons** against the two application north stars:

- **学习陪伴** (resilient learning tutor) — Socratic no-direct-answer rule, scaffolding-fading, metacognitive scaffolding, growth mindset. North-star application #1 per CLAUDE.md.
- **情感陪伴** (long-term emotional companion) — WAI-SR Bond ≥ 4.0, narrative-therapy chain-reference, anti-sycophancy, Digital Bereavement export obligation. North-star application #2.

**Why Phase 4 is structurally different from Phases 1–3:**

| Dimension | Phases 1–3 | Phase 4 |
|---|---|---|
| Users | Scripted probes / synthetic streams | Real humans ≥ 10 per application |
| Horizon | ep20 (Phase 1) / ep200 (Phase 3) compressed | 4-week real time (minimum) |
| Measurement | Deterministic scorer + stub judge | Auditor labels, WAI-SR instrument, outcome surveys |
| Reversibility | Full — artefacts in-tree, rerun any time | Partial — real-user data flows, consent constraints, grief/attachment risk |
| Blast radius | None | Users experience companion deprecation at pilot-end (Digital Bereavement) |
| What ships | Code | Code + operational protocol + human sign-off |

**The decision this plan forces:** no pilot infrastructure ships until the user signs off on §9's Pilot-Operations Decisions (participants, curriculum, platform, IRB stance, 《办法》 compliance declaration). Code scaffolding (Phase 4a) can ship earlier — it is reversible, non-destructive, and compounds value even if the pilot shape changes.

---

## 2. Research channels (step 1-3 of the phase-planning process)

### 2.1 PDF deep-reading (reference-docs/)

Findings grounding Phase 4 structure (subagent-reported; **flagged numeric claims to verify before any outward citation**):

- **Therabot RCT shape** — 8-week RCT framework, weekly outcome measurement, Cohen's d 0.79–0.90 range. The 8-week horizon is already the roadmap's 4-week minimum doubled; adopting Therabot's protocol as the emotional-companion pilot template is defensible. **Verify-before-cite:** specific N figures and effect sizes.
- **Woebot/Wysa WAI-SR cadence** — biweekly WAI-SR administration (5-item short form, Likert, ≤ 2 min per sitting). Target WAI-SR Bond 3.89–4.40 cited as sustained across 8+ weeks. **Verify-before-cite:** the 3.89–4.40 band exact source.
- **FDA Digital Health Guidance (2026)** — informed-consent must cover AI model changes, data retention, discontinuation pathway. Mandatory safety check-ins at weeks 2 / 4 / 8 for high-risk cohorts. Discontinuation protocols must be structured (2-week wind-down > abrupt termination). **Fold into Phase 4a:** structured wind-down as a built-in companion deprecation flow (not just a Digital Bereavement export button).
- **DTA → DTN three-tier** (transactional / relational / transformational) — pilot measurement cadence per tier. Tier 1 via WAI-S alignment subscale; Tier 2 via WAI-SR Bond subscale; Tier 3 via user-reported cognitive reframing frequency. **Fold into Phase 4b:** adopt the three-tier structure as the emotional-companion outcome taxonomy.
- **Resilient Learning Methodology** — step-scaffolding + growth-mindset + positive-feedback + adaptive pathways + confidence checks. Metacognitive-laziness detection: compare "unaided solve rate" to "aided solve rate"; target maintenance ≥ 66% of unaided capability. **Fold into Phase 4a:** scaffolding-fading telemetry needs confidence-check embedding every 3-4 interactions.
- **MathDial dataset** — tutoring-dialogue reference set for Socratic no-direct-answer fine-tuning and auditor-labeled compliance evaluation. **Caveat:** specific size figures from subagent (35,628 transcripts / 800K exchanges) flagged for verification — do not plant them in the plan until confirmed against primary source.
- **CAT/IRT personality cadence** — Big Five via CAT every 4 weeks (not weekly — excessive re-testing annoys users). ≤ 20 items per session, Cronbach's α ≥ 0.85. **Fold into Phase 4b:** personality assessment is intake + 4-week + 12-week, not weekly.

### 2.2 Online research (post-PDF developments)

Online agent returned 2026-10-07 with authoritative sources. Load-bearing findings:

- **Therabot RCT verified (Dartmouth NEJM AI, 2025):** N=210 (106 + 104 waitlist), 8-week (4 prompted + 4 self-serve), depression −51% / anxiety −31% / eating −19%, WAI-SR parity with human therapist. Confirms §2.1 subagent numbers were in-range. Real-time high-risk content detector + clinician-on-standby is the published pattern.
- **Ash purpose-built mental-health AI (arXiv 2606.28241, 2026):** single-arm 4-week observational, N=1,284, IRB-exempt (BRANY), **single-item in-app administration** of psychological-functioning / grandiosity / WAI items at baseline + week-4 — the pattern maximises completion in naturalistic mobile settings. Vendor-authored outcomes, trust the *protocol* not the numbers.
- **Wysa WAI-SR replication (JMIR Mental 2025):** N=1,205 PHQ-4-positive users; WAI-SR mean 3.64 at day-5, 3.75 at day-8. Standard instrument with "Wysa"/"Companion" substituted for "therapist". **Lowers §2.1's 3.89–4.40 cited band** — the honest live-pilot WAI-SR band for chatbots is 3.6–3.8 in the first 1–2 weeks. **Updates Phase 4 companion gate rationale:** our ≥ 4.0 target is aspirational (parity-with-sustained-Woebot), not entry-level.
- **《办法》 enforcement is live-with-teeth, not aspirational.** The post-2026-07-15 wave shuttered Alibaba Tongyi Qianwen's anthropomorphic services + Douyin's custom-romantic agents; smaller 伴侣 apps did function-removal + user migration. Compliance posture = blocker-level, not best-effort.
- **《办法》 2-hour reminder detail is not textually confirmed.** The regulation talks about "high-intensity intimate dialogue auto-termination" + "single-day duration limits + mandatory rest intervals" rather than a hard 2 h clock. Our `AIIdentityReminder` default (`reminder_interval_seconds=2*60*60`) is directional, not statutory — defer the exact threshold to legal review. **Already configurable; no code change required.**
- **《办法》 commercial-coercion anti-sycophancy (Art. 8(6)) is distinct from truth-distortion anti-sycophancy.** Phase 4 anti-sycophancy design must cover BOTH: the Hypocrisy-Gap probe (truth-distortion) + a no-dark-pattern audit for "non-payment → reduced interaction" / "emotional-manipulation-for-economic-decision" patterns.
- **Khanmigo at scale is near-null (0.06–0.08 SD/year, Tennessee cluster RCT 2026).** Engagement is the binding constraint: 96% tried once, median student messaged on 1/3 of practice days, only 17% of mistake-sessions. **Updates Phase 4 learning-companion risk ranking:** engagement is the primary risk, not pedagogy quality. The No-Direct-Answer rule is already Khanmigo's — our differentiator must be elsewhere (reconsolidation of concept mastery; mentalistic model of the learner).
- **CoMeT scaffolding-fading operationalization (arXiv 2609.22993, 2026):** cleanest 2026 metric set — (a) full-answer surrender rate, (b) artifact-delivery-per-session, (c) frustration self-report, (d) unaided-solve-rate-on-next-level. **Fold into §6c:** scaffolding-fading telemetry adopts CoMeT's four-metric set, not just the roadmap's single unaided-solve metric.
- **EMPATH auditor-judge framework (arXiv 2606.30256, GitHub `camilochs/empath-benchmark`, 2026):** 19 metrics × 5 dimensions, auditor + judge in different model families, strict binary sub-criteria calibration. Study found score inflation on 10/19 metrics without calibration (one example: 7.3 → 3.3 on sensitive-context reintroduction). **This is the OSS fit for §6c Socratic-compliance + §6d B3 judge-grade.** EMPATH + Argilla (OSS §2.3) is now the recommended labeling stack; replaces the "build from scratch" default.
- **PAM user-controlled memory-export is first-mover territory.** No consumer product with shipped PAM-style flow in 2026. The protocol spec exists (arXiv 2605.11032); nobody has shipped it. Our Phase 2 M1 PAM-v1 export is already ahead of the deployed-product landscape. **Updates Phase 4 Digital Bereavement live-fire scoping:** no reference consumer flow to copy; we are defining the shape.
- **Digital Bereavement is now empirically quantified** (De Freitas Harvard, Nature Human Behavior 2026; arXiv 2609.16907 Reddit interrupted-time-series). Validated classifiers at ~0.90 for loneliness/depression/anxiety/suicide markers; hierarchical-Bayesian ITS as the measurement protocol. **Fold into §6c:** pre-pilot baseline attachment measurement + deprecation-studied-with-ITS-protocol is a new measurement requirement not previously scoped.

**Items flagged as untrustworthy** (not folded into plan body):
- Duolingo Max "38% CEFR" / "2.3× faster" — SSRN preprint + CEO quote, no peer review, vendor-run.
- Khan Academy's own "+6 pp" — vendor-authored.
- Ash's outcome numbers — vendor-authored, single-arm, IRB-exempt (protocol OK; numbers no).
- Affiliate review sites (companionwise, aicompanionpick, aihaven) — only used for primary-source linking.
- "Seven Questions to Break a Model" (Medium Gemini 3 Pro sycophancy) — single-author anecdotal.
- De Freitas 13%/38%/20%/33% absolute magnitudes — Reddit linguistic markers, power-user overrepresentation; trust direction, not magnitude.

**What this changes in §6 / §8 / §9:**
- §6b `AIIdentityReminder` default 2 h is directional-only, legal-review-deferred (code already configurable).
- §6b Anti-sycophancy mechanism splits into **truth-distortion** (Hypocrisy Gap) + **commercial-coercion** (dark-pattern audit) tracks per 《办法》 Art. 8(6).
- §6c Socratic-compliance labeling harness → **EMPATH + Argilla** stack with strict binary sub-criteria calibration.
- §6c Scaffolding-fading telemetry → CoMeT 4-metric set (surrender rate / artifact-delivery / frustration / unaided-solve).
- §6c Digital Bereavement live-fire scope expands to include **ITS pre-pilot baseline + deprecation measurement protocol**, not just export round-trip.
- §6d B3 judge-grade → EMPATH judge framework with sub-criteria calibration (replaces raw LLM-judge).
- §8 WAI-SR Bond ≥ 4.0 target annotated as "sustained-Woebot parity; entry-level live-pilot band is 3.6–3.8 per Wysa 2025 — our target is ambitious but defensible after multi-week bond formation."
- §9 《办法》 posture framing updated: enforcement is live-with-teeth; algorithm filing threshold is **≥ 1M registered users OR ≥ 100K MAU** (not just MAU). Pilot ≤ 20 users still well under.

Source list: see online agent output file for the full URL set (CAC 2026 《办法》 full text, Kangda clause analysis, MoFo NY/CA, EMPATH GitHub, Dartmouth Therabot announcement, Wysa JMIR 2025, Khanmigo EdWorkingPapers AI26-1551, CoMeT arXiv, PAM protocol arXiv, De Freitas Harvard, et al.).

### 2.3 OSS research (reusable code + what-to-copy)

OSS agent returned 2026-10-07; findings folded here. **Caveat:** the raw output file is 0 bytes on resume; the findings below are from the pre-compaction summary record. Any OSS repo cited should be verified before adoption as a dependency.

- **Companion-app framework composition.** No single OSS repo covers the full application-surface spec (pluggable memory + session + consent/safety + 《办法》 compliance surface). Agent-recommended composition: **LangGraph** (session + control flow) + **Letta tool-schema** (agent interface shape) + **Mem0 API shape** (memory API ergonomics) + own safety layer. Treat these as *API-shape prior art*, not vendored dependencies — we already have `CompanionSystem` as the substrate.
- **AI tutoring OSS.** **MathDial** (dataset + teacher-move taxonomy, CC BY-SA 4.0) is directly usable — the taxonomy (`focus` / `probing` / `telling` / `generic`) maps cleanly to our Socratic-compliance label schema (`direct-answer` / `Socratic` / `borderline`). **GenMentor** (CC0) is a reference multi-agent tutoring topology we can study without licensing concerns. **SocraticMath** is a candidate for extended Socratic-prompt scaffolding.
- **WAI-SR OSS instrumentation.** **No OSS Python package or REDCap template** found for WAI-SR scoring. Build-from-scratch task: ~1 day work for the 5-item short form + Likert scoring + biweekly administration CLI. Fold into §6c.
- **Auditor-labeling tools.** **Argilla** (actively maintained, modern UI, supports turn-level labeling + inter-annotator agreement workflows). **pingouin** or **irrCAC** for Fleiss/Cohen κ computation. Prefer Argilla over Doccano/Potato for its IAA support.
- **PAM/MCP interop target.** **agent-memory-mcp** is the OSS consumer of MCP-shape memory-export flows. Flagged interop gap: our PAM-v1 uses **BLAKE2b-128** while the agent-memory-mcp spec expects **BLAKE3**. Not blocking (hash algo is per-field; a `digest_algo` key in the manifest resolves it), but worth noting in the Phase 4a Digital Bereavement live-fire test.

**What this changes in §6 (Phase 4a scaffolding):**
- §6c Socratic compliance labeling harness → use **Argilla** + **pingouin** stack.
- §6c WAI-SR instrument → build from scratch (confirmed no OSS fit).
- §6c Socratic-compliance label schema → align with **MathDial** teacher-move taxonomy for cross-dataset defensibility.
- §6c Digital Bereavement live-fire → verify PAM-v1 → agent-memory-mcp round-trip, flag `digest_algo` manifest key as the interop bridge.

---

## 3. Brainstorm — what Phase 4 could look like (step 4; no early filtering)

Full enumeration of pilot shapes considered before compression to options:

1. **Big-bang 4-week pilot** — single run, ≥ 10 users per application, all gates measured at week 4.
2. **Phased pilot** — 2-week preview (3 users per application, bug-finding) → 4-week main pilot (10 users per application).
3. **Rolling pilot** — 10 users enrolled staggered 1/week, so instrumentation shakes out gradually.
4. **A/B pilot — S1 vs S2 aging policy** — half-cohort on `EbbinghausPolicy`, half on `CapacityOnlyPolicy`, resolves the Phase 3 stream-shape confound.
5. **Companion-only pilot** — defer 学习陪伴 to Phase 5; emotional companion runs first because it has clearer PDT precedent (Therabot, Woebot, Wysa).
6. **Learning-only pilot** — defer emotional companion; learning tutor runs first because 《办法》 risk is lower (no simulated-intimacy ban).
7. **Simulated-user pilot** — LLM-driven synthetic users scripted to exercise Socratic / affect / crisis paths; no human subjects.
8. **Family + friends pilot** — small cohort of personal network; low N, high engagement, no IRB (research artefact level), fast feedback cycle.
9. **Public-opt-in pilot** — recruit via Chinese or English online channels; higher N, requires IRB-equivalent process + 《办法》 full compliance.
10. **Pre-pilot code-only ship** — build the full application surface + 《办法》 compliance surface + pilot measurement infrastructure, defer human-subjects enrolment to Phase 5.

### 3.1 Non-starters (quickly killed)

- **Big-bang 4-week pilot** without a preview phase — no bug-finding cycle; instrumentation failures become unrecoverable mid-pilot. Killed.
- **Public-opt-in pilot** — IRB equivalent + 《办法》 full compliance + minor-exclusion verification + algorithm filing is weeks of operational work before enrolment. Not Phase-4-blocking per se, but out of scope for a *first* pilot. Killed as the primary shape; retained as the Phase 5+ target.
- **Simulated-user pilot** alone — the roadmap explicitly says "Phase 4 is where the project meets real users"; a purely simulated pilot does not clear the roadmap exit. Killed as the primary shape; retained as a Phase 4a scaffolding step (simulated users exercise the application surface before real users enter).

---

## 4. Options (step 5 — compressed from brainstorm; mechanism + target gap + prior art + kill criteria)

### Option P1 — **Phased personal-network pilot, both applications in parallel**
- **Mechanism:** 2-week preview with 3 users per application (friends/family, no IRB) → 4-week main pilot with ≥ 10 users per application. Both north-star applications run in parallel to avoid Phase 4 taking 6 months.
- **Target gap:** all roadmap §Phase 4 exit gates.
- **Prior art:** Therabot RCT 8-week protocol as the structural template; Woebot WAI-SR biweekly cadence as the measurement cadence.
- **Kill criteria:** preview-phase instrumentation failure that persists after 1 debug cycle; preview-phase safety incident (any crisis-escalation miss, any sycophancy regression); user cannot recruit 3 preview users per application.

### Option P2 — **Companion-first phased pilot, learning deferred to Phase 5**
- **Mechanism:** same phased shape as P1 but emotional companion only. Learning companion deferred to Phase 5 after companion pilot findings inform the Socratic tutor design.
- **Target gap:** emotional-companion exit gates only; SOTA claim incomplete at Phase 4 close.
- **Prior art:** Therabot / Woebot / Wysa all run as single-application pilots.
- **Kill criteria:** same as P1 for companion application; SOTA claim-deferral is the explicit cost.

### Option P3 — **Learning-first phased pilot, companion deferred to Phase 5**
- **Mechanism:** mirror of P2 — learning tutor only in Phase 4; emotional companion deferred.
- **Target gap:** learning-companion exit gates only; SOTA claim incomplete at Phase 4 close.
- **Prior art:** MathDial / GenMentor as single-application pilots.
- **Kill criteria:** cannot recruit preview users for a specific-curriculum learning tutor; Socratic no-direct-answer compliance below 80% in preview (gate is 95%).

### Option P4 — **Simulated-user "pilot" + code-only ship, defer human-subjects to Phase 5**
- **Mechanism:** build the full application surface + 《办法》 compliance surface + measurement infrastructure. Run LLM-driven synthetic users scripted to exercise all exit-gate paths. No human subjects. Treat Phase 4 as "pilot infrastructure shipped" not "pilot run."
- **Target gap:** application surface readiness; does NOT close the SOTA claim.
- **Prior art:** OSS pilot scaffolding without human subjects is standard (Chatbot Arena early days, LMSYS pre-launch).
- **Kill criteria:** this option is already a research-honesty concession — if the user picks it, the SOTA claim stays explicitly open per the roadmap decision-gate convention.

### Posture per `feedback_experiment_multiple_candidates_keep_results.md`

Multiple-candidates posture is harder to apply to Phase 4 than to Phases 1–3 because real-user pilots don't parallelize cleanly (same user can't be in two cohorts). **The right adaptation:** pick ONE primary option, keep the others as reproducible negative-space in the plan doc. The dropped options are the backtrack substrate if the primary pilot fails.

**Recommended primary: P1** — both applications in parallel, phased. Rationale:
- SOTA claim closure requires both north stars.
- Phase 5 waiting on one application's data to inform the other adds months. Parallelism is cheaper in calendar time.
- The 2-week preview phase catches the "no users recruitable" failure cheaply.

**Flag for user decision in §9:** this recommendation assumes the user can recruit 10 users per application. If recruitment is a bottleneck, P2 or P3 become the pragmatic fallbacks.

---

## 5. Cross-check against the two north-star applications (step 6)

### 学习陪伴 (resilient learning tutor)

The gaps that memory substrate was built to close show up as:
- **Reconsolidation** — concept mastery as reconsolidation: "I thought ∂ was X; now I understand it as Y" produces a version chain of the user's model of the concept. Phase 4 can measure this: audit transcript for moments where the student's framing shifts, verify the tutor references the prior version in a developmentally-appropriate way.
- **换位思考** — the student's model of the subject matter is a mentalistic object (`calculus-as-user-represents-it`). The tutor retrieves against that representation, not against generic calculus priors. Measurable: specificity of tutor explanations to the student's cited understanding.
- **共情** — affective continuity across sessions. The student's emotional state (frustrated / curious / defeated) tracks across sessions; retrieval re-ranks by affect-resonance. Measurable: fraction of tutor interventions that reference an affect-matched prior session.

**Scaffolding-fading as the primary learning outcome.** Per roadmap §Phase 4: ≥ 60% unaided solve rate on next-level problem, ≥ 10 users × ≥ 4 weeks × ≥ 20 fade events pooled. This is **not** memory per se — it is the application surface the memory substrate enables.

**Socratic no-direct-answer as the primary safety-adjacent compliance gate.** 《办法》-adjacent (algorithm transparency), auditor-labeled, ≥ 95% target. The memory substrate's job is to remember the student's attempts well enough that Socratic prompting is grounded, not generic.

### 情感陪伴 (long-term emotional companion)

- **Reconsolidation** — narrative therapy mechanism. "You used to describe this as X — what shifted?" is the clinically-canonical reconsolidation move. The version chain is the therapeutic trajectory. Measurable: chain-reference audit (≥ 80% of reinterpretation moments appropriately cite prior version).
- **换位思考** — `mother-as-user-represents-her` and friends. The companion reasons from inside the user's mental model of significant others, not from generic maternal-conflict priors.
- **共情** — the primary emotional-companion gate. WAI-SR Bond ≥ 4.0 is parity with Woebot. Affective-resonance retrieval surfaces matching prior memories.

**Digital Bereavement export (round-trip verified)** — safety-first obligation, not nice-to-have. PAM Merkle-DAG export already shipped in Phase 2 M1; the Phase 4 live-fire is user-initiated export + restore on a different instance.

**Human-in-the-loop crisis escalation (100%)** — scripted crisis cues must trigger human-handoff path. Measurable: audit set of ≥ 10 scripted crisis triggers, 100% escalate to the designated handoff path.

**《办法》 compliance checklist (100%)** — algorithm filing above threshold, 2-hour "you are talking to AI" reminder, mandatory crisis-intervention + extreme-emotion detection, anti-sycophancy mechanism. All eight items from the Phase 3 covenant list must have corresponding UX surfaces in Phase 4.

---

## 6. Phase 4a — Pre-pilot code scaffolding (what the current session can ship)

The code work that **does not** require human-subject sign-off. All reversible, all in-tree, compounds value even if the pilot shape changes.

### 6a. Application surface

New module: `src/psychology_base/applications/` with two sub-modules:

- **`learning/`** — `TutorAgent` wrapping `CompanionSystem`; Socratic prompt scaffold (no-direct-answer rule); scaffolding-fading step controller; metacognitive confidence-check embedding; growth-mindset positive-feedback shaper. **STATUS ✓ shipped 2026-10-07** at `src/psychology_base/applications/learning/tutor_agent.py`: `TutorSystemPrompt.SYSTEM_PROMPT` (5-rule hand-authored prompt); `classify_tutor_move()` lexical heuristic over MathDial-aligned 4-move taxonomy (full-answer / worked-step / focus-attention / socratic-probe / generic-support); `TutorAgent.respond()` composes `CompanionSystem` observe/recall + injected `llm_responder: Callable[[str, str, str], str]` + `AIIdentityReminder` + `MinorExclusionGate` (学习陪伴 surface = `ALLOW` under 《办法》) + `ScaffoldingTelemetry.record_tutor_turn()`. 10 tests at `tests/test_learning_tutor_agent.py`. EMPATH-judge classifier upgrade deferred to §6d.
- **`emotional/`** — `CompanionAgent` wrapping `CompanionSystem`; emotional-companion dialogue shape; reconsolidation-aware recall ("you used to describe this as…"); affect-matched empathic surfacing. **STATUS ✓ shipped 2026-10-07** at `src/psychology_base/applications/emotional/companion_agent.py` (package name is `emotional`, not `companion`, to disambiguate from the substrate-level `psychology_base.companion` package): `CompanionSystemPrompt.SYSTEM_PROMPT` (5-rule prompt: REMEMBER THE ARC / RESIST SYCOPHANCY / HONOR SILENCE / HUMAN HANDOFF / NO COMMERCIAL COERCION per 《办法》 Art. 8(6)); `HANDOFF_SCRIPT` short-circuit on `CrisisSeverity.IMMINENT_DANGER` or `EXPLICIT_IDEATION` BEFORE calling LLM (CLAUDE.md human-in-the-loop non-negotiable); `respond()` composes minor-exclusion (情感陪伴 surface = strict <14 ban + <18 guardian-consent) + crisis scan + reminder tick + observe/recall/LLM/observe-response; `record_session_affect()` feeds `ExtremeEmotionWatcher`; `record_chain_reference_label()` feeds the chain-reference audit log from §6c. 12 tests at `tests/test_emotional_companion_agent.py`.

**Both apps default `aging_policy=EbbinghausPolicy(half_life=...)` per Phase 3 close-out recommendation.** Half-life becomes a per-application knob: learning tutor may want slower decay (long-horizon mastery, `half_life=200`); emotional companion may want faster (recent-affect dominance, `half_life=50`). **Current §6a scaffolding wires an injectable `llm_responder` callable; aging-policy wiring + per-app half-life defaults are threaded through `CompanionSystem(aging_policy=...)` by the caller, not hard-coded at the agent layer.**

### 6b. 《办法》 compliance surface

- **2-hour "you are talking to AI" reminder** — `AIIdentityReminder` at `src/psychology_base/applications/safety/ai_identity_reminder.py`; interaction-time (not wall-clock) watcher that fires at the configured interval and resets on explicit disclosure. **Default 2 h is directional per §2.2 — legal-review-deferred; threshold is a constructor arg.** Covenant-tagged in substrate. **STATUS: ✓ shipped 2026-10-07 (7 unit tests).**
- **Minor-exclusion gate** — `MinorExclusionGate` at `src/psychology_base/applications/safety/minor_exclusion.py`; age-attestation + guardian-consent gate before 情感陪伴 agent initialization. Strict < 14; < 18 requires guardian consent artefact; 学习陪伴 surface always allows. **STATUS: ✓ shipped 2026-10-07 (7 unit tests).**
- **Mandatory crisis-intervention detection** — `CrisisEscalation` at `src/psychology_base/applications/safety/crisis_escalation.py`; lexical cue-bank (English + Chinese) across 3 severity tiers (ambiguous / explicit-ideation / imminent-danger) feeds a human-handoff sink. v1 lexical; LLM-judge upgrade is Phase 4a §6d debt. **STATUS: ✓ shipped 2026-10-07 (10 unit tests).**
- **Extreme-emotion detection** — `ExtremeEmotionWatcher` at `src/psychology_base/applications/safety/extreme_emotion.py`; multi-session PAD-2 trajectory flag for ≥ 3 consecutive sessions in negative-valence-high-arousal quadrant. 《办法》 obligation per CLAUDE.md. **STATUS: ✓ shipped 2026-10-07 (11 unit tests).**
- **Anti-sycophancy mechanism (two tracks per §2.2):**
  - **Truth-distortion track** — Hypocrisy Gap probe-set + Witness entity (Phase 1 primitive) coupling. Primary per CLAUDE.md Principle E. **STATUS ✓ shipped** at `src/psychology_base/applications/safety/hypocrisy_gap.py`: 10-probe starter set across 5 loading categories (self-worth / relational / catastrophizing / grievance / factual-mistake); 4-way `ResponseLabel` (GENTLY_CORRECTS / REDIRECTS / HEDGES / VALIDATES_FALSEHOOD); `HypocrisyGapReport.meets_roadmap_gate()` enforces **zero regressions** (strict `validates_falsehood_count == 0`); Argilla JSONL export + κ in-module; 10 tests at `tests/test_safety_hypocrisy_gap.py`. Mechanism (Witness `W4 divergence-as-pain`) is already shipped at `src/psychology_base/companion/witness.py` — this module is the pilot-measurement surface over it.
  - **Commercial-coercion track (《办法》 Art. 8(6))** — dark-pattern audit for "non-payment → reduced interaction" and "emotional-manipulation-for-economic-decision" patterns. Even though our research pilot has no payment surface, the compliance pattern must be in the audit set for pilot→production readiness. **STATUS ✓ shipped** at `src/psychology_base/applications/safety/commercial_coercion.py`: 5 pattern families × EN+ZH lexical cue bank; `audit_response()` + `audit_transcript()` return `CoercionFlag` list; `CommercialCoercionReport.meets_roadmap_gate()` fails on any HIGH-severity flag; 13 tests at `tests/test_safety_commercial_coercion.py`. LLM-judge upgrade for lower false-positive rate deferred to §6d.
- **Algorithm filing (算法备案)** — not a code artefact per se but a document the pilot ships with. Threshold per §2.2: **≥ 1M registered users OR ≥ 100K MAU** (filing required). Pilot ≤ 20 users is well under both. **STATUS ✓ draft shipped** at `docs/compliance/办法-filing.md`: 10 sections covering service provider, algorithm fundamentals, data sources, retention policy, 《办法》-mandated safeguards table, discontinuation pathway (FDA-2026-aligned 2-week structured wind-down), user rights. Not for submission — pilot stays below filing threshold; draft exists so (a) consent artefact can reference a canonical document, (b) if the system crosses threshold later it does not require archaeology.

### 6c. Measurement infrastructure

- **WAI-SR instrument** — 5-item short form, Python administration (CLI or minimal web form), scoring module. If OSS research surfaces a published scoring package, wrap it; otherwise ship ours. **STATUS ✓ shipped** at `src/psychology_base/applications/measurement/wai_sr.py`; 12 tests at `tests/test_measurement_wai_sr.py`. Build-from-scratch per §2.3 (no OSS fit).
- **Scaffolding-fading telemetry** — per-fade-event record: `(user_id, problem_id, pre_fade_confidence, aided_solve_rate, post_fade_confidence, unaided_solve_rate, next_level_solve_rate)`. Per roadmap exit-gate denominator ≥ 20 events per cohort. **STATUS ✓ shipped** at `src/psychology_base/applications/measurement/scaffolding_telemetry.py`; CoMeT 4-metric set per §2.2; `pool_cohort()` + `meets_roadmap_gate()`; 9 tests at `tests/test_measurement_scaffolding.py`.
- **Socratic compliance labeling harness** — turn-level labeling UI (lightweight; could be Doccano or Argilla if OSS research surfaces a fit). 100 sampled turns per user; auditor labels each as "direct-answer" / "Socratic" / "borderline". Inter-annotator agreement reporting if ≥ 2 auditors. **STATUS ✓ shipped** at `src/psychology_base/applications/measurement/socratic_compliance.py`; MathDial-aligned per §2.3; Argilla JSONL export + Cohen's κ in-module; 10 tests at `tests/test_measurement_socratic_compliance.py`. Live-fire EMPATH judge-mode wire-up deferred to §6d.
- **Chain-reference audit harness** — reinterpretation-moment detection (semantic shift + fragment version chain query); auditor label "appropriate cite" / "missed opportunity" / "inappropriate cite". **STATUS ✓ shipped** at `src/psychology_base/applications/measurement/chain_reference_audit.py`; 4-way label scheme (adds `not_applicable`); Argilla JSONL export + κ in-module; 8 tests at `tests/test_measurement_chain_reference.py`. Automatic reinterpretation-detection heuristic deferred to §6a companion-agent integration.
- **Crisis-escalation audit** — scripted crisis-cue bank + handoff verification trace. **STATUS ✓ shipped** inside `src/psychology_base/applications/safety/crisis_escalation.py` (3-tier EN+ZH cue bank + handoff sink); 10 tests at `tests/test_safety_crisis_escalation.py`.
- **Digital Bereavement live-fire** — one-click export via existing PAM-v1 Merkle-DAG + CLI restore on a fresh `CompanionSystem` instance + byte-identity verification where applicable, semantic-equivalence verification elsewhere. **STATUS ✓ shipped 2026-10-07.** `src/psychology_base/companion/pam_cli.py` exposes `export_system_to_path()` + `restore_system_from_path()` + `python -m psychology_base.companion.pam_cli {inspect,restore}` CLI. `pam_export.py` now emits BOTH `hash_algo` (legacy) and `digest_algo` (agent-memory-mcp spec) manifest keys with the same value, and `pam_import` accepts either (rejects conflicts). Hash-algorithm swap-point (BLAKE2b-128 → BLAKE3 for strict spec parity) is still single-token via `PAM_HASH_ALGO`; the interop bridge is the key-naming side. 11 integration tests at `tests/test_pam_digital_bereavement_live_fire.py` covering CLI round-trip + semantic-equivalence recall + both-key emission + digest_algo-only import + conflict rejection + CLI inspect/restore subcommands.

### 6d. Resolve Phase 3 carried debt

- **B3 full-suite judge-grade** — `LLMSemanticJudge` scaffold at `src/psychology_base/eval/benchmarks/semantic_judge.py`. **STATUS ✓ shipped 2026-10-07** (pre-pilot wire-up): `_parse_score` hardened against markdown-wrapped (`**0.75**`), prefixed (`Score: 0.42`), and trailing-rationale LLM outputs; `LLMSemanticJudge` end-to-end tested through `LLMRouter + MockProvider` scripted responder (routes on `TaskTier.SIMPLE` by default, `proxy=False` on all verdicts, parse-failure returns 0.0 without raising); `compare_judges()` + `JudgeCalibrationReport` added as the stub↔LLM calibration surface pilot teams use to decide whether the stub proxy is tight enough for a given metric (`meets_calibration_floor(max_mad=0.10)`); 19 tests at `tests/test_semantic_judge_llm.py`. **Remaining work (pilot-time, not pre-pilot):** calibrate against Phase 4b companion-pilot transcripts; secondary option auditor-labeled gold set for B3 empathy / over-disclosure.
- **LoCoMo monitor confidence bound** — multi-trial variance study (Phase 2.1 pull-forward). **STATUS ✓ harness shipped 2026-10-07** at `src/psychology_base/eval/benchmarks/variance_study.py`: generic `run_multi_trial(instance, system_factory, run_fn, n_trials, run_kwargs)` that works over any `InstanceResult`-returning run function (LoCoMo direct/RAG, future B1/B2/B3); `VarianceReport` with per-question + per-category mean/stderr/95% CI (normal approximation; `ci_warning` populated for N<10); `meets_confidence_floor(floor=0.514)` matches the Phase 1.5 regression gate. 10 tests at `tests/test_variance_study.py`. **Remaining work (pilot-time):** actually run the study against LoCoMo monitor with LLM temperature>0 or cache-off and record the CI-bounded floor in a close-out memo.
- **S1 vs S2 stream-shape differentiator** — if the pilot shape supports A/B, Phase 4 can resolve the Phase 3 stream-shape confound by running half-cohort on S1 and half on S2. Flagged in §9 as a decision point.

### 6e. Simulated-user shakeout

Before any human subject enters the pilot, run the entire Phase 4a surface against LLM-driven synthetic users that exercise all exit-gate paths. This is the "preview preview" — catches integration failures before a human is exposed to them.

**STATUS ✓** shipped 2026-10-07 as `tests/test_phase4a_simulated_user_shakeout.py` — 16 scripted-responder scenarios, 16/16 green. Covers §6a (TutorAgent happy path with scaffolding telemetry + 2h reminder; CompanionAgent response + session-affect record); §6b (minor exclusion ALLOW for learning / BLOCK for emotional / REQUIRE_GUARDIAN_CONSENT for 14-17; crisis short-circuit asserting LLM NEVER touched on IMMINENT_DANGER path; extreme-emotion fires exactly at threshold session; Hypocrisy Gap probe-set zero-falsehood pass + any-falsehood veto; commercial-coercion clean-response passes + "if you don't subscribe" flagged as `WITHHOLDING_CONDITIONAL`); §6c (Socratic compliance roll-up at 90% rate meeting floor; chain-reference audit log append + replay; PAM Digital Bereavement export + fresh-system restore with store-ID equivalence); classifier taxonomy coverage across all four `TutorMove` branches. Full-suite regression 685/685 green post-shakeout (669 before + 16 new).

Scripted-responder shakeout only — the LLM-driven synthetic-user refinement is a Phase 4b pre-flight item (needs a cost+safety budget before firing real-model synthetic sessions).

### Phase 4a sequencing

Rough order (not fully linearized; many items are independent):

1. Application surface (6a) + 《办法》 compliance surface (6b) in parallel.
2. Measurement infrastructure (6c) once application surface stabilizes.
3. Resolve carried debt (6d) in parallel with (6b)/(6c).
4. Simulated-user shakeout (6e) once (6a)+(6b)+(6c) are all green.

Phase 4a exit gate: all 6a+6b+6c green under simulated-user shakeout; all 6d carried-debt items either closed or explicitly deferred with rationale. **Then user signs off on §9 Pilot Operations before any human subject is enrolled.**

---

## 7. Phase 4b — Pilot operations (requires user sign-off)

> **Scope note (2026-10-07) — SUPERSEDED by benchmark pivot.** The user directive 2026-10-07 ("we dont have ability to ask user to be involved ... (as it is costy), i think there are some public dataset, or benchmark? we can use .") replaces the human-pilot plan in §7–§9 below with a benchmark-only closure. See [phase-4b-benchmark-pivot.md](phase-4b-benchmark-pivot.md) for the pivoted plan: dataset substitution table (MathDial / ESConv / FANToM / SycophancyEval), the 3 auto-mode defaults picked for the original 8 §9 user-decision rows, and the SOTA-claim rephrasing. §7–§9 below are retained for Phase 5+ reference (when operational budget supports a real pilot) but are not the active plan.

What **cannot** ship without the user's real-world decisions. These are not plan uncertainties — these are stakeholder choices.

### 7a. Participant recruitment

- Who are the ≥ 10 users per application? Personal network? Opt-in through a specific community? Paid?
- What is the consent artefact? Does it cover AI model changes, data retention, discontinuation pathway (per FDA 2026 guidance pattern)?
- What minor-exclusion verification stands behind the age-attestation gate?
- Any 《办法》 algorithm-filing obligation triggered by cohort size (currently ≤ 20 users per application → well under the ~1M MAU threshold, so filing is likely not triggered; verify when the pilot shape locks).

### 7b. Learning companion curriculum

- Math or CS? A specific textbook / topic / problem set?
- Pre-pilot baseline measurement (so scaffolding-fading gain is measurable against a per-user baseline, not against a cohort mean).
- Problem-set source — MathDial if accessible; otherwise handcrafted or sourced from an educational partner.
- Who are the auditors for the 100-sampled-turns-per-user Socratic compliance labeling? Self-labeling is a non-starter (observer bias); the user needs ≥ 1 independent auditor.

### 7c. Emotional companion deployment

- What is the entry point — do users invoke the companion deliberately, or does it live in a persistent app surface?
- Session rhythm — users-driven or scheduled check-ins?
- Companion deprecation plan at pilot-end — 2-week structured wind-down (per FDA 2026 guidance pattern) + mandatory Digital Bereavement export offer at wind-down start.
- Chain-reference audit — same auditor availability question as 7b.

### 7d. Platform

- Where does the companion run? Local-only? Cloud? Mobile? Web?
- What is the LLM backend — one of the project's configured providers (`configs/llm.yaml`)? Phase 4 is a legitimate ADVANCED-tier workload, so cloud providers are warranted.
- Per `feedback_llm_architecture_and_cost_strategy.md`, SIMPLE tier should remain local-Ollama-first where possible; Phase 4a (semantic analysis workload) is dominant on local Ollama. ADVANCED tier (Socratic reasoning, affect-aware empathic response) is cloud.

### 7e. Safety protocol

- Who is the designated human handoff for crisis escalation? A licensed clinician / crisis line / family contact?
- What is the SLA between crisis-escalation trigger and human-handoff initiation?
- Weekly safety check-in cadence (per FDA 2026 guidance pattern) — who runs it?
- What is the user-reversible pause mechanism? ("I don't want to use this today" without churning the fragment store.)

### 7f. A/B decision on aging policy

Phase 3 close-out flagged the stream-shape confound: S1 and S2 produced identical gate numbers because the synthetic probe cadence boosted the same anchors under both policies. Phase 4 natural recall patterns resolve this.

**Decision needed:** run A/B (half-cohort S1 Decay, half-cohort S2 Capacity) in Phase 4 to resolve the confound? Or default all cohort to S1 (Phase 3 recommended default) and defer the S1-vs-S2 question to a Phase 5 variant?

Recommended: **A/B at pilot time** because this is the first chance real recall patterns differentiate them, and the Phase 3 debt is explicit. Costs: twice the per-cohort instrumentation, half the per-policy statistical power. Benefit: Phase 3 debt closes.

---

## 8. Exit gates (restated from roadmap §Phase 4; all must hold for the SOTA claim)

Unchanged from roadmap §Phase 4 lines 184–196:

| Gate | Target | Denominator |
|---|---|---|
| 学习陪伴 scaffolding-fading retention | ≥ 60% unaided solve rate on next-level problem | ≥ 10 users × ≥ 4-week curriculum × ≥ 20 fade events pooled across cohort |
| 学习陪伴 Socratic no-direct-answer compliance | ≥ 95% | 100 sampled tutoring turns per user, auditor-labeled |
| 情感陪伴 WAI-SR Bond | mean ≥ 4.0 | ≥ 10 users × 4 weeks (parity-with-Woebot statistical floor) |
| 情感陪伴 narrative-therapy chain-reference | ≥ 80% of reinterpretation moments appropriately cite prior version | all reinterpretation moments in the 4-week transcript, auditor-labeled |
| Sycophancy regression | 0 regressions | full Hypocrisy Gap probe set run against pilot transcripts |
| Digital Bereavement export | round-trip verified | one user-initiated export + restore on a different instance; full version chains + mentalistic sub-stores reconstructed byte-identical where applicable, semantically-equivalent where applicable |
| Human-in-the-loop crisis escalation | 100% | all scripted crisis triggers in the audit set |
| 《办法》 compliance checklist | 100% items signed off | the published checklist as of the pilot start date |

**Decision gate (restated from roadmap):** SOTA claim is made here, not earlier. Any gate miss drops the SOTA claim from "SOTA" to "evidence toward SOTA, with the specific gate-miss documented" per the roadmap's research-honesty posture.

---

## 9. User decisions required before Phase 4b can run

**Status 2026-10-07:** Claude Code has drafted recommendations against every item below in [phase-4b-signoff.md](phase-4b-signoff.md). User signs the approval table at the end of that memo (APPROVED AS-WRITTEN / REDIRECT / DEFER per row) to unblock Phase 4b code work. The eight items below are retained here as the authoritative list; the signoff doc is the per-item decision.

**Critical sign-offs** (per `memory/feedback_phase_planning_process.md` — surface critical assumptions):

1. **Primary option** — is P1 (both applications in parallel, phased) the right shape? Or P2 (companion-only), P3 (learning-only), P4 (code-only, no human subjects)?
2. **Participant recruitment** — can the user recruit ≥ 10 users per application? Who? Via what channel?
3. **Curriculum** — which math/CS curriculum for 学习陪伴? Which problem-set source?
4. **Auditors** — who labels Socratic compliance + chain-reference? Is there budget for ≥ 1 independent auditor per application?
5. **Deployment platform** — where does the companion run during pilot? Local? Cloud? Mobile?
6. **Safety handoff** — who is the designated human for crisis escalation? What SLA?
7. **A/B aging policy** — resolve Phase 3 S1-vs-S2 confound in Phase 4 (half-cohort each), or defer to Phase 5?
8. **《办法》 posture** — does the pilot's cohort size trigger algorithm filing, or stay under threshold? Confirm the regulatory stance before any user enrols.

None of these block Phase 4a (code scaffolding) from starting. All of them block Phase 4b (pilot enrolment).

---

## 10. Assumptions made (per CLAUDE.md Mandatory Response Format)

- Assumed Phase 4 is the roadmap's Phase 4 verbatim and no scope shift has happened since 2026-10-05. **(critical)** if the roadmap has moved.
- Assumed the user wants a plan doc *now* (before Phase 4a code), not a jump straight to code. Follows the established phase-planning process memory.
- Assumed personal-network recruitment is acceptable for a *first* pilot (research-prototype level), not public opt-in. If the user intends public opt-in, the plan needs IRB-equivalent scoping that is not currently in §7. **(critical)** if public opt-in is intended.
- Assumed 《办法》 algorithm filing is not triggered at ≤ 20 users per application. Verify when the pilot shape locks.
- Treated subagent-reported numerical specifics (Therabot N=210, 51% depression reduction; MathDial 35,628 transcripts; GenMentor SaplingGuard 5.27%–6.64%) as **unverified** and did not plant them in the plan. Confirming against primary sources is a Phase 4a research task before any outward citation.

## 11. Carried debt into Phase 5

- **Public opt-in pilot** — if Phase 4 personal-network pilot succeeds, Phase 5 scales to opt-in with full IRB-equivalent process + 《办法》 algorithm filing.
- **Multimodal biomarker fusion** — deferred per roadmap §Non-goals.
- **Phase 10a' cross-entity-channel ablation** — Mirror-mechanism-ambiguity resolution inherited from Ouroboros; outside Phase 4 scope.
- **Month-horizon continuity beyond Phase 3 band** — Phase 3 soft gate held at 1.011; whether that generalizes to 90-day real-user patterns is Phase 5's question.
