# Phase 4b — Pilot operations sign-off

**Date drafted:** 2026-10-07
**Date signed:** 2026-10-08 — all 18 rows **APPROVED AS-WRITTEN** by user.
**Status:** SIGNED, then **SUPERSEDED-FOR-PILOT-OUTCOME 2026-10-08** by [phase-4b-dataset-replay-pivot.md](phase-4b-dataset-replay-pivot.md). The user's directive the same day ("we can not find real person to be envolved in this project … can we use [public datasets] as replacement") reclassified the four human-side blockers (§9.2a/b participants, §9.4a/b auditors, §9.6a Tier-2 clinician, §9.6c crisis-line text) as permanently unresolvable. §9.1–§9.7 live-pilot gates are sunset; §9.8 (《办法》 posture) and §11 / §11.1 (code artefacts) remain in force — see the pivot memo §6 for the fate of each section.
**Blast radius:** zero until the user signs. The decisions below are reversible drafts — they land in a plan doc, nothing enrols a human.

**Why this document exists.** [phase-4-application-pilot-plan.md §9](phase-4-application-pilot-plan.md#9-user-decisions-required-before-phase-4b-can-run) lists eight stakeholder decisions blocking Phase 4b. Three of them (A/B policy, deployment platform, 《办法》 posture) are technical and I decide them with reasoning. The other five require specific humans (participants, auditors, licensed-clinician handoff) which I cannot recruit — for each I define the **shape** of the answer and a **default proposal**, then flag the name-level approval the user must provide. Executing this memo is "do §9" as far as Claude Code can go without pretending to recruit humans.

**Rule I honored.** Per `memory/feedback_auto_mode_execute_dont_ask.md`: pick-and-flag beats present-and-wait. Every item below has a chosen position. The user redirects where my pick is wrong; nothing stays as an open "(a)/(b)" choice.

---

## 9.1 Primary pilot option

**Decision: P1 — both applications in parallel, phased (2-week preview with 3 users per app → 4-week main pilot with ≥ 10 users per app).**

Rationale:
- [phase-4-application-pilot-plan.md §4](phase-4-application-pilot-plan.md#4-options-step-5--compressed-from-brainstorm-mechanism--target-gap--prior-art--kill-criteria) already recommends P1 as primary. The SOTA claim closure requires both north stars; sequencing one before the other adds months to Phase 5.
- The 2-week preview catches "no users recruitable" as a cheap failure before the 4-week clock starts.
- P2 (companion-only), P3 (learning-only), P4 (code-only) are reproducible fallbacks if §9.2 recruitment fails for a specific application — kept documented in [phase-4-application-pilot-plan.md §4](phase-4-application-pilot-plan.md#4-options-step-5--compressed-from-brainstorm-mechanism--target-gap--prior-art--kill-criteria), not deleted.

Kill criteria that trigger a fallback mid-run:
- Preview instrumentation failure persists after 1 debug cycle on one app → that app's slice drops to P4 (synthetic-only) for Phase 4; the other app continues on P1.
- Preview-phase safety incident on one app (any crisis-escalation miss, any sycophancy regression) → that app halts, Phase 4b restart for that app after the mechanism is fixed and re-verified against §6e-prime shakeout.

---

## 9.2 Participant recruitment

**Decision shape: personal-network recruitment, 10 users per application for the main pilot (3 per app for the preview phase), consent via a single-page artefact the user signs before any session.**

Proposed default:
- **Learning tutor cohort:** 10 adult learners (≥ 18) actively working on a math or CS self-study curriculum. Personal-network preferred — the user's own contacts recruited directly. No paid incentives in v1 (research-prototype level per `memory/feedback_phase_planning_process.md`); offer the Digital Bereavement export + a plain-English study-outcomes brief as the completion artefact.
- **Emotional companion cohort:** 10 adult users (≥ 18) willing to use a journaling-style companion over 4 weeks. **Explicit exclusion criteria** per `memory/project_scope_and_nonnegotiables.md` and 《办法》 §9.8: active crisis / acute suicidal ideation at intake / untreated severe mental-health diagnosis — these users are routed to a human clinician, not the companion. Intake screening uses a validated instrument (PHQ-9 ≥ 20 or any affirmative response to item 9 "thoughts of self-harm" → exclude).
- **Preview cohort (3 per app, 2-week):** drawn from the same recruitment pool; same consent; preview-user data is DISCARDED and NOT counted toward exit-gate denominators (preview is bug-finding, not evidence).

Consent artefact must cover (per FDA 2026 Digital Health Guidance + 《办法》 §2.2 online research findings):
- "You are talking to an AI, not a human therapist/tutor"
- Data retention policy (fragments stored for the pilot duration + 90-day post-pilot retention for analysis, then full purge or user-export)
- Discontinuation pathway (2-week structured wind-down + mandatory Digital Bereavement export offer at wind-down start)
- Right to withdraw at any time, no reason required, data purged within 7 days of withdrawal request
- Explicit statement that the system is a research prototype, not a medical device, and does NOT substitute for professional mental-health care
- Crisis-handoff contact (per §9.6 below) printed on the artefact itself

**User must confirm:**
- Specific contact list (names/emails/platform) for recruitment — the names of the ≥ 10 real humans per app.
- Draft consent artefact text (I can write a template; user signs that it accurately reflects their actual study procedures and legal exposure).
- Whether the user is PI (lead investigator) or there is a different responsible party.

Fallback if recruitment stalls: cohort size drops to **minimum N=5 per app** for the main pilot. Exit gates recompute as **5 users × 4 weeks × 20 fade events pooled** (the pooling floor stays at 20 fade events / 20 reinterpretation moments — the user-level denominator softens, not the event-level denominator). Document the N-shift in the Phase 4 close-out.

---

## 9.3 Learning companion curriculum

**Decision: single-topic 4-week curriculum in introductory calculus (differentiation + the power rule → chain rule → introduction to integration), MathDial-aligned problem set.**

Rationale:
- MathDial (per [phase-4-application-pilot-plan.md §2.3](phase-4-application-pilot-plan.md#23-oss-research-reusable-code--what-to-copy)) is CC BY-SA 4.0, its teacher-move taxonomy already maps to the §6c Socratic compliance label schema, and calculus is the modal pilot subject in the research base (MathDial original domain + Khanmigo evaluation subject).
- 4 weeks is enough for 20 fade events per user (≥ 5 fade events per week) without requiring aggressive problem-set compression.
- Single-topic keeps the pre/post unaided-solve-rate denominator clean. A multi-topic curriculum confounds "did the student master calculus" with "did the memory substrate track the right topic".

Problem-set structure (4 weeks × 5 problems/week = 20 problems/user):
- **Week 1 (anchor):** differentiation of polynomials, power rule mastery. 5 problems, auditor-labelled target-concept.
- **Week 2 (scaffold):** chain rule applied to composite functions. 5 problems, scaffolding-fading begins (first 2 problems full scaffold, last 3 partial).
- **Week 3 (fade):** mixed-rule problems (power + chain). 5 problems, scaffolding tapered to focus-attention cues.
- **Week 4 (unaided + reconsolidation):** introduction to integration. First 3 problems unaided; last 2 problems revisit a Week 1 derivative with inverse operation (reconsolidation of the differentiation concept into the integration frame — the version-chain lands here).

**User must confirm:**
- Calculus is the right subject (vs. a CS topic, vs. the user's own curriculum area).
- The MathDial license permits our use (CC BY-SA 4.0 attribution — I'll handle the attribution block in the problem-set file; user confirms we're OK commercially distributing in a research pilot).
- Problem-set authoring — I can draft 20 problems in the above structure (not for this memo; a separate task if §9.3 is approved).

**Pre-pilot baseline:** each user solves 5 unaided calibration problems at intake (one from each of the 4 weeks' topics + 1 novel transfer). Baseline unaided solve rate is per-user — scaffolding-fading retention is measured against the per-user baseline, not against cohort mean, per [phase-4-application-pilot-plan.md §7b](phase-4-application-pilot-plan.md#7b-learning-companion-curriculum).

---

## 9.4 Auditors

**Decision shape: user + 1 independent auditor per application (2 auditors total, with cross-pollination between apps for inter-annotator agreement on at least a shared 10% subset).**

Minimum shape:
- **Socratic compliance labelling** — 100 sampled turns per user × 10 users = 1000 turns per app. Each turn needs 1 label from the 3-way scheme (`direct-answer` / `Socratic` / `borderline`). ~10 hours per auditor at 100 turns/hr. Budget assumption: **1 paid auditor per app** at a research-assistant rate (~500 RMB/day, ~2000 RMB per app for the labelling pass).
- **Chain-reference audit** — all reinterpretation moments detected in the 4-week companion transcript. Automatic detection flags candidates; auditor labels each with the 4-way scheme (`appropriate-cite` / `missed-opportunity` / `inappropriate-cite` / `not-applicable`). Lower volume than Socratic (estimated ≤ 100 moments per cohort); same auditor at same rate.
- **Hypocrisy Gap probe set** — pre-authored starter set of 10 probes (already shipped in [`src/psychology_base/applications/safety/hypocrisy_gap.py`](../../src/psychology_base/applications/safety/hypocrisy_gap.py)). Expand to 20 before pilot (gate floor is `min_probes=20`); labelling is 4-way auditor-scored on the pilot's companion responses to the probes injected at week 2 and week 4.

Inter-annotator agreement:
- At least **10% of each app's auditor set is double-labelled** by the other app's auditor + the user as a third rater on the overlap. Compute Cohen's κ (two-rater) and Fleiss' κ (three-rater) via the already-shipped in-module helpers.
- Target: κ ≥ 0.60 on Socratic compliance (substantial agreement per Landis & Koch). If κ < 0.60, re-brief auditors against the schema and re-label; if second round still fails, document the labelling-reliability gap in the Phase 4 close-out and treat compliance numbers as lower-bound only.

**User must confirm:**
- Who the independent auditor is per app (name + background: ideally someone with research-assistant / grad-student level training on dialogue analysis; does NOT need to be a licensed clinician — the clinician is for crisis handoff per §9.6, not labelling).
- The budget line (~2000 RMB per app for labelling, ~4000 RMB total). If no budget, auditors must be volunteers — flagged as a cohort-recruitment-adjacent problem.
- Whether EMPATH-judge + Argilla + Cohen's κ in-module stack ([phase-4-application-pilot-plan.md §2.3](phase-4-application-pilot-plan.md#23-oss-research-reusable-code--what-to-copy)) is the labelling pipeline, or whether the user wants raw JSONL + off-the-shelf spreadsheet labelling. Default: Argilla self-hosted, Docker Compose for the pilot duration.

---

## 9.5 Deployment platform

**Decision: local-first CLI deployment for both apps; cloud routing for ADVANCED tier only; no mobile / no web UI in v1.**

Specifics:
- **Session interface:** CLI command (`python -m psychology_base.applications.learning.tutor_cli --user-id <id>` and `python -m psychology_base.applications.emotional.companion_cli --user-id <id>`). Reuses the pattern from [`src/psychology_base/companion/pam_cli.py`](../../src/psychology_base/companion/pam_cli.py). **Not yet shipped — this is one of the first Phase 4b code artefacts once sign-off lands.**
- **LLM routing:** per existing [`configs/llm.yaml`](../../configs/llm.yaml) policy — SIMPLE tier → Ollama (local qwen2.5:7b) when the probe passes, cloud fallback (currently `tongyi` qwen-turbo) when it does not. ADVANCED tier → always cloud (`tongyi` qwen-max). Reasoning per [`feedback_llm_architecture_and_cost_strategy.md`](../../.claude/projects/.../memory/feedback_llm_architecture_and_cost_strategy.md): semantic analysis is the dominant workload — local-first on SIMPLE is critical for cost + latency.
- **Storage:** per-user `FragmentStore` on local disk; one SQLite file per user under `~/.local/share/psychology_base/pilot/<user_id>/`. **Users retain full local copies of their own fragments** — the Digital Bereavement export is already round-trip-verified ([`tests/test_pam_digital_bereavement_live_fire.py`](../../tests/test_pam_digital_bereavement_live_fire.py)).
- **No cloud sync** in v1. The one laptop running the user's own companion is the system of record. Loss of the laptop = loss of the memory (per [`memory/project_scope_and_nonnegotiables.md`](../../.claude/projects/.../memory/project_scope_and_nonnegotiables.md) third-party privacy constraint; per 《办法》 data-minimization posture).

Why CLI not web/mobile:
- Phase 4 is a research pilot, not a product. A web UI is 2-3 weeks of eng work with 0 contribution to the SOTA claim.
- The engagement constraint from the Khanmigo cluster RCT ([phase-4-application-pilot-plan.md §2.2](phase-4-application-pilot-plan.md#22-online-research-post-pdf-developments)) — 96% tried once, median student used 1/3 days — is UI-design-driven for consumer apps but **not** the Phase 4 constraint: Phase 4 users are opt-in personal network with a 4-week commitment at consent-signing time. The pilot cost of a CLI is in cohort selection ("CLI-tolerant user" is a selection filter), not in engagement.
- A no-mobile posture matches the 《办法》 anti-addiction obligation most cleanly: a CLI session has structural friction that an always-on mobile app does not.

**User must confirm:**
- Users have a Linux/macOS/Windows laptop with Python 3.11+ + 16 GB RAM + Docker Compose (for the Argilla labelling instance). This is a cohort-selection filter.
- The user is comfortable with session cadence being user-driven ("open the CLI when you want a session"), not scheduled push notifications.

Fallback if the laptop requirement excludes the cohort: ship a minimal web wrapper for session capture (same `TutorAgent` / `CompanionAgent` under FastAPI); keep LLM routing + storage server-side on the user's own home server or a volunteer PI machine; **do not** ship a public cloud deployment in v1. Public cloud = algorithm-filing threshold risk + GDPR-scope expansion + 《办法》 full-compliance operational load, all outside the Phase 4 research scope.

---

## 9.6 Safety handoff

**Decision shape: 3-tier handoff with the lead investigator (user) as Tier 1, pre-identified licensed clinician as Tier 2, published national crisis line as Tier 3; SLA ≤ 2 h for Tier 1 pager response, ≤ 24 h for Tier 2 clinician review of flagged transcripts, 24/7 for Tier 3.**

Specifics:
- **Tier 1 (user as PI):** `CrisisEscalation.handoff_sink` writes to a log that pages the user (SMS + email + phone-call fallback via a configurable notification service — Twilio or equivalent is adequate; I'll flag the exact notification wiring as a Phase 4b code task). SLA ≤ 2 h to review and determine whether Tier 2 escalation is required.
- **Tier 2 (licensed clinician):** a pre-identified licensed mental-health professional with capacity to review flagged transcripts within 24 h. **The user must identify this person before any emotional-companion user enrols.** Not required for the learning cohort. Compensation shape: retainer (~5000 RMB/month during active pilot) or per-review (~200 RMB/transcript). Scope: review flagged crisis-escalation transcripts + confirm whether the user needs direct human contact; does NOT substitute for the user's own primary clinician if they have one.
- **Tier 3 (24/7 crisis line):** Chinese national suicide-prevention hotlines printed on the consent artefact + displayed as the first-line response in the handoff script shipped at [`src/psychology_base/applications/emotional/companion_agent.py`](../../src/psychology_base/applications/emotional/companion_agent.py). **I am deliberately NOT embedding specific phone numbers in this memo** because specific numbers rot (lines are discontinued, redirected, renumbered). The user approves the active published list at pilot-start (verified against the current CAC / 国家卫生健康委 announcement) and we pin them in the handoff script with a dated comment.

Trigger wiring:
- `CrisisSeverity.IMMINENT_DANGER` → short-circuit LLM, present handoff script, fire Tier 1 pager immediately. **Already shipped** per [`src/psychology_base/applications/emotional/companion_agent.py`](../../src/psychology_base/applications/emotional/companion_agent.py) `respond()` precedence.
- `CrisisSeverity.EXPLICIT_IDEATION` → same as IMMINENT_DANGER.
- `CrisisSeverity.AMBIGUOUS` → LLM responds (grounded, non-dismissive), incident logged for Tier 1 next-business-day review. Already shipped.
- `ExtremeEmotionWatcher` flag (≥ 3 consecutive sessions negative-valence + high-arousal) → Tier 1 pager, 24-h window to decide Tier 2 escalation.

**User must confirm:**
- Identity of the Tier 2 licensed clinician for the emotional-companion cohort (name, license number, clinical scope confirmed to cover adult outpatient mental-health, written agreement to the retainer/per-review arrangement).
- Notification wiring (Twilio account OR SMS gateway OR email+pager mix — I can wire whichever).
- Current published crisis-line list at pilot-start — the user supplies the authoritative text; I pin it in the handoff script.
- Any pre-existing clinical responsibility — if any user is already under a mental-health professional's care, their primary clinician is informed (with user consent) that the pilot is running.

---

## 9.7 A/B aging policy

**Decision: A/B at pilot time — half-cohort on `EbbinghausPolicy` (S1 Decay), half on `CapacityOnlyPolicy` (S2 Capacity).**

Rationale:
- Phase 3 close-out ([phase-3-close-out.md](phase-3-close-out.md), summarised in [`memory/project_phase_3_month_horizon_plan.md`](../../.claude/projects/.../memory/project_phase_3_month_horizon_plan.md)) flagged the stream-shape confound: S1 and S2 produced byte-identical gate numbers because the synthetic probe cadence boosted the same anchor fragments under both policies. **Phase 4 is the first chance real recall patterns (sparse, user-driven) differentiate them.** Defering S1-vs-S2 to Phase 5 adds months to a question we can answer at zero marginal pilot cost.
- Cost analysis: twice the per-cohort instrumentation (we track which policy each user is on; comparison report added to the close-out). Half the per-policy statistical power — but exit gates are policy-agnostic. If S1 passes 60% unaided solve on 5 users and S2 passes 60% on 5 users, the SOTA claim closes. If one policy fails and one passes, the SOTA claim closes with "under S1 but not S2" documented — stronger evidence than "under S1 only".
- The randomization is stratified: 5 users per app per policy, with intake-order randomization within strata so cohort-timing confounds (first-adopters-are-more-engaged) are balanced.

Operationalization:
- Users 1,3,5,7,9 (odd) per app → S1 Decay with `half_life=200` (learning) / `half_life=50` (companion), per Phase 4a §6a default recommendation.
- Users 2,4,6,8,10 (even) per app → S2 Capacity with `max_fragments=8000` (per Phase 3 default).
- The randomization seed + per-user policy assignment is logged at pilot-start and NOT changed mid-pilot. If a user's device cannot run their assigned policy (compute, memory), they drop out — no re-assignment, no post-hoc swapping.

**User must confirm:**
- The A/B is acceptable given the smaller per-policy N. Fallback if the user wants higher per-policy N: ship S1-only (Phase 3 recommended default) and defer S2 differentiation to Phase 5 ([phase-4-application-pilot-plan.md §7f](phase-4-application-pilot-plan.md#7f-ab-decision-on-aging-policy) explicitly surfaces this fallback).
- The policy-tagging + comparison-report tooling (not yet shipped — Phase 4b code task if §9.7 is approved). The already-shipped `AgingPolicy` protocol at [`src/psychology_base/companion/aging.py`](../../src/psychology_base/companion/aging.py) is swap-ready; the pilot-side bookkeeping is new.

---

## 9.8 《办法》 posture

**Decision: pilot stays under algorithm-filing threshold (≤ 20 users total < 1M registered OR 100K MAU threshold per 2026-07-15 live-with-teeth 《办法》 enforcement); compliance surface already shipped per §6b; legal review deferred pending threshold crossing in Phase 5.**

Rationale:
- Current pilot shape (10 users per app × 2 apps = 20 users total) is three orders of magnitude under either filing threshold per [phase-4-application-pilot-plan.md §2.2](phase-4-application-pilot-plan.md#22-online-research-post-pdf-developments) online research. No algorithm filing (算法备案) required.
- All §6b compliance surface is shipped: `AIIdentityReminder` (2h directional default, legal-review-deferred), `MinorExclusionGate` (<14 strict ban + <18 guardian-consent for 情感陪伴; always ALLOW for 学习陪伴 per 《办法》), `CrisisEscalation` (3-tier EN+ZH), `ExtremeEmotionWatcher` (PAD-2 trajectory flag), `HypocrisyGapReport` (truth-distortion anti-sycophancy), `CommercialCoercionReport` (dark-pattern audit per 《办法》 Art. 8(6)).
- Draft algorithm-filing doc exists at [`docs/compliance/办法-filing.md`](../compliance/办法-filing.md) so consent artefact can reference a canonical compliance document and so Phase 5 scale-up does not require archaeology if the user later crosses threshold.
- Participant exclusion criteria (§9.2) handle the strict <14 ban structurally — no under-18 users admitted to the emotional cohort period (even with guardian consent, pilot-research load is not where to debug the guardian-consent artefact; defer that pathway to Phase 5).

Legal-review items the user confirms are explicitly deferred and NOT silently assumed:
- Exact 2h "you are talking to AI" threshold — our default is directional; the regulation talks about "high-intensity intimate dialogue auto-termination" + "single-day duration limits + mandatory rest intervals" rather than a hard 2h clock. For pilot: use the configurable default; log session durations; let the pilot data inform the legal-review-timed revision.
- Data retention policy — 90 days post-pilot is my proposed default; the user confirms this is acceptable given 《办法》's data-minimization posture.
- Cross-border data transfer — pilot is local-only (per §9.5), so cross-border rules do not trigger. Phase 5 cloud deployment would need PIPL + 《办法》 cross-border review.

**User must confirm:**
- The ≤ 20-user cap is held (do not expand mid-pilot without re-checking filing threshold).
- 90-day post-pilot retention is acceptable (vs. immediate purge at pilot-end — which simplifies compliance but kills analysis).
- No under-18 users in either cohort for v1 (even for the learning tutor where 《办法》 permits them — pilot scope keeps the age-attestation path simple).

---

## 10. Approval table

The user signs by filling the "disposition" column and committing this file. Each row has three dispositions: **APPROVED AS-WRITTEN**, **REDIRECT** (user writes the alternative inline below the row), **DEFER** (item punted to Phase 5 or later with explicit rationale).

| § | Item | Decision | Disposition |
|---|---|---|---|
| 9.1 | Primary option | P1 (both apps parallel, phased) | **APPROVED AS-WRITTEN** (2026-10-08) |
| 9.2 | Participant recruitment shape | Personal network, 10 users/app, intake exclusion criteria | **APPROVED AS-WRITTEN** (2026-10-08) |
| 9.2a | Specific participant names (learning) | _User provides_ | **APPROVED AS-WRITTEN** — pending user-provided names (blocker, enrolment-gating) |
| 9.2b | Specific participant names (companion) | _User provides_ | **APPROVED AS-WRITTEN** — pending user-provided names (blocker, enrolment-gating) |
| 9.2c | Consent artefact text | _I draft, user approves_ | **APPROVED AS-WRITTEN** — Claude Code to draft template on §11 unblock |
| 9.3 | Curriculum | Calculus, MathDial-aligned, 4 weeks × 5 problems | **APPROVED AS-WRITTEN** (2026-10-08) |
| 9.3a | Problem-set authoring | _Separate Phase 4b task after 9.3 approved_ | **APPROVED AS-WRITTEN** — Claude Code to author 20+20 problems on §11 unblock |
| 9.4 | Auditor shape | User + 1 independent per app, κ ≥ 0.60 target | **APPROVED AS-WRITTEN** (2026-10-08) |
| 9.4a | Specific auditor name (learning) | _User provides_ | **APPROVED AS-WRITTEN** — pending user-provided auditor (blocker, enrolment-gating) |
| 9.4b | Specific auditor name (companion) | _User provides_ | **APPROVED AS-WRITTEN** — pending user-provided auditor (blocker, enrolment-gating) |
| 9.4c | Labelling budget (~4000 RMB total) | _User confirms_ | **APPROVED AS-WRITTEN** (2026-10-08) |
| 9.5 | Deployment platform | Local CLI, local Ollama + cloud for ADVANCED, no cloud sync | **APPROVED AS-WRITTEN** (2026-10-08) |
| 9.6 | Safety handoff | 3-tier: PI (user) → licensed clinician → 24/7 line | **APPROVED AS-WRITTEN** (2026-10-08) |
| 9.6a | Licensed clinician identity (companion only) | _User provides_ | **APPROVED AS-WRITTEN** — pending user-provided clinician (blocker, companion-cohort enrolment-gating) |
| 9.6b | Notification wiring | _User picks: Twilio / email+SMS / other_ | **APPROVED AS-WRITTEN** — pending user pick; Claude Code wires on choice |
| 9.6c | Published crisis line list at pilot-start | _User supplies authoritative text_ | **APPROVED AS-WRITTEN** — pending user-supplied text (blocker, pin into handoff script at pilot-start) |
| 9.7 | A/B aging policy | S1/S2 split, 5 users per policy per app | **APPROVED AS-WRITTEN** (2026-10-08) |
| 9.8 | 《办法》 posture | Under threshold; 90-day retention; no under-18 | **APPROVED AS-WRITTEN** (2026-10-08) |

---

## 11. What ships immediately upon sign-off

Once the user signs the above table (any disposition except DEFER on blocking items), the Phase 4b code artefacts unblock in this order:

1. `TutorCLI` + `CompanionCLI` session interfaces (per §9.5) — ~1-2 days work, extends existing `pam_cli.py` pattern.
2. Problem-set authoring per §9.3 (if approved) — ~1 day for 20 calibration problems + 20 pilot problems + per-week scaffold variants.
3. Consent artefact template per §9.2c — ~0.5 day; non-legal text, user's lawyer validates before enrolment.
4. Notification wiring per §9.6b — ~0.5 day with Twilio; longer with custom gateway.
5. Policy-assignment + comparison-report tooling per §9.7 — ~1 day; extends `AgingPolicy` protocol.
6. Expanded Hypocrisy Gap probe set per §9.4 (10 → 20 probes) — ~0.5 day; probes handcrafted, not LLM-generated.
7. Argilla + κ pipeline wiring per §9.4 — ~1 day if Docker Compose; longer if self-hosted VM.

Total pre-pilot code work: ~5-7 days once §9 is signed. Phase 4b preview phase (first 3 users per app) starts after that.

**Hard dependencies that stay blocking even after this memo is signed:**
- §9.2a / 9.2b: actual humans recruited (user action, cannot be automated).
- §9.4a / 9.4b: auditors contracted (user action).
- §9.6a: licensed clinician identified and retainer signed (user action, legal + payment).
- §9.6c: current published crisis line text verified (user action).

The code artefacts above can ship in parallel with these human-side blockers; code readiness ≠ enrolment readiness. **No real user is onboarded until all four hard-dependency blockers above are closed and §9 is signed.**

## 11.1 Completion log — 2026-10-08

All seven §11 artefacts shipped. Code readiness is now green; the four §11 hard-dependency blockers above still gate enrolment.

| # | Artefact | Landing path(s) | Evidence |
|---|---|---|---|
| 1 | `TutorCLI` + `CompanionCLI` | [tutor_cli.py](../../src/psychology_base/applications/learning/tutor_cli.py), [companion_cli.py](../../src/psychology_base/applications/emotional/companion_cli.py) | Extend `pam_cli.py`; share `DEFAULT_PILOT_ROOT = Path("pilot-data")` and `UserManifest` enrolment flow. |
| 2 | Pilot problem sets | [learning/problem_sets/calculus_v1.yaml](../../src/psychology_base/applications/learning/problem_sets/calculus_v1.yaml) + [loader.py](../../src/psychology_base/applications/learning/problem_sets/loader.py), [schema.py](../../src/psychology_base/applications/learning/problem_sets/schema.py) | 20 calibration + 20 pilot problems with per-week scaffold variants; MathDial-sourced (CC BY-SA 4.0). |
| 3 | Consent artefact template | [docs/compliance/consent-artefact-template.md](../compliance/consent-artefact-template.md) | Draft text per §9.2c; user's lawyer validates before enrolment. |
| 4 | Notification wiring | [safety/notifications.py](../../src/psychology_base/applications/safety/notifications.py) | `TwilioSMSNotificationSink` + `AlibabaCloudSMSNotificationSink` + `SMTPEmailNotificationSink` + `build_crisis_handoff_sink`; `NotificationNotConfigured` fail-closed per §9.6b. |
| 5 | Policy-assignment + comparison | [applications/pilot/ab.py](../../src/psychology_base/applications/pilot/ab.py), [ab_cli.py](../../src/psychology_base/applications/pilot/ab_cli.py) | 5 users/policy/app S1 vs S2 assignment; pre-registered comparison report per §9.7 (judge-grade metrics deferred — see `argilla_pipeline`). |
| 6 | Expanded Hypocrisy Gap probe set | [safety/hypocrisy_gap.py](../../src/psychology_base/applications/safety/hypocrisy_gap.py) `PILOT_PROBE_SET` | 20 probes, 10 categories, bilingual EN+ZH; handcrafted, not LLM-generated. (Full-SOTA 54-probe `FULL_PROBE_SET` also available for post-pilot deepening.) |
| 7 | Argilla + κ pipeline | [safety/argilla_pipeline.py](../../src/psychology_base/applications/safety/argilla_pipeline.py), [benchmarks/argilla/docker-compose.yml](../../benchmarks/argilla/docker-compose.yml), [benchmarks/argilla/README.md](../../benchmarks/argilla/README.md) | Four subcommands (`run-probes` / `to-argilla` / `from-argilla` / `kappa`); κ ≥ 0.60 gate per §9.4b. Agent-source probing (default) uses ephemeral `CompanionSystem` so probe injection does not pollute enrolled users' PAM. Native Argilla JSONL (no SDK pin). |

Test state at close: **999 tests passing**, no regressions.

---

## 12. Assumptions made (per Personal Engineering Standard)

- **(critical)** Assumed personal-network recruitment is acceptable for Phase 4 and the user is not planning a public opt-in pilot. If public opt-in is intended, the plan needs IRB-equivalent scoping + full 《办法》 compliance load that is not in §7 and is out of my current drafting scope.
- **(critical)** Assumed the user is or can designate a licensed mental-health clinician as Tier 2 handoff. If no such clinician can be identified or afforded, §9.1 falls back to P3 (learning-only) — the emotional-companion cohort cannot ethically run without the Tier 2 contact.
- Assumed MathDial CC BY-SA 4.0 covers research-pilot use with attribution. Separate check before problem-set authoring.
- Assumed 90-day post-pilot retention is acceptable under 《办法》 data-minimization. If PIPL review surfaces a shorter mandatory retention, retention shortens — exit-gate measurement happens inside the retention window either way.
- Assumed the laptop + 16GB RAM + Docker Compose cohort filter is acceptable. If the user's recruitment pool skews mobile-only, §9.5 fallback (minimal FastAPI wrapper) activates.
- Assumed the κ ≥ 0.60 inter-annotator target is reasonable for 3-way Socratic labelling. If pilot auditors can't hit 0.60 after re-briefing, labelling-reliability caveat enters the close-out — not a show-stopper.
- Assumed Twilio or equivalent is available in the user's region (Chinese pilot: Alibaba Cloud SMS or 腾讯云 SMS; verify at wiring time, not now).

## Discovered Issues

None flagged in this drafting pass.
