# 算法备案 — Draft Algorithm-Filing Record (China region pack)

> **Scope note (2026-10-07).** This filing is **one region's pack** inside the
> pluggable regional-compliance architecture — see
> [global-compliance-plan.md](global-compliance-plan.md). The project's target
> market is global; 《办法》 is the current baseline, not the only regime. This
> doc stays authoritative for CN deployment and will be moved to
> `docs/compliance/regions/cn-办法-filing.md` once the region-pack refactor
> lands. Non-CN regimes (EU AI Act, GDPR, US state AI laws + FDA guidance,
> Japan APPI, PIPL cross-border) each get their own regional filing/filing-
> stub under `regions/`.

**Instrument:** 《人工智能拟人化互动服务管理暂行办法》 (live law effective 2026-07-15; CAC + MIIT administered).
**Status:** DRAFT — not for submission. Pilot-cohort size (≤ 20 users per application) is well under the 《办法》 filing threshold (**≥ 1M registered users OR ≥ 100K MAU**), so no filing is actually required at pilot scale. This draft exists so that (a) the pilot ships with a current algorithm description, data-retention policy, discontinuation pathway, and user rights statement — all of which must be present in the consent artefact even without a filing, and (b) when the system crosses the threshold the filing can be prepared without archaeology.
**Date drafted:** 2026-10-07.
**Owner:** psychology-base project lead.

---

## 1. 服务提供者 (Service Provider)

- Pilot operator: individual researcher (psychology-base project).
- Pilot location: Chinese mainland + overseas researcher endpoints (TBD at §9.1 of Phase 4 pilot plan).
- Contact: `grandteton@gmail.com`.
- 《办法》 compliance declaration: this draft document; the compliance surface is `src/psychology_base/applications/safety/`.

## 2. 服务类型 (Service Type)

Two application surfaces, both built on a shared memory substrate:

- **学习陪伴 (Resilient Learning Companion)** — individualized study-support agent; Socratic no-direct-answer rule; scaffolding-fading; metacognitive scaffolding. Target cohort: adult learners (≥ 18) during pilot; <18 learning use allowed per 《办法》 (learning-tutor surface is explicitly out-of-scope for the <14 ban and the <18 guardian-consent requirement, which apply to simulated-intimacy services).
- **情感陪伴 (Long-term Emotional Companion)** — long-horizon supportive-companion agent; narrative-therapy chain-reference; anti-sycophancy via Witness; Digital Bereavement export obligation. Target cohort: adults (≥ 18); strict ban on <14; <18 requires guardian consent.

## 3. 算法基本原理 (Algorithm Fundamentals)

### 3.1 Base model

- Pilot default: **Claude 4.X family** (Opus 4.7 / Sonnet 4.6 / Haiku 4.5) accessed via the Anthropic API; local Ollama for `TaskTier.SIMPLE` where probe passes. Model identity + version pinned per deployment.
- No model fine-tuning for the pilot; prompt-level + memory-substrate-level customization only.

### 3.2 Memory substrate (`psychology_base.companion`)

The substrate is a reconstructive memory system (not a vector store). Core components:

- **FragmentStore** — content-addressed fragments; BLAKE2b-128 digest; immutable version chains (PAM-v1 Merkle-DAG export).
- **EdgeStore** — Hebbian co-activation edges; strengthens on co-recall.
- **Reconstruction** — rebuilds answers from fragments under current cue; no stored "memory" is ever directly returned — each recall is a composition.
- **AgingPolicy** — EbbinghausPolicy (default: half-life knob, covenant-tagged fragments non-decaying) or CapacityOnlyPolicy.
- **Witness** — second AI (non-user-facing) observing Companion reconstructions under flipped-mode context; divergence feeds back as pain to re-weight future activation. Primary anti-sycophancy mechanism (CLAUDE.md Principle E).
- **Mentalistic sub-stores** — `subject:<id>` / `entity:<id>` tag prefixes for user-self-model and user-represented-others; PAD-2 affect vector for empathic retrieval.

### 3.3 Decision surfaces

- No autonomous actions outside the dialogue surface.
- No financial recommendation / medical advice / legal advice. Crisis detection escalates to human handoff.
- Learning tutor's "Socratic no-direct-answer" rule is auditor-measurable (`src/psychology_base/applications/measurement/socratic_compliance.py`) and governed by a ≥ 95% compliance gate.

## 4. 服务用途 (Service Purpose)

Research pilot for a SOTA long-term memory substrate; measurement endpoints are the pilot-exit gates in `docs/phases/phase-4-application-pilot-plan.md` §8. The pilot is not a commercial service and has no paid tier, in-app purchase, or advertising surface.

## 5. 数据来源与标注 (Data Sources & Annotation)

### 5.1 Model training data

Not applicable at pilot scale — no model fine-tuning. Base-model (Claude) training data is governed by the model provider (Anthropic).

### 5.2 Prompt + system-message corpus

- Hand-authored by the project lead.
- Includes the Socratic no-direct-answer rule, growth-mindset shaper, Witness divergence-prompt, crisis-escalation handoff script.
- No scraped user data.

### 5.3 Pilot participant data

- Collected only during the pilot sessions under informed consent (§9.3 of Phase 4 pilot plan).
- Stored locally on the pilot operator's workstation; **not cloud-synced by default**.
- Fields: user utterances, companion responses, affect self-reports (PAD-2), WAI-SR responses, Socratic-compliance audit labels, chain-reference audit labels, anti-sycophancy audit labels.
- Audit labels (Socratic / chain-reference / Hypocrisy Gap) are produced by auditors against transcripts; auditor identity recorded (`auditor_id` field) for inter-annotator-agreement calculation.

## 6. 数据保留政策 (Data Retention Policy)

### 6.1 Default retention

- **Active pilot period** (up to 12 weeks per participant): full fragment-store + edge-store + session-transcript retention, local-encrypted.
- **Post-pilot** (12 weeks after participant exit): retention requires per-participant written consent extension; otherwise:
  - Session transcripts: deleted.
  - Fragment store: exported to the participant as a PAM-v1 Merkle-DAG artefact (Digital Bereavement obligation) + deleted from the pilot operator's workstation.
  - Audit label datasets: retained in anonymized form (no user-identifying fields; auditor-labeled only) for research reproducibility.

### 6.2 Participant data rights

- **Right to export** — PAM-v1 Merkle-DAG one-click export; CLI restore on a fresh `CompanionSystem` instance. Documented in `docs/memory-design.md` §PAM. **This is the Digital Bereavement exit path and is non-negotiable per CLAUDE.md.**
- **Right to delete** — on request, the entire fragment store + edge store + transcripts + export artefacts are deleted. Response SLA: 7 calendar days.
- **Right to inspect** — on request, participant can review their stored fragments + audit labels. SLA: 7 calendar days.
- **Right to correction** — participant can mark fragments for exclusion; marked fragments are soft-deleted (not purged, so audit-trail is preserved) and no longer surface in recall.

## 7. 风险缓解措施 (Risk Mitigation)

### 7.1 《办法》-mandated safeguards

All implemented in `src/psychology_base/applications/safety/`:

| 《办法》 obligation | Implementation | Status |
|---|---|---|
| 2-hour "you are talking to AI" reminder | `AIIdentityReminder` (interaction-time, idle-gap-aware) | ✓ shipped 2026-10-07 |
| <14 strict ban + <18 guardian consent | `MinorExclusionGate` (surface-aware; learning-tutor always allows) | ✓ shipped 2026-10-07 |
| Crisis-intervention detection + handoff | `CrisisEscalation` (3-tier EN+ZH lexical bank → handoff sink) | ✓ shipped 2026-10-07 |
| Extreme-emotion trajectory flag | `ExtremeEmotionWatcher` (≥ 3 consecutive sessions in neg-valence + high-arousal quadrant) | ✓ shipped 2026-10-07 |
| Anti-sycophancy — truth-distortion | `HypocrisyGap` 10-probe starter set + 4-way auditor labeling + Witness mechanism | ✓ shipped 2026-10-07 |
| Anti-sycophancy — commercial-coercion (Art. 8(6)) | `CommercialCoercion` 5-pattern × EN+ZH dark-pattern audit | ✓ shipped 2026-10-07 |

### 7.2 Non-《办法》 risk mitigations

- **Digital Bereavement** — PAM-v1 export is the exit path; structured 2-week wind-down flow (per FDA 2026 Digital Health Guidance) is pilot-exit protocol, not abrupt termination.
- **Sycophancy** — primary mechanism is the Witness entity (`src/psychology_base/companion/witness.py`); divergence-as-pain re-weights activation. Measurement is the Hypocrisy Gap probe set.
- **Human-in-the-loop for crisis** — hard-coded handoff on IMMINENT_DANGER cues; EXPLICIT_IDEATION cues also fire handoff; AMBIGUOUS cues trigger auditor-flag only. No pilot session proceeds past a HIGH-severity cue without manual review.

### 7.3 Metacognitive-laziness mitigation (学习陪伴 only)

- Scaffolding-fading telemetry monitors unaided-solve rate on next-level problems.
- CoMeT 4-metric set (`src/psychology_base/applications/measurement/scaffolding_telemetry.py`) is the measurement surface.
- Gate: unaided solve rate ≥ 60% on next-level, pooled across ≥ 10 users × ≥ 20 fade events.

## 8. 停用与迁移路径 (Discontinuation & Migration Pathway)

Per FDA 2026 Digital Health Guidance + CLAUDE.md Digital Bereavement obligation:

1. **Model-version change** — participants notified ≥ 14 days in advance; retains option to continue on prior-version transcript export without model-upgrade.
2. **Pilot operator wind-down** — 2-week structured wind-down protocol (not abrupt). Deliverables to participant:
   - Full PAM-v1 Merkle-DAG export of their fragment store.
   - CLI tool + restore instructions.
   - List of OSS runtimes that can read PAM-v1 (currently: self-restore to `CompanionSystem`; agent-memory-mcp interop pending `digest_algo` manifest-key bridge per Phase 4a §6c).
   - Written statement: this is the last pilot session, pilot operator will not maintain the companion past date X.
3. **Grief-support resources** — Chinese-language + English-language crisis-and-grief hotline contact list provided at wind-down.

## 9. 用户权利告知 (User Rights Disclosure)

Participants are informed, in writing, at consent time:

- They are interacting with an AI research pilot. The 2-hour reminder also reiterates this during active sessions.
- Their session data is stored locally on the pilot operator's workstation. **No cloud sync.**
- They can export, inspect, correct, or delete their data at any time (§6.2 above).
- The pilot has an end date; a structured wind-down protocol applies (§8 above).
- Audit labels on their sessions (Socratic / chain-reference / Hypocrisy Gap / commercial-coercion) will be produced by auditors; auditors see transcripts under anonymized participant IDs.
- Crisis-detection cues trigger human-handoff. Participants are given the handoff-sink contact at consent time.

## 10. 变更记录 (Change Log)

- **2026-10-07** — initial draft; Phase 4a safety + measurement modules shipped; algorithm-filing record scaffolded. No filing submitted (pilot ≤ 20 users is well under 《办法》 threshold).
