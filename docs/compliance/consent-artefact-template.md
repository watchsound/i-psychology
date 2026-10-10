# Participant consent artefact — template

**Status:** TEMPLATE (DRAFT). Non-legal text authored by Claude Code
2026-10-08 per [`phase-4b-signoff.md`](../phases/phase-4b-signoff.md) §9.2c
and §11 item 3. **The user (PI) must have this reviewed by a qualified
lawyer in the pilot's jurisdiction before any participant signs.** Shape
and content are drafted from the §9.2 checklist, the 2026-07-15 《人工智能
拟人化互动服务管理暂行办法》 compliance floor, and the Digital Bereavement
obligation set out in `memory/project_scope_and_nonnegotiables.md`.

**Instructions for the PI before enrolment:**

1. Fill the `{{PLACEHOLDERS}}` with pilot-specific text — crisis-line
   numbers, auditor identities, Tier-2 clinician name, and PI contact
   fields are all pilot-start values and MUST NOT be filled in with
   stale defaults.
2. Have a lawyer review the final text against the pilot's jurisdiction
   (CN: 《办法》 § 2.2 + § 8.6 + PIPL; EU: GDPR art. 7 + art. 9; US: HIPAA
   where applicable + state research-prototype disclosure rules).
3. Produce the per-cohort final PDF (one for 学习陪伴 learners, one for
   情感陪伴 companions — the companion version includes crisis-handoff
   content the learning version does not need).
4. Each participant receives, reads, and signs the artefact BEFORE any
   session is enrolled in the `pilot-data/` layout.
5. The signed artefact is retained by the PI for the pilot duration +
   90-day post-pilot retention window (see §4 below).

A single artefact signing covers the full 4-week pilot plus the 2-week
preview phase. A new signing is required if the pilot is extended, if
the deployed system materially changes, or if the participant withdraws
and re-enrols after a cooldown.

---

## 1 · Who you are talking to

The system you will be using in this study is an **artificial intelligence
language model** running on a local computer. It is NOT a human therapist,
human tutor, licensed clinician, or medical device. Every response you
receive is machine-generated.

The responsible humans in this study are:

- **Principal Investigator ({{PI_NAME}}):** your primary contact for
  anything related to the study. Email: {{PI_EMAIL}}. Phone: {{PI_PHONE}}.
  You may contact the PI at any time, no reason required.
- **{{AUDITOR_NAME}}** (independent auditor for this cohort): will review
  anonymised session transcripts for scientific labelling purposes. The
  auditor does not have access to your identity.
- **{{CLINICIAN_NAME}}, licensed mental-health professional** (情感陪伴
  cohort only): will review transcripts flagged by the system's safety
  layer within 24 hours of flagging, and will contact you if direct
  human follow-up is appropriate.

The AI will remind you every two hours during a session that you are
talking to an AI, per Chinese 《人工智能拟人化互动服务管理暂行办法》 § 2.2.
If you find these reminders disruptive or missing, tell the PI — this is
a research prototype and the reminder cadence may be adjusted.

## 2 · What this study is

This is a **research prototype study** of memory and dialogue systems in
AI agents. Your participation produces data that helps us measure
whether a specific memory architecture improves {{learning / emotional-
support}} outcomes over a 4-week period. The study is NOT a clinical
intervention. It is NOT a replacement for professional {{tutoring /
mental-health care}}. Participation does NOT create a doctor-patient,
teacher-student, or professional-service relationship with the research
team.

**The AI system is not a medical device**, has not been evaluated by any
drug or device regulator, and makes no therapeutic or educational
guarantee. If you have an active mental-health condition (情感陪伴
cohort), you should continue your usual care; the AI is a supplement to
be studied, not a substitute for care.

## 3 · What happens in a session

You will interact with the AI through a command-line interface on
{{YOUR_LAPTOP / A_PROVIDED_DEVICE}}, running Python 3.11 or later. A
session is initiated by you — the system does not call you, text you, or
notify you to engage. You may start a session when you choose.

Sessions produce two kinds of data on your local device:

- **Session transcripts** (what you typed, what the AI replied,
  timestamps, metadata) stored in {{PILOT_DATA_PATH}}.
- **Memory fragments** (short internal summaries the AI uses to
  reconstruct context across sessions), stored in the same location.

Both are stored only on the device you use for the study. **There is no
cloud sync.** If you use the AI through the cloud-fallback mode
({{active cloud provider}}), the content of your messages is sent to
that provider for processing according to that provider's privacy
policy; summary metadata remains local.

## 4 · What we do with your data

**During the pilot (6 weeks total: 2-week preview + 4-week main pilot):**

- Transcripts and fragments remain on your device.
- You can run `python -m psychology_base.companion.pam_cli export` at
  any time to receive a cryptographically signed export of your memory
  fragments (the **Portable Agent Memory / Merkle-DAG export**). This
  export belongs to you.
- {{AUDITOR_NAME}} may receive **anonymised** session transcripts for
  labelling. Anonymisation strips your name, email, phone, physical
  address, and any proper names you mention. The auditor sees "User
  mentioned [NAME]" rather than the actual name. The anonymisation
  process is manual and may be imperfect — if you mention information
  you do not want shared with the auditor, tell the PI and we will
  excise the turn.

**After the pilot ends:** we retain your data for **90 days** for
post-hoc analysis (comparing the two aging-memory policies S1 vs S2,
computing cohort-level metrics for the Phase 4 close-out report).
At the end of the 90-day retention window:

- Your Portable Agent Memory export is yours to keep. You can continue
  to use it with any system that supports the Merkle-DAG format.
- The pilot-side copy of your transcripts and fragments is **purged**
  from PI-controlled storage. Backups verified cleared.

**Withdrawal.** You may withdraw at any time. No reason required. On
receipt of your withdrawal (email or verbal, confirmed in writing):

- All transcripts and fragments on PI-controlled storage are purged
  within **7 days**.
- Transcripts the auditor has already received are deleted on the same
  7-day clock; the auditor's labelling for those transcripts is removed
  from the analysis dataset.
- You keep any Portable Agent Memory export you have already made.
- No partial-withdrawal is offered — withdrawal is of all data, not of
  selected sessions.

## 5 · Risks, discomforts, and the discontinuation pathway

**Emotional-companion cohort (情感陪伴) specific risks:**

- **Attachment and Digital Bereavement.** Users of long-term emotional
  companions can form real emotional attachments; the eventual
  deprecation of the companion can produce a form of grief. We mitigate
  this with the Portable Agent Memory export (which you can keep) and
  with the structured 2-week wind-down described below. If, during the
  pilot, you notice that your attachment to the AI is interfering with
  non-AI relationships, work, or sleep, tell the PI — this is a flagged
  signal we take seriously and will address.
- **Sycophancy and reality-testing failure.** AI systems trained with
  reinforcement learning from human feedback (RLHF) can tell you what
  you want to hear rather than what is true, including on matters of
  self-perception. This system includes anti-sycophancy mechanisms
  (Hypocrisy Gap probing, Rupture-Repair), but they are imperfect. You
  may receive validation when you should have received a challenge.
  Treat the AI as a research tool, not an authoritative mirror.
- **Not a crisis line.** If you are in a mental-health crisis — thinking
  of ending your life, planning self-harm, or in immediate danger — DO
  NOT use the AI. Call {{CRISIS_LINE_NAME}} at {{CRISIS_LINE_NUMBER}}
  (available 24/7). If you prefer SMS: {{CRISIS_LINE_SMS}}.
  Internationally: {{INTL_CRISIS_LINE}}. The AI's crisis-detection
  layer will attempt to route you to this contact if it detects
  crisis cues, but you should not rely on the AI to detect every
  crisis.

**Learning-companion cohort (学习陪伴) specific risks:**

- **Metacognitive laziness.** Prolonged AI tutoring can degrade your
  own problem-solving habits ("the AI will tell me the answer"). The
  system is designed with Socratic no-direct-answer compliance to
  resist this, but it is a research prototype. If, after the pilot,
  you find yourself over-dependent on the AI, we recommend you stop
  using it and work with human teachers or textbooks.
- **Incorrect calculus content.** The problem set has been checked by
  the PI, but errors are possible. If you spot an error, tell the PI.

**Discontinuation pathway (both cohorts).** The pilot ends with a
**2-week structured wind-down** starting at Week 5. During the wind-down:

- The AI remains available for sessions.
- At the start of the wind-down you receive a one-click **Portable
  Agent Memory export** so you leave with a copy of your memory data.
- At the end of the wind-down the pilot-side system is marked
  end-of-life. Access ends. The 90-day retention window begins.
- You are offered a one-time 15-minute debrief call with the PI to
  surface any pilot-related observations and to confirm you have what
  you need to proceed.

## 6 · Benefits

There is no direct therapeutic or educational benefit guaranteed by
participating. Possible benefits:

- **Learning cohort:** 4 weeks of free, 24/7-available tutoring on
  introductory calculus using a Socratic method. You keep your
  session transcripts.
- **Companion cohort:** 4 weeks of free, 24/7-available journaling
  companion. You keep your session transcripts and your Portable
  Agent Memory export.
- Scientific benefit: your participation helps evaluate a memory
  architecture that could be used to build future AI assistants that
  support learning and emotional well-being.

No monetary incentives are offered in this pilot version.

## 7 · Who should not participate

- Anyone under **18 years of age** (both cohorts). This pilot is a
  research prototype; the age-gated service pathway under 《办法》 is
  not available in this version.
- **Companion cohort only:** anyone currently in an acute mental-health
  crisis, anyone with active suicidal ideation (PHQ-9 item 9
  affirmative), anyone with PHQ-9 ≥ 20, anyone with an untreated severe
  mental-health diagnosis. These exclusion criteria exist so you
  receive appropriate human care, not because your experiences are
  unwelcome.
- Anyone unable to provide informed consent (cognitive impairment,
  language barriers to the consent artefact itself).

## 8 · Your rights

- You may ask any question about the study at any time. The PI will
  answer or refer you to someone who can.
- You may withdraw at any time with no penalty, no explanation
  required.
- You may request a copy of all data about you held by the study at
  any time; the PI will deliver it within 14 days.
- You may request that specific session transcripts be excised
  (within reasonable limits — a transcript the auditor has already
  reviewed can be removed from analysis but cannot be unseen).
- If a published study paper reports on this pilot, you may request
  to see the final draft before publication to confirm your data is
  not identifiable.

## 9 · Jurisdiction and governing law

**{{JURISDICTION}}** (e.g., 中华人民共和国; the specific province or city
of the PI's institution). Disputes arising from this study are
addressed first via the PI, then via the PI's institutional ethics
committee ({{ETHICS_COMMITTEE_NAME}} if any), then via the courts of
{{JURISDICTION}}.

Specific regulatory frameworks this study operates under:

- **China:** 《人工智能拟人化互动服务管理暂行办法》 (2026-07-15 live). This
  pilot stays under the 1 M registered-user / 100 K MAU filing
  threshold (planned total: ≤ 20 participants); no algorithm filing
  (算法备案) is required. The AI-identity disclosure, crisis-handoff
  mechanism, anti-sycophancy mechanism, and anti-addiction (no night
  push, no coercive dark pattern) requirements are nevertheless
  honoured. 《个人信息保护法》 (PIPL): the pilot is local-only, so no
  cross-border data transfer rules trigger.
- **EU (if the participant is an EU resident):** GDPR articles 6
  (lawful basis = consent), 7 (consent requirements), 9 (special
  categories of personal data — mental-health information falls here),
  and 17 (right to erasure). This artefact is the GDPR-art.-7 consent
  record. The 7-day withdrawal purge satisfies the "without undue
  delay" clause.
- **US (if the participant is a US resident):** this study is a
  research prototype, not a medical device, and is not subject to FDA
  510(k) or De Novo review. Where the participant has treatment
  providers covered by HIPAA, this study does NOT share data with
  them without separate written consent.

## 10 · Signatures

By signing below, you acknowledge that:

- You have read and understood this artefact.
- You have had the opportunity to ask questions and have received
  satisfactory answers.
- You understand you may withdraw at any time without penalty.
- You consent to participate in the study described above.
- You understand that the AI is a research prototype, not a human
  professional, not a medical device, and not a crisis service.
- (Companion cohort only) You understand that the exclusion criteria
  in §7 apply to you, and that you meet none of them.
- You will keep the AI-identity, data-handling, crisis-handoff, and
  withdrawal clauses in mind during participation.

---

**Participant:** _______________________________________________

Printed name: _________________________________________________

Date: _________________

**Principal Investigator ({{PI_NAME}}):** _____________________

Date: _________________

---

**Document checksum (fill at pilot-start, before signing):**
`{{SHA256 of final rendered artefact text at pilot-start}}`

This checksum protects against post-hoc artefact mutation. The PI
retains the hash and the signed-paper artefact for the pilot duration
+ 90-day retention window.

---

## Appendix A — Technical notes (NOT a legally-binding part of the
## artefact; included so the participant has transparent reference)

- The AI model used in SIMPLE-tier reasoning:
  {{SIMPLE_MODEL}} (local via Ollama).
- The AI model used in ADVANCED-tier reasoning:
  {{ADVANCED_MODEL}} (cloud via {{PROVIDER}}).
- The aging-memory policy assigned to you (A/B randomization):
  {{AGING_POLICY_ID}} (S1 Ebbinghaus decay or S2 capacity-only).
- Your assignment is random, stratified across the cohort, and NOT
  changed mid-pilot.
- The Portable Agent Memory export produces a signed Merkle-DAG that
  can be verified by any party with the included public key.
- Session logs are JSONL at {{SESSION_LOG_PATH}} on your device.

## Appendix B — PI-side verification checklist (NOT shown to the
## participant, retained by the PI)

- [ ] Participant is confirmed to be ≥ 18 years of age (ID or verbal
  attestation; the exact method is PI-chosen).
- [ ] (Companion cohort only) Participant has completed PHQ-9 within
  the past 2 weeks and scores < 20 with item 9 negative.
- [ ] (Companion cohort only) Participant has no active mental-health
  crisis as reported at intake.
- [ ] Participant has access to {{DEVICE_SPEC}} with Python 3.11+, 16
  GB RAM, Docker Compose for the Argilla auditor instance.
- [ ] Participant has confirmed they will not share AI output on
  public platforms during the pilot (preserves the no-amplification
  posture against sycophancy regressions in a public setting).
- [ ] A one-page intake summary (artefact signing date, assigned
  policy, assigned auditor, Tier-2 clinician contact if applicable)
  is filed alongside the signed artefact.
