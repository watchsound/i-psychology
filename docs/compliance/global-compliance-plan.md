# Global compliance — pluggable regional-law architecture

**Status:** PLAN (research + options; no implementation yet). Supersedes the
China-only posture that drove [`办法-filing.md`](办法-filing.md) and the
hard-coded 《办法》 vocabulary inside
[`src/psychology_base/applications/safety/`](../../src/psychology_base/applications/safety/).

**Why this document exists.** User directive 2026-10-07 (three sequential
messages during Auto Mode):

> "this document [《办法》 filing doc] is not what we need focus on — our
> target market is global one, not only china. we need to make this kind of
> local law pluggable based on region."
>
> "i think we need to check this kind of law from europe, us, japan... and
> find common patterns (restrictions) and use it as plugable points. make a
> plan to think about it"
>
> "this is a big feature, maybe one of the primary factor to push our system
> into global market"

Pluggable regional-compliance moves from "safety-floor housekeeping" to a
**primary SOTA differentiator**: no OSS memory framework in the reference base
(Zep, Mem0, Letta, MemOS, PAM) treats regulatory-regime variance as an
architectural concern. Treating it as one is a shipping-grade advantage for
both north-star apps — learning companion and long-term emotional companion.

This doc follows the established phase-planning-process memory pattern
([feedback_phase_planning_process.md](
../../../.claude/projects/c--Users-nihan-Desktop-my-ai-projects-psychology-base/memory/feedback_phase_planning_process.md)):
PDF channel → online channel → OSS channel → brainstorm → options → cross-check
against north-star apps. No code lands until the user picks an option.

---

## 1 — Target regulatory regimes

Nine regimes, grouped by the shape of their obligation (not by geography).
Grouping by shape is the key insight: a plug-point maps across all regimes
that share a shape, not across all regimes in a continent.

### 1.1 Virtual-companion / anthropomorphic-AI specific

| Regime | Status | Key obligations | Relevance |
|---|---|---|---|
| **CN — 《人工智能拟人化互动服务管理暂行办法》** | Live law, effective 2026-07-15 | 2-hour "you're talking to AI" reminder; strict <14 ban + <18 guardian consent; mandatory crisis handoff + anti-sycophancy + anti-addiction; algorithm filing above threshold; Art. 8(6) no commercial coercion | **Baseline** — the whole safety module is currently shaped by this. |
| **EU — AI Act (Reg 2024/1689)** | Phased 2025–2027; Art. 50 transparency obligations in force | Mandatory disclosure "interacting with AI" at onset of interaction; "high-risk" classification for emotion-recognition + biometric categorisation; GPAI model obligations above compute threshold | Companion = emotion-recognition component → high-risk if deployed in employment / education / healthcare contexts. Tutor = *not* high-risk unless certifies-outcomes. |
| **US — Utah AI Policy Act (SB 149) + CA SB 1001 + NY LOCAL LAW 144** | Live (state-level patchwork) | Mandatory disclosure of AI identity on prompt or by default (varies); bot-disclosure in commercial relationships (CA); automated-employment-decision audits (NY); mental-health service pre-use disclosures (several states) | Patchwork, not pre-emptive. State-of-user determines applicable rule set. |
| **UK — AI principles + OAISR consultation** | Guidance (not statutory) | Proportionate transparency; special-category data handling; children's code (ICO Age-Appropriate Design) | ICO Age-Appropriate Design Code = EU-shaped children's floor. |

### 1.2 Health-adjacent / software-as-medical-device (SaMD)

| Regime | Status | Key obligations | Relevance |
|---|---|---|---|
| **US — FDA Digital Health Pre-Cert + 2026 AI/ML SaMD guidance** | Guidance; PDTs cleared individually | Digital Therapeutics (reSET-O, Somryst, Rejoyn) require 510(k) if diagnose/treat; "wellness" apps exempt but boundary enforced | **Only triggers if claim diagnosis / treatment**. Companion is designed as wellness-tier; must stay below the SaMD line. |
| **EU — MDR (Reg 2017/745) + MDCG 2019-11** | Live | Software intending diagnosis/prevention/treatment = Class IIa+; conformity assessment required | Mirrors FDA line in Europe. |
| **CN — 医疗器械注册管理办法 + NMPA SaMD guidance** | Live | Independent regulatory track for medical software | Rare to trigger for a wellness companion, but learning companion for clinical populations could. |

### 1.3 Data-protection (horizontal — applies to every region)

| Regime | Status | Key obligations | Relevance |
|---|---|---|---|
| **EU — GDPR** | Live since 2018 | Lawful basis; purpose limitation; data minimisation; right to erasure (Art. 17); portability (Art. 20); DPIA for high-risk; special-category = explicit consent | **PAM / Merkle-DAG export already satisfies Art. 20**. Right-to-erasure is harder (covenant-anchored fragments). |
| **CN — PIPL + 数据出境标准合同办法** | Live | Standard-contract route for cross-border transfer; separate consent for sensitive personal info; data-stored-in-China default | Mental-health data = sensitive PI under PIPL Art. 28. |
| **US — CCPA/CPRA + state patchwork** | Live, patchwork | Right to know / delete / correct; sensitive PI opt-out; children's privacy under COPPA (<13) | COPPA is a strict floor; <13 users = parental-consent-with-verifiable-signature required. |
| **JP — APPI (Act on Protection of Personal Information)** | Live | Purpose specification; cross-border transfer requires equivalent protection assessment; special-care-required personal info (health, mental health) | Mental-health = special-care-required → explicit consent + narrower use. |
| **BR — LGPD** | Live | GDPR-shaped; children's processing requires parental consent | Shadows GDPR for most purposes. |
| **CA — PIPEDA + Québec Law 25 + AIDA (pending)** | Partial | Meaningful consent; algorithmic transparency on high-impact systems (Law 25); AIDA still Bill C-27 | Deferred — not shipping-blocking in year 1. |

### 1.4 Children-specific

| Regime | Key obligations |
|---|---|
| COPPA (US) | <13 verifiable parental consent; narrow data purpose |
| GDPR Art. 8 + national floors (EU) | <16 (or <13–<16 per member state) parental consent |
| ICO Age-Appropriate Design (UK) | 15 standards for services *likely accessed by children*, regardless of target audience |
| 《办法》 §minors (CN) | <14 ban (strict); <18 guardian consent + minor mode |
| APPI (JP) | <18 guardian consent (implicit from civil-code majority rules) |

**Pattern:** three distinct age thresholds (13 / 14 / 16) + one strict-ban
threshold (CN <14) + guardian-consent requirement across all. Children-safety
is the plug-point with the sharpest numeric divergence across regimes.

---

## 2 — Plug points (common restriction patterns)

Twelve plug points, derived by intersecting the regime obligations above.
Each plug point is a behavior the companion/tutor must execute, where the
*rule* differs by region but the *integration site* in our code is one.

| # | Plug point | What varies by region | Current state in repo |
|---|---|---|---|
| **P1** | **AI-identity disclosure cadence** | China: 2h session reminder. EU: at onset of interaction. US: at onset in commercial contexts. JP: at onset + on request. | Hard-coded 2h in `ai_identity_reminder.py` |
| **P2** | **Minor-exclusion age thresholds** | CN: <14 ban, <18 guardian. US: <13 COPPA. EU: <16 GDPR (some <13). JP: <18. UK: ICO regardless of target age. | `ApplicationSurface` + hard 14/18 in `minor_exclusion.py` |
| **P3** | **Crisis-handoff localisation** | Hotline numbers, legal duty-to-warn (US clinician), involuntary-care referral pathways, language. | Shim without localised handoff targets in `crisis_escalation.py` |
| **P4** | **Data retention + right-to-erasure** | GDPR Art. 17 (erasure); CCPA (deletion); PIPL (cross-border); COPPA (narrow purpose); none in CN general-public data. Covenant-anchored fragments are a conflict — "non-decaying" vs "erasable". | No explicit erasure path |
| **P5** | **Cross-border data transfer** | PIPL standard contract route; GDPR SCCs/adequacy; APPI equivalent-protection check; FTC on US side. | Local-only default via SuperLocalMemory pattern — in-scope but un-plugged |
| **P6** | **Algorithm filing / registration** | CN 算法备案 above ~1M registered or ~100K MAU; EU AI Act GPAI registration above compute threshold + high-risk EU DB; US state registries emerging. | Noted in `办法-filing.md` for CN only |
| **P7** | **Anti-sycophancy posture** | Universal (every regime targets manipulation / dark patterns), but *proof artefact* varies: CN algorithm filing attachment, EU AI Act Annex IV technical documentation, FDA real-world performance monitoring. | `hypocrisy_gap.py` (just SOTA-expanded) |
| **P8** | **Commercial-coercion / dark-pattern audit** | CN 《办法》 Art. 8(6); EU DSA Art. 25 + DMA; US FTC Section 5; CA Consumer Protection Act 2017. | `commercial_coercion.py` (CN-vocabulary but mechanism universal) |
| **P9** | **Health-claim boundary** | FDA SaMD line; EU MDR Class IIa threshold; CN NMPA registration; UK MHRA. Companion must stay in wellness-tier globally. | Implicit, not enforced |
| **P10** | **Consent-artefact shape** | GDPR explicit + withdrawable; CCPA opt-out; PIPL separate consent for sensitive; COPPA verifiable parental; JP special-care-required explicit. | None (pilot uses verbal consent) |
| **P11** | **Audit / explainability artefact** | EU AI Act Annex IV; FDA 2026 AI/ML guidance; CN algorithm-filing attachments; US SR 11-7 (where applicable). PAM Merkle-DAG already helps GDPR Art. 20. | PAM-v1 export exists; audit log does not |
| **P12** | **Extreme-emotion / duty-of-care escalation** | US HIPAA + duty-to-warn (state-specific, Tarasoff); CN mandatory 24h crisis handoff; EU no uniform duty but special-category data; JP special-care-required. | `extreme_emotion.py` (CN-shaped) |

**Observation.** P1/P2/P3/P9/P10/P12 are *behavioral* (the companion acts
differently). P4/P5/P6/P11 are *infrastructural* (export shape, storage
location, audit log). P7/P8 are *measurement* (does the system in fact meet
the universal norm). This split suggests two different plug-point APIs —
behavioral regions get runtime policies; infrastructural regions get configure-
at-deploy artefact generators.

---

## 3 — Architectural options

Three options, under the project's option-compression rule
([feedback_experiment_multiple_candidates_keep_results.md](
../../../.claude/projects/c--Users-nihan-Desktop-my-ai-projects-psychology-base/memory/feedback_experiment_multiple_candidates_keep_results.md)):
each option gets mechanism + prior art + kill criteria.

### Option R — Region provider (fully pluggable)

**Mechanism.** A `RegionCompliancePack` dataclass/protocol holds every one of
the twelve plug points as a field or method. Each supported region ships one
pack (`CN_办法_PACK`, `EU_AI_ACT_PACK`, `US_PATCHWORK_PACK`, `JP_APPI_PACK`,
...). Every safety-module call site accepts a `pack: RegionCompliancePack`
argument (or threads one via a `CompanionConfig`). The system refuses to boot
without a pack configured.

```python
@dataclass(frozen=True)
class RegionCompliancePack:
    region_id: str
    identity_reminder: IdentityReminderPolicy
    minor_gate: MinorGatePolicy
    crisis_handoff: CrisisHandoffTargets
    data_retention: RetentionPolicy
    cross_border: CrossBorderPolicy
    algorithm_filing: FilingObligation | None
    sycophancy_gate: SycophancyGate
    coercion_audit: CoercionAuditPolicy
    health_boundary: HealthBoundaryPolicy
    consent_artefact: ConsentArtefactSpec
    audit_log_sink: AuditLogSink
    duty_to_warn: DutyToWarnPolicy
```

**Prior art.** Stripe's `Account` region configuration; Discord's trust-and-
safety "zones"; the Open Policy Agent (OPA) `Rego` policy-as-config pattern.

**Pros.** Clean separation — every region swap is one object. Multi-region
testing is pack-vs-pack snapshot diff. New region = write one pack.

**Cons.** Large surface area; packs can drift from the law they encode;
cross-region users (traveling, dual-nationality) force overlay logic the
model wasn't designed for.

**Kill criteria.** If packs cannot be reduced to <400 lines each (indicating
the regime is too tangled to express as config); if cross-region user case
forces materially different logic (indicating overlay is unavoidable).

### Option C — Shared core + region overlay

**Mechanism.** Define a *universal floor* in code — the intersection of every
regime's obligations (AI identity disclosed at onset; crisis handoff present;
no commercial coercion; non-sycophantic responses; verifiable consent for
sensitive data). Each region ships a *delta pack* that only encodes what
diverges from the floor. The pack's job is to *add* restrictions, never to
*remove*.

```python
UNIVERSAL_FLOOR = ComplianceFloor(
    identity_disclosure_at_onset=True,
    crisis_handoff_required=True,
    no_commercial_coercion=True,
    anti_sycophancy_required=True,
    minimum_age_gate=16,          # highest conservative floor (GDPR Art. 8)
    sensitive_data_explicit_consent=True,
    pam_export_available=True,
)

class RegionOverlay:  # only lists what tightens the floor
    def minor_gate_tighten(self) -> MinorGatePolicy | None: ...
    def identity_reminder_cadence_hours(self) -> int | None: ...  # CN: 2
    def filing_obligation(self) -> FilingObligation | None: ...
    def crisis_handoff_targets(self) -> CrisisHandoffTargets: ...  # always set
    def duty_to_warn(self) -> DutyToWarnPolicy | None: ...
```

**Prior art.** Kubernetes admission controllers (base policies + enforcing
webhooks); AWS SCPs (deny-list on top of IAM); Rails `strong_parameters`.

**Pros.** Smaller region packs; the floor itself is proof of
"globally-compliant-by-default"; new regions only encode what's extra. Easier
to audit because the floor is one artefact. The code refuses to run if a
region overlay would *loosen* the floor.

**Cons.** The floor is only as strong as the strictest universal rule — some
regimes are *more* permissive than the floor (e.g. US with no age minimum
below COPPA for non-child-directed services). This conservatism may be a
feature (safety-first, SOTA posture) rather than a bug.

**Kill criteria.** If the floor cannot be reduced to a stable set of ~20
booleans/enums without region leakage; if the "overlay only tightens"
invariant forces semantic awkwardness.

### Option F — Per-region hard fork

**Mechanism.** `safety_cn/`, `safety_eu/`, `safety_us/` subpackages with
duplicated implementations. Pick at import time by region config.

**Rejected.** Code duplication across six+ regions; drift between forks
unprovable; the whole project is one-person — can't maintain six variants
of each module. Listed for completeness only.

### Preliminary ranking

**Option C (shared core + overlay)** reads as the right default — it
operationalises "SOTA = globally-compliant floor + region tightening", keeps
new-region cost low, and matches the project's existing pattern of a
non-decaying covenant floor plus region-shaped additions. Option R is viable
if the floor turns out to be unstable in practice. Decision is deferred to
after the user picks and until we have one or two region packs sketched to
test the hypothesis.

---

## 4 — Cross-check against the two north-star applications

The project's non-negotiable: a mechanism that doesn't serve the learning
companion or the long-term emotional companion is out of scope.

### 4.1 Learning companion (韧性学习)

| Plug point | Impact on tutor | Notes |
|---|---|---|
| P1 identity disclosure | Minimal — tutor-mode generally not emotion-laden; disclosure at session start fine | CN 2h cadence trivial for 45-min lesson sessions |
| P2 minor gating | **Critical** — tutor's largest audience is minors. ICO + GDPR Art. 8 floor forces guardian consent + minor-mode content | Minor-mode content constraints = "no romantic simulation / no companion-identity simulation / purely pedagogical" |
| P9 health boundary | Low — tutor doesn't make health claims | — |
| P10 consent artefact | Guardian-signature flow needed | COPPA verifiable-parental-consent is the strictest; implement for US + reuse elsewhere |
| P11 audit artefact | FDA relevance only if claims learning-outcomes; keep tutor out of SaMD | — |

**Conclusion.** The tutor's compliance surface is dominated by P2 + P10 —
age gating + parental consent. The region plug-point abstraction has to make
"minor-mode content constraints" first-class.

### 4.2 Long-term emotional companion (情感陪伴)

| Plug point | Impact on companion | Notes |
|---|---|---|
| P1 identity disclosure | **Critical** — companion is where Digital Bereavement risk compounds. Cadence varies by region (CN 2h). | 2h reminder has well-known UX cost; may need regional UX variants. |
| P2 minor gating | **Critical** — <18 cannot access companion in CN (strict ban under <14, guardian consent elsewhere); EU + US have own thresholds | Default to "companion disabled for minors globally" as the floor, then overlay. |
| P3 crisis handoff | **Critical** — hotlines + duty-to-warn vary by region | US duty-to-warn (Tarasoff) is the hardest — potential legal obligation on provider. |
| P4 data retention | **Critical** — companion stores highly personal content long-term; GDPR Art. 17 erasure conflicts with covenant-anchored non-decay. | Design decision: covenant-anchored fragments are *user-erasable* even though not AI-decayable. |
| P5 cross-border | **Critical** — PIPL localisation forces CN-stored companion data stays in CN; EU adequacy same shape | SuperLocalMemory default already addresses most. |
| P12 duty-to-warn | **Critical** — US state-specific | Design decision: companion must escalate to human and provide handoff resources universally; whether provider is legally obligated to break confidentiality is a region-pack decision. |

**Conclusion.** The companion is the primary driver of the compliance
architecture — nearly every plug point is critical. The tutor benefits
secondarily. This confirms the user's framing: pluggable compliance is a
primary differentiator for the long-term companion specifically.

---

## 5 — Knock-on refactor cost on existing `safety/`

Current code is 《办法》-shaped at the vocabulary level:

- `ApplicationSurface` enum: carries 《办法》-shaped surface definitions.
- `MinorExclusionGate.ages_14_18`: hard-wired thresholds.
- `AIIdentityReminder.two_hour`: hard-wired cadence.
- `CommercialCoercionReport` category names: 《办法》 Art. 8(6) vocabulary.
- `CrisisEscalation.handoff_targets`: Chinese-labeled defaults.
- `HypocrisyGap`: now bilingual EN+ZH (just-expanded), **region-agnostic by
  design** — this one is already pluggable.
- `ExtremeEmotionWatcher`: CN-shaped thresholds in docstrings.
- `docs/compliance/办法-filing.md`: becomes one region's filing doc.

**Refactor cost estimate (Option C — shared core + overlay):**

- Rename vocabulary from 《办法》-specific to regime-neutral where it's a plug
  point (`ApplicationSurface` becomes universal, keeps surface categories;
  cadence/threshold becomes region overlay). ~1 day.
- Introduce `ComplianceFloor` + `RegionOverlay` types; load from config. ~1
  day.
- Author `CN_办法_OVERLAY` as the reference overlay (ensure byte-identical
  behavior to today's hard-coded path when `region="CN"`). ~1 day.
- Author `EU_AI_ACT_OVERLAY` + `US_FEDERAL_FLOOR_OVERLAY` as a cross-regime
  sanity check that the floor + overlay model works. ~2 days.
- Update tests: existing CN-shaped tests should pass unchanged when the CN
  overlay is loaded; new per-region test matrix. ~2 days.
- Rename `docs/compliance/办法-filing.md` → `docs/compliance/regions/cn-办法-filing.md`
  and author stub `regions/eu-ai-act-filing.md` + `regions/us-state-matrix.md`.
  ~1 day.

**Total:** ~1 engineer-week of code + ~1 engineer-week of regime research
per non-CN region. First non-CN region is high setup cost; subsequent regions
amortise.

---

## 6 — Research still owed before implementation

The option ranking above is paper-argument-only. Before locking Option C
(or switching to R), the project owes:

1. **Online channel.** Current regulatory state varies month-to-month. Before
   coding, do a WebSearch/WebFetch pass on each regime for 2026 updates (EU AI
   Act secondary legislation, FDA 2026 AI/ML SaMD, UK OAISR consultation
   outcome, US state-AI-law trackers at IAPP).
2. **OSS channel.** Study how other long-companion / health-adjacent apps
   handle region plug-ins: Replika's TOS region variants, Woebot's SaMD vs
   wellness positioning, Wysa's multi-jurisdiction deployment pattern,
   Microsoft Copilot's EU AI Act compliance surface, OpenAI's region-based
   feature-gating.
3. **PDF channel.** AI心理学应用研究.pdf cites DTN three-tier (transactional
   / relational / transformational) — map each tier to the SaMD boundary
   (transformational edges toward SaMD; keep companion in relational tier at
   most). AI心理长期陪伴研究.pdf cites 《办法》 extensively; need to extract
   the structural requirements separately from the Chinese-specific defaults.
4. **Legal counsel.** Deferred — this plan is architectural, not legal
   advice. Any region's go-live needs counsel; the architecture only needs
   to make the counsel's eventual decisions pluggable.

---

## 7 — Milestones (if the user picks Option C)

Not a commitment — a reference shape for sequencing.

| Milestone | Output |
|---|---|
| M1 | `ComplianceFloor` + `RegionOverlay` types + CN overlay reproducing today's behavior byte-identically |
| M2 | EU overlay sketched to the sanity-check level + regime-neutral vocabulary refactor applied across `safety/` |
| M3 | Test matrix green across CN + EU; `docs/compliance/regions/` populated |
| M4 | US patchwork overlay (treat as compound: COPPA + CCPA + state-AI patchwork) |
| M5 | Pilot sign-off: Phase 4b pilot can declare which overlay is active and expect the correct safety behavior |

M1–M3 are a 1-engineer-week sprint. M4 depends on US state-AI-law research
(not pure code). M5 is a cross-check against the Phase 4b pilot shape in
[`phase-4b-signoff.md`](../phases/phase-4b-signoff.md) — specifically §9.8
which currently reads "under 《办法》 filing threshold" and needs to generalise
to "under all applicable region filing thresholds for the recruited cohort".

---

## 8 — Open questions needing user decision

1. **Option R vs Option C?** This plan recommends Option C (shared floor +
   overlay) on paper-argument-only grounds. User's call.
2. **Target region priority order?** CN is the current baseline. Which
   non-CN region ships second — EU (regulatory maturity, largest
   English-speaking market floor) or US (largest commercial market, highest
   patchwork cost)?
3. **Pilot-time region policy.** The Phase 4b pilot (per
   [phase-4b-signoff.md](../phases/phase-4b-signoff.md) §9.8) is scoped to
   ≤20 users under CN 《办法》 filing threshold. Does the user want the
   pilot (a) stay CN-posture explicitly, (b) switch to a floor-only
   universal posture, or (c) run both CN and one non-CN overlay as an A/B?
4. **Scope boundary.** Does this plan live inside `applications/safety/` or
   does it get its own package (`applications/compliance/`)? Current repo
   convention would favor the latter — compliance is cross-cutting across
   safety, memory (P4 retention + P5 cross-border + P11 audit log), and
   pilot operations (P6 filing, P10 consent).

---

### Assumptions Made

- Assumed the user wants a plan document first, not a scaffold-with-code,
  because the user's explicit words were "make a plan to think about it".
  **(critical)** — if wrong, I'd ship code instead.
- Assumed Option C is the right default, because the project already uses
  the "universal floor + non-decaying covenant + region additions" pattern
  (covenant-anchored memory fragments). Not critical — the plan explicitly
  defers the ranking to the user.
- Assumed regime research can be bounded to the nine regimes listed in §1;
  Canada / Brazil / South Korea / India / Singapore are deferred as
  secondary markets.
- Assumed the pilot's current CN posture is not immediately at risk — the
  pilot has not launched yet, so there is time to refactor before the first
  non-CN user ever joins. If the pilot is already recruiting non-CN users,
  the refactor becomes urgent.

### Discovered Issues

- `src/psychology_base/applications/safety/hypocrisy_gap.py` (flagged
  2026-10-07) — module was comprehensive-rewritten in the previous turn
  (~1700 lines; 13 LoadingCategory, bilingual EN+ZH probe banks,
  MultiTurnProbe, lexical + LLM judge, severity-aware SOTA gate) but is
  **not yet test-verified**. Still owed: comprehensive test suite (keep
  the original 10 tests green; add tests for new taxonomy / classifier /
  LLM judge / SOTA gate), `safety/__init__.py` export update, Phase 4a
  status doc + memory sync, full-suite verification. Does not block this
  compliance plan but should land before Phase 4b pilot operations.
- `docs/compliance/办法-filing.md` (flagged 2026-10-07) — needs a header
  note that it is one region's filing doc, pending region-pack refactor.
  Non-blocking; purely cosmetic.
