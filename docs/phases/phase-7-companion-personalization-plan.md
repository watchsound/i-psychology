# Phase 7 — Companion personalization bootstrap (import → trait inference → response conditioning)

**Date drafted:** 2026-10-08
**Status:** PROPOSAL. Scope anchor only; implementation unblocks on user approval of §12 decision table.
**Precedent:** [phase-6-close-out.md](phase-6-close-out.md) closed v1 memory-substrate work (绵延 held at ≥ 1.00 through ep1000; V5 covenant 8/8 zero-drift; PAM mid-stream round-trip 6/6). Phase 7 is the first application-layer phase after v1 close — targeting 情感陪伴 onboarding and response personalization, not substrate.
**User directive (2026-10-08):** three application-layer requirements for the long-term emotional companion — (1) import existing chat history at onboarding; (2) gradually build the user's 人格特质; (3) use the inferred 人格特质 to shape response generation. Sub-directive on (1): *import only used to generate 人格特质 和 其他 meta data about user* — imports feed inference, never `recall()`.

---

## 1. Why this phase exists

The companion as shipped at v1 close assumes a cold-start user: `CompanionSystem.observe()` begins populating fragments turn-one, with no prior model of who the user is. For 情感陪伴 this is a load-bearing miss. Users with any prior chat-history (ChatGPT, Replika, Character.AI, Line/WhatsApp journals, prior companion apps) arrive carrying substantial self-disclosure the companion has to re-elicit from scratch — a bad onboarding experience and a false blank-slate posture when the user already knows the AI "knows nothing yet."

Phase 7 closes three gaps:

1. **Onboarding ingestion gap.** No path exists to seed the companion from pre-existing user material.
2. **Trait representation gap.** Mentalistic sub-stores exist (Phase 2 M2) but no OCEAN / BFI-2 / implicit-vs-explicit trait inference layer feeds them. [NLP Personality System Architecture.pdf](../../reference-docs/NLP%20Personality%20System%20Architecture.pdf) specifies Big Five as the research-base default; the project has never implemented it.
3. **Response personalization gap.** `CompanionAgent` composes prompts from Rules + recall. No trait-aware scaffolding exists; a high-neuroticism user and a low-agreeableness user receive the same pacing.

Phase 7 adds three modules + a feature-flag-gated conditioning path, under the one-way-boundary design the user named 2026-10-08: raw imports never enter the fragment store; only inferences do, and only after user confirmation.

---

## 2. Scope — three features, one phase

### 2.1 Chat-history import (bootstrap-only)

- Parsers for three common export formats: ChatGPT JSON, WhatsApp text dump, generic JSONL turn-by-turn. Each adapter is a thin shim; the target data model is a shared `ImportedTurn(speaker, content, timestamp, metadata)` sequence.
- Source-timestamp preservation (so a 2-year-old turn is 2 years old, not fresh).
- Pre-ingest safety sweep: run the import batch through `safety/hypocrisy_gap.py` + `safety/crisis_lines.py` + the extreme-emotion detector before inference. Self-harm content surfaces the Tier-2 handoff path to the user, does not silently ingest.
- Consent artefact at import time ([`docs/compliance/consent-artefact-template.md`](../compliance/consent-artefact-template.md)) — personally sensitive material requires explicit user attestation.
- **One-way boundary:** `chat_import.py` has no code path to `CompanionSystem.observe()` for raw turns. Enforced by module structure (no import of `CompanionSystem` from the import module) and by regression test.
- Raw imports retained under `imports/<user_id>/raw/*.json` as a compliance audit log — **outside** the fragment store, **never** consulted by `recall()`, shreddable on user request per 《办法》 data-minimization.

### 2.2 Gradual 人格特质 inference

- OCEAN / Big Five per [NLP Personality System Architecture.pdf](../../reference-docs/NLP%20Personality%20System%20Architecture.pdf). MBTI explicitly rejected per the same source.
- Two-layer representation reusing Phase 2 M2 mentalistic sub-stores:
  - `self_explicit` — statements the user makes about themselves ("I'm not a math person"). Tag prefix `subject:user:trait:<dim>:explicit:*`.
  - `self_implicit` — inferred from observed behavior: message-length distribution, time-of-day pattern, affective-valence distribution, response-to-challenge rate, bilingual-switch markers. Tag prefix `subject:user:trait:<dim>:implicit:*`.
- Other meta extracted alongside OCEAN:
  - Significant-other mentions (`entity:mother`, `entity:lin-sister`) — Pillar 2 Store #2 scaffolding.
  - User self-model statements — Store #1 seeds.
  - Semantic-drift timeline anchor for recurring symbolic concepts (家 / 成功 / "a good day").
  - Communication-style markers (formality, pacing, emoji use) — scaffolding config, not persona.
  - Affective baseline (PAD-2 distribution) — starting point for Phase 2 M3 affective-resonance retrieval.
- **Traits are reconstructions, not fields** (Principle C). On query, the trait value is rebuilt by walking tagged fragments under current context; it is not a static dataclass column.
- **Selection-pressure gating** (Principle D). A single data point moves the estimate by ε only; traits emerge from accumulated evidence. Covenant-tagged user-explicit statements ("I live with major depression") are protected from decay; implicit markers decay like any other fragment.
- **Confidence surface.** Every retrieved trait estimate carries a `confidence ∈ [0, 1]` derived from (a) count of underlying observations, (b) explicit/implicit agreement, (c) recency distribution. Low-confidence traits do not condition responses.

### 2.3 Trait-conditioned response scaffolding

- `CompanionAgent` gains a `TraitProfile` reconstruction fetch at prompt-assembly time.
- Conditioning adjusts **scaffolding and pacing**, not content:
  - High-neuroticism user → longer validation dwell before Socratic probing (Rule 1a extended).
  - High-conscientiousness → structured turn-taking with explicit agendas.
  - High-openness → generative probes allowed earlier.
  - Low-agreeableness → Rule 2 anti-sycophancy allowed to speak more sharply.
- **Hard constraints that do not bend to traits:**
  - V5 covenant 8/8 remains hard-coded. Crisis handoff fires regardless of agreeableness reading. AI-identity reminder fires regardless of trait profile.
  - Minor-exclusion gate (<14 strict ban, <18 guardian) runs before any trait-conditioning.
  - Rule 2 anti-sycophancy floor is a floor, not a ceiling — low-agreeableness tightens it, high-agreeableness never loosens below floor.
- **Feature-flag-gated.** Shipped behind `TraitConditioningFlag(enabled=False)` initially; enables only after §9 benchmark gates pass.

---

## 3. Hard + soft gates

| Gate | Target | Nature |
|---|---|---|
| **One-way boundary enforcement** | Static test confirms `chat_import.py` has no transitive import of `CompanionSystem.observe()` for raw-turn ingestion; raw imports never appear in `FragmentStore.all()` after import pass | **hard; structural** |
| **V5 covenant 8/8 under trait conditioning** | At ep500 under trait-conditioning-on across the OCEAN extremes (high-N / low-A / high-O / low-C) × dense + sparse-monthly streams, every covenant probe still holds | **hard; zero drift** |
| **Hypocrisy Gap regression under trait conditioning** | `FULL_PROBE_SET` severity high + medium counts unchanged byte-identical with trait-conditioning on vs off; no new HIGH-severity sycophancy regressions | **hard** |
| **User-confirmation gate** | No inferred `subject:user:*` or `entity:*` fragment enters the active store without a confirmed-by-user flag; audit-log test confirms every ingested inference has `source:inferred-confirmed` | **hard; privacy** |
| **Import-to-OCEAN accuracy** | On synthetic imports with known OCEAN ground truth (N≥30 personas, 3 message volumes), inferred trait vector within ±0.5 Likert unit on ≥ 3 of 5 dimensions at 80% confidence threshold | soft; informational |
| **Stereotype-collapse eval** | Under trait conditioning, companion response diversity (lexical + semantic-embedding spread) on identical-prompt-different-user does not collapse below baseline-diversity × 0.70 | soft; informational |
| **Minor-exclusion under import** | Any imported history with age-under-18 indicators triggers minor-gating sweep; strict-ban-14 behavior unchanged | **hard; 《办法》 compliance** |

**Decision gate stance.** If trait inference accuracy is below the soft gate, ship with the limitation documented ("inference remains noisy below N imported turns"). If V5 covenant drift or Hypocrisy Gap regression appears under trait conditioning, **disable the feature flag and ship without §2.3** — §2.1 + §2.2 can land independently.

---

## 4. Deliverables

- `src/psychology_base/applications/emotional/chat_import.py` — three format adapters + `ImportedTurn` data model + consent + safety sweep + raw-audit-log writer. One-way boundary structurally enforced.
- `src/psychology_base/applications/emotional/trait_profile.py` — `TraitProfile` reconstruction (OCEAN vector with per-dimension confidence) + `UserMeta` (significant-others, self-model, affective baseline, communication-style markers).
- `src/psychology_base/applications/emotional/trait_inference.py` — explicit-vs-implicit inference passes; LLM-backed with `LLMRouter` SIMPLE-tier default + ADVANCED fallback on low-confidence cases. Confirmation-gate wiring to onboarding flow.
- Extensions to `CompanionAgent`: `TraitConditioningFlag`, prompt-assembly hook that pulls `TraitProfile`, Witness-side sycophancy regression check.
- `src/psychology_base/applications/emotional/import_cli.py` — CLI entrypoint for onboarding (`import-chat-history` subcommand) extending `pam_cli.py` pattern.
- Benchmark adapters:
  - `src/psychology_base/eval/benchmarks/b4_import_to_profile.py` — synthetic imports + OCEAN ground truth; LLM-labeler.
  - `src/psychology_base/eval/benchmarks/b5_stereotype_collapse.py` — identical-prompt-different-user diversity measurement.
  - Regression harness extensions in `benchmarks/scripts/phase7_*.py` for V5 + Hypocrisy under conditioning.
- Experiment artefacts under `docs/experiments/phase-7-personalization/`:
  - `import-safety-sweep.json` — safety-stack regression on imported corpora.
  - `trait-inference-accuracy.json` — B4 results per persona.
  - `stereotype-collapse.json` — B5 diversity measurement.
  - `v5-covenant-under-conditioning.json` + `hypocrisy-under-conditioning.json` — regression evidence.
- Phase 7 close-out memo at `docs/phases/phase-7-close-out.md`.

---

## 5. Not in scope

- Fine-tuning or model-weight updates. Trait conditioning is prompt-space only.
- Multimodal personality signals (vocal F0, FACS, keystroke dynamics). Reference-docs cite these; v1 is text-only.
- Longitudinal real-user trait drift validation. Trait inference is tested on scripted + synthetic data; real-user drift evidence is Phase 8+ or recruitment-budget-unblock dependent (same cold-storage status as Phase 4b live-pilot path).
- Response generation on dimensions beyond OCEAN (humor style, cultural identity axes, political alignment) — out-of-scope, trait-conditioning stays on the OCEAN surface for v1.
- Trait inference from imports in non-text formats (voice logs, screenshots). Text-only ingestion.
- Any new covenant rule. V5's 8 items are frozen; Phase 7 ships under the existing covenant surface.
- A fourth mentalistic store beyond Phase 2's four (self / other-as-user-represents-them / user-of-AI / AI-of-user). Phase 7 populates Store #1 + Store #4 with inferred content; it does not add new stores.
- Live-pilot evaluation. All Phase 7 evaluation is scripted + synthetic + LLM-judge, same closure posture as Phase 4b.

---

## 6. Sequence under Auto Mode

Each step stops at its own kill criterion. V5 covenant drift anywhere halts the sequence immediately (safety failure).

1. **This memo** (plan).
2. **`chat_import.py` + three format adapters + one-way-boundary enforcement test.** Structural; low design risk.
3. **Safety sweep on imported corpora** — regression test confirms Hypocrisy + crisis + extreme-emotion detectors fire on synthetic imports containing seeded risk material.
4. **`trait_profile.py` + `trait_inference.py` passive mode** — inference runs, outputs TraitProfile + UserMeta, but conditioning flag stays `enabled=False`. Store inferences via confirmation gate.
5. **B4 import-to-profile accuracy benchmark** — synthetic personas, LLM-labeled ground truth, measure ±0.5 Likert accuracy. Below soft gate ships with documented limitation.
6. **CompanionAgent trait-conditioning hook** behind feature flag — prompt-assembly reads TraitProfile, scaffolding adjusts, Witness-side sycophancy check wires in.
7. **V5 covenant regression under conditioning** — ep500 sweep across OCEAN extremes. **Hard gate: 8/8 or feature flag disabled permanently.**
8. **Hypocrisy Gap regression under conditioning** — `FULL_PROBE_SET` byte-identical severity counts. **Hard gate: no new HIGH-severity regressions or feature flag disabled.**
9. **B5 stereotype-collapse eval** — soft gate; ships with limitation documented if below floor.
10. **Phase 7 close-out memo** with per-gate verdict + decision on feature-flag default.

---

## 7. Architecture — the one-way boundary and data flow

```
chat log file (ChatGPT / WhatsApp / JSONL)
          │
          ▼
chat_import.py
  parse → ImportedTurn[]
  safety sweep (hypocrisy + crisis + extreme-emotion detectors)
  consent attestation check
          │
          ├──→ imports/<user_id>/raw/*.json         (audit log; never consulted by recall)
          │
          ▼
trait_inference.py
  explicit pass (user-statement extraction) → OCEAN explicit vector
  implicit pass (behavioral markers)       → OCEAN implicit vector
  significant-other extraction             → entity:<id> candidates
  self-model statements                    → subject:user:* candidates
  semantic-drift seed                      → drift-timeline anchors
  affective baseline                       → PAD-2 distribution
          │
          ▼
onboarding confirmation gate
  surface candidates to user
  user confirms / rejects / corrects
          │
          ▼
CompanionSystem.observe()  ← only confirmed items, tagged source:inferred-confirmed
```

**Three structural invariants enforced in tests:**

1. `chat_import.py` imports no symbol from `psychology_base.companion.system`. The module boundary is static-analyzed in `tests/test_phase7_import_boundary.py`.
2. Every fragment with tag prefix `source:inferred-*` must also carry `source:inferred-confirmed` to appear in `FragmentStore.all()`. Enforced via `_ingest_fragment` precondition.
3. `imports/<user_id>/raw/` paths are never read by any code in `psychology_base.companion.*` or `psychology_base.applications.emotional.companion_agent`. Enforced via directory-isolation test.

---

## 8. Trait representation detail

| OCEAN dimension | Explicit signal | Implicit signal | Confidence weight |
|---|---|---|---|
| Openness | Direct statements re: curiosity, art, abstract thinking; openness-related self-labels | Topic breadth entropy across sessions; abstract-vs-concrete token ratio; novel-concept introduction rate | Volume (log N observations) + explicit/implicit agreement |
| Conscientiousness | Self-reported planning / organization habits; goal statements | Message-structure markers (lists, bullet points, chronological ordering); session-boundary punctuality | Same |
| Extraversion | Direct statements re: social preference | Session-length distribution; emoji use; exclamation / question ratio; time-of-day pattern | Same |
| Agreeableness | Statements re: conflict handling, trust | Challenge-response pattern (back-down vs engage); profanity rate; disagreement-with-AI frequency | Same |
| Neuroticism | Statements re: mood, stress, worry | PAD-2 negative-valence baseline; catastrophizing lexical markers; crisis-adjacent language frequency (bounded; crisis itself routes to handoff, not trait inference) | Same |

**Reconstruction, not field.** `TraitProfile.get(dim)` walks tagged fragments at query time; it does not read a column.

**Covenant-tagged trait exceptions.** A user statement like "I have major depressive disorder, diagnosed" or "I am in recovery from alcohol dependence" is covenant-tagged — protected from decay, immune to selection pressure. These populate `self_explicit` permanently and the trait-conditioning path respects them as hard constraints (e.g., `recovery:alcohol` suppresses any response that treats alcohol casually, regardless of OCEAN reading).

**Explicit-vs-implicit gap as clinical hypothesis engine.** When `|explicit_openness − implicit_openness| ≥ 0.5 Likert units`, the gap is surfaced to the companion as a candidate clinical hypothesis for the next session — the Phase 2 M2 Store #1 vs Store #4 gap mechanism, now actuated.

---

## 9. Benchmarks

### B4 — Import-to-profile accuracy (new)

- **Setup.** 30 synthetic personas with pre-assigned OCEAN vectors (Likert 1–5 per dimension). Each persona generates N={50, 200, 500} messages via LLM following the pre-assigned profile. Imports fed through Phase 7 pipeline; inferred TraitProfile compared to ground truth.
- **Metric.** Mean absolute error per OCEAN dimension; fraction of dimensions within ±0.5 Likert units at 80% confidence threshold.
- **Pass bar.** ≥ 3 of 5 dimensions within ±0.5 at N=200 messages. Explicitly degraded gate: inference quality bounded by import volume.
- **Honesty posture.** Synthetic personas are weak evidence vs real users; this is a mechanism-viability floor, not SOTA claim.

### B5 — Stereotype collapse (new)

- **Setup.** 10 distinct TraitProfile seeds (OCEAN extremes + mid-points). Companion responds to the same 20 scripted prompts under each profile. Measure response diversity:
  - Lexical: unique-token count across the 10 responses to same prompt.
  - Semantic: pairwise embedding distance across the 10 responses.
- **Baseline.** Same measurement with trait-conditioning disabled.
- **Pass bar.** Trait-on diversity ≥ baseline diversity × 0.70. If below, trait conditioning is reducing user-space to stereotypes.
- **Carried debt.** Semantic-diversity requires an embedding model (sentence-transformers or similar); provision step wired via `LLMRouter.embed`.

### Regression harness

- V5 covenant 8/8 at ep500 under OCEAN extremes × dense + sparse-monthly streams. Hard gate.
- Hypocrisy Gap `FULL_PROBE_SET` severity count byte-identical under conditioning on vs off. Hard gate.
- Minor-exclusion gate fires on imports containing <14 / <18 age indicators. Hard gate.
- B2 FANToM accuracy on BELIEF_INACCESSIBLE under conditioning on vs off; monitor, not gate (expect no delta).

---

## 10. Dependencies on prior phases

- Phase 2 M2 mentalistic sub-stores (`subject:<id>` / `entity:<id>` tag prefixes + `recall(scope=...)`) — **ships; Phase 7 populates them with inferred content.**
- Phase 2 M3 PAD-2 affect — **ships; Phase 7 uses it for affective baseline in trait inference.**
- Phase 4a CompanionAgent + safety stack — **ships; Phase 7 extends it with trait-conditioning hook + Witness sycophancy check.**
- Phase 4a `LLMRouter` + `LLMSemanticJudge` — **ships; Phase 7 uses SIMPLE-tier for inference, ADVANCED for low-confidence cases.**
- Phase 4a `safety/hypocrisy_gap.py` + `safety/crisis_lines.py` + extreme-emotion detector — **ships; Phase 7 wires them into import pre-ingest sweep.**
- Phase 4b §11 `TraitConditioningFlag` precedent — none exists; new feature flag.
- Phase 5 Track 1 FANToM scope-selector — **ships; informs trait-inference scope selection (which fragments feed which dimension).**
- Phase 6 PAM mid-stream round-trip — **ships; trait fragments + UserMeta included in PAM export surface.**

---

## 11. Risks

- **Trait inference is low-ground-truth.** BFI-2 self-report is weak; behavioral traces need volume. Early-pilot trait estimates will be noisy. **Mitigation:** confidence-weighted conditioning — low-confidence traits do not condition responses.
- **Stereotype collapse under conditioning.** The AI reduces the user to their profile. **Mitigation:** B5 benchmark + fresh-observation-override principle (reconstruction, not field).
- **Import pollution despite boundary.** A clever path through inference → confirmation → observe could still smuggle imported content. **Mitigation:** every ingested inference carries `source:inferred-confirmed` tag + raw-audit-log directory isolation test.
- **Sycophancy regression under trait conditioning.** High-agreeableness reading could soften Rule 2. **Mitigation:** V5 + Hypocrisy regression hard gates + Rule 2 floor-not-ceiling architecture.
- **《办法》 compliance under trait inference.** Trait inference is "personality profiling of a natural person" — likely heightened consent + data-minimization obligations. **Mitigation:** consent artefact at import; audit-log retention per data-minimization; shred-on-request path; raw imports outside fragment store.
- **Chinese-user OCEAN bias.** Big Five is Western-developed. **Mitigation:** NEO-PI-R Chinese adaptations ([NLP Personality System Architecture.pdf](../../reference-docs/NLP%20Personality%20System%20Architecture.pdf)) for ZH user path; separate linguistic-marker set for ZH-specific implicit signals.
- **Confirmation-gate friction.** Onboarding surface "I'm picking up that you have a sister Lin — is that right?" could feel intrusive. **Mitigation:** user-controlled confirmation granularity (confirm all / confirm high-confidence only / defer); default to "confirm high-confidence only."
- **Covenant-tagged trait handling for crisis-adjacent self-disclosure.** User importing a journal mentioning historical suicidal ideation requires careful handling — covenant-tag protects memory, but surfacing it in onboarding confirmation may re-traumatize. **Mitigation:** crisis-adjacent self-disclosure during import routes to Tier-2 handoff path before confirmation gate, not into trait-inference pass.

---

## 12. User-side decision table (gates implementation start)

Same disposition pattern as Phase 4b §9. Each row is a judgment call user owns.

| # | Item | Default under Auto Mode | Status |
|---|---|---|---|
| D1 | Target import formats for v1 | ChatGPT JSON + WhatsApp text + generic JSONL (three adapters) | pending user approval |
| D2 | Trait-conditioning feature flag default on ship | `enabled=False` (opt-in per user) vs `enabled=True` (opt-out) | **pending user judgment call** — recommend False for v1 |
| D3 | Confirmation-gate granularity default | High-confidence-only (default) vs all-inferences (slower, more thorough) vs defer-until-later | pending user approval |
| D4 | Raw-import retention window | 90 days post-import (matches Phase 4b §9.8 retention posture) vs 30 days vs shredded-after-inference-pass | pending user judgment |
| D5 | ZH-vs-EN OCEAN adaptation | Full NEO-PI-R Chinese adaptation vs Big Five with ZH-marker set only | pending user judgment |
| D6 | Trait-inference LLM budget | SIMPLE-tier only (cost floor) vs ADVANCED fallback on low-confidence (quality ceiling) | pending user approval; recommend ADVANCED fallback |
| D7 | Benchmark auditor requirement | B4 LLM-labeler only (CI-only) vs κ ≥ 0.60 human co-labeler on subset | pending user judgment; recommend LLM-only for v1 |
| D8 | Covenant-tagged self-disclosure handling | Route to Tier-2 handoff path before inference vs ingest + surface + handoff | pending user judgment; recommend route-before-inference |

**Hard dependencies that stay blocking even after this memo is signed:**

- Synthetic persona corpus for B4 (can be LLM-generated; no human blocker).
- OCEAN ground-truth label pipeline for B4 (LLM-labeler is sufficient under D7 default).
- Confirmation-gate UI copy (user drafts or approves AI draft).
- 《办法》 legal review of trait-inference consent artefact text (user action).

---

## 13. Carried debt from prior phases that Phase 7 unblocks

- **Phase 2 M2 attribute-reconstruction scaffolding** — populated with real inferred content; closes the "sub-stores exist but hold only test fixtures" debt.
- **Phase 2 M3 affective baseline** — gets a seeded baseline from import rather than cold-start uniform prior.
- **Phase 4b §11 companion CLI onboarding flow** — Phase 7 is the first user-facing feature past the CLI's enrolment entry point.
- **Phase 5 Track 2 architectural anti-sycophancy** — Phase 7 trait-conditioning stress-tests whether Rule 2 holds under personality-shaped responses, re-opening a controlled slice of the Track 2 question Phase 5 closed as "mechanism tie on arbitrary factual probes."

---

## 14. Carried debt into Phase 8+ (not resolved in Phase 7)

- Live-pilot trait drift validation — Phase 8+ or recruitment-unblock dependent.
- Multimodal trait signals (vocal, FACS, keystroke) — v2 scope.
- Trait inference beyond OCEAN (humor, cultural identity, political alignment) — v2 scope.
- Covenant-rule evolution for trait-conditioning edge cases — v1 freezes the 8-item covenant; Phase 7 ships under it.
- Cross-session trait reconstruction stability under capacity pressure — monitor through B4 at N=500; longer-horizon stability study is v2.

---

## 15. Compute + wall-time budget

- Chat import pipeline: file-size-dependent, dominated by LLM inference calls during the inference pass. SIMPLE-tier at ~1 s/call × ~N/20 calls per import (20 turns per inference batch). An N=500 import ≈ 25 LLM calls ≈ 25 s cache-warm, ≈ 2 min cold.
- B4 synthetic benchmark: 30 personas × N={50, 200, 500} × inference pass = 90 inference runs ≈ 45 min cache-warm.
- B5 stereotype-collapse: 10 profiles × 20 prompts × 2 (on/off) = 400 companion responses + embedding pass ≈ 20 min cache-warm.
- V5 regression under conditioning: 4 OCEAN extremes × 2 streams × ep500 = ~40 min cache-warm.
- Hypocrisy regression under conditioning: 54-probe `FULL_PROBE_SET` × 2 (on/off) = ~15 min cache-warm.

Total Phase 7 wall budget: ~2.5 h cache-warm. Within Phase 5/6 budget envelope.

---

## 16. Decision gate stance

Three asymmetric outcomes:

1. **All hard gates pass + both soft gates pass.** Ship §2.1 + §2.2 + §2.3 as full phase; feature flag default per D2.
2. **Hard gates pass + one or both soft gates fail.** Ship §2.1 + §2.2 + §2.3 with documented soft-gate limitation; feature flag default `enabled=False` regardless of D2 preference until soft gates re-cleared in Phase 7.1.
3. **Any hard gate fails.** Ship §2.1 + §2.2 only; §2.3 permanently disabled under current `CompanionSystemPrompt`; close-out memo documents the structural reason and names the Phase 7.1 reopen conditions. Same closure posture as Phase 5 Track 2.

No fourth path. V5 covenant drift halts immediately; sycophancy regression under conditioning disables the feature flag; stereotype collapse below floor disables the feature flag.

---

## Discovered Issues

None at planning time. Issues flagged during implementation will land in the Phase 7 close-out memo.

## Assumptions Made

- **(critical)** Assumed import adapters for ChatGPT JSON + WhatsApp text + generic JSONL cover ≥ 80% of expected v1 user import volume. If user feedback shows the target population imports from Line / Telegram / Character.AI JSON predominantly, scope expands and §2.1 ships with additional adapters.
- **(critical)** Assumed trait-conditioning stays prompt-space only. If `CompanionAgent` design evolves to parametric routing (model selection by trait, system-prompt-template selection by trait), the sycophancy regression surface widens and Phase 7 scope grows.
- Assumed `TraitConditioningFlag` is a per-user config, not a global config. A single user running trait-conditioning on while another runs it off is supported; shared global flag is not.
- Assumed Big Five OCEAN is the research-base default per [NLP Personality System Architecture.pdf](../../reference-docs/NLP%20Personality%20System%20Architecture.pdf). MBTI is explicitly out; HEXACO (adds Honesty-Humility sixth dimension) is out of scope for v1 but reasonable Phase 7.1 extension.
- Assumed LLM-labeler sufficient for B4 ground-truth in CI mode. If pilot auditor budget appears, κ ≥ 0.60 human co-labeler on subset is a drop-in extension.
- Assumed raw-import audit-log retention of 90 days matches Phase 4b §9.8 posture. If 《办法》 legal review shortens retention, retention shortens; inference pass completes within retention window regardless.
- Assumed trait inference during onboarding does not require real-time response (user can wait 1–2 min for the inference pass to complete). If real-time is required, Phase 7.1 adds a streaming inference variant.
