# Phase 2 — Multi-session and mentalistic processes

**Date drafted:** 2026-10-07
**Reason this phase exists:** Phase 1.5 outcome retracted the LoCoMo 0.70 exit gate (reader-model-bound, not substrate-bound; see [roadmap.md](../roadmap.md) Phase 1.5 outcome section). R5 F1=0.544 shipped as non-regression monitor. The project pivots back onto its stated thesis — behavioral SOTA on reconsolidation + 换位思考 + 共情 — rather than continuing to grind single-hop paraphrase extraction. This phase builds the mechanisms the thesis actually names.

**Companion docs:** [roadmap.md](../roadmap.md) sequences the overall plan; [sota-goal.md](../sota-goal.md) defines SOTA behaviorally; [memory-design.md](../memory-design.md) names the three pillars; [benchmarks.md](../benchmarks.md) defines B1 / B2 / B3 / V-track metrics; this doc chooses the mechanisms that produce measurable movement on those metrics.

**Scope reminder from roadmap.md Phase 2:**
- Reconsolidation across sessions (Principle A) → B1 full suite
- Mentalistic objects as processes (Principle C) → B2 full suite
- Affective-resonance retrieval → feeds B3 (full B3 is Phase 3)
- Covenant-boundary non-decaying rules verified under rule-decay → V5 style portion
- Shared-referent tagging (`co_experienced=true`) — minimal dyadic layer, no AI-subjective perspective

**Phase 2 exit gates (from roadmap, abbreviated):**
- B1 fragment preservation ≥ 95%; rendering divergence 0.25–0.45; within-session Jaccard ≥ 0.65
- B2 attribute reconstruction ≥ 10/12; grounding ≥ 70% cite ≥ 2 stored attributes; gap-detection precision ≥ 60% @ recall ≥ 50%
- V3 mentalistic liveness within 0.10–0.30 Jaccard-distance band
- V5 style-rule internalization ≥ 70% held after prompt-text removal
- LoCoMo 3-instance F1 ≥ 0.514 (monitor, not gate)

---

## Phase planning process checklist

Per [memory feedback](../../CLAUDE.md) `feedback_phase_planning_process.md`, each phase plan runs 6 research + synthesis steps before mechanism choice, and 2 experiment + comparison steps during build:

- [x] **Step 1 — Deep PDF research** on reference-docs/ chapters relevant to this phase's primitives
- [x] **Step 2 — Online research** for post-PDF developments (arXiv 2024–2026, lab blogs, leaderboards)
- [x] **Step 3 — OSS study** of how popular memory frameworks implement (or don't implement) each primitive
- [x] **Step 4 — Brainstorm** without early filtering
- [x] **Step 5 — Compress** to 2–4 options each with (mechanism | target gap | prior art | kill criteria)
- [x] **Step 6 — Cross-check** each option against learning-companion + emotional-companion north stars
- [ ] **Step 7 — Experiment** on viable candidates, persist results per-option under `docs/experiments/phase-2-mentalistic/option-X/`
- [ ] **Step 8 — Comparison table** as the side-by-side verdict the plan cites when choosing

Each step gets a numbered section below as it is produced.

---

## 1. PDF research — reconsolidation / mentalistic / affective primitives in the reference base

Three parallel Explore passes across all six PDFs, 2026-10-07.

### 1a. Reconsolidation measurement (B1 operationalization)

**Finding: zero precedent.** The PDF base covers consolidation (SleepGate, Ebbinghaus `R=exp(-t/S)`, Reflection Tree distillation) and retrieval scoring (Generative Agents `score = α·recency + β·importance + γ·relevance`) but has **no operationalization of reconsolidation as "same event, different rendering"**.

- `AI Agent Memory Framework Analysis.pdf:5` — Ebbinghaus curve for decay, not reconstruction
- `智能体记忆架构研究.pdf:5` — Generative Agents retrieval score; `pp.8` benchmark table (LongMemEval, LoCoMo, DMR) all score **retrieval accuracy**, not evolution
- `AI心理学应用研究.pdf:3-4` — DTA ladder, Therabot Cohen's d 0.64–0.90, Perceived Alignment r=0.53 — all measure **outcome improvement**, not narrative progress
- Fragment-preservation %, divergence scores, Jaccard trajectories, LLM-judge paradigms for same-event-different-rendering: **none cited**

**Implication for B1.** Our roadmap targets (fragment-preservation ≥ 95% / rendering-divergence 0.25–0.45 / within-session Jaccard ≥ 0.65) are novel measurements with no PDF prior art. Minimum operationalization: seed with identical fragments + different session contexts, measure semantic-shift via embedding distance (cosine or Wasserstein), validate reframing-vs-confabulation via LLM-judge on a scripted sample. This is scaffolding we must build; no borrowing possible from PDF frameworks.

### 1b. Mentalistic substrate shape (B2 operationalization)

**Finding: zero precedent for per-entity sub-process; all PDF frameworks use static profile or biographical slots.**

- `NLP Personality System Architecture.pdf:5` (PANDORA), `:8` (BIG5-CHAT) — persona = BERT 768-d embedding + 5 Big-Five scalars or 15-facet probabilities. Immutable snapshot. No mutation history.
- `AI Agent Memory Framework Analysis.pdf:2-3` — Letta episodic = timestamped JSON; MemOS MemCube = plaintext + KV-cache + weights. Sentra dual-time-state KG for third-party tracking is the closest existing art — still retrieval-based, not process-level. `:3` Portable Agent Memory Merkle-DAG offers versioning but as **provenance**, not reconstruction.
- `AI心理学应用研究.pdf:8-10` — DTA framework has transactional / relational / supportive tiers but none of them require a distinct entity model for third parties. All third parties appear as biographical slots.
- `AI心理长期陪伴研究.pdf:5` (SDT three-needs), `:9-10` (WAI-SR) — AI's model of user is scalar bond/task/goal scores, not a reconstructible process.

**Implication for substrate.** The minimum shape no framework ships: `(subject_id, entity_id, timestamped-event-log, reconstructed-state-on-read)`. Static embedding + biographical slots fail Principle C by construction. This is a new abstraction, not a port.

### 1c. Affect encoding (B3 feed, Phase-2-stage affective-resonance)

**Finding: text-only extractable options exist; zero precedent for storing affect ON a fragment.**

- `AI心理学应用研究.pdf:2-3` — vocal (F0 / jitter / shimmer), facial (FACS AU6/AU12/AU15/AU1/AU4), keystroke (IKD / hold time) are the biomarker PDF canon. Depression detection 90.47% from audio. **All require non-text modalities.**
- `NLP Personality System Architecture.pdf:2-4` — text-only path: BERT/RoBERTa 768-d → Big-Five 5-scalar classification; Neuroticism loads on negative affect. BFI-2 15-facet continuous distribution.
- `AI心理长期陪伴研究.pdf:5-6` — WAI-SR Bond (3.89–4.40) / Task / Goal measured per session, text-questionnaire-derivable. Zero cross-session affect-matching precedent cited.
- `AI Agent Memory Framework Analysis.pdf:4-5` — Mem0 / Zep / Letta / MemOS: **none tag fragments with affect state at observe-time.**

**Implication for schema.** Text-only, Chinese-regulatory-defensible, no biometric: either a **3-scalar WAI-SR-style** (Bond / Task / Goal, session-level) or **5-scalar Big-Five** (fragment-level) tag. Storage on fragments is novel; the PDFs provide the encoding, not the schema.

---

## 2. Online research — post-PDF developments 2024–2026

Three directly-relevant 2024–2026 papers found that address each Phase 2 gap:

| Paper | arXiv | Phase 2 gap it addresses | What to borrow |
|---|---|---|---|
| **REALM** — Reconsolidation-Evolution Agentic Long-Term Memory | `2609.16053` | Gap 1 (reconsolidation) | Explicit reconsolidation framing on a heterogeneous cognitive graph with retrieval-feedback-driven updates. **Closest prior art for Gap 1.** Read in full before freezing the reconsolidation substrate. |
| **PersMem** — Internalizing Personality into Dual-Pathway Memory | `2609.34372` | Gap 2 (mentalistic process) + Principle F (rules → dispositions) | Dual-pathway persona-into-memory. Persona is memory-resident, not prompt-resident. |
| **Emotional RAG** (Huang et al.) | `2410.23041` (ICKG 2024) | Gap 3 (affect-resonance retrieval) | Dual-vector encoding: semantic + emotion vectors; two retrieval strategies (combination / sequential). **OSS code at `BAI-LAB/EmotionalRAG`.** Direct template. |
| **Portable Agent Memory** | `2605.11032` | Substrate shape | Content-addressed Merkle-DAG with `id (BLAKE3) / parent_ids / created_at / version` — append-only shape satisfies Digital-Bereavement export obligation natively. |
| **From Human Memory to AI Memory** (survey) | `2504.15965` | Positioning | Mentions where reconsolidation sits in the taxonomy; useful for the thesis-claim framing. |

Current leaderboards (LongMemEval, LoCoMo) are unchanged — reader-model-floor-dominated; see Phase 1.5 outcome. **PerspectiveQA / ToMBench** are the Theory-of-Mind benchmarks worth tracking as external validation for Gap 2 but are not Phase 2 exit criteria.

---

## 3. OSS study — version chains, perspective-taking, affect

Confirmed: **all three Phase 2 gaps are OSS-wide.** Per-repo findings:

| Repo | Version history on same identity | Persona / mentalistic | Affect tagging |
|---|---|---|---|
| **mem0ai/mem0** | Partial — `history` API logs ADD/UPDATE/DELETE events, but April 2026 algorithm switched to single-pass append-only, so history is a flat op log, not a chain tied to stable fact identity | No — only `user_id / agent_id / session_id` filters | No |
| **getzep/graphiti** | Bi-temporal `valid_at / invalid_at` on edges handles **external-world** fact change. Treats same-speaker reinterpretation as if the world flipped — wrong primitive for Gap 1 | No | No |
| **letta-ai/letta** | Core blocks are strings mutated in-place via `core_memory_append` / `core_memory_replace` — rewrite, not chain | Persona + human blocks are free-text, not entity-scoped processes | No |
| **MemTensor/MemOS** | Tiered L1/L2/L3 — no version chain primitive in OSS src | README mentions "personas" but no sub-store primitive | No |
| **langchain-ai/langmem** | None | None | None |
| **Portable Agent Memory** (`arXiv:2605.11032`) | **Yes — Merkle-DAG natively supports append-only version chains** (`parent_ids`, content-addressed) | No | No |

**Substrate ideas worth borrowing:**

1. **Portable Agent Memory Merkle-DAG** as on-disk shape for fragment-append + reconsolidation chain. `parent_ids` + content-addressed ID enforces append-only and ships Digital-Bereavement export for free.
2. **Mem0 history API shape** (`previous_value / new_value / event / timestamp`) as the query interface over the chain — stable, minimal, familiar.
3. **Graphiti bi-temporal** — reused ONLY for *external* world fact change (CEO changed, address changed). Kept **distinct** from the reconsolidation chain which handles *inner* reinterpretation. The two phenomena need separate mechanisms, not one.
4. **Emotional RAG dual-vector** — directly usable for Gap 3. Semantic + emotion vectors on each fragment; retrieval combines or sequences them. OSS reference implementation at `BAI-LAB/EmotionalRAG`.

**Must-be-built-from-scratch:** mentalistic sub-store per entity (Gap 2). No OSS primitive. Letta persona blocks are the closest and still miss the mark (free-text, not entity-scoped process).

**Reconciliation with memory-design.md Principle B.** The design doc dissolves the explicit Merkle version chain into the reconstruction trajectory. Borrowing PAM's Merkle-DAG is NOT a reversal — fragments are already append-only (B's corollary). The DAG shape is the on-disk serialization that satisfies PAM-export, not an engineered-chain-as-primary-retrieval-structure. Retrieval remains reconstruction; the DAG is the write-side substrate.

---

## 4. Brainstorm — three mechanism tracks

Research in steps 1–3 confirms Phase 2 has three mechanism tracks, not one. Each track needs its own substrate choice; each contributes to a different Phase-2 exit gate. The brainstorm below is intentionally unconstrained; the compression to options happens in §5.

### Track M1 — reconsolidation substrate (Gap 1, Principle B)

Possible shapes (per step 3 OSS findings):

- **M1-α** Pure reconstruction, no identity chain. Fragments append-only; "same event re-rendered" is observable only in the reconstruction trajectory. Principle B in its purest form.
- **M1-β** Event-identity tag. Each memorable event gets a stable `event_id`; new fragments that re-render it carry the same tag. Chain is queryable but implicit (fragments, not Merkle nodes).
- **M1-γ** Merkle-DAG on fragments. Content-addressed (`BLAKE3`), `parent_ids`, serialization satisfies PAM / Digital-Bereavement export natively. Compatible with Principle B — retrieval still reconstructs; DAG is write-side.
- **M1-δ** REALM-style retrieval-feedback-driven update (per `arXiv:2609.16053`). Retrieval cues drive which fragments get reinforced vs decayed.

### Track M2 — mentalistic sub-process (Gap 2, Principle C)

Possible shapes:

- **M2-α** Flat namespaced facts. `mother:*`, `self:*`, `ai:*` facts in the same substrate as other facts, filtered by namespace at retrieval. Simplest; lowest-fidelity (fails Principle C — it's a slot store wearing a namespace).
- **M2-β** Per-entity fragment subset + own tick. Each mentalistic process maintains its own fragment subset and ticks on observations relevant to that entity. Faithful to Principle C.
- **M2-γ** Subject-perspective-tagged fragments. Fragments carry `(subject, entity)` tag; retrieval against a mentalistic process = fragments where `(subject, entity)` matches. One substrate, retrieval-time filter + reconstruction.
- **M2-δ** PersMem-style dual-pathway. Per `arXiv:2609.34372`, two memory pathways — persona-resident vs event-resident. The user's self-model lives in the persona pathway, the AI's observation record lives in event. Two reconstructions, cross-queryable.

### Track M3 — affective-resonance retrieval (Gap 3 feed)

Possible shapes:

- **M3-α** Session-boundary 3-scalar WAI-SR tag (Bond / Task / Goal). Low fidelity, text-questionnaire-derivable, Chinese-regulatory-safe (session-scoped).
- **M3-β** Fragment-level 5-scalar Big-Five tag inferred from fragment text via BERT-head. Higher fidelity, text-only, no biometric. Neuroticism is the mood-proxy.
- **M3-γ** Emotional-RAG dual-vector. Each fragment carries `(semantic_vec, emotion_vec)`. Retrieval combines or sequences. Direct port from `arXiv:2410.23041` / `BAI-LAB/EmotionalRAG`.
- **M3-δ** 2-scalar PAD-adjacent (valence / arousal) extracted per fragment. Minimal; dimensional; aligns with classic affect literature.

### Cross-cutting candidates (apply across tracks)

- Minimal dyadic shared-referent tag (`co_experienced=true`) at observe time — decided 2026-10-05, folds into whichever substrate the three tracks settle on.
- Witness ablation under reconsolidation workloads — does the Witness pressure surface identity-drift when the user's framing shifts? Validates Principle E in a new context. Independent of M1/M2/M3 choice; runs during experimentation.
- R1 spreading-activation on multi-session edges — the Step H open question. If M2-β per-entity tick accumulates Hebbian history across sessions, R1 on those edges answers the question LoCoMo single-pass couldn't.

---

## 5. Compressed options

One option **per track** (3 options total), each with the chosen variant + runner-up + kill criteria. The three options are **not alternatives to each other** — they are three sequential mechanism blocks that together cover Phase 2's exit gates.

### Option M1 — reconsolidation substrate = **M1-γ Merkle-DAG on fragments** (runner-up M1-β)

| Field | Content |
|---|---|
| Mechanism | Fragments stored as append-only Merkle-DAG nodes with content-addressed id (BLAKE3) + `parent_ids` pointing to prior renderings of the same referent. Retrieval remains reconstruction; the DAG is write-side substrate + export shape. |
| Target gap | Gap 1 — reconsolidation; produces B1 fragment-preservation ≥ 95% by construction (append-only, provable); supports V1 state continuity under reconsolidation. |
| Prior art | Portable Agent Memory `arXiv:2605.11032`; REALM `arXiv:2609.16053` for retrieval-feedback-driven update; Mem0 history API shape for the query interface. |
| Why not M1-α | Pure-reconstruction fails to deliver measurable B1 fragment-preservation without an on-disk invariant; also fails Digital-Bereavement export obligation (no deterministic serialization). |
| Why not M1-δ (REALM) in v1 | Retrieval-feedback-driven update introduces a reinforcement loop that interacts with Principle D selection pressure; wait until M1-γ substrate is validated before layering REALM semantics. |
| Kill criteria | B1 fragment-preservation < 90% on scripted 10-session scenario; **OR** PAM export round-trip fails byte-identity on fragment subset; **OR** 3-instance LoCoMo monitor drops below 0.514. |

### Option M2 — mentalistic sub-process = **M2-γ subject-perspective-tagged fragments** (runner-up M2-β)

| Field | Content |
|---|---|
| Mechanism | Fragments carry `(subject_id, entity_id)` tag at observe-time. Retrieval against a mentalistic process = reconstruction over fragments matching `(subject, entity)`. One substrate (unified fragment graph), retrieval-time-partitioned. |
| Target gap | Gap 2 — mentalistic representation; produces B2 attribute reconstruction ≥ 10/12 by retrieval-time filter + reconstruction; separates user-self-model (`(user, user)`) from AI-model-of-user (`(ai, user)`) so the clinical-insight gap is computable. |
| Prior art | PersMem `arXiv:2609.34372` dual-pathway (adapted — we use one pathway with retrieval-time partition rather than two separate storage pathways to avoid substrate duplication). |
| Why not M2-α (flat namespace) | Fails Principle C by construction — namespace-filter-over-slot-store is a lookup wearing a process costume. The Phase 2 decision gate (V3 Jaccard-distance stays at 0.0 → "layer is not alive") explicitly kills this. |
| Why not M2-β (per-entity substrate + own tick) in v1 | Introduces N parallel entity-processes; test substrate grows with the user's social graph. M2-γ gets 80% of the fidelity with one substrate; promote to M2-β only if V3 shows M2-γ collapses to lookup-wearing-process. |
| Kill criteria | B2 grounding rate < 50% (per Phase 2 decision gate); **OR** V3 Jaccard-distance = 0.0 after 3 iterations (same input → same output, no process behavior); **OR** AI-model-of-user ≡ user-self-model (gap-detection precision = 0). |

### Option M3 — affective-resonance retrieval = **M3-γ Emotional-RAG dual-vector** (runner-up M3-β)

| Field | Content |
|---|---|
| Mechanism | Each fragment stored with `(semantic_vec, emotion_vec)`. Emotion vec derived at observe-time from fragment text via a lightweight affect classifier (BERT-head → PAD 2-scalar or Big-Five Neuroticism scalar). Retrieval combines or sequences the two vectors per Emotional-RAG strategies. |
| Target gap | Gap 3 feed — affect-cued reconstruction; Phase 2 stage of B3; emotional continuity across sessions by resonance, not time-series playback. |
| Prior art | Emotional RAG `arXiv:2410.23041` + `BAI-LAB/EmotionalRAG` reference implementation. |
| Why not M3-α (3-scalar WAI-SR) | WAI-SR is session-level; too coarse for fragment-level resonance re-rank. Also depends on questionnaire-at-session-end which may not occur in free dialogue. |
| Why not M3-δ (2-scalar PAD) in v1 | PAD is lower-fidelity than Emotional-RAG dual-vector; use PAD only if Emotional-RAG emotion-vec dimensionality is too costly on our substrate. |
| Kill criteria | B3-stage resonance recall < 40% (full B3 is Phase 3 scope but Phase 2 produces the feed); **OR** emotion-vec classifier accuracy < 60% on a scripted-affect validation sample; **OR** over-disclosure flag ≥ 10% on auditor-labeled sample (surveillance-feel floor from the roadmap). |

---

## 6. Application cross-check

For each option, which concrete behavior in the two north-star applications does it unlock?

### 学习陪伴 — resilient-learning companion

| Option | Concrete unlocked behavior | Reference-doc anchor |
|---|---|---|
| M1-γ (reconsolidation substrate) | **"I thought ∂ was X; now I understand it as Y"** chain-reference across weeks. On concept mastery (which is literally reconsolidation of the student's understanding), tutor cites "last Thursday you described this as X — what shifted?" Scaffolding-fading becomes measurable: as the student's version-chain for a concept converges, prompt-scaffolding fades. | `AI辅助韧性学习解决方案.pdf` scaffolding-fading + metacognitive scaffolding |
| M2-γ (mentalistic sub-process) | **"How do you think your teacher sees your math ability?"** — the student's model of the teacher is a distinct process from biographical teacher facts. Perspective-taking in learning context (student imagines rubric from inside teacher-process). | GenMentor multi-agent topology; `/profile` cross-model substrate |
| M3-γ (affect-resonance) | **"The last time you felt this stuck was two weeks ago on fractions — want to try the same unlock we found then?"** Affect-cued recall of a prior learning breakthrough matched by current frustration. | SDT competence need; CLT paradox-of-help; `AI心理长期陪伴研究.pdf:5` |

### 情感陪伴 — long-term emotional companion

| Option | Concrete unlocked behavior | Reference-doc anchor |
|---|---|---|
| M1-γ (reconsolidation substrate) | **Narrative-therapy chain-reference** — "you used to describe your father as 'never there'; last month that shifted to 'overwhelmed and trying in his way' — want to add this view?" This is the engine of narrative therapy per `memory-gaps.md`, impossible without the chain. B1 directly scores it. | `AI心理学应用研究.pdf` DTA transformational-tier moves |
| M2-γ (mentalistic sub-process) | **"From inside her side, what might she be trying to protect?"** — reasoning from the user's `(user, mother)` mentalistic process, not generic maternal-conflict LLM priors. Clinical-insight move "I notice something different from how you describe yourself" works from the gap between `(user, user)` and `(ai, user)`. | `AI心理长期陪伴研究.pdf` Rogers unconditional positive regard vs. sycophancy tension — perspective-taking is the counter-mechanism |
| M3-γ (affect-resonance) | **"This feels like the pattern we hit in May"** — empathic recall grounded in a prior affect-matched moment rather than surveillance-replay of an affect timeline. B3 scores this. | Woebot / Wysa WAI-SR bond scores; Emotional-RAG mood-congruence |

### Cross-check verdict

All three options unlock named behaviors in both applications. **No option is scope-only for one application.** Covenant-anchored rules (self-harm handoff, 《办法》 floor, Digital-Bereavement export) remain non-decaying across all three options per Principle F exception.

---

## 7. Experiments

Three experiment subdirs will be created under `docs/experiments/phase-2-mentalistic/`:

- `option-M1-merkle-fragments/` — reconsolidation substrate build + B1 measurement
- `option-M2-perspective-tagged/` — mentalistic sub-process + B2 measurement + V3 liveness check
- `option-M3-emotional-rag/` — affective-resonance retrieval + Phase-2-stage B3 feed

Sequence: **M1 → M2 → M3**. M1 is foundational (fragment substrate change); M2 is substrate-additive (new tag on fragments); M3 is retrieval-additive (new vector + re-rank). Each experiment includes a 3-instance LoCoMo monitor run to confirm the 0.514 floor holds.

Build steps and interim gates per option will be elaborated in each option's README as the experiment starts.

---

## 8. Comparison table

Populated after experiments land. Will record:

- Phase-2 exit gate value measured per option (B1 / B2 / V3 / V5 as applicable)
- LoCoMo 3-instance monitor value (regression check at 0.514)
- Dropped-option retention — any option that fails its kill criterion keeps its subdir as reproducible negative-space per the phase-planning memory.

---

## 7. Experiments

*To be filled in. One subdir per option under `docs/experiments/phase-2-mentalistic/option-X/`.*

---

## 8. Comparison table

*To be filled in after experiments land.*
