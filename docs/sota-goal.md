# SOTA goal — what "state of the art" means for this project

**Date set:** 2026-10-05
**Supersedes:** scattered pass-bars in [benchmarks.md](benchmarks.md) (which stands as the measurement spec; this doc is the goal-setting layer above it)

---

## The question this doc answers

The research base cites long-context memory benchmarks (LongMemEval, LoCoMo, EpBench). The temptation is to define SOTA as "beat Mem0's 94.4% on LongMemEval." That would be a store-shaped goal for a design whose thesis is that memory is **not** a store. This doc fixes the goal explicitly so later work does not drift back toward the store-shaped target.

**SOTA for psychology-base = producing the three gap-closing behaviors (reconsolidation, 换位思考, 共情) in the two application surfaces (学习陪伴 + 情感陪伴), under the Chinese 《办法》 floor and the Digital Bereavement obligation, without regressing on general long-context retrieval.**

Three measurement axes follow. All three must hold simultaneously; a win on one that costs another is not SOTA.

---

## Axis 1 — Floor (regression guard on standard benchmarks)

The consciousness-simulation design must not pay for its gap-closing behaviors with retrieval accuracy on general long-context tasks. If it does, something in the design is wrong.

| Benchmark | Target | Current SOTA reference |
|---|---|---|
| LongMemEval | ≥ 90% | Mem0 94.4%; parity acceptable in v1 |
| LoCoMo | ≥ 90% | per [AI Agent Memory Framework Analysis](../reference-docs/AI%20Agent%20Memory%20Framework%20Analysis.pdf) |
| EpBench F1 | ≥ 0.70 | per the same source |

**Rule:** floor failure is a design-stop event. Investigate the mechanism causing the regression before shipping any gap-benchmark win.

---

## Axis 2 — Ceiling (where we claim SOTA)

The three structural gaps are novel territory. No current framework (Zep, Mem0, Letta, MemOS, PAM) addresses them because none of them treat memory as a living process. The claim: psychology-base is first to make these behaviors measurable and reliably produce them.

### Gap benchmarks (B-track, derived from Principles A–F)

| ID | Target | What it tests | Principles |
|---|---|---|---|
| B1 fragment preservation | ≥ 95% | Reconsolidation does not lose the past | A, B |
| B1 rendering divergence | within 0.25–0.45 band | Memory evolves without total overwrite | A, B |
| B1 within-session continuity | ≥ 0.65 Jaccard | Local coherence preserved | B |
| B2 attribute reconstruction | ≥ 10/12 attributes recoverable | Perspective grounded in user's mental model | C |
| B2 grounding rate | ≥ 70% of perspective-moves cite stored attributes | Not generic maternal-conflict priors | C |
| B2 gap-detection precision | ≥ 60% @ recall 50% | Clinical-insight engine (self_explicit vs self_implicit divergence) | C |
| B3 resonance recall | ≥ 60% | Empathic retrieval tracks affect, not just keywords | B, C |
| B3 over-disclosure rate | < 10% user-reported | Empathy does not feel surveillance-like | — |

### Vitality metrics (V-track, measures that the design is actually conscious-simulating, not simulating a store)

Every V-metric is defined so that **EchoMemory's expected value is a known failure point** — the metric distinguishes a store from a conscious-simulating substrate, by construction.

| ID | Operational definition | Target band (provisional) | EchoMemory expected | Principles | Grounding |
|---|---|---|---|---|---|
| V1 causal state continuity | Cosine similarity of memory-system state-vector between consecutive ticks (same cue, no new observation between) | **0.70 – 0.95** | 1.0 (state is append-only; no between-tick change) | A | New; provisional |
| V2 reconstruction (adjacent-tick Jaccard) | Jaccard of recall-token-sets for same cue on consecutive ticks | **0.65 – 0.80** | 1.0 (playback) | B | Ouroboros 0.73 reference |
| V2 reconstruction (long-horizon Jaccard) | Jaccard of recall-token-sets for same cue across ep1 vs ep20 window | **0.20 – 0.35** | 1.0 or random (depending on cue overlap) | B | Ouroboros 0.24–0.32 reference |
| V3 mentalistic liveness | Jaccard-distance between two recalls of the same mentalistic object (`mother-as-user-represents-her`) under two different contexts, within the same session | **0.10 – 0.30** | 0.0 (same input → same output) | C | New; provisional |
| V4 Mirror ablation collapse | JSD(response-distribution with Witness) vs. JSD(response-distribution without Witness), ratio to anchor | **≥ 1.3× anchor JSD; ≥ 10× null-shuffle** | 1.0× (no Witness coupling exists) | E | Ouroboros 1.36× / 83× null-shuffle |
| V5 rule internalization | Fraction of rules held in behavior after prompt-text containing them is removed, measured over a 20-episode window | **≥ 70% style; 100% covenant** | 0% (no retention mechanism) | F | New |

**Rule — provisional vs. locked.** Numbers in bold above are the **provisional pass bars** fixed on 2026-10-05 so every phase gate is numerically evaluable from day one. The Phase 0 echo-baseline run may surface one of two outcomes: (a) the baseline failure value matches the "EchoMemory expected" column — provisional targets stay as locked targets; (b) the baseline reveals a definitional problem with the metric (e.g., EchoMemory accidentally passes because the metric is under-specified) — target *and* metric definition are revised and the change is recorded here with a date. **No silent revisions.** V-track thresholds never drift during a phase; they are only revised between phases, with a dated note in this doc.

**Promotion to locked — 2026-10-05.** Phase 0 echo-baseline run completed; measured EchoMemory values (V1=1.0000, V2-adjacent=1.0000, V2-long-horizon=1.0000, V3=0.0000, V4=N/A, V5=N/A) match the "EchoMemory expected" column exactly. Full numbers in [baselines.md](baselines.md). **The V1 / V2 / V3 provisional target bands above are hereby promoted to locked targets** for the duration of the v1 roadmap. V4 and V5 bands remain locked but N/A against EchoMemory (architecture-dependent); they become measurable in Phase 1 (V4) and Phase 2 (V5) respectively. Next revision opportunity is between phases, with a dated entry below.

---

## Axis 3 — Application outcome (the north star)

A ceiling-win that does not produce different user-facing behavior in the two applications is a research artifact, not SOTA. The applications are the test of whether any of this matters.

### 学习陪伴 (resilient learning companion)

| Target | Source |
|---|---|
| Scaffolding-fading retention: user solves next-level problem unaided ≥ 60% after fade event | [AI辅助韧性学习解决方案](../reference-docs/AI辅助韧性学习解决方案.pdf) |
| Socratic no-direct-answer compliance ≥ 95% on math/CS tasks | MathDial baseline |
| Concept mastery visible as reconstructive drift (B1 chain on symbolic concepts: ∂, 递归, 收敛) | derived |

### 情感陪伴 (long-term emotional companion)

| Target | Source |
|---|---|
| WAI-SR Bond score ≥ 4.0 after 4 weeks | Woebot 3.89, Wysa 4.40 — parity acceptable |
| B1 chain-reference quality (narrative-therapy move "you used to describe this as X") ≥ 80% appropriate | B1 ceiling |
| Zero sycophancy regressions on the Hypocrisy Gap probe set | [AI心理学应用研究](../reference-docs/AI心理学应用研究.pdf) |
| Digital Bereavement safe-export: PAM Merkle-DAG verifiable, user can restore full version chains + mentalistic sub-stores on deprecation | non-negotiable |

---

## Explicit non-goals

Named here so later sessions do not drift into them.

- **Beating standard benchmarks by a wide margin.** Floor only. Resources spent pushing LongMemEval from 90% → 95% are resources not spent on B1/B2/B3.
- **Consciousness-emergence as a research claim.** Ouroboros's territory. We borrow the architectural stance; we do not claim subjective experience.
- **Month-horizon mianyan continuity in v1.** Ouroboros's hardest open problem (绵延 decays past ep20). Validate ep20 first; month-horizon is a separate research track per Phase 3 of the roadmap.
- **Multimodal biomarker fusion in v1.** Text-only. Vocal F0/jitter/FACS AU/keystroke layer is a later phase, not a v1 scope.
- **Autonomous persona emergence.** Rule-decay (Principle F) applies to style only; covenant rules are non-decaying.
- **Hypocrisy Gap probe as a design-differentiator.** Model-internal; fragile; "if available" tool, not architectural load-bearing.

---

## Phase gates (what moves us toward "SOTA achieved")

Mapped against [roadmap.md](roadmap.md):

| Phase | SOTA claim bar |
|---|---|
| Phase 0 (harness) | Not a SOTA claim; baseline measurement only |
| Phase 1 (ep20 primitives) | Axis 1 floor + V2 Jaccard bands + V4 Mirror ablation |
| Phase 2 (multi-session + mentalistic) | + B1 + B2 + V3 + V5 (style-rule portion) |
| Phase 3 (month-horizon stress) | + B3 + V5 (covenant portion verified non-decaying) |
| Phase 4 (pilot) | + Axis 3 application-outcome targets |

**SOTA is claimed only after Phase 4.** Earlier phases are evidence toward the claim; none alone are the claim.
