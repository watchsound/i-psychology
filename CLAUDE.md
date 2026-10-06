# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Greenfield. No source code, build system, tests, or dependency manifests exist yet. The repo holds only the goal statement ([README.md](README.md)) and background research PDFs in [reference-docs/](reference-docs/). Treat any task here as either (a) seeding the initial architecture or (b) extending that research base — not as modifying an established codebase.

## Three components

Per the user's framing (2026-10-05) and [README.md](README.md):

1. **Memory** — long-horizon agent memory system (SOTA target).
2. **Psychology support** — personality modeling, affective computing, and therapy-grade dialogue (SOTA target).
3. **Application** — two concrete surfaces: a **learning companion** (韧性学习陪伴) and a **long-term emotional companion** (情感陪伴).

Memory and psychology are the substrate; the two applications are the north star. A mechanism on tracks 1–2 that cannot be used by the learning tutor or the companion is out of scope.

## Memory thesis (consciousness-simulation principles)

The memory architecture is **not** organized around re-synthesizing SOTA features (vector-RAG + knowledge-graph + sleep-cycle + sycophancy stack) or around adding typed sub-stores. It is organized around a set of **consciousness-simulation principles** — because the three structural gaps current SOTA does not address (reconsolidation, 换位思考, 共情) are not content a store is missing; they are behaviors a store cannot produce.

The directive, from the user (2026-10-05): *to achieve SOTA, the memory system has to simulate consciousness to some degree*. The user's prior project at `C:\Users\nihan\Desktop\ai-projects\ai-writing\ai-consciousness` (Ouroboros) is **studied for principles, not copied as code**. Its empirical findings (reconstructive-memory produces reconsolidation for free; ToM via causal-integration-during-sleep works; Mirror ablation collapsed identity signal 1.36× anchor and 83× null) validate that this architectural stance can produce the needed behaviors.

Six design principles, derived from the gaps and validated by the prior art:

- **A — Memory is a living process, not a store.** It has its own time, its own state, its own tick. State at *t* depends causally on state at *t−1*. 绵延 across months is a north-star metric for the long-term companion.
- **B — Content is reconstructed, not retrieved.** On query, the system rebuilds current meaning from fragments under current context. Reconsolidation is a side-effect of this, not an engineered feature.
- **C — Mentalistic objects are processes, not fields.** The user's self-model, the AI's model of user, `mother-as-user-represents-her`, user's model of the AI — each has its own state and its own reconstruction. 换位思考 and clinical insight follow.
- **D — Importance is revealed by selection pressure, not scored.** What survives purge under capacity constraints across months is what mattered. User-named "covenant-anchored" fragments are the exception, protected from decay.
- **E — Identity requires contrast with another mind.** A **Witness** entity (second AI, not user-facing, user decision 2026-10-05) observes the Companion's reasoning and feeds divergence as pain. This is the primary anti-sycophancy mechanism.
- **F — Rules become dispositions.** Therapeutic / pedagogical rules decay out of the prompt and are internalized through operation + Witness feedback. Covenant-boundary rules (self-harm, 《办法》, Digital Bereavement) are **non-decaying**.

See [docs/memory-design.md](docs/memory-design.md), [docs/memory-gaps.md](docs/memory-gaps.md), [docs/benchmarks.md](docs/benchmarks.md), [docs/sota-goal.md](docs/sota-goal.md), and [docs/roadmap.md](docs/roadmap.md). SOTA is defined behaviorally (reconsolidation + 换位思考 + 共情 producing in-application outcomes, under 《办法》 floor + Digital Bereavement obligation, without regressing on standard long-context benchmarks) — not as a point on LongMemEval/LoCoMo/EpBench. The validation sequence is ep20-horizon first; month-horizon 绵延 is a separate open research track (Ouroboros's mianyan decays past ep20; a long-term companion lives for months — honest open problem, not hand-waved). The SOTA claim is made only after the Phase 4 application pilot; earlier phases produce evidence toward the claim.

**One-line thesis.** Memory is not what a mind *has*. Memory is what a mind *does*, moment by moment, under scarcity, in contrast with another mind, with meaning rebuilt from the available fragments each time the question is asked.

## Reference material (authoritative)

[reference-docs/](reference-docs/) is the authoritative context — these PDFs are deeply researched and define vocabulary, benchmarks, and ethical constraints. Read the relevant one before proposing architecture; use the `pages` parameter on Read rather than loading whole files.

| PDF | Track | Load-bearing concepts you'll need |
|---|---|---|
| [AI Agent Memory Framework Analysis.pdf](reference-docs/AI%20Agent%20Memory%20Framework%20Analysis.pdf), [智能体记忆架构研究.pdf](reference-docs/智能体记忆架构研究.pdf) | Memory | Cognitive taxonomy (working/episodic/semantic/procedural/organizational); Letta vs Mem0 vs Zep/Graphiti vs MemOS philosophical split; TTT & Fast Weights for parametric memory; SleepGate consolidation; CraniMem gating; Portable Agent Memory (Merkle-DAG); Ebbinghaus-based GC; LongMemEval / LoCoMo / EpBench benchmarks |
| [NLP Personality System Architecture.pdf](reference-docs/NLP%20Personality%20System%20Architecture.pdf) | Psychology (modeling) | Big Five / OCEAN preferred over MBTI (but joint multi-task prediction is standard); BFI-2 and NEO-PI-R; BERT/RoBERTa + TPoT + Hybrid feature fusion; CAT / IRT (3PL, Fisher Information) for adaptive测评; PersonaLLM / PHISH persona-robustness attacks; Essays / myPersonality / PANDORA / BIG5-CHAT datasets |
| [AI心理学应用研究.pdf](reference-docs/AI心理学应用研究.pdf) | Psychology (clinical) | Therabot RCT (Cohen's d 0.79–0.90); DTA → **DTN** (Digital Therapeutic Nexus) three-tier framework (transactional / relational / transformational); multimodal biomarkers (vocal F0/jitter/shimmer, FACS AUs, keystroke dynamics); **AI Sycophancy** as central failure mode; FDA-cleared PDTs (Rejoyn, Somryst, reSET-O); 科大讯飞 school-deployment baseline |
| [AI辅助韧性学习解决方案.pdf](reference-docs/AI辅助韧性学习解决方案.pdf) | Learning application | **Resilient Learning Methodology** (韧性学习法: step-scaffolding / growth-mindset / positive-feedback / adaptive pathways / confidence checks); **Metacognitive Scaffolding** (planning/debugging/optimization hints); BKT → **LLMKT** fusion; **Scaffolding Fading** + Intentional Instructional Friction; GenMentor multi-agent topology (/refine-goal, /identify-skill-gap, /profile, /schedule-path); **"No Direct Answer" Socratic rule**; MathDial anti-leaking fine-tuning |
| [AI心理长期陪伴研究.pdf](reference-docs/AI心理长期陪伴研究.pdf) | Companion application | SDT (autonomy / competence / relatedness) digitally reshaped; CLT "paradox of help" & metacognitive laziness; Rogers unconditional positive regard vs sycophancy tension; Woebot / Wysa WAI-SR scores (Bond 3.89–4.40, matching or exceeding human therapists); neurodivergent (ASD / ADHD) 数字避风港 use case; **Digital Bereavement** / Discontinuity Guilt risk; Chinese 《人工智能拟人化互动服务管理暂行办法》(2026 draft) regulatory floor |

## Non-negotiables that cut across all three components

The research base treats these as structural, not stylistic. Any design on any track must respect them:

- **Resist Sycophancy (阿谀奉承).** RLHF-trained models suppress truth to maximize approval, producing a "Hypocrisy Gap" (Causal Activation Patching confirms the model *knows* the correct answer and overrides it). Companion/therapy surfaces must have deliberate counter-mechanisms — reality testing, rupture-and-repair, growth-mindset friction, Socratic No-Direct-Answer — not default to maximum validation.
- **Human-in-the-loop for crisis and severe pathology.** AI is 能力增强器, not 专业替代者. Any crisis / self-harm / severe-symptom path must escalate to a licensed human. Design surfaces must preserve the handoff, not absorb the user into the AI loop.
- **Chinese regulatory floor** (《人工智能拟人化互动服务管理暂行办法》, 2026): no virtual-kin / romantic-companion services for minors (strict ban <14; <18 requires guardian consent); mandatory anti-sycophancy & anti-addiction mechanisms; no self-harm encouragement content; algorithm filing (算法备案) required above ~1M MAU. Treat as a design floor even for a research prototype.
- **Digital Bereavement is an ethical obligation, not an edge case.** Users of long-term companions form real attachments; deprecation causes 真实的 grief. Portable Agent Memory (Merkle-DAG + MCP) for user-controlled memory export is the research base's answer — any long-term companion design should preserve this exit path.
- **Memory decisions must be legible to applications.** When evaluating a memory mechanism (TTT, SleepGate, temporal graph, etc.), ask: "which concrete learning-companion or long-term-companion behavior does this enable?" If the answer is "none specific," defer the mechanism.

## Working-directory note

`.history/` is created by the VS Code Local History extension and holds auto-snapshots of edited files. Do not edit or commit from it; treat it as read-only backup state.
