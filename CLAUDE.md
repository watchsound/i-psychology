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
