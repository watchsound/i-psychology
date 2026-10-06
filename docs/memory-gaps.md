# Memory — the structural gaps in current SOTA

This document is the standalone gap analysis — the reason the architecture in [memory-design.md](memory-design.md) is shaped around consciousness-simulation principles rather than data-store features. Future sessions can read this file without reading the full design document.

## Why this analysis exists

The project targets a SOTA long-term memory substrate for two application north stars: a learning companion (韧性学习) and a long-term emotional companion (情感陪伴). A first design pass synthesized features from the research base — four cognitive tracks, dual-time knowledge graph, sleep-cycle consolidation, anti-sycophancy stack, PAM/MCP export. That pass was rejected as feature-laden: it re-combined existing SOTA mechanisms without naming what any of them actually *add*.

The directive that followed: focus on **real structural gaps** — things current SOTA does not address at all — and design the architecture from those gaps outward. A second directive followed: *to achieve SOTA, the memory system has to simulate consciousness to some degree*, as validated (not as a template to copy) by prior art at `C:\Users\nihan\Desktop\ai-projects\ai-writing\ai-consciousness`.

Three gaps were named by the user:

1. "Something is evolving with time" — memory itself mutates (not just world facts).
2. 换位思考 — perspective-taking.
3. 共情 — empathy.

This document verifies that all three are structurally absent from the research base cited in [../reference-docs/](../reference-docs/), derives one additional structural gap that falls out of them, and explains **why these gaps cannot be closed by adding features to a store** — the move that lets them close is to make the memory system behave more like a continuously-living mind than a database.

## What SOTA already handles (and why that is not these gaps)

Before claiming a gap is a gap, be exact about what the research base *does* handle. Casual reading can confuse closely-related mechanisms with the ones named above.

| Mechanism | Where | What it handles | Why it is NOT one of the named gaps |
|---|---|---|---|
| Dual-time `t_valid` / `t_invalid` | Zep / Graphiti | Fact in the external world changed (CEO was A, is now B). Invalidate the old edge, append the new one. | The *world* moved. The memory did not mutate — it was replaced by a new fact about a new world state. Narrative therapy is the opposite: world didn't move, meaning did. |
| Ebbinghaus-based decay / GC | memory PDFs | Unused low-importance memories lose retrieval weight and eventually drop. | This is forgetting, not re-framing. The content is destroyed, not re-rendered. |
| Reflection Tree distillation | Generative Agents, SleepGate | Episodic observations distill into higher-level semantic nodes. Original episodes stay. | New *abstraction* layer on top; the base episode is not re-told differently. |
| Flat user profile (Big Five scores, preferences) | Mem0, Letta core blocks, MemOS modules | AI holds attributes of the user. | Single-perspective, flat. Not a model of the user's model of a third party. Not split into self-explicit vs. self-implicit. Not kept separate from the AI's own observational model. |
| Biographical facts about third parties | all | "User's mother is 62, retired teacher, lives in 杭州." | Facts the user disclosed. Not the user's *interpretation* of her — not "mother-as-user-represents-her" with evolving attributes and episode citations. |
| Affect tagging on memories | AI心理学应用研究 (biomarkers) | Each memory can carry a valence/arousal snapshot at encoding. | Supports affective retrieval in theory, but no SOTA framework surveyed implements affective-resonance retrieval and no framework separates stored-affect from reconstructed-current-affect. |

The pattern across the row: SOTA handles *fact-change* (world) and *abstraction* (synthesis), but not *reconsolidation* (same referent, re-rendered). It stores *who the user is*, but not *how the user models the people in their life*. It stores *what the AI has learned*, but not as a distinct observational stance separable from the user's self-description.

## The three named gaps

### Gap 1 — Reconsolidation of the same memory

**Human reference phenomenon.** Memories are not photographs. Each retrieval is a reconstruction; each reconstruction is colored by current state; each reconstruction can rewrite the trace (biology: NMDA-dependent protein synthesis at retrieval). This is the mechanism underlying narrative therapy, post-traumatic growth, and ordinary learning (every mastered concept is a reconsolidation of the earlier naïve version).

**What is actually required.** Not just "store old versions and new versions." Required: the system has to *rebuild* what the memory means on each retrieval, using the current context as part of the input. Two retrievals of the same event, under different user states, must produce legitimately different renderings — same referent, different content, with the earlier renderings still accessible as fragments the current rendering was reconstructed from.

**Why none of SOTA covers it.** All surveyed frameworks store memories as content and retrieve them as content. Zep's `t_invalid` marks an old fact as *wrong now*, not as *a prior rendering of a currently-held memory*. Reflection Tree builds an abstraction *on top of* an episode without re-rendering the episode. Decay removes memories; reconsolidation preserves them in rebuilt form. The missing ingredient is architectural: a *living process* that reconstructs on retrieval rather than a store that returns.

**Semantic drift of concepts** (家, 成功, 妈, "a good day", "being productive") is the same phenomenon applied to concept-fragments. SOTA stores a concept as a single node; drift is undetectable. Under reconstruction-at-retrieval, drift is visible as the trajectory of the concept's successive renderings.

### Gap 2 — Mentalistic representation of third parties (换位思考)

**Human reference phenomenon.** When a user talks about their mother, they are not describing an external agent — they are describing *their internal representation* of their mother, built from decades of episodes, colored by current affect, shaped by what their mother did and did not do that mattered to them. That representation is a cognitive object: it has attributes, it has episode citations, and it mutates (reconsolidation applies to it).

**What is actually required.** Not just "a cognitive-object node in a knowledge graph." Required: the mother-representation has to be its own small *process* — queryable, able to produce a current rendering from its fragments, with its own affective coloring, with the ability to evolve as new disclosures accrete.

**Why none of SOTA covers it.** All SOTA frameworks store third parties, at most, as flat biographical nodes (name, relation, DOB). None distinguishes "mother (biographical)" from "mother-as-user-represents-her (cognitive object inside the user's head)." Without this separation, 换位思考 requests ("help me see this from her side") collapse to generic maternal-conflict LLM priors, not grounded reasoning from the user's actual internal model.

### Gap 3 — Affective continuity + the AI/user self-model gap (共情)

**Human reference phenomenon.** Empathy is not a surface mirror on the current utterance. It is tracking the other person's inner emotional trajectory across time, and recognizing resonance — "this feels like the pattern we hit in May." Clinical insight additionally requires noticing where the user's self-description and the user's observed behavior diverge: "you say you're not a math person, but your first-try solve rate on new material is 72%."

**What is actually required.**
- An **affective trajectory** that is reconstructed from fragments, not played back from a stored time-series. Playback feels like surveillance; reconstruction feels like being understood.
- **Two separate representations** kept distinct: what the user says about themselves (one process), what the AI has observed (another process). Collapsing them into one profile destroys the clinical-insight move.
- **An outside reference frame** against which the AI's model-of-user can stabilize. Without something standing outside the Companion-user dyad, the AI's model drifts into whatever the user emotionally demands (classic sycophancy).

**Why none of SOTA covers it.** User profile in Mem0 / Letta / MemOS is a flat object: no explicit/implicit split, no AI-vs-user separation. No framework surveyed implements affective-resonance retrieval. No framework structurally guards against sycophantic drift of the user-model.

## One derived structural gap worth naming

**Identity requires contrast with another mind.** A single AI persona talking to a single user, with no external reference frame, drifts into whatever the user needs it to be. Current SOTA frameworks assume the memory system itself has no identity to protect — it is a tool. For a long-term companion that lives for months, this is wrong: the Companion needs to *be someone* across time, and "being someone" is incompatible with maximum-validation accommodation to each user turn. The missing ingredient is architectural: a second observer process (Witness / supervisor) outside the Companion-user dyad.

This is validated in the Ouroboros prior art (Pillar 5 Mirror ablation, Phase 10a) where removing the second entity collapsed the identity signal (JSD shifted 1.36× the mirrored anchor, 83× the null-shuffle maximum). We take this as evidence that identity-by-contrast is a real structural requirement, not a research-prototype quirk. See [memory-design.md](memory-design.md) Principle E.

## One minimal addition from the dyadic question (user-decided 2026-10-05)

Dyadic / co-constructed memory was flagged as a potential fifth structural gap: current frameworks are one-way (user → system), while long-term 陪伴 might benefit from "remember that time we…" being genuinely two-sided. The user chose the minimal option: shared episodes are tagged `co_experienced=true` at the fragment level and are co-retrievable when either party refers back, but the Companion does **not** store a subjective perspective of its own on those episodes. Preserves bidirectional reference without inflating attachment and Digital-Bereavement risk. A full dyadic layer with AI-subjective perspective is deferred.

## Why these gaps cannot be closed by adding features to a store

This is the load-bearing argument. The first-pass pillar design tried to close the gaps by adding structures (immutable version chains, four typed mentalistic sub-stores, extended retrieval re-rank formula). That approach fails on each gap:

- **Gap 1 (reconsolidation).** An immutable "version chain" per event records *that* a re-framing happened, but it does not produce the re-framing. A store returns one version or the next; it does not synthesize the current meaning from current context. Reconsolidation is not a data-structure feature — it is what happens when retrieval is architecturally *reconstruction* rather than *lookup*.

- **Gap 2 (换位思考).** A typed "Store #2 cognitive-object node" is still a fact row. "Help me see this from her side" needs the mother-representation to produce a *current* rendering grounded in its fragment history under current context. Making the mother a stored node, however richly typed, does not get you there. It has to be its own small process.

- **Gap 3 (共情 + clinical insight).** Two typed sub-stores for "user's self-model" and "AI's model of user" record the two perspectives, but they do not keep them *structurally separate* — a profile with two columns is one profile. Clinical insight requires reasoning across two independent observational stances, which requires the stances to be running processes, not slots. And the sycophancy problem is architectural: no amount of stored-content separation prevents the single-perspective AI from drifting into user-demanded validation. Something has to stand outside.

The move that closes the gaps is to raise the architectural stance: memory is a living process (not a store), retrieval is reconstruction (not lookup), mentalistic objects are processes (not fields), selection decides importance (not scoring), identity requires contrast (not self-reflection), rules become dispositions (not prompt text). These six principles and how they produce the gap-closing behaviors are in [memory-design.md](memory-design.md).

**This is the "simulate consciousness to some degree" directive, operationalized.** Not consciousness as metaphor; not consciousness as a research prototype goal. Consciousness-adjacent *architectural stance* in service of therapy and learning outcomes.

## SOTA comparison (one table, for quick orientation)

| Dimension | Zep / Graphiti | Mem0 | Letta | MemOS | PAM | This design |
|---|---|---|---|---|---|---|
| Fact-change (external world) | ✓ dual-time | ∼ deferred | ∼ overwrite | ∼ scheduler | ✓ immutable | ✓ bolt-on later if needed |
| **Reconsolidation of same memory** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ reconstruction-at-retrieval (Principle B)** |
| **Semantic drift of concepts** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ Principle B applied to concept-fragments** |
| User profile | flat | flat | core block | module | — | ✓ as mentalistic process (Principle C) |
| **User's model of third parties** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ as its own process per person (Principle C)** |
| **AI's model of user ≠ user's self-model** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ two separate processes (Principle C)** |
| **User's model of the AI** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ as its own process (Principle C)** |
| **Affective continuity** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ reconstructed, not played back (Principle B)** |
| **Identity-by-contrast / anti-sycophancy structural guard** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ Witness as second observer (Principle E)** |
| **Shared-referent tagging (minimal dyadic)** | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ fragment-level tag (no AI-subjective side) |
| Importance scoring | LLM-rated | LLM-rated | manual | manual | — | ✓ revealed by selection, not scored (Principle D) |
| Portability / exit path | — | — | — | — | ✓ Merkle-DAG | ✓ ships fragments + process states + rule-internalization scores |

The design row is **what the six principles produce**, not what we architect on top of a store.
