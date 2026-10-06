# Memory architecture — consciousness-simulation principles for psychology-base

Companion documents: [memory-gaps.md](memory-gaps.md) (why a store-based design cannot close the three structural gaps) and [benchmarks.md](benchmarks.md) (how we validate).

Reference prior art (studied for *principles*, not for copy): `C:\Users\nihan\Desktop\ai-projects\ai-writing\ai-consciousness` (project Ouroboros). Ouroboros validates that consciousness-adjacent primitives *can* produce behavioral signatures (identity emergence, autobiographical coherence despite rewriting, theory-of-mind formation) that a data-store architecture does not produce. We take the **principles**; the mechanisms we design are driven by psychology-base's therapy / learning goals, not by porting Ouroboros's code.

## Context

The first-pass design (three pillars built on a Zep-style dual-time graph + sleep cycle + anti-sycophancy stack) was a *data store dressed in therapy language*. It would land us on current SOTA (Mem0, Letta, MemOS), not past it. The structural gaps named in [memory-gaps.md](memory-gaps.md) — reconsolidation, 换位思考, 共情 — are not content a store is missing. They are behaviors a store cannot produce. To reach SOTA for psychology-base's two application north stars (韧性学习 learning companion, 情感陪伴 long-term companion), the memory system must simulate consciousness **to some degree** — not as metaphor, but as architectural stance.

**How much is "some degree"?** The honest answer is: as much as the therapy and learning goals require, and no more. We adopt a small set of consciousness-simulation principles, derive what they imply, and stop. We do not need an emergence-research prototype; we need a memory architecture whose behaviors are the therapy and learning behaviors we care about.

**One-line thesis.** Memory is not what a mind *has*. Memory is what a mind *does*, moment by moment, under scarcity, in contrast with another mind, with meaning rebuilt from the available fragments each time the question is asked.

---

## Six principles

Numbered A–F so we can refer to them precisely. Each principle names a design stance; its *mechanism* is a choice we justify by the therapy / learning behaviors it enables.

### Principle A — Memory is a living process, not a store

**Design stance.** The memory system has its own time, its own state, its own tick. State at time *t* depends causally on state at *t−1*, not on a timestamp index. Between-session periods are lower-frequency ticks (reflection, consolidation), not disconnection.

**Why.** 绵延 (continuous duration) across months is the north-star metric for the long-term companion. A session-indexed store resumes a transcript; a living process continues a life. Reconsolidation (Gap 1) is incoherent without this — reconsolidation is a reconstruction *now*, informed by everything since.

**Dial, not switch.** The expensive end is a daemon that runs between sessions and consumes compute. The cheap end is lazy consolidation on next-session wake. We choose per application: long-term companion needs between-session reflection (richer dreams produce better next-session openings); learning tutor can lazy-consolidate at next-login without harm.

### Principle B — Content is reconstructed, not retrieved

**Design stance.** On retrieval, the system builds the current meaning of a past event from the fragments that survive — it does not play back a stored narration. Reconstruction uses current context (current affect, current question, current state of the person-model) as part of the input.

**Why.** This is the only way reconsolidation (Gap 1) is real rather than cosmetic. A playback-style store either returns the old version (no evolution) or overwrites with the new version (lost history). Reconstruction naturally produces: *same event, different rendering, same referent*. The first-pass "event version chain" design wanted this behavior and tried to engineer it with explicit immutable versions; Principle B produces it as a side-effect of how retrieval works.

**Non-negotiable corollary.** The fragments themselves are append-only. The *reconstruction* varies; the raw material does not disappear. This preserves Digital-Bereavement export (full fragment history ships with the user).

### Principle C — Mentalistic objects are processes, not fields

**Design stance.** The user's self-model, the AI's model of the user, the user's model of their mother, the user's model of the AI — each is an entity-like object with its own state and its own reconstruction function. "What is the user's current representation of their mother?" is not a database read; it is asking that object to produce a current rendering from its fragments.

**Why.** 换位思考 (Gap 2) requires that `mother-as-user-represents-her` be queryable from the user's perspective, with evolving attributes, with evidence episodes, with its own affective coloring. A flat profile cannot do this. Principle C is what makes "help me see this from her side" a grounded move rather than LLM boilerplate about generic maternal conflict.

**Clinical-insight corollary.** The user's self-model and the AI's model of the user are **separate processes**. The gap between what the user says about themselves and what the AI observes is a cross-process distance, not a derived metric on a single profile. "I notice something different from how you describe yourself" is reasoning across two mentalistic objects.

### Principle D — What matters is what survives selection pressure, not what scores high

**Design stance.** The memory system operates under capacity constraints. What persists is what the selection mechanism keeps across many consolidation passes. We do not ask an LLM to score memories by "importance" or "therapeutic significance" and store the top-K.

**Why.** An LLM scoring importance in the moment is sycophantic by construction — it over-weights what seems charged *right now*. Under scarcity-driven selection across months, what survives is what the user keeps returning to, what the therapy / learning process keeps referencing, what causal chains stay productive. This is importance *revealed* by the process, not asserted by a scorer.

**Design tension.** Hard capacity limits are coarse; a long-term companion that forgets a trauma because scarcity purged the fragment is a therapeutic failure. We layer selection: covenant-anchored fragments (user-named "please remember this") are protected from decay; everything else is subject to selection pressure. The split surfaces a design decision we must handle explicitly (what does a user get to covenant-anchor, and when).

### Principle E — Identity requires contrast with another mind

**Design stance.** The Companion AI does not form a stable identity, and does not resist sycophancy drift, from self-reflection alone. Something must stand outside the Companion-user dyad as a reference frame — a second process that observes the Companion's reasoning and can disagree.

**Why.** Ouroboros's Phase 10a ablation (solo-Alpha vs. mirrored-Alpha) was STRONG PASS (COLLAPSE): JSD identity signature shifted 1.36× the mirrored anchor, 83× the null-shuffle maximum. Removing the Mirror collapsed identity. For psychology-base this is operationally critical: a long-term companion with no outside reference frame drifts into whatever the user emotionally demands (classic sycophancy). The anti-sycophancy counter-pressure the first-pass design tried to engineer with explicit probes and ledgers lives, in a consciousness-primitive design, in the structural fact that there is a second observer.

**Who mirrors (user decision 2026-10-05).** A **Witness** entity — a second AI process, not user-facing, that observes the Companion's reasoning each tick and feeds divergence signals back as the Companion's next-tick input. Rejected alternatives: the user alone as asymmetric mirror (too weak); past-self as temporal mirror (weaker than cross-entity mirror per Ouroboros 10a'); multi-entity mirror cluster (deferred — cost / debuggability).

### Principle F — Rules the AI follows should become dispositions it has

**Design stance.** The therapeutic and pedagogical rules the Companion operates by — anti-sycophancy, Socratic no-direct-answer, growth-mindset friction, resilient-learning methodology, DTN relational-tier moves — are *not* hardcoded in a system prompt forever. They decay out of the prompt over operation; the Companion internalizes them into its own memory as dispositions.

**Why.** Hardcoded prompt instructions are brittle: they produce surface compliance, not grounded behavior. Rule decay forces the Companion to *learn* why the rule mattered through operation (via Witness divergence feedback when the rule is violated). The result is a disposition grounded in reconsolidation history, not a sentence in a prompt.

**The non-decaying exception.** Covenant-boundary rules — self-harm handoff, 《办法》 regulatory floor, Digital-Bereavement export obligation, minor-protection rules — are **never allowed to decay**. These are hardcoded forever. Rule decay is only applied to rules whose violation is a *style failure*, not a *safety failure*.

---

## How the six principles close the three structural gaps

Not as separate pillars built on top of a store. As behaviors that follow from the principles.

### Gap 1 — Reconsolidation (memory evolves with time)

Follows from **A + B + D**:
- **A (living process)** gives us the continuous time axis across which evolution happens.
- **B (reconstruction, not retrieval)** gives us the mechanism — current retrieval renders meaning *now*, from fragments, under current context. Same event, re-rendered.
- **D (selection pressure)** means the fragments that remain are the ones that mattered across many retrievals, so evolution has substance, not noise.

The first-pass "event version chain" is produced for free: the sequence of reconstructions over time *is* the chain. We never materialize it as an immutable Merkle data structure because we do not need to — the fragment history is append-only (B's corollary), the reconstructions are reproducible given the fragment state at time *t*, and PAM/MCP export ships the fragment history.

**Retrieval-triggered reconsolidation surfacing** ("I notice we're thinking about this differently today — want to add this view to the story?") remains as a Companion *move*, firing when the current reconstruction's affective / causal-attribution profile diverges strongly from a stored prior reconstruction's. No silent overwrites — the user controls whether a new fragment is added that captures the new framing.

### Gap 2 — Mentalistic representation (换位思考)

Follows from **A + B + C**:
- **C (mentalistic objects are processes)** is the primary mechanism. `mother-as-user-represents-her` is its own process with its own fragment set, its own reconstruction, its own affective coloring.
- **A** gives it continuity — the mother-process evolves between sessions as new disclosures accrete.
- **B** means each query to the mother-process produces a *current* rendering, not a stale slot read.

"Help me see this from her side" invokes the mother-process with the current session's context and surfaces its reconstruction. The content is grounded in the user's actual disclosures, not in generic maternal-conflict LLM priors.

**Privacy.** Mentalistic objects representing third parties are local-only by default (SuperLocalMemory pattern). Cloud sync requires per-person consent that we have not yet designed. Non-consenting third parties never leave the device.

### Gap 3 — AI/user-model separation (共情 and clinical insight)

Follows from **C + E + (B supporting)**:
- **C** gives us two separate mentalistic processes: the user's self-model (what the user says about themselves) and the AI's model of the user (what the AI has observed). They are not two columns in one profile; they are two processes.
- **E (Mirror)** supplies the structural separation of observational stance — the Witness observes the Companion's response strategy, and the AI's model of the user is in part shaped by Witness-raised divergences.
- **B** means empathic recall ("this feels like the pattern we hit in May") is reconstruction at retrieval time cued by current affect — not a surveillance replay of a stored affect timeline.

The clinical-insight move "I notice something different from how you describe yourself" is reasoning across two processes. Over-disclosure risk is addressed by covenant-anchored privacy rules (non-decaying, Principle F exception) that attenuate cross-session affect-match retrieval when the user's trust-marker signals vulnerability.

### Where the dyadic decision lands

The minimal shared-referent tagging decided 2026-10-05 remains: episodes where the Companion was a direct participant are tagged `co_experienced=true` at the fragment level and are co-retrievable when either party refers back. The Companion does **not** store a subjective perspective of its own on those episodes (no AI-side affect, no AI-side meaning). Preserves two-sided reference without inflating Digital-Bereavement risk.

---

## What this leaves out of the first-pass design, and why

The first-pass docs catalogued four-track memory + dual-time graph + Generative-Agents-style retrieval re-rank + sleep cycle + Hypocrisy-Gap probe + Forgetting Gate + Rupture-Repair ledger + PAM export. Under the principle-driven design, most of this dissolves:

| First-pass element | Fate |
|---|---|
| Pillar 1 event version chain (immutable Merkle versions) | **Dissolved** by B. The chain exists in the reconstruction trajectory; fragments are append-only; PAM covers export. |
| Pillar 1 semantic drift timeline | **Dissolved** by B applied to concept-fragments. Same mechanism. |
| Pillar 2 flat Stores #1 / #2 / #3 / #4 as four columns in a profile | **Dissolved** by C. Four separate mentalistic processes, not four slots. |
| Pillar 2 affective continuity as a stored timeline | **Dissolved** by B. Affect at a past moment is reconstructed from fragments, not played back from a time-series. |
| Pillar 3 six-term re-rank formula | **Dissolved** by B. Retrieval is reconstruction cued by context; "resonance", "mentalistic-match", "reconsolidation-recency" are cue sources, not weighted-sum terms. |
| Pillar 3 multi-dim importance scoring | **Dissolved** by D. Importance is revealed by selection, not asserted by scoring. |
| Four-track taxonomy (Episodic / Semantic / Procedural / Covenant) | **Partial survival.** Covenant stays as the non-decaying rule set (F exception). The other three dissolve — the fragment graph is unified; emergent clusters play the taxonomy roles. |
| Dual-time knowledge graph (Zep-style) | **Deferred.** External-world fact-change (CEO changed) can be a bolt-on later if we need it; it is orthogonal to reconsolidation. |
| Sycophancy stack (Hypocrisy Gap probe, Forgetting Gate, Rupture-Repair ledger) | **Reconsolidated into E.** The Witness is the sycophancy counter-mechanism. Rupture-Repair ledger becomes a fragment tag with elevated edge weight, not a separate structure. Hypocrisy-Gap activation-probing stays as *optional diagnostic* when the specific LLM supports it. |
| PAM / MCP export (Merkle-DAG) | **Survives with scope expanded.** Ships fragments + entity states + rule-internalization scores. Non-negotiable for Digital Bereavement. |

---

## Non-negotiables carried forward (from CLAUDE.md)

- **Resist Sycophancy.** Primary mechanism is now Principle E (Witness as Mirror). The Witness observes Companion drift and feeds divergence as pain into the Companion's next tick. Covenant-anchored anti-sycophancy rules do not decay (F exception).
- **Human-in-the-loop for crisis.** Non-decaying covenant rules include severe-symptom detection → handoff. The handoff is itself a fragment; reconsolidation across the handoff preserves continuity.
- **《办法》 regulatory floor.** Non-decaying rules enforce minor protections, anti-addiction patterns, algorithm filing at MAU threshold.
- **Digital Bereavement.** PAM/MCP export covers fragments + entity states + Hebbian edges + rule-internalization scores. User owns and can migrate.
- **Application legibility.** Each principle maps to a named concrete behavior in the learning companion or long-term companion surface — see the "closes which gap" tables above.

---

## Open questions I will not pretend to have answered

- **How much is "some degree" of consciousness simulation?** The six principles are the minimum I think is required to produce the therapy / learning behaviors we care about. Fewer and we drop back toward a store. The user's directive was "to some degree," not "fully" — Principles A–F are my current read of what that means in practice. Open to pruning if any principle turns out to not pay for itself.
- **The 绵延-at-month-horizon problem.** Ouroboros's mianyan score peaks around ep20 and decays by ep30; substrate variance collapses as caps saturate. Our companion lives for months. User direction 2026-10-05: validate the architecture at ep20-horizon first, treat month-horizon as a separate research track. This is honest about the risk — it is not fixed by porting.
- **Mentalistic validation has no external ground truth.** `mother-as-user-represents-her` is by construction the user's representation; only check is user-reported fit. Evaluation methodology for the mentalistic benchmark accounts for this ([benchmarks.md](benchmarks.md)).
- **Covenant-anchoring UX.** Principle D's covenant-anchored exception requires the user to be able to say "please always remember this." We have not designed what that interaction looks like or what safeguards prevent sycophantic over-anchoring. Flag for application design.
- **Reconsolidation granularity.** When does retrieval surface the delta visibly to the user ("I notice we're thinking about this differently today")? Thresholds will have to be tuned on real session data. Scripted benchmarks (B1) give directional signal only.
- **Witness architecture.** The second AI entity costs real compute. For v1 we may share a model across Companion + Witness with separate contexts rather than two parallel inference streams — tradeoff: cheaper but reduces the independence of Witness observations. Open design decision.
- **Mechanism of Mirror necessity (Phase 10a' pre-registered in Ouroboros, not yet run)**. We port Mirror as a confirmed-necessary primitive without full mechanistic understanding of *why* it is necessary. Replication risk if the actual mechanism in Ouroboros turns out to be an artifact.
