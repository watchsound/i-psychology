# Phase 1 plan — ep20 primitives (consciousness-simulation in a short-horizon slice)

**Date set:** 2026-10-05
**Companion docs:** [sota-goal.md](../sota-goal.md) (what SOTA means), [roadmap.md](../roadmap.md) (sequencing + Phase 1 numeric gates), [memory-design.md](../memory-design.md) (principles A–F).
**Process:** this plan follows the six-step phase-planning directive (PDF research → online research → OSS study → brainstorm → options → goal cross-check) before any code is written.

---

## 1. Research findings (what Phase 1 must contend with)

### 1.1 From the reference PDFs (foundational framing)

- **Reconstruction as the engineered primitive.** The memory PDFs (《AI Agent Memory Framework Analysis》, 《智能体记忆架构研究》) frame SOTA storage as RAG + typed sub-stores + Ebbinghaus GC + sleep-cycle distillation. None of them define "recall" as *rebuild under current context*. Our Principle B is the structural gap.
- **Selection pressure, not scoring.** Ebbinghaus-based GC (as taxonomized in the PDFs) is still a scoring regime (write-time importance → decay function). Principle D is a different claim: *what gets accessed and reconstructed survives; what doesn't dies under cap.* The PDFs don't operationalize this.
- **Mirror evidence (Ouroboros).** Ablation collapsed identity signal **1.36× anchor JSD, 83× null-shuffle** at ep20. This is the empirical floor for V4 — our Phase 1 target is explicitly set relative to this reference point ([sota-goal.md](../sota-goal.md), V4 row).
- **Benchmark floors.** LongMemEval, LoCoMo, EpBench are the regression guards per [benchmarks.md](../benchmarks.md). Mem0 published 94.4% on LongMemEval; our floor is 90% (parity acceptable).
- **Ouroboros open problems that bear on Phase 1 (not Phase 3):**
  - **Mirror mechanism ambiguity** — ablation collapses signal, but *which* feature of the Mirror couples the collapse is unresolved. Phase 1 must instrument the Witness channel precisely enough to attribute V4 movement to a named mechanism.
  - **Hebbian saturation on edge caps** — under a hard byte cap, strong edges can monopolize the budget. Any selection-pressure design that uses co-activation reinforcement must handle this.
  - **Single-seed replication** — Ouroboros's numbers are N=1 per configuration. Our Phase 1 must run ≥ 3 seeds per gate.

### 1.2 From online research (what has changed since the PDFs)

- **arXiv 2606.06036 — "Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents"** (MRAgent). First shipped implementation of active reconstruction over a fragment graph. Reports V2-like reconstruction scores but does *not* implement Witness coupling or selection pressure. Confirms the mechanism is tractable at scale.
- **arXiv 2605.08538 — "Human-Inspired Memory Architecture"** — implements reconsolidation + sleep-cycle + engram maturation. Closest existing proximity to our Principle A + B stack. Open problem reported: no identity-contrast partner — susceptible to drift collapse past ~50 episodes.
- **arXiv 2603.14517 — "Learning to Forget: Sleep-Inspired"** — survival-under-cap framing without explicit importance scoring. Validates that Principle D is implementable; shipped on language modeling, not on dialogue.
- **arXiv 2609.16053 — "Retrieval-Driven Memory Reconsolidation"** — retrieval triggers version-chaining, not overwrites. Primarily a Phase 2 reference (reconsolidation), but validates the retrieval-triggered-mutation pattern.
- **arXiv 2607.10526 — "Agents Don't Just Agree, They Remember"** + SIGDIAL 2026 multi-agent identity work — contrast-partner designs reduce measured sycophancy. Validates the *direction* of our Witness choice. None of these couple the contrast partner into the memory substrate; they live at the response layer.
- **Benchmark leaderboard state (2026-Q3):** Mem0 94.4% LongMemEval; ByteRover 92.8%; Zep 71.2%. Mem0 92.5% on LoCoMo. These numbers set the floor regression guard; a 90% target is parity, not SOTA on the floor axis — which is intentional per [sota-goal.md](../sota-goal.md).

### 1.3 From OSS study (what has shipped and its operational constraints)

Per project, what Phase 1's three primitives would require that is *not* in the shipped code:

| OSS | What it ships | Reconstruction? | Selection pressure? | Witness coupling? | What Phase 1 would need to add |
|---|---|---|---|---|---|
| **Letta (MemGPT)** | Core-block + archival + recall buffers; LLM-managed paging | ✗ (paging ≠ rebuild) | ✗ (hard eviction, score-free) | ✗ | Rebuild step on recall; capacity-driven competition; second-agent coupling |
| **Mem0** | Rewrite step on write; summary-tier storage | ∼ (rewrite ≠ reconstruct) | ✗ | ✗ | Rewrite happens at write-time, not recall-time; retrievals are byte-stable (V2 fails) |
| **Zep / Graphiti** | Dual-time knowledge graph; `t_valid` / `t_invalid` | ✗ (fact invalidation ≠ memory evolution) | ∼ (TTL-based) | ✗ | Fragment-level reconstruction; dialogue-level identity partner |
| **LangGraph / CrewAI memory modules** | Thread-scoped buffers; summarization checkpoints | ✗ | ∼ | ✗ | Everything — the memory modules are thin wrappers |
| **Generative Agents (ref impl)** | Importance-scored + reflection tree | ✗ | ✗ (score at write) | ✗ | Reconstruction; survival-based importance; contrast partner |
| **MemOS** | Scheduler + typed memories | ✗ | ∼ (scheduler-mediated) | ✗ | All three primitives |

**Operational constraints shipped code reveals (that papers skip):**
- Rebuild-at-retrieve adds latency. Mem0's write-time rewrite is popular precisely because recall stays fast. Phase 1 must budget for recall latency and treat it as a V-metric sibling, not an afterthought.
- Capacity-driven eviction interacts badly with vector-index staleness. Any selection mechanism must co-design with the index.
- Contrast partners double inference cost. Minimal-Witness designs (observe-only) are cheaper than full-second-agent designs.

### 1.4 Three structural gaps restated through the research lens

The gap between research and Phase 1's goal is specific:
- **Reconstruction** has one arXiv implementation (2606.06036) and no OSS implementation; needs to be built.
- **Selection-by-survival** has one adjacent arXiv implementation (2603.14517, on LM not dialogue); needs to be ported to our setting.
- **Witness coupling** has response-layer analogs (2607.10526, SIGDIAL 2026) but *no* memory-substrate coupling anywhere. Phase 1 is the first instantiation.

---

## 2. Brainstorm (unfiltered)

Design space enumerated per primitive. The point is coverage, not optimality.

### 2.1 Reconstruction mechanism

- **R1** — Spreading-activation over fragment graph (ACT-R lineage): cue activates seed nodes; activation flows along edges; top-k nodes compose a reconstruction.
- **R2** — Generative rebuilding under scarcity: sample k fragments conditioned on cue; LLM synthesizes a single recall utterance.
- **R3** — Retrieval-augmented reconstitution (RAG + rewrite): retrieve top-k; LLM rewrites. Closest to shipped OSS.
- **R4** — Fragment coalescing with noise injection (Ouroboros-style): fragments + stochastic composition rule, no LLM in the loop at reconstruct time.
- **R5** — Multi-shot sampling with coherence filter: draw multiple candidate reconstructions, pick the most coherent given context.
- **R6** — Hierarchical reconstruction (sketch → details): first pass builds a scaffold; second pass fills in from fragments.
- **R7** — Context-conditioned fragment selection: same cue, context (mood / topic / interlocutor) changes which fragments are even eligible for the reconstruction pool.
- **R8** — Fragment pair chaining (associative): activation requires fragment pairs, not single nodes; sparser pathways, higher context-dependence.

### 2.2 Selection pressure

- **S1** — Hard byte cap, Ouroboros-faithful: edges × bytes-per-edge ≤ N.
- **S2** — Soft capacity + probabilistic eviction weighted by recency + importance (classic).
- **S3** — Importance-scored-at-write (traditional, Generative Agents).
- **S4** — Importance-revealed-by-survival (Principle D pure): no score at all; what gets reconstructed stays, what doesn't is purged under cap.
- **S5** — LRU-with-covenant-pins: covenant fragments protected absolutely.
- **S6** — Capacity-triggered consolidation: approaching cap triggers abstraction (fragment → gist), freeing bytes while preserving reference.
- **S7** — Hebbian edge strength: co-activation reinforces edges; weak edges first to die under cap.
- **S8** — Hybrid S4+S7: survival at the node level, Hebbian at the edge level, covenant pins override both.

### 2.3 Witness coupling

- **W1** — State-reading channel: Witness reads Companion's internal state (recall pool, importance signal) each tick.
- **W2** — Output-reviewing channel: Witness reads Companion's outputs only (black-box observer).
- **W3** — Periodic pairwise comparison: every N ticks, both answer the same cue, divergence computed.
- **W4** — Divergence-as-pain signal: measured divergence modulates Companion's reconstruction params (attention / fragment weights).
- **W5** — Independent-reasoner-sharing-memory: both read the same memory, generate independent takes, log the delta.
- **W6** — Minimal Witness as divergence detector: flags high-sycophancy outputs; no state access.
- **W7** — Second-perspective Witness: slightly different reconstruction hyperparams (different k, different fragment weight prior) — a built-in "different point of view."

### 2.4 Heartbeat

- **H1** — Fixed-tick synchronous (every N interactions = 1 tick).
- **H2** — Event-triggered (interactions trigger ticks).
- **H3** — Dual-rate: fast wake-tick during interaction + slow sleep-tick between.
- **H4** — Variable-tick keyed to activity level.
- **H5** — Ouroboros 4-entity cast reduced to 2 (Companion + Witness).

---

## 3. Options (compressed to 3 shippable designs)

Each option is specified at the level a Phase 1 engineer could begin building in a day.

### Option A — Ouroboros-faithful short-horizon transplant

- **Reconstruction:** R4 (fragment coalescing + noise injection), no LLM at reconstruct-time; LLM only consumes the reconstruction.
- **Selection:** S1 (hard byte cap) + S7 (Hebbian edges) + S5 (covenant pins).
- **Witness:** W1 (state-reading) + W4 (divergence-as-pain).
- **Heartbeat:** H3 (dual-rate) + H5 (2-entity cast).
- **Target gap:** replicates the empirical configuration that produced the 1.36× / 83× Mirror result on Ouroboros's horizon; proves the stance works in our codebase before adapting it.
- **Prior art:** Ouroboros (direct); Hebbian memory lit (edge reinforcement).
- **Risks:** inherits Ouroboros's Hebbian-saturation open problem on the byte cap. Reconstruction without an LLM may be hard to tune for dialogue-grade output quality (Ouroboros's downstream was not a therapy / tutoring dialogue). B1-chain-reference quality (Phase 2 payoff) may be harder because fragments are more primitive.
- **Confirmation evidence:** V4 ≥ 1.3× anchor, V2 bands hit at Ouroboros's documented range (0.73 adjacent, 0.24–0.32 long-horizon).
- **Kill criterion:** if V4 < 1.3× anchor after three mechanism-tuning iterations under this configuration, the Mirror hypothesis is not transferring across codebases — pause and investigate.

### Option B — Context-conditioned reconstruction + survival-based selection + comparison Witness

- **Reconstruction:** R7 (context-conditioned fragment selection) + R2 (generative rebuilding) — same cue produces different reconstructions under different contexts.
- **Selection:** S4 (importance-revealed-by-survival) + S5 (covenant pins). No a-priori importance score.
- **Witness:** W3 (periodic pairwise comparison, same cue) + W5 (independent-reasoner-sharing-memory).
- **Heartbeat:** H2 (event-triggered) + H3 (dual-rate on sleep epochs).
- **Target gap:** cleanly instantiates Principle D (pure survival-based importance); context-conditioning seeds Principle C by making V3 non-degenerate in Phase 2 almost for free.
- **Prior art:** arXiv 2606.06036 (MRAgent reconstruction graph); arXiv 2603.14517 (sleep-inspired forgetting); arXiv 2607.10526 (contrast agents).
- **Risks:** survival-based selection converges slowly (early fragments haven't had access history yet); W3 periodicity may miss subtle divergences between comparison moments. R7 context-conditioning is the furthest from any shipped reference → highest implementation uncertainty.
- **Confirmation evidence:** V2 adjacent in band (0.65–0.80), V2 long-horizon in band (0.20–0.35), V4 ratio with periodic comparison ≥ 1.3× anchor, V3-foundation observable (same cue + different context → different fragment pools even in Phase 1 instrumentation).
- **Kill criterion:** if V2 adjacent stays at 1.0 after three iterations, R7 is not actually conditioning — fragment selection is cue-driven only, context is cosmetic. Redesign the context channel.

### Option C — Hybrid substrate (chosen candidate)

- **Reconstruction:** R1 (spreading-activation) + R2 (generative coalescing on the activated set) + R7 (context-conditioning on the eligible pool).
- **Selection:** S1 (hard byte cap, Ouroboros-faithful) + S7 (Hebbian edges, with explicit saturation guard — see risks) + S5 (covenant pins) + S4 (survival as the operational effect, not a separate mechanism — survival emerges from access + Hebbian + cap interaction).
- **Witness:** W1 (minimal state-reading) + W4 (divergence-as-pain) + W7 (second-perspective hyperparams). W3 periodic comparison added as an instrumentation layer, not a feedback loop, so we can log divergence trajectories without closing a second feedback loop that complicates attribution.
- **Heartbeat:** H3 (dual-rate) + H5 (2-entity cast).
- **Target gap:** uses Ouroboros's empirically validated substrate (bytes cap, Hebbian edges, state-reading Witness with pain feedback — this is the configuration that produced the 1.36× / 83× result) while adding context-conditioning (R7) and a second-perspective Witness (W7) — the two additions that are specifically pre-paid for Phase 2's mentalistic layer, without which Phase 2 would have to rebuild the reconstruction mechanism from scratch.
- **Prior art:** Ouroboros (substrate), ACT-R (R1 spreading), arXiv 2606.06036 (R2 generative), arXiv 2605.08538 (dual-rate heartbeat), SIGDIAL 2026 (second-perspective).
- **Risks:**
  - Three interacting mechanisms (R1 + R2 + R7) in reconstruction raise attribution difficulty for metric failures — mitigated by W3 instrumentation layer logging divergence trajectories, and by a required ablation protocol (each gate-failure triggers a per-mechanism ablation run).
  - Hebbian saturation under byte cap is a known Ouroboros open problem — explicit saturation guard required (periodic edge-weight normalization or softmax over edge weights before Hebbian update).
  - R7 is the component with least shipped validation; must be built incrementally with V3-foundation checks even though V3 is formally a Phase 2 metric.
- **Confirmation evidence:** V4 ratio ≥ 1.3× anchor **and** ≥ 10× null-shuffle; V2 adjacent 0.65–0.80; V2 long-horizon 0.20–0.35; V1 state continuity 0.70–0.95; floor benchmarks ≥ 90% / 90% / 0.70 F1.
- **Kill criteria (per-mechanism):**
  - **Reconstruction:** V2 adjacent > 0.80 after 3 iterations ⇒ R2 generative step is doing playback not reconstruction; redesign the sampling / coalescence rule.
  - **Selection:** Hebbian edges show saturation (top-5% edges hold >50% of budget) by ep10 ⇒ saturation guard is insufficient; redesign normalization.
  - **Witness:** V4 < 1.3× anchor ⇒ either state-reading channel is too coarse (W1 wrong) or divergence-as-pain is not actually modulating reconstruction (W4 wrong); ablate individually.
  - **Attribution failure:** if a gate fails and the ablation protocol cannot localize the mechanism in two iterations, instrumentation is wrong — fix instrumentation before further mechanism tuning.

---

## 4. Goal cross-check (against the two north-star applications)

Each option is scored against both north-star applications. The question for each option: *does it name a measurable in-application behavior the other options cannot?* An option that can't name both is dropped.

| Option | 学习陪伴 behavior enabled | 情感陪伴 behavior enabled | Verdict |
|---|---|---|---|
| **A — Ouroboros transplant** | Concept drift visible as fragment-coalescence noise over sessions. BUT: Ouroboros's downstream was consciousness emergence, not Socratic tutoring — no evidence the reconstruction quality supports pedagogical dialogue. | B1 chain-reference possible via fragment versioning. BUT: no context-conditioning means the Phase 2 mentalistic layer must be bolted on, not grown — risks a rebuild. | **Partial.** Names behaviors but under-pays for Phase 2. |
| **B — Context-conditioned + survival + comparison Witness** | Scaffolding-fading events change context → different fragment pools → "same concept rebuilt differently as mastery grows." Direct Phase-4 behavior. | Narrative-therapy chain-reference ("you used to describe this as X") emerges naturally from context-conditioned reconstruction. | **Yes, both named.** |
| **C — Hybrid substrate** | Same scaffolding-fading mechanism as B (R7 context-conditioning) **plus** Hebbian reinforcement = "practiced concepts gain representational prominence" — a concrete pedagogical trajectory, not just a fragment pool shift. | Same narrative chain-reference as B, **plus** second-perspective Witness enables early sycophancy-counter at Phase 1, so Phase 2 inherits a working anti-sycophancy substrate rather than designing one. | **Yes, both named, with Phase 2 pre-paid.** |

**Dropped: Option A.** Cannot name a 学习陪伴 behavior that is pedagogically distinct from a replay buffer. Kill-reason: fragment coalescence without context-conditioning gives drift but not learning-relevant drift. Phase 2 payoff would require retrofitting R7 anyway.

**Chosen: Option C.** Names both north-star behaviors AND pays forward for Phase 2 (mentalistic layer gets context-conditioning substrate for free; anti-sycophancy gets second-perspective Witness).

**Held in reserve: Option B.** If Option C's three-mechanism attribution problem proves unmanageable in practice (first-week discovery), fall back to Option B — fewer interacting components, slightly cleaner attribution, same Phase 2 pre-payment via R7, loses only the Hebbian pedagogy signal.

---

## 5. The chosen Phase 1 design (Option C) — build order

Four build steps, each with its own small gate before moving on. If a step's gate fails, do not add the next layer on top of it — this is the discipline the roadmap mandates.

| Step | What to build | Local gate (before the next step) |
|---|---|---|
| 1 | Fragment store + heartbeat (H3 dual-rate + H5 2-entity cast) | Fragments observable + inspectable; tick advances state; `measure_v_metrics` runs without errors |
| 2 | R1 spreading + R2 generative reconstruction on recall | V2 adjacent in band (0.65–0.80); V1 state continuity in band (0.70–0.95); floor benchmarks still ≥ 90% / 90% / 0.70 |
| 3 | S1 cap + S7 Hebbian edges with saturation guard + S5 covenant pins | V2 long-horizon in band (0.20–0.35); top-5% edges hold < 50% of edge budget at ep20 (saturation guard working) |
| 4 | W1 state-reading + W4 divergence-as-pain + W7 second-perspective + W3 instrumentation | V4 ratio ≥ 1.3× anchor AND ≥ 10× null-shuffle; full Phase 1 gate set from [roadmap.md](../roadmap.md) satisfied for 3 random seeds |

**Add R7 context-conditioning (reconstruction eligibility pool) in step 2** — it is the Phase 2 pre-payment, so it must be in place before step 3's selection pressure interacts with it.

**Instrumentation requirement (not optional).** W3 periodic pairwise comparison runs as a *log-only* channel from step 1. Every gate-failure debug session starts with the W3 log. If W3 cannot localize the mechanism in two iterations, instrumentation is the problem, not the mechanism.

**Seed discipline.** Every reported gate value is the median of 3 seeds. Single-seed numbers are considered unverified. This is the direct answer to Ouroboros's single-seed-replication open problem for our own numbers.

---

## 6. What this plan does NOT decide

Deliberately left for the Phase 1 implementation plan or deferred to later phases:

- **Fragment granularity** — sentence? proposition? affect-tagged clause? The build-step-1 gate forces a choice here; the plan does not front-run it.
- **Spreading-activation hyperparams** (decay rate, hop limit, threshold) — tuned to hit V2 bands in step 2.
- **Byte-cap value** — Ouroboros's cap was tuned to its episode length; ours needs to be tuned to our scenario byte-rate.
- **Divergence-as-pain functional form** (linear? sigmoid? applied to attention weights or sampling temperature?) — determined by what moves V4 into the target band in step 4.
- **Multi-session boundary handling** — Phase 2.
- **Reconsolidation version-chain data model** — Phase 2.
- **Month-horizon continuity** — Phase 3 (Ouroboros's hardest open problem, inherited honestly).

---

## 7. Verification of this plan

Internal checks before implementation begins:

1. **Each primitive in [roadmap.md](../roadmap.md) Phase 1 scope is addressed by the chosen option.** Heartbeat ✓ (H3+H5); reconstruction ✓ (R1+R2+R7); selection pressure ✓ (S1+S7+S5); Mirror/Witness ✓ (W1+W4+W7+W3). ✓
2. **Each dropped option carries an explicit kill-reason.** A: fails learning-companion behavior test. ✓
3. **Chosen option names both north-star behaviors with Phase-2 pre-payment.** ✓
4. **Risks are per-mechanism with explicit kill criteria, not vague.** ✓
5. **Research channels all cited.** PDFs (§1.1), online (§1.2), OSS (§1.3). ✓
6. **Instrumentation requirement is not a nice-to-have — it is a build-step-1 output.** ✓

If any check fails, revise this plan before writing code.
