# Phase 1.5 — Scale-sensitive recall

**Date drafted:** 2026-10-06
**Reason this phase exists:** the Phase 1 real-data floor-regression gate on LoCoMo failed catastrophically (α-direct = 0.016, γ-RAG = 0.004 vs 0.90 floor on 43 single-hop Q, [comparison.md](../experiments/phase-1-benchmark-adapters/comparison.md)). The ep20-tuned primitives do not scale to 500+-turn dialogues. This phase closes the scale gap before Phase 2 adds reconsolidation / mentalistic structure on top.

**Companion docs:** [roadmap.md](../roadmap.md) sequences the overall plan; this phase sits between Phase 1 (ep20 primitives, done + partially gated) and Phase 2 (multi-session + mentalistic). [comparison.md](../experiments/phase-1-benchmark-adapters/comparison.md) holds the measured gate numbers.

---

## 1. Problem statement — the measured gap

Introspection on real LoCoMo instance `conv-26` (419 history turns) after full ingest, `CompanionSystem(seed=0, witness_active=False)`:

```
fragment_count=14   byte_cap=4096   edge_count=89
```

Two structural failures compound:

1. **Byte-cap under-sizes for long dialogues.** Selection pressure (S1 priority / S7 purge / S5 covenant) compressed 419 observations into 14 fragments at 4 KB. 13 are from `session_1`; 1 is from `session_17`. Every gold-bearing fact in sessions 2–19 was purged.
2. **Recall is cue-insensitive.** Measured on 4 single-hop questions with distinct golds (`Adoption agencies`, `Transgender woman`, `Single`, `Sweden`), the Q0-vs-Q1 fragment-set overlap is **6/6 = 1.000**. The reconstruction string is 95%+ identical across questions. The reader correctly refuses (`"not stated"`) on 19/20 because the surfaced fragments contain zero gold tokens.

Neither failure is fixable in the adapter / reader / RAG-wrapper layer. Enlarging `byte_cap` without cue-sensitivity would produce "larger uniform retrieval" rather than "right retrieval."

**Numeric target this phase must hit:** LoCoMo single-hop F1 ≥ 0.70 on the full 10-instance real-data gate ([option-gamma/run_locomo_gate.py](../experiments/phase-1-benchmark-adapters/option-gamma/run_locomo_gate.py), `--categories single-hop`). This is below Mem0's 92.5% blog number but above the SOTA 2025-paper mid-tier (Mem0 single-hop J=67.13, Zep ~76). We will not chase 92.5% in this phase; we will clear the ceiling that currently blocks Phase 2.

---

## 2. PDF review — how reference frameworks handle long-dialogue retrieval

Source: `reference-docs/AI Agent Memory Framework Analysis.pdf`, `reference-docs/智能体记忆架构研究.pdf`. Load-bearing claims:

- **Mem0** uses passive fact extraction via a backend LLM, writes extracted facts to a vector DB, retrieves via semantic search + BM25 hybrid. The agent itself has no control over what gets written. **Reported: 92.5% LoCoMo single-hop F1, 94.4% LongMemEval.**
- **Zep / Graphiti** uses a temporal knowledge graph with bi-temporal edges (`t_valid`, `t_invalid`). Retrieval combines dense vectors + BM25 + graph traversal. **Reported: 94.8% LongMemEval, P95 < 200 ms.**
- **Letta (MemGPT)** uses three tiers: Core (pinned in system prompt) + Recall (sliding window) + Archival (vector-indexed cold storage). The LLM decides retrieval via tool calls. **Reported: ~74% LoCoMo self-managed, 92–94% with explicit retrieval tools.**
- **MemOS** standardizes memory as "MemCubes" with lifecycle-managed tiers. Introduces predictive preloading on-inference. **Reported: +159% relative on LoCoMo vs OpenAI global memory baseline.**
- **CraniMem** gates fragments at entry (goal-conditioned mask) *before* they enter the reasoning window. **Reported: +14.3% AppWorld task completion.**
- **SleepGate** uses offline consolidation + Forgetting Gate during sleep epochs to prune low-value entries and distill episodic → semantic. No standalone retrieval story; conditions the retrieval surface.
- **Ebbinghaus GC** downweights, does not delete. Different memory types get different half-lives. Research cited: 13% accuracy loss vs 39% for naive full-record storage.

**PDF gaps (filled by §3):** actual embedding model + chunk-size + hybrid-fusion weights are not specified for any framework. No framework is tested at 10K+ fragment contexts in the PDFs.

---

## 3. OSS implementation review (what each project actually does)

Source: direct code / docs survey, 2026-10-06. Full report in session research log; key extractions below.

### Mem0 ([mem0ai/mem0](https://github.com/mem0ai/mem0))

- **Storage:** vector store (Qdrant practical default). Memories are **LLM-extracted facts, not raw messages**.
- **Ingest pipeline** (`mem0/memory/main.py:~1378`):
  1. Semantic-search top-k=10 existing memories using the current turn as query
  2. LLM called with `ADDITIVE_EXTRACTION_PROMPT` → emits ADD/UPDATE/DELETE/NOOP ops
  3. Batch-embed resulting fact strings
- **Query:** `vector_store.search(query, vectors, top_k=200, filters)`. Default embedder in LoCoMo eval: **OpenAI `text-embedding-3-small`**. Newer release adds BM25 + entity-overlap as **boost signals, not recall expanders** ("BM25 is a boost signal"). Optional cross-encoder reranker (disabled by default).
- **The write path itself retrieves + consolidates** — this is how the memory list stays bounded at ~hundreds of facts regardless of dialogue length.
- **Paper (2025) numbers:** single-hop F1 **38.72** / J **67.13** (Mem0); the "92.5" is from a 2026 blog on an updated system. Full-context baseline hits J ≈ 72.90 at ~600k tokens vs Mem0 ~7k.

### Letta / MemGPT ([letta-ai/letta](https://github.com/letta-ai/letta))

- **Four tiers:** Core blocks (pinned in prompt, agent-editable), Recall (full-text searchable raw history), Archival (**PostgreSQL + pgvector + HNSW**, default embedder `BAAI/bge-small-en-v1.5`), FIFO queue.
- **Query:** LLM emits a tool call like `archival_memory_search(query, top_k=5–10)` → pure cosine KNN. **No cross-tier scoring**; tiers consulted independently.
- **Overflow policy:** recursive summarization — oldest FIFO entries summarized into a block and evicted to archival.

### Zep / Graphiti ([getzep/graphiti](https://github.com/getzep/graphiti))

- **Storage:** temporal KG in Neo4j. Entity + episodic + community nodes; valid-time-stamped edges. Both nodes and edges embedded (`name` + `summary` for nodes, `fact` for edges).
- **Retrieval dispatch** (`search_config_recipes.py`): three raw methods (cosine KNN, Neo4j BM25, BFS graph walk) + five rerankers (RRF, MMR, cross-encoder, node-distance, episode-mentions).
- **Pre-built recipes:** `COMBINED_HYBRID_SEARCH_RRF`, `EDGE_HYBRID_SEARCH_NODE_DISTANCE`, etc. Default search limit ~10.
- Paper claims Zep marginally beats Mem0g on LoCoMo open-domain at ~100× the token cost per conversation.

### LangGraph / LangMem

- `BaseStore` + namespace tuples. Vector search enabled by configuring an `index` with `dims` + `embed` + `fields`. Default when configured: OpenAI `text-embedding-3-small` 1536-d. No BM25, graph walk, or built-in re-rank. No published LoCoMo numbers.

### Cross-cutting findings

1. **None of the four retrieve raw dialogue fragments at query time.** Mem0 stores LLM-extracted facts; Letta stores LLM-written passages; Graphiti stores extracted triples; LangMem stores structured memory objects. **Our 4KB raw-fragment store is the odd one out.**
2. **All four embed with cue-sensitive semantic vectors.** A top-K that returns the same fragments regardless of query implies no embedding or query-independent scoring.
3. **Consolidation (ADD/UPDATE/DELETE) is where Mem0 wins on long dialogues, not retrieval tricks.** It keeps the memory list small so top-K is meaningful.
4. **Hybrid (vector + BM25 + graph) only helps once the base signal is cue-sensitive.** Mem0 docs are explicit that BM25 is a boost, not a recall expander.

---

## 4. Brainstorm — mechanism space (unfiltered)

Mechanisms that could address the measured gap, before compression:

- **M1 Raise byte_cap** to 32 KB / 64 KB / 256 KB. Trivial. Addresses purge but not cue-insensitivity.
- **M2 Add semantic index** over fragments: embed on write, cosine re-rank on recall. Addresses cue-insensitivity.
- **M3 LLM-extraction consolidation layer** (Mem0 pattern): sleep-tick runs an extraction LLM over recent raw fragments → emits structured facts (ADD/UPDATE/DELETE) into a separate fact store. Recall queries the fact store, not the fragment store.
- **M4 Hierarchical tiering** (Letta pattern): core / working / archival. Different capacity + different retention. Current `byte_cap` becomes the working-tier cap only.
- **M5 Temporal knowledge graph** (Zep pattern): extract entity + relationship triples with t_valid / t_invalid. Retrieval does hybrid vector + BM25 + graph walk.
- **M6 Fragment summaries** (SleepGate pattern): sleep-tick compresses raw session into summary fragment; raw fragments age out but summary is retained. Changes what survives selection pressure, not how recall ranks.
- **M7 Cue-conditioned selection pressure**: instead of applying one static priority function at purge time, parameterize the purge by a learned "what-is-likely-to-be-asked" prior. Experimental; not in any OSS reference.
- **M8 Predictive preload** (MemOS pattern): on each turn, speculatively preload fragments for likely next question. Adds state outside the recall path.
- **M9 BM25 keyword index** (simple sparse retrieval): no embedding model, no LLM calls, pure Lucene-style. Addresses cue-insensitivity cheaply.
- **M10 Hybrid of M3 + M2**: facts in a vector-indexed store, raw fragments still in substrate for Principle B reconstruction behind the facts.

Discarded during brainstorm:
- Parametric memory (TTT / Fast Weights) — out of scope for v1 per [memory-design.md](../memory-design.md) deferral; depends on model internals.
- CraniMem-style gate at entry (M11 candidate) — gates writes, doesn't solve retrieval; our ingest already applies selection pressure.
- Cross-encoder reranker as mainline — Mem0 disables by default (slow); available as optional add-on later.

---

## 5. Options (compressed to four)

Each option specifies: **mechanism**, **principles touched**, **what Phase 1 keeps/changes**, **kill criteria**.

### Option 5.A — Scale-up primitives (minimal change)

**Mechanism.** M1 + M2 + M9. Raise `byte_cap` to 64 KB. On write, embed each fragment with a local SIMPLE-tier embedder (BAAI/bge-small-zh-v1.5 is already cached — see [`project_llm_architecture_and_cost_strategy.md`](../../.claude/.../feedback_llm_architecture_and_cost_strategy.md) memory). On recall, score fragments by `α·cosine(cue, fragment) + β·recency + γ·importance`, return top-K. Keep raw fragments as substrate.

**Principles touched.** B (reconstruction) extended: fragment selection at recall becomes cue-conditioned, matching Principle B's "rebuild from fragments under current context" more faithfully than the current cue-insensitive version. D (selection pressure) unchanged on write side, extended on read side.

**What changes.** Fragment store gains an embedding column; `recall` gains a scoring function. No new store, no new LLM calls at ingest.

**What Phase 1 primitives keep.** Fragment+heartbeat (unchanged), R1/R2/R7 reconstruction (unchanged — just better fragment selection feeding into it), S1/S7/S5 selection pressure (write-side unchanged; read-side gains re-rank), W1/W4/W7/W3 Witness (unchanged).

**Kill criteria.** If LoCoMo single-hop F1 < 0.50 on the full real-data gate even after re-rank, this option fails — the issue is that raw dialogue fragments carry too much noise regardless of selection, and we need the extraction layer. Also: if `byte_cap=64KB` on `conv-26` still produces < 50 surviving fragments after full ingest, selection pressure is purging too aggressively and the mechanism needs revisit at the write side.

**North-star fit.**
- Learning companion: adequate — short-form fact recall benefits directly.
- Long-term companion: adequate for single-session recall; weak for cross-session semantic grouping (no fact-level structure).

### Option 5.B — Add consolidation layer (Mem0 pattern)

**Mechanism.** M3 + M2. Raw fragments stay as substrate. Add a sleep-tick extraction pass: SIMPLE-tier LLM reads recent raw fragments, emits structured facts (ADD/UPDATE/DELETE/NOOP operations against existing facts). Facts live in a separate vector-indexed store. Recall queries facts (top-K cosine); facts carry back-pointers to source fragments so Principle B reconstruction can still ground meaning in raw text when needed.

**Principles touched.**
- A (living process): facts have their own lifecycle (ADD/UPDATE/DELETE); updates are versions, not overwrites — natural Phase 2 reconsolidation substrate.
- B (reconstruction): meaning is rebuilt from facts; facts are themselves rebuilt from raw fragments. Two-level reconstruction, which arguably matches biology better (semantic memory vs episodic memory).
- D (selection pressure): operates at both tiers — fragments still purge under capacity, facts purge under ADD/UPDATE/DELETE policy.
- C (mentalistic processes): facts can be typed (`entity=mother`, `relation=user-represents-as`), giving Phase 2 a natural attachment point for mentalistic sub-stores.

**What changes.** New `FactStore` component. New sleep-tick extraction step (SIMPLE-tier LLM calls). Recall path dispatches: cue → facts (primary) → optional fragment lookup via back-pointers.

**What Phase 1 primitives keep.** All kept. Fact store is additive, not replacement. The fragment store remains the "episodic deep layer"; facts are the "semantic surface layer".

**Kill criteria.** If LLM-extraction consolidation produces fewer than 50 facts per 500-turn dialogue (over-aggressive consolidation) or more than 2000 facts (under-consolidation), the extraction prompt needs revisit — one iteration allowed. If facts-based retrieval still returns Q0=Q1 overlap > 0.5 (cue-insensitivity persists), the embedder is wrong. If total ingest cost exceeds $1 per 500-turn dialogue (SIMPLE-tier should be ~$0.10), consolidation is too chatty.

**North-star fit.**
- Learning companion: strong — facts naturally represent learned concepts; version chains = concept mastery evolution (Phase 2 B1 benchmark substrate).
- Long-term companion: strong — facts about third parties = Phase 2 Store #2 attachment point.

**Cost.** SIMPLE-tier extraction every ~50 turns = ~10 LLM calls per 500-turn dialogue × $0.001 (local Ollama or qwen-turbo) = ~$0.01–0.10 per dialogue. Within the "LLM budget legible to applications" rule from CLAUDE.md.

### Option 5.C — Hierarchical tiering (Letta pattern)

**Mechanism.** M4 + M2 + M6. Three tiers: **core** (covenant + current session, pinned, no purge), **working** (last 3 sessions, current `byte_cap` → scaled to 32 KB), **archival** (vector-indexed older fragments). Sleep-tick compresses older working fragments into session-summary fragments (M6) which move to archival. Recall queries all three in parallel + fuses by scope-weighted score.

**Principles touched.**
- D (selection pressure): now tiered — survives working-tier purge → promoted to archival with summary.
- B (reconstruction): fuses across tiers; current session grounds in raw, old sessions ground in summaries.
- F (rule decay): covenant lives in core; style rules decay through tiers.

**What changes.** Three stores, three recall paths, cross-tier fusion function. Significant structural shift.

**What Phase 1 primitives keep.** Fragment+heartbeat kept (per tier). Selection pressure reinterpreted. Reconstruction extended.

**Kill criteria.** If cross-tier fusion weights are ad-hoc enough that LoCoMo tuning becomes application-specific, this is a store-shape disguised as consciousness-simulation — not acceptable per CLAUDE.md "Memory decisions must be legible to applications" rule reversed. If archival summaries introduce the kind of reconsolidation Phase 2 wants to design intentionally, scope is leaking.

**North-star fit.**
- Learning companion: adequate — tiering matches natural session cadence.
- Long-term companion: partial — hierarchy by time doesn't capture mentalistic structure; Phase 2 would still need Option 5.B/D on top.

### Option 5.D — Temporal knowledge graph (Zep pattern)

**Mechanism.** M5 + M2. Extraction at sleep tick emits entity + relationship triples with `t_valid` / `t_invalid`. Graph built in SQLite + a lightweight graph layer (not Neo4j; keep deployable). Recall does hybrid: entity embedding search + 1-2 hop graph walk + optional BM25 boost.

**Principles touched.**
- A (living process): `t_invalid` naturally represents fact-change over time (Zep's dual-time story).
- C (mentalistic processes): entity nodes are first-class — `mother-as-user-represents-her` is a node with attributes; the "process" is the retrieval re-evaluation across the node + its edges.
- B (reconstruction): fuse graph context (node + edge summaries) with raw fragment lookup.

**What changes.** Entirely new component. Heavy build cost. Graph tooling (even in SQLite) is non-trivial.

**What Phase 1 primitives keep.** Fragments kept as source-of-truth for Principle B grounding; graph is derived.

**Kill criteria.** If build cost exceeds 2× Option 5.B, abandon — the Phase 2 mentalistic layer was planned without a graph, which means the graph is adding scope. If graph-walk at retrieval doesn't materially improve over Option 5.B's cosine-only recall on LoCoMo single-hop (< 5 pt F1 gain), the graph complexity isn't buying anything.

**North-star fit.**
- Learning companion: overkill — procedural + fact knowledge doesn't need bi-temporal edges.
- Long-term companion: strongest fit — bi-temporal + entity graph = natural substrate for Phase 2 reconsolidation + mentalistic processes.

---

## 6. Cross-check against the two north-stars

| Dimension | 5.A Scale-up | 5.B Consolidation | 5.C Hierarchy | 5.D KG |
|---|---|---|---|---|
| **LoCoMo single-hop ≥ 0.70 (gate)** | likely 0.4–0.6 | likely 0.6–0.8 | likely 0.5–0.7 | likely 0.6–0.8 |
| **Preserves Phase 1 primitives** | fully | fully (additive) | partially (reinterpret) | partially (fragments → source-of-truth only) |
| **Enables Phase 2 reconsolidation** | weak (fragment-level only) | strong (facts = version chain target) | weak | strongest (bi-temporal edges) |
| **Enables Phase 2 mentalistic** | weak | strong (typed facts) | weak | strongest (entity nodes) |
| **Enables Phase 2 affective-resonance** | adequate | strong | adequate | strong |
| **Cost per 500-turn dialogue** | ~$0 (local embedder) | ~$0.01–0.10 (SIMPLE LLM) | ~$0.01 (local embedder) | ~$0.10–0.50 (extraction LLM) |
| **Build complexity** | 2–3 days | 5–7 days | 5–7 days | 10–14 days |
| **Learning companion fit** | adequate | strong | adequate | overkill |
| **Long-term companion fit** | weak | strong | partial | strongest |
| **Risk of scope-leak into Phase 2** | low | medium (fact versioning is near-reconsolidation) | low | high (bi-temporal is Phase 2 territory) |

---

## 7. Recommendation and open questions

**Recommended path:** **Option 5.B (consolidation layer)**, with the explicit discipline that fact versioning in this phase is "fact UPDATE replaces the old fact" (no version chain). Phase 2 adds the version chain on top of this substrate — natural layering, no scope-leak.

**Reasoning.**
- 5.A under-shoots: raw dialogue fragments carry too much noise; the SOTA consensus (Mem0, Letta, Zep) is "retrieve over structured facts, not raw turns."
- 5.D over-shoots: bi-temporal edges + entity graph are Phase 2 mentalistic + reconsolidation substrate; building the graph now before we know what mentalistic structure we need risks shaping Phase 2 to fit the graph rather than the gap.
- 5.C hybridizes with other options poorly and introduces a store-shape (tier hierarchy) that leaks into Phase 2 design.
- 5.B is the Mem0 pattern — the project with the highest LoCoMo number — adapted to our substrate: facts are the semantic layer; fragments remain the episodic deep layer that Principle B reconstruction grounds in. The two-level reconstruction matches biology (semantic vs episodic) better than the one-level fragment store currently does.

**Open questions to resolve before implementation:**

1. **Extraction cadence.** Mem0 extracts per-turn. We could extract per-session (cheaper, more coherent) or per-sleep-tick (more Phase 1 principle-aligned). Likely per-sleep-tick, but needs explicit choice.
2. **Fact typing.** Flat strings (Mem0 style) or typed entity+relation+value (Zep style)? Flat is simpler; typed sets up Phase 2 mentalistic cleanly. Likely typed-but-minimal.
3. **Fragment back-pointers.** Every fact points back to the source fragments it was extracted from, so Principle B grounding still works. Required for the learning-companion use case ("show me what we actually said about ∂").
4. **Embedder choice.** Local BAAI/bge-small-zh-v1.5 (already cached) vs qwen-text-embedding (DashScope, cost). Likely local for ingest, cloud for ADVANCED-tier re-ranking only.
5. **Recall path dispatch.** When does the system fall back to raw-fragment search? Only on fact-store miss, or always as a boost? Likely only on miss, to preserve Mem0's finding that extracted facts are the signal.
6. **Interaction with Phase 1 selection pressure.** When a fragment is extracted into a fact, does it become safe from purge (because its content now lives in the fact)? Or does the fragment still purge independently? Likely: fragments purge normally; facts have their own capacity cap. This separates episodic decay from semantic retention — biology-shaped.

**Deferred experiments (step 7 of the phase-planning process):**

- E1. Prototype 5.B on `conv-26` only (1 instance, 154 Q): expected to lift single-hop F1 from 0.016 to ≥ 0.40. If it doesn't, 5.B is wrong; try 5.A as a sanity check or 5.D as the harder redesign.
- E2. If E1 passes threshold: run the full 10-instance LoCoMo gate. Decision: ≥ 0.70 → Phase 1.5 exits; 0.50–0.70 → iterate on consolidation prompt (one iteration budget); < 0.50 → re-open brainstorm.
- E3. Comparison table in [docs/experiments/phase-1.5-scale-recall/comparison.md](../experiments/phase-1.5-scale-recall/comparison.md) once E2 completes.

---

## 8. Exit criteria

Phase 1.5 exits when **all** of the following hold:

| Gate | Target |
|---|---|
| LoCoMo single-hop F1 (real 10-instance full gate) | ≥ 0.70 |
| LongMemEval accuracy (real smoke subset, 50 instances) | ≥ 0.70 |
| EpBench F1 (real smoke subset, 50 instances) | ≥ 0.50 |
| V1/V2 Phase 1 bands held (no regression on ep20 primitives) | all in-band per [baselines.md](../baselines.md) |
| V4 Witness ablation still ≥ 1.3× anchor / ≥ 10× null | consciousness-simulation substrate not broken |
| Cost per 500-turn dialogue | ≤ $0.50 |
| Build time | ≤ 10 days of implementation |

**Note on the LoCoMo target.** 0.70 is below Mem0's blog 92.5% and below the Phase 1 original 0.90 floor. Rationale: Phase 1's 0.90 was "SOTA regression guard"; the measured reality is that Mem0's own paper reports J=67.13 single-hop (0.90 was aspirational per the Mem0 blog, which uses an updated system). 0.70 clears the fail-mode ceiling (currently 0.004) without chasing an unverified blog number. If the final consolidation implementation surprises us upward, great; if 0.70 is itself unreachable, we have a bigger problem than scale.

---

## 9. What this phase does NOT do

- Does not add reconsolidation version chains (Phase 2 scope).
- Does not add mentalistic sub-stores (Phase 2 scope — but fact typing is the hook).
- Does not add affective-resonance retrieval (Phase 2 scope — but recall-side scoring is the hook).
- Does not add long-horizon continuity / 绵延 (Phase 3 scope).
- Does not change the Witness or covenant primitives.

---

## 10. Decision needed from user before implementation starts

1. **Option choice** — recommended 5.B. User should name the chosen option (5.A/5.B/5.C/5.D or a composite).
2. **Fact typing** — flat strings or minimal typing (entity+relation+value)?
3. **Experiment commitment** — run E1 on 1 instance before full E2 gate (recommended, saves API cost), or go straight to full gate?

Once decided, I draft a `phase-1.5-build-plan.md` with concrete steps + file layout + per-step tests, matching the pattern of [phase-1-plan.md](phase-1-plan.md) and [phase-1-benchmark-adapters-plan.md](phase-1-benchmark-adapters-plan.md).
