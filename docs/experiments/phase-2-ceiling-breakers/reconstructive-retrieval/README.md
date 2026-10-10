# Reconstructive-retrieval architecture (Phase 2 ceiling-breaker)

Goal: close the 71% `session_present_gold_absent` structural ceiling
confirmed by multi-instance validation (3-instance R5 aggregate
F1=0.544 vs 0.70 gate, −0.156). Chosen scope: **best** mechanism
class, not cheapest — the user's directive after seeing the
cross-channel research synthesis (reference-PDFs + arXiv + OSS).

## One-line thesis

The reader needs **facts expanded to their source turns**, **the
right session fully in context**, and **a recall-biased filter
before the precision-biased reader**. Each lever stacks on
existing substrate; two more levers (reconsolidation hook,
reconstructive assembly) carry the project past the SOTA
Mem0/Zep ceiling into thesis territory.

## Substrate already in place

- `Fact.source_fragment_ids` back-pointers ([facts.py:84](../../../src/psychology_base/companion/facts.py#L84))
  — facts know which turns produced them. Today these are
  **stored but not used for retrieval lift**.
- `FactStore.search` cosine over fact embeddings
  ([facts.py:156-179](../../../src/psychology_base/companion/facts.py#L156-L179)).
- Extraction prompt already requests "ONE fact PER ITEM"
  granularity ([extraction.py:59-61](../../../src/psychology_base/companion/extraction.py#L59-L61)).
- Hierarchical dual-tier reader context: top-150 facts + top-50
  cross-encoder-reranked fragments ([system.py:62-63](../../../src/psychology_base/companion/system.py#L62-L63)).
- Fragment-level reconstruction (R1 spreading / R2 generative / R7
  context-conditioning) in [reconstruction.py](../../../src/psychology_base/companion/reconstruction.py)
  — **unused by the LoCoMo eval path**; wired for Phase 3.

## What this plan adds

Eight sequential steps. The first six get to the Mem0/Zep LoCoMo
ceiling (reported 0.75–0.92; third-party reproductions ~0.75);
steps 7–8 are the two thesis-only levers that attack what the
top SOTA stacks don't.

Each step has:
- **Mechanism** (what it does)
- **Why this step now** (dependency on prior steps)
- **Measurement** (3-instance F1 + recall-diagnostic bucket shift)
- **Kill criteria** (back out if lift < 2× noise band 0.03 and no
  bucket migration)

### Step A — Fact-to-turn lift (promote source fragments)

**Mechanism.** When a fact lands in the top-K fact tier, its
`source_fragment_ids` get PROMOTED into the fragment tier ahead of
cosine-only candidates. The reader sees the gold turn directly, not
just the fact text. This is the exact structural answer to
`session_present_gold_absent`: the fact matches the cue; its source
turn is the gold turn; the back-pointer exists; use it.

**Why first.** Zero schema change — back-pointers already exist.
Pure retrieval-side lift. If this alone moves the needle, we have
budget to breathe.

**Measurement.** 3-instance (conv-26 + conv-30 + conv-41, ~74 Q)
F1 + recall-diagnostic buckets. Target: ≥ +0.05 F1 and
`session_present_gold_absent` incorrects drop ≥ 2.

**Kill criteria.** F1 lift < 0.03 AND no bucket migration → the
fact tier isn't catching gold-adjacent cues. Pivot to Step C
(entity graph) directly, with Step B as a probe only.

### Step B — Hybrid retrieval over facts (BM25 + dense + entity overlap, RRF)

**Mechanism.** Fact text is short (`"Caroline origin Sweden"`) and
cosine-only misses proper-noun matches when query uses the same
noun in a different syntactic frame. Add BM25 over fact text,
entity-overlap scoring (query entity → fact.entity match), fuse via
Reciprocal Rank Fusion. Mem0 production stack pattern.

**Why after A.** Fact-side signal diversification only matters if
the fact tier itself is load-bearing (established by A).

**Measurement.** Same as A. Target: ≥ +0.03 F1 on top of A, or
`gold_in_pool` incorrects drop ≥ 2 (lexical matches catching what
dense missed).

**Kill criteria.** Null-op within noise. Fallback: trust dense +
move to Step C.

### Step C — Entity co-occurrence graph + seed-expand

**Mechanism.** Build a graph over `Fact.entity` with edges weighted
by same-session co-occurrence. Seed with query-mentioned entities;
expand N hops (or PageRank with teleport) over the graph; lift any
fact mentioning expanded entities. HippoRAG / MRAgent pattern.

**Why after A+B.** Entity-level reasoning depends on fact structure
being retrievable at all (A+B). This step adds the "right session,
wrong turn" bridge: gold turn's entity may not match query
embedding, but IS reachable from query entity graph.

**Measurement.** 3-instance F1 + bucket. Target: ≥ +0.03 F1 OR
`session_present_gold_absent` incorrects drop ≥ 3.

**Kill criteria.** Lift < 0.02 AND no bucket migration → entity
extraction is too coarse. Pre-flight: dump facts across 3
instances, inspect entity vocabulary distribution. If entities
degenerate to "user" / "assistant" / "it", Step C is dead on
arrival; richer entity extraction prompt is a prerequisite.

### Step D — Episode-mention reranker

**Mechanism.** Count facts per entity per session. Boost facts
whose entities appear in multiple sessions ("important across
corpus"). Graphiti pattern (`episode_mentions`); Zep attributes
part of its 18.5% LongMemEval lift to it.

**Why after C.** Depends on multi-session entity index already
being built for C.

**Measurement.** 3-instance F1 + bucket. Target: ≥ +0.02 F1.

**Kill criteria.** Null-op. Keep configurable off-by-default,
revisit at month-horizon.

### Step E — Localize-filter LLM stage before reader

**Mechanism.** Between cross-encoder rerank and the final reader
prompt, insert a SIMPLE-tier LLM pass with a **recall-biased**
prompt: "Which of these fragments COULD support answering Q?
Include anything plausibly relevant." Output = filtered pool for
reader. EMem ablation: this stage alone is worth ~4.7 LoCoMo
points.

**Why after D.** Reader-side intervention; works even if A-D leave
`gold_in_pool` reader-precision failures.

**Measurement.** 3-instance F1 + bucket. Target: ≥ +0.03 F1 AND
`gold_in_pool` incorrects drop ≥ 2 (addresses v3 reader refusals
on `conv-30-q9` "Rome" and `conv-41-q49` "Cakes").

**Kill criteria.** Lift < 0.02. Fallback: Step F directly.

### Step F — Full-session reader context on winning session

**Mechanism.** Once localize-filter (or cross-encoder) identifies a
winning session, the reader sees that **whole session's turns** as
context, not just top-K fragments. Fragment-ranking is used to
*choose the session*, not to choose turns within it. Reader-
precision failures from truncated context (over-generalization,
count mismatches on `conv-41-q18`/`conv-41-q62`) resolve.

**Why after E.** Requires localize-filter to produce a confident
session choice. Cost: context window pressure — qwen-max 32K easily
holds a single ~5K-turn session; multi-session fallback for
multi-hop deferred.

**Measurement.** 3-instance F1 + bucket. Target: ≥ +0.03 F1,
`gold_in_pool` incorrects → 0 on the 3-instance set.

**Kill criteria.** Lift < 0.02 AND token cost 2× R5. Fallback:
hybrid — session context + top-K from other sessions.

### Step G (thesis) — Reconsolidation hook: version chain on framing divergence

**Mechanism.** `Fact.upsert` currently does full overwrite
([facts.py:131-141](../../../src/psychology_base/companion/facts.py#L131-L141)). Replace with append-to-version-chain
when the new fact's framing (affect vector + attribution)
diverges from the current head beyond threshold. Latest version
is default; full chain available on request. Direct read of
CLAUDE.md principle B (reconstruction) and the reconsolidation gap
Zep/Mem0 don't close.

**Why after A-F.** Needs retrieval already working so a
reconsolidated fact is actually reachable; needs facts carrying
affect/attribution tags (Phase 2 extraction prompt extension).

**Measurement.** B1 Narrative Reinterpretation Benchmark (not
LoCoMo — this is beyond-SOTA territory). Scripted therapy
sequence; chain preservation ≥ 95%, chain-reference quality ≥ 80%.

**Kill criteria.** Chain preservation < 90% or user-reported
confusion in pilot. Fallback: user-invited explicit versioning
only (no auto-detect).

### Step H (thesis) — Reconstructive answer assembly

**Mechanism.** Wire `reconstruction.py` R1+R2+R7 into the eval
reader path. The reader is explicitly prompted to **rebuild**
current meaning from fragments under current context, not retrieve
a stored answer. Intermediate reasoning step (chain-of-fragments)
before the final answer. CLAUDE.md principle B realized.

**Why last.** Depends on all retrieval infrastructure (A-F)
feeding high-quality fragments to reconstruct from; depends on
reconsolidation (G) for the "current context" to be meaningful.

**Measurement.** B2 Perspective-Taking Benchmark (specificity ≥
70%); 3-instance LoCoMo as regression floor (no F1 regression ≥
0.03 vs post-F stack).

**Kill criteria.** LoCoMo regression ≥ 0.05 → reconstructive
assembly adds noise, not signal. Keep behind a flag; disable in
the shipped stack; preserve for Phase 3.

## Application cross-check (CLAUDE.md non-negotiable)

Each step must enable a concrete learning-companion or long-term-
companion behavior. If "none specific," the step is deferred.

| Step | Learning companion | Emotional companion |
|---|---|---|
| A fact→turn lift | Prerequisite recall (concept X depends on prerequisite Y, surfaced via back-pointer) | Narrative continuity (user mentions "the Rome trip" → the actual Rome turn surfaces, not a paraphrase) |
| B hybrid fact retrieval | Named-concept queries ("tell me about derivatives") hit exact-concept facts | Named-person queries ("remember when I told you about Lin?") hit by lexical match |
| C entity graph | Concept-dependency graph (prereq / corequisite / builds-on) | Mentalistic graph foundation: `mother-as-user-represents-her` entity node with edges to episodes |
| D episode-mention | High-mention concepts = user's current focus curriculum | High-mention people = central figures in user's inner model |
| E localize-filter | Socratic question generation (recall-biased filter = "what might the user need?") | Clinical insight gate (recall-biased filter surfaces candidates for empathic callback) |
| F full-session reader | Context-aware tutoring (whole-lesson coherence) | Full-conversation empathic continuity (not snippet-only) |
| G reconsolidation | Concept mastery = version chain of "my understanding of X" | Narrative therapy = version chain of "how I describe this event" |
| H reconstructive assembly | Mastery check = can user reconstruct X from fragments | 共情 realized: AI rebuilds the user's mind-model fresh each turn |

## Expected trajectory

**Floor case** (A-F all deliver minimum thresholds): 0.544 + 0.05 +
0.03 + 0.03 + 0.02 + 0.03 + 0.03 = **0.734**. Clears 0.70 gate.

**Realistic case** (A, C, E, F fire; B, D null-op): 0.544 + 0.05 +
0.03 + 0.03 = **0.654**. Short of gate; triggers Step G/H early as
remediation.

**Floor case regresses on multi-hop / temporal / open-domain**:
run those categories at the end of each step and record. If
multi-hop regresses ≥ 0.05, blocker for shipping that step.

## Not in scope

- Late chunking (Jina pattern). Overlaps with Step F full-session
  reader. Revisit only if F stalls.
- RAPTOR tree. Overlaps with Step D + Step C summarization. Marginal
  on top of these; defer.
- Cross-encoder scaling (more layers, bigger reranker). Tuning, not
  structural. Already have gte-rerank-v2.
- Multi-hop retrieval expansion. Phase 1.5 gate is single-hop only;
  multi-hop is Phase 3.

## Decision log

- 2026-10-06: user chose BEST (not cheapest) after seeing research
  synthesis. Scope is the full A-H stack, not a single-lever MVP.
- 2026-10-06: steps ordered by dependency, not by expected lift.
  A/B are cheap foundation; C-F are the real SOTA floor work; G/H
  are thesis-only.
- 2026-10-07: **Step A ✗ regressed** (F1=0.486, −0.058). Root cause:
  `source_fragment_ids` is window-wide, not per-fact (extraction.py:270).
  Fix = Step A' (per-fact `source_turn_hint`, extraction prompt v2 +
  cache regeneration). Step A code shipped off-by-default; mechanism
  correct, provenance granularity wrong. See [step-A/README.md](step-A/README.md).
- 2026-10-07: **Step B ∅ null-op** (F1=0.554, +0.010 — within noise).
  `gold_in_pool` 5→2 but offset by `session_present_gold_absent`
  12→15; same questions wrong, pool composition shifted sideways.
  Confirms the ceiling is downstream (within-session turn ranking
  or reader precision), not fact-tier recall. Flag retained
  off-by-default. See [step-B/README.md](step-B/README.md).
- 2026-10-07: Re-order remaining steps. A/B were cheap probes; both
  established that the ceiling bucket (`session_present_gold_absent`)
  is not what A/B attack. Pivot sequence: **E (localize-filter)** and
  **F (full-session reader)** before C/D — E/F directly attack the
  measured bottleneck without requiring per-fact provenance.
  A' remains foundational for G/H but is scheduled after E/F.
- 2026-10-07: **Step F REPLACE ✗ regressed** (F1=0.497, −0.047). Top-1
  session often wrong — cross-encoder's confident top-1 reflects
  per-turn paraphrase match, not whole-session relevance. REPLACE drops
  correct cross-session evidence; target bucket did not migrate. Backed
  out in favor of ADDITIVE variant.
- 2026-10-07: **Step F ADDITIVE ∅ null-op within noise** (F1=0.557,
  +0.013 vs R5). Preserves reranked top-K + unions winning-session
  extras at tail. **First measured partial bucket resolution of Phase
  2b**: `session_present_gold_absent` 12→10 (−2), `gold_in_pool` 5→4
  (−1), total incorrect 17→14. Lift under 0.03 threshold so retained
  off-by-default per null-op protocol. 4/4 residual `gold_in_pool`
  failures (conv-30-q9 Rome, conv-41-q18 Rob, conv-41-q49 Cakes,
  conv-41-q62 "two") are reader-side (gold in pool with
  substring+coverage=1.0+session-in-top-K, reader refuses or
  over-generalizes). Motivates Step E (recall-biased LLM filter) as the
  next lever — three retrieval-side steps (A/B/F) now confirm the
  residual ceiling is reader-side, not retrieval-side. See
  [step-F/README.md](step-F/README.md).
- 2026-10-07: **Step E ∅ null-op in both configurations.** Isolation
  (E only, no widen) F1=0.537, −0.007 vs R5 — slight regression driven
  by precision gate on already-reranked pool; refusals 9.5% (highest
  Phase 2b; introduced new refusals on conv-26-q3 Adoption and
  conv-41-q40 Kyle Sara that R5 had correct). Pivoted pipeline to
  EMem widen-then-filter order (F ADDITIVE widens, then E filters).
  Stack F+E F1=0.551, +0.007 vs R5 — widen pass restored E's isolation
  losses (refusals 9.5%→6.8%, q3 Adoption back correct) but net ~0.006
  below F alone. **Same 4 residual gold_in_pool failures as F alone
  (Rome/Rob/Cakes/"two") — neither widening nor filtering touches the
  reader-side pathology.** Four sequential retrieval-side levers
  (A/B/F/E) have now each landed as regression or null-op. Both E
  flags retained off-by-default; shipped floor remains R5 (F1=0.544);
  Step F ADDITIVE remains single-lever best observed (F1=0.557).
  **Next lever: Step A' (per-fact `source_turn_hint`, foundational
  for G/H regardless) or reader-side intervention (prompt v4,
  few-shot grounding, forced-citation decode).** See
  [step-E/README.md](step-E/README.md).
