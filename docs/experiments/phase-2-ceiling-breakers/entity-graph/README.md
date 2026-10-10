# Step C — Entity-graph fragment lift

## Hypothesis

Six sequential Phase 2b levers (Step A fact-to-turn lift, Step B hybrid
fact retrieval, Step F ADDITIVE full-session reader, Step E localize-
filter, reader-prompt v4, reader-evidence rerank-first) landed as
regression or null-op. The dominant failure bucket on R5 is
`session_present_gold_absent` (12/17 incorrect, 71%): the right session
is picked by cross-encoder rerank but the specific gold turn ranks past
the reader's top-50 cut-off.

Hypothesis: a fact-derived **entity graph** gives a structural recall
channel orthogonal to cosine/BM25/paraphrase signals. Cue entities →
1-hop graph neighbors → fragments mentioning any expanded entity become
a bounded top-N tier appended to the fragment pool. This should surface
turns that mention cue-relevant entities regardless of embedder rank,
attacking `session_present_gold_absent` without touching per-fact
provenance (independent of Steps A / A').

## Config

- New module [entity_graph.py](../../../src/psychology_base/companion/entity_graph.py)
  with `EntityGraph`, `build_entity_graph`, `score_fragments`.
- `entity_graph_retrieval: bool = False` + `entity_graph_cap=20`,
  `entity_graph_seed_cap=10`, `entity_graph_min_hits=1` added to
  `CompanionSystem`.
- CLI flag `--entity-graph-retrieval`.
- Lazy build on first recall; rebuild when `len(fact_store)` changes
  (same pattern as Step B's fact BM25 index).
- Boost tier dedup'd against cosine / substring / source_lift so no
  double emission. In legacy ordering, inserted between substring and
  cross-encoder-reranked fragments. In rerank-first ordering, after
  source_lift and before facts' window-sources.
- Stacked on R5 shipped baseline (consolidation + rerank + reader
  prompt v3 + judge-mode). All other Phase 2b flags OFF for isolation.

## Result — ∅ null-op

- **F1 = 0.547** (+0.003 vs R5 0.544) — within noise band (0.03)
- 3-instance (74 Q, wall 338.8 s, cache hits 61 / misses 13):
  conv-26 0.484, conv-30 0.409, conv-41 0.661
- Reader refusal rate **12.2% (9/74)**, up from R5 8.1%
- Diagnostic: 17 incorrect, 6 `gold_in_pool` (+1 vs R5's 5),
  **11 `session_present_gold_absent` (−1 vs R5's 12)**, 0
  `recall_missing`

### Per-instance

| Instance | Questions | F1 (Step C) | F1 (R5) | Δ |
|---|---|---|---|---|
| conv-26 | 32 | 0.484 | 0.469 | +0.015 |
| conv-30 | 11 | 0.409 | 0.545 | **−0.136** |
| conv-41 | 31 | 0.661 | 0.621 | +0.040 |
| **Aggregate** | **74** | **0.547** | 0.544 | +0.003 |

- conv-30 regressed heavily (−0.136 on 11 Q = 1-2 row flips) —
  likely entity-boost tier polluted the pool with entity-dense but
  gold-irrelevant turns, pushing a previously-correct answer into
  refusal.
- conv-41 lifted (+0.040) — the mechanism works on the instance where
  entity-expansion matches question structure.
- conv-26 marginal (+0.015) — within single-roll judge noise.

### Target-bucket migration

| Bucket | R5 | Step C | Δ |
|---|---|---|---|
| `gold_in_pool` | 5 | 6 | +1 |
| `session_present_gold_absent` | 12 | 11 | **−1** |
| `recall_missing` | 0 | 0 | 0 |
| **Total incorrect** | **17** | **17** | 0 |

The mechanism IS doing targeted work (−1 on the attacked bucket) but
the lift is offset 1-for-1 by a new `gold_in_pool` reader-side failure
AND +4 refusals net — the entity-boost tier shifts pool composition
without a corresponding F1 lift because the reader can't disambiguate
the added entity-dense distractors.

### Row-level pattern

- **conv-30 regression** (R5 0.545 → Step C 0.409, 1-2 row swap):
  consistent with the pool-pollution hypothesis — conv-30 is a small
  instance (11 Q) where a single flipped row swings aggregate F1 by
  0.09.
- Target bucket migration (session_present_gold_absent 12→11) means
  one previously-residual case recovered — not visible in F1 due to
  compensating losses elsewhere.

## Verdict — ∅ null-op, keep flag off-by-default

**F1 lift +0.003 is within noise.** Target bucket moved by 1 row but
reader-side refusals jumped (8.1% → 12.2%) washing out the gain.
Mechanism is doing *targeted* work (bucket migration confirms) but
pool-pollution offset is complete.

Flag retained off-by-default as a reproducible negative-space entry.
Shipped floor remains R5 (F1=0.544).

### Why this landed null-op (not a win)

- Entity-expansion widens the pool with turns mentioning
  cue-relevant entities. In the attacked `session_present_gold_absent`
  bucket, this surfaces the gold turn. But the same expansion also
  surfaces entity-dense distractors in nearby sessions.
- The reader (v3) handles added distractors by refusing more often,
  not by picking the gold more precisely. +4 refusals (8.1% → 12.2%).
- 1 row recovered on `session_present_gold_absent` ↔ 1 row lost on
  `gold_in_pool` + refusal budget → net F1 flat.

### What this leaves on the table

Seven sequential Phase 2b levers (A / B / F / E / v4 / rerank-first /
C) have each landed as regression or null-op on the shipped substrate.
`session_present_gold_absent` has now dropped from R5 12 → F ADDITIVE
10 → Step C 11 but F1 is stuck at 0.544-0.557. The reader-side
pollution offset is structural: every widening tier adds signal AND
noise in near-equal measure.

Remaining candidates:

- **Step H — Reconstructive assembly** (R1 spreading-activation over
  Jaccard-seeded EdgeStore). Thesis-aligned structural recall path
  built from co-occurrence, orthogonal to all embedder-based signals.
  Running next.
- **Step A'** — extraction prompt v2 + per-fact `source_turn_hint`
  + fact-cache regen (~15h wall). Foundational for G/H but no longer
  blocks forward progress.

## Shipped state

- `entity_graph_retrieval` flag on `CompanionSystem` + gate CLI,
  default `False`.
- Full `_recall_hierarchical` + `_hybrid_rank_facts` interaction
  tested; 13 unit tests on `entity_graph.py` passing + 344 full
  companion test suite passing.
- R5 remains the shipped floor.

## Artefact

- [step-c-first3.json](step-c-first3.json) — full gate report.
- [step-c-first3.log](step-c-first3.log) — stdout from the run.

## CLI reproduction

```
python docs/experiments/phase-1-benchmark-adapters/option-gamma/run_locomo_gate.py \
  --consolidation --cross-encoder-rerank \
  --reader-prompt-variant v3 --judge-single-hop \
  --mode rag --instance-limit 3 \
  --entity-graph-retrieval \
  --out docs/experiments/phase-2-ceiling-breakers/entity-graph/step-c-first3.json
```
