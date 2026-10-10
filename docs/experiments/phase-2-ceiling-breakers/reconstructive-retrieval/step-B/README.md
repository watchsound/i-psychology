# Step B — Hybrid fact retrieval (BM25 + cosine + entity-overlap, RRF)

## Hypothesis

Fact text is short (`"Caroline origin Sweden"`). Pure cosine over
fact embeddings misses proper-noun / named-entity cues that use
the same noun in a different syntactic frame. Adding BM25
(literal-token recall) and entity-overlap (cue tokens matching
`fact.entity` tokens), RRF-fused with cosine, should diversify
the fact tier so proper-noun cues reach it even when cosine
buries them. Mem0 production stack pattern. Target: lift the
`gold_in_pool` reader failures caused by the right fact
not reaching the FACTS block in the first place.

## Config

- `--fact-hybrid-retrieval` flag added to `CompanionSystem` + gate CLI.
- Channels: BM25 over `embed_text_for(fact)` + dense cosine over
  fact embedding + entity-overlap on `fact.entity` tokens.
- RRF k = 60 (standard literature value).
- Entity-overlap token min length = 2 (drops single-char junk).
- Stacked on R5 shipped baseline (consolidation +
  cross-encoder-rerank + reader prompt v3 + judge-mode).
- Step A `fact_source_turn_lift` OFF (Step A regressed; measured
  independently against R5).

## Result

**F1 = 0.554 across 3 instances (74 Q, wall 262 s)** vs R5 baseline
0.544. **Δ = +0.010.** Within noise band (0.03). **Null-op.**

### Per-instance

| Instance | Q | Step B F1 | R5 F1 | Δ |
|---|---|---|---|---|
| conv-26 | 32 | 0.484 | 0.469 | +0.015 |
| conv-30 | 11 | 0.636 | 0.545 | +0.091 |
| conv-41 | 31 | 0.597 | 0.621 | −0.024 |
| **Aggregate** | **74** | **0.554** | **0.544** | **+0.010** |

conv-30 shows a legitimate +0.091 lift — the 11-question instance
is small enough that a 1-question swing is +0.091, so the signal is
inside its own noise band. conv-26 and conv-41 land within noise.

### Recall diagnostic

| Bucket | Step B | R5 | Δ |
|---|---|---|---|
| Incorrect total | 17/74 | 17/74 | 0 |
| `gold_in_pool` | 2 | 5 | **−3** |
| `session_present_gold_absent` | 15 | 12 | **+3** |
| `recall_missing` | 0 | 0 | 0 |
| `no_evidence_meta` | 0 | 0 | 0 |

Reader refusals held steady (6/74, 8.1% vs R5 7/74, 9.5%). The
`gold_in_pool` drop looks like the stated target met (plan called
for ≥ 2 drop) — but the drop is **100% offset** by a matching rise
in `session_present_gold_absent`. Total incorrect is identical:
*the same questions are still wrong, just with a different pool
composition.*

What's happening: the three new channels promote different facts
into top-150. For 3 rows previously classified `gold_in_pool` (the
gold text was in the pool, reader missed it), the new pool
composition drops the gold text → the row reclassifies to
`session_present_gold_absent`. The reader doesn't start getting
these right; the diagnostic bucket just slides sideways.

## Verdict — ∅ NULL-OP (within noise, reclassification only)

Kill criteria from the plan:
> Null-op within noise. Fallback: trust dense + move to Step C.

F1 lift +0.010 is well within the 0.03 noise band. Bucket "drop"
is reclassification, not resolution. Total incorrect unchanged.
**Decision: back out as default; retain flag off-by-default as
reproducible negative result.**

## Why Step B failed to break through

The target bucket on R5 was `session_present_gold_absent` =
12/17 incorrect (71%). This is a *within-session fragment-ranking*
failure, not a *fact-tier recall* failure. The fact tier was
already surfacing the right facts for most of these rows — the
problem is that the specific gold turn ranks past the top-50
reranked fragment pool within its session. No amount of fact-tier
diversification helps when the bottleneck is downstream.

The 3-channel RRF *does* diversify fact-tier recall (gold_in_pool
drops), it just redistributes the ceiling into a different bucket
rather than lifting the ceiling. This confirms the Step A diagnosis:
**the ceiling is not fact-tier recall; it's the gap between fact and
turn provenance.** Step B is a lateral move; the real fix is still
Step A' (per-fact `source_turn_hint`) or Step E (localize-filter) or
Step F (full-session reader).

## Decision — null-op, keep off-by-default, proceed to A' or E

Options considered:

1. **Null-op; keep flag retained** — document, proceed to the next
   independent lever. The hybrid-fact code is harmless at default
   off; retains optionality if future steps change the equation.
2. **Delete the code** — cheaper to maintain. But the mechanism is
   correct and the Mem0 production pattern cites it as load-bearing;
   deleting is throwing away a lever that may fire differently once
   extraction has per-fact provenance (Step A').
3. **Tune RRF k / channel weights** — no evidence the shape is
   wrong; the signal just hits a downstream bottleneck. Tuning
   lipstick won't close the structural gap.

**Chosen:** option 1. Flag `fact_hybrid_retrieval` retained
off-by-default; mechanism is sound, signal-to-noise is capped by
the downstream bottleneck Step B isn't attacking.

## Shipped state

- `fact_hybrid_retrieval` flag on `CompanionSystem` + gate CLI,
  default `False`.
- 8 unit tests in [test_companion_fact_hybrid_retrieval.py](../../../../../tests/test_companion_fact_hybrid_retrieval.py)
  cover: requires-consolidation, defaults-off, BM25-only match,
  entity-overlap match, index rebuild on fact-count change, empty
  fact store, cosine-floor gate, stacks with Step A.
- R5 remains the shipped floor.

## Implied next moves

Per the composed plan dependency order (A/B → C → D → E → F → G/H):

1. **Step A' — per-fact `source_turn_hint`** (extraction prompt v2).
   Rebuild fact caches (~97 min wall × 3 instances). Unblocks Steps
   C/D/G which assume per-fact provenance. This is the foundational
   schema change.
2. **Step E — localize-filter LLM stage** can run independent of
   A'. Targets `gold_in_pool` (reader-precision) directly rather
   than fact-tier recall. Lower cost than A' (no cache regeneration;
   SIMPLE-tier LLM call per recall).
3. **Step F — full-session reader context** also independent of A'.
   Highest expected lift on `session_present_gold_absent` (the
   actual ceiling bucket): if the right session is in the pool and
   the whole session is in the reader context, the gold turn is
   guaranteed to be visible to the reader.

Recommendation: Step E or F before A', since both attack the
measured bottleneck directly and don't require 5h of cache
regeneration. A' remains foundational for G/H but can be scheduled
once a working E/F lands.
