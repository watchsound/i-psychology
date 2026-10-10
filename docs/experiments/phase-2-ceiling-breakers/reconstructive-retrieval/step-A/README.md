# Step A — Fact-to-turn lift (back-pointer promotion)

## Hypothesis

`Fact.source_fragment_ids` back-pointers already exist on every fact
(facts.py:84), stored but not used for retrieval lift. Prepending the
top-N facts' source turns to the DIALOGUE block ahead of cosine-ranked
turns should surface gold turns the within-session cosine ranker buries
— directly attacking the 71% `session_present_gold_absent` ceiling.

## Config

- `--fact-source-turn-lift` flag added to `CompanionSystem` + gate CLI.
- Top-N fact hits = 15.
- Fragment lift cap = 30.
- Stacked on R5 shipped baseline (consolidation + cross-encoder-rerank
  + reader prompt v3 + judge-mode).

## Result

**F1 = 0.486 across 3 instances (74 Q, wall 351 s)** vs R5 baseline
0.544. **Regressed by 0.058.**

### Per-instance

| Instance | Q | Step A F1 | R5 F1 | Δ |
|---|---|---|---|---|
| conv-26 | 32 | 0.422 | 0.469 | −0.047 |
| conv-30 | 11 | 0.409 | 0.545 | **−0.136** |
| conv-41 | 31 | 0.581 | 0.621 | −0.040 |
| **Aggregate** | **74** | **0.486** | **0.544** | **−0.058** |

### Recall diagnostic

| Bucket | Step A | R5 | Δ |
|---|---|---|---|
| Incorrect total | 20/74 | 17/74 | +3 |
| `gold_in_pool` | 8 | 5 | **+3** |
| `session_present_gold_absent` | 12 | 12 | 0 |
| `recall_missing` | 0 | 0 | 0 |

Reader refusals held steady (7/74 vs ~7/74 R5). Target bucket
`session_present_gold_absent` did **not** drop. The regression came
entirely from new `gold_in_pool` incorrect rows — the gold turns are
now reaching the DIALOGUE pool but the reader is failing to extract
them, likely because the pool is being diluted by non-gold back-
pointer turns.

## Verdict — ✗ FAIL (regression, not null-op)

Kill criteria from the plan: F1 lift < 0.03 AND no bucket migration.
Observed: F1 lift = **−0.058**, bucket migration = **adverse**
(+3 `gold_in_pool`). Decisively fails.

## Root-cause diagnosis

The extractor at [extraction.py:270](../../../../../src/psychology_base/companion/extraction.py#L270)
sets `source_ids = tuple(f.id for f in window)` — **every fact
extracted from a window gets the ENTIRE window as its
`source_fragment_ids`**, not the specific turn it came from. A window
of 10 turns producing 5 facts gives all 5 facts the same list of 10
turn IDs. There is no per-fact provenance in the current data model.

So Step A, when it lifts "the source turns for the top 15 fact hits",
is really lifting *the entire extraction windows those facts came
from* — ~30 arbitrary turns from potentially 15 different windows.
Prepending these to DIALOGUE ahead of the cross-encoder-reranked top
turns displaces the actual gold from the reader's context.

The back-pointer field was designed in Phase 1.5 as a book-keeping
artifact for audit, not as a retrieval primitive. Step A assumed
per-fact granularity that the extraction prompt does not produce.

## Decision — back out + redesign extraction before retrying

Options considered:

1. **Narrow the lift** — only lift first source ID per fact, or gate
   by cue cosine. Cheap, but still rides window-level noise; expected
   null-op at best.
2. **Fix extraction** (Step A') — change the extraction prompt to
   require `source_turn_hint` per fact; regenerate fact cache (~97
   min fresh ingestion × 3 instances). Produces per-fact provenance,
   makes Step A sound.
3. **Skip to Step B** — hybrid fact retrieval (BM25 + dense + entity
   overlap, RRF). Doesn't depend on back-pointer granularity.

**Chosen:** option 2 (fix extraction with per-fact `source_turn_hint`)
in parallel with option 3 (Step B as an independent lever). Per-fact
provenance is foundational to the whole reconstructive-retrieval
thesis — Steps C/D/G all assume it. Fixing it now unblocks the rest
of the stack. Step B is independent and can run in parallel.

Step A code shipped off-by-default. The flag + CLI + tests remain as
the mechanism; only its signal-to-noise ratio is bad on the current
extraction schema.

## Shipped state

- `fact_source_turn_lift` flag retained on `CompanionSystem` + gate
  CLI, default `False`. Not deleted — becomes correct once extraction
  provides per-fact provenance (planned Step A').
- 8 unit tests in [test_companion_fact_source_lift.py](../../../../../tests/test_companion_fact_source_lift.py)
  verify the lift mechanism works mechanically; a schema change to
  per-fact source IDs will not require re-writing them (they operate
  on `Fact.source_fragment_ids` which is schema-stable).
- R5 remains the shipped floor.

## Implied next moves

1. **Step A' — extraction prompt v2 with per-fact `source_turn_hint`.**
   The extractor emits one line per fact; add a 6th field pointing
   at the specific turn id in the window. Fresh fact cache run.
2. **Step B — hybrid fact retrieval** can proceed in parallel; it
   does not depend on per-fact provenance.
3. **Audit the Step A implementation** once Step A' lands — the
   mechanism should produce +0.03–0.05 F1 and ≥2 bucket drops under
   per-fact granularity.
