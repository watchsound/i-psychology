# Step F — Full-session reader context on winning session

## Hypothesis

The 71% `session_present_gold_absent` ceiling (R5 3-instance:
12/17 incorrect) is a *within-session fragment-ranking* failure:
the right session reaches cross-encoder top-50 but the specific
gold turn ranks past that cap within its session. If the top-1
reranked fragment's **whole session** becomes the reader's
DIALOGUE pool — chronologically ordered — the gold turn is
guaranteed visible to the reader even when its within-session
rank is past 150.

Two variants explored (2026-10-07):

- **REPLACE**: replace `fragment_hits` with the top-1 session's
  turn set. Pure session-expansion; cross-session reranked
  evidence is dropped.
- **ADDITIVE**: keep every reranked top-K hit, UNION in every
  un-reranked turn of the top-1 session at the tail. Reader sees
  strictly more context than R5.

## REPLACE variant — ✗ REGRESSED, backed out

**F1 = 0.497 across 3 instances (74 Q)** vs R5 baseline 0.544.
**Δ = −0.047.** Clear regression beyond noise.

### Diagnosis

Top-1 session is often not the right session. The cross-encoder's
confident top-1 score reflects paraphrase match on a single turn,
not whole-session relevance. REPLACE drops correct cross-session
rerank evidence; the gold turn ends up in the wrong session's
expanded pool. `session_present_gold_absent` **did not migrate**
(12 → 12).

Decision: back out REPLACE; try ADDITIVE.

## ADDITIVE variant — ∅ NULL-OP (within noise)

**F1 = 0.557 across 3 instances (74 Q, wall 307 s)** vs R5 baseline
0.544. **Δ = +0.013.** Within noise band (0.03). **Null-op.**

### Per-instance

| Instance | Q | Step F ADDITIVE F1 | R5 F1 | Δ |
|---|---|---|---|---|
| conv-26 | 32 | 0.484 | 0.469 | +0.015 |
| conv-30 | 11 | 0.545 | 0.545 | 0.000 |
| conv-41 | 31 | 0.637 | 0.621 | +0.016 |
| **Aggregate** | **74** | **0.557** | **0.544** | **+0.013** |

Consistent small positive signal across instances — no single
instance drives the aggregate. Below the 0.03 noise threshold.

### Recall diagnostic

| Bucket | Step F ADDITIVE | R5 | Δ |
|---|---|---|---|
| Incorrect total | 14/74 | 17/74 | **−3** |
| `gold_in_pool` | 4 | 5 | −1 |
| `session_present_gold_absent` | 10 | 12 | **−2** |
| `recall_missing` | 0 | 0 | 0 |
| `no_evidence_meta` | 0 | 0 | 0 |

Both target buckets moved in the right direction — the strongest
bucket signal from any Phase 2b step so far. Three rows flipped
incorrect → correct. The target bucket
(`session_present_gold_absent`) dropped from 12 → 10 (−2), the
first measured partial resolution rather than reclassification.

### Residual `gold_in_pool` incorrect rows (4)

| Instance | Q | Gold | Candidate | Failure mode |
|---|---|---|---|---|
| conv-30 | q9 | Rome | not stated | Reader refusal (gold in pool, substring present, token coverage 1.0) |
| conv-41 | q18 | Rob | a colleague | Reader over-generalization |
| conv-41 | q49 | Cakes | not stated | Reader refusal |
| conv-41 | q62 | two | 1 | Reader count mismatch |

All 4 are reader-side failures: gold is present in the pool
(`gold_substring_in_top_k=True`, `gold_token_coverage=1.0`,
`evidence_session_in_top_k=True`) but the reader refuses or
over-generalizes. This is the exact failure mode Step E
(localize-filter, recall-biased) is designed to address.

## Verdict — ∅ NULL-OP, retain off-by-default

Kill criteria from the plan:
> Lift < 0.02 AND token cost 2× R5. Fallback: hybrid — session
> context + top-K from other sessions.

The ADDITIVE variant IS the hybrid fallback (session context +
reranked top-K from other sessions). It landed at +0.013 lift
— below the 0.02 threshold. Bucket signal is present but F1 did
not clear the 2× noise bar.

**Decision: retain flag `full_session_reader` off-by-default;
mechanism is sound and showed the first partial bucket
resolution of Phase 2b, but the resolution isn't large enough to
ship on its own.** Three sequential steps (A/B/F) have now
confirmed the ceiling is deeply resistant to any single lever on
the retrieval side.

## Why Step F didn't fully break through

The 4 residual `gold_in_pool` failures are READER-SIDE. The pool
already has:
- gold-containing turn (substring present, coverage 1.0)
- correct session (evidence session in top-K)
- full session chronology (Step F adds the un-reranked turns)

...and the reader still refuses or over-generalizes. More
retrieval-side context doesn't help when the reader's precision
prior overrides a visible answer. The next lever is
**reader-upstream recall-biasing** (Step E), not more retrieval
widening.

## Shipped state

- `full_session_reader` flag on `CompanionSystem` + gate CLI,
  default `False`.
- `full_session_reader_cap_turns = 300` (safety cap; typical
  LoCoMo session is 20-100 turns so cap rarely binds).
- 10 unit tests in [test_companion_full_session_reader.py](../../../../../tests/test_companion_full_session_reader.py)
  cover: requires-rerank, mutex with within-session-rerank,
  defaults-off, preserve-reranked + add-session-extras, order
  (reranked first then chronological), score reuse, cap
  binding, cap-zero-budget still keeps reranked, empty input,
  untagged winning fragment.
- Mutex retained with `within_session_rerank` (both widen the
  pool via different strategies).
- R5 remains the shipped floor.

## Implied next move

Three retrieval-side steps (A/B/F) have landed at: regress /
null-op / null-op-with-partial-bucket-signal. The residual
failures are now dominated by reader-side issues (4/4 `gold_in_pool`
failures above are reader refusals or over-generalizations with
gold visible in the pool).

**Proceed to Step E — localize-filter LLM stage.** Step E
inserts a recall-biased SIMPLE-tier LLM filter between
cross-encoder rerank and the reader. Target: reduce reader
refusals / over-generalizations on the 4 residual `gold_in_pool`
rows. EMem reports ~4.7 LoCoMo points for this stage alone;
plausibly closes the current 0.013-above-R5 gap AND the 4
residual rows.

If Step E stacks positively with Step F ADDITIVE (the pool
widening + the filter re-prioritization), that's the first
candidate stack that might clear the 0.70 gate.
