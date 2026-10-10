# Step H — Reconstructive assembly (R1 spreading-activation retrieval)

## Hypothesis

Seven sequential Phase 2b levers (Steps A / B / F / E, reader-prompt
v4, reader-evidence rerank-first, Step C entity graph) have each
landed as regression or null-op. The retrieval-side ceiling is
structural: every widening tier adds signal AND noise in near-equal
measure because the reader can't disambiguate added distractors
without a corresponding precision signal.

Hypothesis: wire the R1 spreading-activation primitive from
[reconstruction.py](../../../src/psychology_base/companion/reconstruction.py)
into the LoCoMo eval path as a NEW recall channel. R1 ranks fragments
by activation profile over the Jaccard-seeded `EdgeStore` (built at
observe-time, not from embeddings). This is a *structural* recall
signal, co-occurrence-based, orthogonal to every embedder / BM25 /
entity-overlap tier tried so far. If any single-lever lift exists on
the current substrate, a channel built from a fundamentally different
information source is the strongest candidate.

Thesis alignment: this wires the reconstructive-memory primitives into
the application eval path, which the architectural plan calls out as
the "thesis-only lever (reconstructive assembly) carries past the
Mem0/Zep ceiling into the consciousness-simulation thesis." Measures
re-measurement of LoCoMo F1 under reconstructive assembly rather than
retrieve-then-compose.

## Config

- `reconstructive_retrieval: bool = False` + `reconstructive_cap=20`,
  `reconstructive_hops=2`, `reconstructive_hop_decay=0.5` added to
  `CompanionSystem`. R1 runs on a one-shot local `ActivationState` so
  the shipped flat-fallback path's `self.activation` is NEVER
  double-mutated (would inflate recall_count and break V-metric
  invariants).
- CLI flag `--reconstructive-retrieval`.
- Does NOT require consolidation (independent of fact-store).
- Boost tier dedup'd against cosine / substring / source_lift /
  entity_boost. In legacy ordering, inserted between entity_boost and
  cross-encoder-reranked. In rerank-first ordering, after
  source_lift / entity_boost and before facts' window-sources.
- Stacked on R5 shipped baseline (consolidation + rerank + reader
  prompt v3 + judge-mode). All other Phase 2b flags OFF for isolation.

## Result — ✗ regressed

- **F1 = 0.490** (−0.054 vs R5 0.544)
- 3-instance (74 Q, wall 354.2 s, cache hits 65 / misses 9):
  conv-26 0.406, conv-30 0.409, conv-41 0.605
- Reader refusal rate 12.2% (9/74), same as Step C
- Diagnostic: **23 incorrect (+6 vs R5)**, 8 `gold_in_pool` (+3),
  **15 `session_present_gold_absent` (+3)**, 0 `recall_missing`

### Per-instance

| Instance | Questions | F1 (Step H) | F1 (R5) | Δ |
|---|---|---|---|---|
| conv-26 | 32 | 0.406 | 0.469 | **−0.063** |
| conv-30 | 11 | 0.409 | 0.545 | **−0.136** |
| conv-41 | 31 | 0.605 | 0.621 | −0.016 |
| **Aggregate** | **74** | **0.490** | 0.544 | **−0.054** |

Every instance regressed. conv-30 took the biggest hit (same as Step
C — small instance = each flipped row swings aggregate ~0.09). conv-41
only edged below R5 (within judge noise) suggesting R1-activation on
that corpus didn't actively hurt — it just didn't help.

### Target-bucket migration — opposite direction

| Bucket | R5 | Step H | Δ |
|---|---|---|---|
| `gold_in_pool` | 5 | 8 | **+3** |
| `session_present_gold_absent` | 12 | 15 | **+3** |
| `recall_missing` | 0 | 0 | 0 |
| **Total incorrect** | **17** | **23** | **+6** |

Both diagnostic buckets moved in the *wrong* direction. The
reconstructive-boost tier pulled fragments into the pool that
displaced gold fragments from the reader's top-50 formatted view
AND added distractors the reader couldn't disambiguate.

### Diagnosis — why R1 regresses here

The R1 primitive was validated in the Ouroboros prior-art context
where edges carry **months of Hebbian + Witness co-activation
updates** that reshape the structural signal. In this LoCoMo
single-session gate:

- `EdgeStore` is populated at `observe()` time with Jaccard token-
  overlap priors only (see `system.py:681` — `jaccard_prior` seed).
- No Hebbian updates during this gate (gates do observe → recall; the
  Hebbian bump path requires `tick()` cycles between observations).
- Spread-activation over Jaccard-only edges therefore degenerates to
  **a quadratic-cost token-overlap ranker**, with 2-hop fan-out
  amplifying the token-centroid signal.

In Ouroboros's ep20 context this co-occurrence signal is informative
because edges reflect repeated co-activation across months. On a
single-pass benchmark it reduces to "fragments with lots of common
words with the cue, through 2-hop paths of word-shared peers" —
which is literally what cosine already does, but noisier.

### Row-level pattern

- **gold_in_pool +3 and session_present_gold_absent +3**: R1
  activation surfaces token-overlap-dense turns, which crowds out
  precision-ranked turns. Fragments previously in the top-50 formatted
  view fall out (because R1-tier appends before fragment_hits in
  legacy ordering), and the reader sees a muddier pool.
- Expected under the hypothesis "R1 only helps when edges carry
  history." On first-pass data the mechanism is correct but the
  signal is uninformative.

## Verdict — ✗ regressed, keep flag off-by-default

**R1 spreading-activation over Jaccard-seeded edges regresses F1 by
−0.054** and inflates BOTH diagnostic buckets. The mechanism is
*structurally* orthogonal to prior Phase 2b levers but the signal is
not informative on single-pass LoCoMo data because edges carry no
recall-history Hebbian updates.

Flag retained off-by-default as a reproducible negative-space entry.
Shipped floor remains R5 (F1=0.544).

### What this falsifies and what it leaves open

Falsified: "a structural recall channel orthogonal to embedder signals
lifts F1 on single-pass benchmarks" — on this substrate, no. The
structural signal needs recall-history reinforcement that LoCoMo's
gate protocol doesn't produce.

Open: whether R1 would lift F1 on **multi-session** gate runs where
Hebbian co-activation accumulates (B2 perspective-taking benchmark,
B3 affective continuity). This is a thesis-aligned measurement that
the current LoCoMo single-hop gate cannot evaluate.

## Shipped state

- `reconstructive_retrieval` flag on `CompanionSystem` + gate CLI,
  default `False`.
- `spread_activation` import from `reconstruction.py` (no edits to
  the primitive — it's reused as-is per the thesis directive "study
  prior art for principles, don't port code").
- 4 unit tests on the flag + state-isolation + default-off behavior
  in [tests/test_companion_reconstructive_retrieval.py](../../../tests/test_companion_reconstructive_retrieval.py);
  344 full companion test suite passing.
- R5 remains the shipped floor.

## Artefact

- [step-h-first3.json](step-h-first3.json) — full gate report.
- [step-h-first3.log](step-h-first3.log) — stdout from the run.

## CLI reproduction

```
python docs/experiments/phase-1-benchmark-adapters/option-gamma/run_locomo_gate.py \
  --consolidation --cross-encoder-rerank \
  --reader-prompt-variant v3 --judge-single-hop \
  --mode rag --instance-limit 3 \
  --reconstructive-retrieval \
  --out docs/experiments/phase-2-ceiling-breakers/reconstructive-assembly/step-h-first3.json
```
