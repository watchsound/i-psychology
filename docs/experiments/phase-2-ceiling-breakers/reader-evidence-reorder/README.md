# Reader-evidence reorder — rerank-first ordering (Phase 2b plumbing)

## Hypothesis

Five sequential Phase 2b levers (Step A fact-to-turn lift, Step B
hybrid fact retrieval, Step F ADDITIVE full-session reader, Step E
localize-filter, reader-prompt v4 few-shot) landed as regression or
null-op. Three failures (conv-30-q9 Rome, conv-41-q49 Cakes,
conv-41-q62 "two") are **prompt-invariant** — same wrong output
across v3/v4/F/F+E. They all have `gold_substring_in_top_k=True` and
`gold_token_coverage=1.0` yet the reader can't extract gold.

Code audit revealed a structural ordering issue in
`_hierarchical_recall`:

- `fragments_used` (what the reader's `_format_fragments(limit=50)`
  surfaces as the formatted list) is built with **facts' window-wide
  `source_fragment_ids` FIRST** (positions 0-150+), then substring,
  then cross-encoder-reranked top-K at the TAIL.
- Each fact's `source_fragment_ids` is a **~30-turn extraction
  window** (per [extraction.py:269](../../../src/psychology_base/companion/extraction.py#L269)).
- With 150 facts × 30-turn windows, after dedup there are 200+
  fragments in `fragments_used`, and the reranked top-K sits in
  positions 150+ — **below the reader's top-50 formatted-list cut-off**.
- DIALOGUE text order mirrors this: substring + reranked at the
  bottom of a long concatenated block.

**Hypothesis**: the reader is lost-in-the-middle. The highest-precision
reranked evidence is where attention is weakest. Reordering so reranked
top-K comes FIRST should restore grounding on the 3 invariant failures.

## Config

- `reader_evidence_rerank_first: bool` added to `CompanionSystem`
  (default False — R5 behavior unchanged).
- When True, `_hierarchical_recall` reorders both DIALOGUE and
  `fragments_used` to: fragment_hits (reranked) → substring →
  source_lift → facts' window-sources (dedup preserved).
- CLI flag `--reader-evidence-rerank-first` on gate runner.
- Stacked on R5 shipped baseline (consolidation + rerank + reader
  prompt v3 + judge-mode). F/E flags OFF for isolation.

## Result — ✗ regressed

- **F1 = 0.510** (−0.034 vs R5 0.544)
- 3-instance (74 Q, wall 480.8 s): conv-26 0.406, conv-30 0.500,
  conv-41 0.621
- Reader refusal rate **18.9% (14/74)** — more than DOUBLE R5
  (8.1%); highest refusal rate observed in Phase 2b
- Diagnostic: 21 incorrect (16 session_present_gold_absent, 5
  gold_in_pool, 0 recall_missing)
- Target bucket: `session_present_gold_absent` 12→16 (+4),
  `gold_in_pool` 5→5 (unchanged)

### Row-level pattern

**Fixed under rerank-first** (2 of 4 reader-side residuals):
- conv-41-q18 Rob (was "a colleague") → correct — specific-name
  with reranked turn at top of context
- conv-41-q62 "two" (was "1") → correct — integer extraction
  worked when reranked turn was at top

**Still broken under all variants** (prompt-invariant AND
evidence-order-invariant):
- conv-30-q9 Rome → "not stated"
- conv-41-q49 Cakes → "not stated"

**Broken BY rerank-first** (regressed from R5-correct → refusal):
conv-26-q3 Adoption, q34 Mentoring, q37 sunset, q51 Horse, q61
Matt Patterson, q78 Figurines, q6 homeless shelter — 7 refusals
on rows that previously extracted correctly.

### Diagnosis

The reader apparently needs the facts' window-context as narrative
scaffolding to extract confidently. Reranked top-K turns are often
short/atomic/decontextualized — great precision signal but weak
narrative frame. When presented first (with facts' window-sources
pushed below the formatted-list cut-off), the reader refuses on
previously-answerable rows because it lacks conversational context
to anchor the extraction.

The current ordering (facts' window-sources first → substring →
reranked last) was likely intuitive-but-structurally-wrong on paper,
but functionally correct in practice: the window-sources are
redundant with DIALOGUE but their *presence first in the formatted
list* gives the reader implicit conversation flow that enables
precise extraction when it does find gold.

## Verdict — ✗ regressed, keep flag off-by-default

**Reordering regresses −0.034 F1** and jumps refusal rate from 8.1% to
18.9%. Hypothesis "reranked-first fixes lost-in-the-middle" is
falsified: the lost-in-the-middle effect is real on *gold* but the
window-context *also* provides essential narrative scaffolding. Can't
reorder in isolation.

Shipped floor remains R5 (F1=0.544). Flag retained off-by-default as
reproducible negative result.

**Rome and Cakes are now confirmed evidence-invariant.** Six
sequential Phase 2b levers (A/B/F/E/v4/rerank-first) all tapped out
on these specific rows. The grounding failure is at the fact-content
layer (what the extraction stored, how the fragment was tokenized, or
how the gold string lexically differs from what the reader sees) —
not at retrieval, pool composition, prompt, or ordering.

## Shipped state

- `reader_evidence_rerank_first` flag on `CompanionSystem` + gate CLI,
  default `False`.
- Full `_hierarchical_recall` ordering branch covered by existing
  companion tests (327 passing).
- R5 remains the shipped floor.

## What this leaves on the table

Six sequential Phase 2b levers each landed as regression or null-op
on the shipped substrate. Current best = Step F ADDITIVE (F1=0.557).
0.70 gate is structurally out of reach on this substrate.

Remaining candidates (per `reconstructive-retrieval/README.md`):

- **Step A'** — extraction prompt v2 with per-fact `source_turn_hint`
  (fact-cache regeneration ~5h × 3 = ~15h wall). Foundational for
  Steps G/H. May null-op on F1 alone (Step A regressed; the
  mechanism needs more than correct provenance to unblock the F1
  ceiling) but UNLOCKS G/H.
- **Step C** — entity graph (attacks session_present_gold_absent,
  the 11-row bucket, independent of per-fact provenance).
- **Step G** — reconsolidation hook (thesis-only; measures
  reconsolidation outcomes on B1 benchmark, not F1 on LoCoMo).
- **Step H** — reconstructive assembly (thesis-only; wires the
  existing R1/R2/R7 reconstruction primitives to the LoCoMo eval
  path — could re-measure F1 under reconstruction instead of
  retrieve-then-compose).

**Honest conclusion**: six sequential F1-attacking levers have
plateaued around 0.544-0.557. The 0.70 LoCoMo gate was a Phase 1.5
floor-check, not the project's north star. The project thesis
(consciousness-simulation principles, reconsolidation + ToM +
reconstructive memory) is validated separately via B1/B2/B3 + V-track
benchmarks per the SOTA goal doc. Moving to Step H
(reconstructive-assembly) directly is a defensible pivot that aligns
F1 measurement with the project's actual claim.
