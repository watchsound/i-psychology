# Phase 2 ceiling breakers — comparison

Phase 1.5-prime cap: F1=0.469 (P3 cross-encoder rerank over top-150) on conv-26
single-hop under judge-mode. Phase 2 attacks the ceiling with four candidate
levers; this doc records the side-by-side verdict as each lands.

**Pivot 2026-10-07 — reconstructive-retrieval architecture.** The R1–R8
sweep + multi-instance validation confirmed the ceiling is structural
(`session_present_gold_absent` = 71% of incorrect across 3 instances;
aggregate R5 F1 = 0.544 vs 0.70 gate). Scope moves from single-lever
tuning to the composed stack the SOTA systems converge on + two
thesis-only levers. Plan: [reconstructive-retrieval/README.md](reconstructive-retrieval/README.md).
R1–R8 remain shipped as reproducible negative-space; R5 shipped as the
floor. New steps A–H are tracked under their own experiment dir.

## Verdict table

| Option | Lever | F1 (judge, standalone) | Δ vs P3 (0.469) | Gate (0.70) | Status |
|---|---|---|---|---|---|
| P3 (baseline) | rerank top-150 → 50 | 0.469 | — | ✗ | shipped |
| R1 | rerank over all 419 fragments | 0.391 | −0.078 | ✗ | killed (serial-bottleneck confound noted) |
| R2 | BM25 + semantic hybrid (RRF k=60) | 0.422 | −0.047 | ✗ | ablation complete — null-op in stacked architecture |
| R3 | reader prompt redesign (v2) | 0.438 | −0.031 | ✗ | measured — refusal 4→2 but introduced hallucination regressions |
| R4 | extractor swap qwen-turbo → qwen-max | — | — | — | de-prioritized (R6 verdict → extractor not on failing path) |
| R5 | reader prompt v3 (precision tightening) | **0.484** | **+0.016** | ✗ | **shipped — matches R6 upper-bound prediction (+0.063 vs R3 baseline)** |
| R6 | recall@50 diagnostic instrumentation | n/a (measurement) | — | — | shipped — reclassified 7-row residue |
| R7 | new recall channel for proper nouns | — | — | — | retired (R6 showed `recall_missing=0`) |
| R8-c | within-session rerank (two-stage, M=3) | **0.500** | **+0.031** | ✗ | **measured — passes minimum, under-shoots target (0.56)** |
| R8-c @ M=5 | within-session rerank (M=5) | 0.438 | −0.031 | ✗ | tuning pilot — regresses below R5 (precision/recall knee) |
| R8-b | fact-density tie-breaker (standalone, shipped) | 0.438 | −0.031 | ✗ | measured — null-op within noise; sentence-start regex fix landed, pre-fix 0.469 kept as audit baseline |
| R8-b + R8-c | stacked within-session + fact-density | 0.438 | −0.031 | ✗ | measured — reader refusals triple; stacking hurts |
| **Multi-instance (R5 across 3)** | shipped R5 on conv-26 + conv-30 + conv-41 | **0.544** (74 Q) | n/a (new scope) | ✗ | **measured — conv-26 is outlier (0.469); conv-41 near-gate (0.621); structural ceiling confirmed** |
| **Step A (reconstructive-retrieval)** | fact-to-turn lift via source_fragment_ids back-pointers | **0.486** (74 Q, 3-inst) | −0.058 vs R5 (0.544) | ✗ | **regressed — back-pointers are window-wide not per-fact; extraction schema fix required (Step A')** |
| **Step B (reconstructive-retrieval)** | hybrid fact retrieval (BM25 + cosine + entity-overlap, RRF k=60) | **0.554** (74 Q, 3-inst) | +0.010 vs R5 (0.544) | ✗ | **null-op within noise — `gold_in_pool` drops 5→2 but offset by `session_present_gold_absent` 12→15 (reclassification, not resolution); flag retained off-by-default** |
| **Step F REPLACE (reconstructive-retrieval)** | full-session reader: replace fragment_hits with top-1 session | **0.497** (74 Q, 3-inst) | −0.047 vs R5 (0.544) | ✗ | **regressed — top-1 session often wrong; REPLACE drops correct cross-session evidence; backed out in favor of ADDITIVE variant** |
| **Step F ADDITIVE (reconstructive-retrieval)** | full-session reader: UNION winning-session extras onto reranked top-K | **0.557** (74 Q, 3-inst) | +0.013 vs R5 (0.544) | ✗ | **null-op within noise — but first measured partial bucket resolution: `session_present_gold_absent` 12→10 (−2) and `gold_in_pool` 5→4 (−1), total incorrect 17→14; 4/4 residual gold_in_pool failures are reader-side (refusal/over-generalization), motivating Step E; flag retained off-by-default** |
| **Step E isolation (reconstructive-retrieval)** | localize-filter LLM stage (SIMPLE-tier qwen-turbo, recall-biased) between rerank and reader, no pool widening | **0.537** (74 Q, 3-inst) | −0.007 vs R5 (0.544) | ✗ | **null-op (regress-leaning) — stacking a precision gate on cross-encoder's already-reranked top-50 removes signal; refusals 9.5% (highest Phase 2b); conv-26-q3 Adoption + conv-41-q40 Kyle Sara previously correct under R5 dropped; flag retained off-by-default** |
| **Step F ADDITIVE + E stack (reconstructive-retrieval)** | widen pool with full-session union, then filter with localize-filter (EMem widen-then-filter order) | **0.551** (74 Q, 3-inst) | +0.007 vs R5 (0.544) | ✗ | **null-op within noise — widen-then-filter restores E's isolation losses (q3 Adoption back correct, refusals 9.5%→6.8%) but net ~0.006 below F alone; same 4 residual gold_in_pool failures (Rome/Rob/Cakes/two) as F alone — filter doesn't resolve reader-side cases; both flags retained off-by-default** |
| **Reader-prompt v4** | few-shot grounding (v3 + 3 worked examples: anti-refusal / specific-name-over-descriptor / integer-count-over-paraphrase) | **0.551** (74 Q, 3-inst) | +0.007 vs R5 (0.544) | ✗ | **null-op SWAP — fixed Rob + bowls/cup via Example 2 but broke 3 rows (q40 "2"→"once or twice a year", q40/q52 Kyle-Sara/Cindy-Laura→refusals); 3 persistent failures (Rome/Cakes/"two") prompt-invariant across v3/v4/v3+F/v3+F+E — grounding fails below the system prompt layer; v3 remains shipped floor** |
| **Reader-evidence reorder (rerank-first)** | reorder DIALOGUE + fragments_used to put cross-encoder-reranked top-K first (then substring, source-lift, facts' window-sources) | **0.510** (74 Q, 3-inst) | −0.034 vs R5 (0.544) | ✗ | **regressed — refusals jump 8.1%→18.9%; reader needs facts' window-context scaffolding to extract confidently. Fixed Rob + "two" via top-focused reranked but broke 7 rows (q3 Adoption, q37 sunset, q51 Horse, q61 Matt Patterson, q78 Figurines, q6 homeless shelter, q34 Mentoring) into refusals. Rome + Cakes still broken (now confirmed evidence-order-invariant). Flag retained off-by-default** |
| **Step C (entity graph)** | fact-derived entity graph (1-hop expand, cap=20 fragment-boost tier dedup'd against cosine/substring/source_lift) | **0.547** (74 Q, 3-inst) | +0.003 vs R5 (0.544) | ✗ | **null-op within noise — target bucket `session_present_gold_absent` 12→11 (−1) but offset by `gold_in_pool` 5→6 (+1) + refusal rate 8.1%→12.2%. Mechanism IS doing targeted work (bucket migration confirms) but reader can't disambiguate added entity-dense distractors. conv-30 regressed −0.136; conv-41 lifted +0.040. Flag retained off-by-default** |
| **Step H (reconstructive assembly)** | R1 spreading-activation over Jaccard-seeded EdgeStore (2-hop, decay=0.5, cap=20, local one-shot ActivationState to avoid double-mutation) | **0.490** (74 Q, 3-inst) | −0.054 vs R5 (0.544) | ✗ | **regressed — both diagnostic buckets moved WRONG direction (`session_present_gold_absent` 12→15 +3; `gold_in_pool` 5→8 +3). Diagnosis: Jaccard-only edges on single-pass data degenerate R1 to quadratic-cost token-overlap ranker. Ouroboros's R1 lift relies on months of Hebbian + Witness co-activation history that LoCoMo gate protocol doesn't produce. Flag retained off-by-default. Open: whether R1 lifts on multi-session gates (B2/B3)** |

## 2×2 ablation (R2 × R3) on conv-26 single-hop, judge mode

Component-level instrumentation (refusal count, refusal rate) added to the
gate runner so each cell reports both F1 and refusals. All runs: 32 Q,
consolidation on, cross-encoder rerank on, judge-single-hop on.

| Cell | R2 | R3 | F1 | Refusals | Δ F1 vs A |
|---|---|---|---|---|---|
| **A** (baseline P3) | off | off | **0.469** | 4/32 (12.5%) | — |
| **B** (R3 only) | off | **on** | **0.438** | 2/32 (6.2%) | −0.031 |
| **C** (R2 only) | **on** | off | **0.422** | 5/32 (15.6%) | −0.047 |
| **D** (R2 + R3 stacked) | **on** | **on** | **0.438** | 2/32 (6.2%) | −0.031 |

**Decisive finding — R2 is a null-op in the stacked architecture.** Cell D
is byte-for-byte identical to Cell B: same F1, same refusals, same seven
incorrect rows, same candidate strings on every incorrect row. BM25 did not
produce *any* behavioral change under reader-v2. The R2 kill verdict is
now honest and measured, not inferred from standalone data: with the
downstream reader unblocked, BM25 produces zero lift.

Why BM25 is null in-stack:
- RRF k=60 weights top-rank evidence from both channels. In practice,
  cosine (via rerank) and BM25 agree on the top of their lists for most
  cues on this corpus, so the fused top-50 is dominated by cosine already.
- The seven persistent broken rows split into (a) fragments the reader
  mis-extracts even with the gold in context (q11 Sweden → "home country",
  q40 "2" → "once or twice a year", q75 "3" → "2") and (b) gold fragments
  that may be missing from the top-50 regardless of recall channel (q7
  Single, q76 "19 October 2023", q34 Mentoring). BM25 cannot help either
  class.

**R3 did its structural job.** Refusals dropped 4 → 2 (−50%). Of the four
original refusals (q3 Adoption agencies, q7 Single, q34 Mentoring, q76
"19 October 2023"), R3 fixed q3 correctly. R3 also fixed q48 bowls/cup
and q66 marshmallows (imprecise-match wins, not refusals). But R3
introduced hallucination regressions on two previously-correct rows:
q11 Sweden → "home country" (over-generic place answer) and q40 "2" →
"once or twice a year" (frequency instead of count). Net Δ F1 = −0.031
despite the structural win.

**Diagnosis for next moves:**
- The ceiling is not single-component. R3 traded refusals for
  hallucinations. The v2 prompt needs tightening (bias toward proper
  nouns for "where" cues, strict numeric for "how many") — this is R3
  v3, not a new option.
- The remaining 7 shared broken rows split roughly into reader-precision
  failures (fixable by prompt tuning or model swap) and recall failures
  where the gold fragment likely never reaches the top-50. The
  component-level instrumentation we just added (refusal count) is
  insufficient to tell which is which per-row — recall@50 for
  gold-containing fragments is the missing metric and should be added
  next.
- **R4 (extractor-swap) is no longer the natural next move.** R4
  attacks fact-extraction quality on the write side. The 2×2 says the
  binding constraints are now (a) reader precision under paraphrase
  pressure and (b) recall of 3 persistent gold fragments the current
  stack doesn't surface. Neither is on the extractor path.

## New candidate options (replace R4 in priority)

- **R5 — reader prompt v3 (precision tightening).** **Shipped 2026-10-06**:
  F1=0.484 on conv-26 single-hop under reader-v3 + rerank + judge, +0.063
  vs R3-only baseline (0.422). Matches R6's +0.0625 upper-bound prediction.
  Refusals 2→1/32. Row-level: fixed q7 Single (single-word refusal gone),
  q11 Sweden (proper-noun bias), q40 "2" (integer bias, surprise win);
  broke q43 abstract art and q66 marshmallows (over-generalization to
  shortest literal). Net +1 binary question with partial-credit shifts
  on borderline rows. Prompt engineering near plateau — do not iterate
  v3 → v4 on the same two-row swap-bucket.
  Artefacts: [option-R5-reader-v3/README.md](option-R5-reader-v3/README.md); [e1-r5-conv26.json](option-R5-reader-v3/e1-r5-conv26.json).
- **R6 — recall@50 instrumentation + diagnostic.** **Shipped 2026-10-06**:
  `_recall_diagnostics` in locomo.py + `_recall_stats` aggregator in the
  gate runner + 13 new tests + `_normalize_evidence_sessions` mapping
  LoCoMo's native `D<N>:<T>` turn-level evidence to our fragment
  `session_<N>` tags. Non-F1 lever by design; the measurement that
  reclassified the 7-row residue (see "R6 outcome" section below).
  Artefacts: [option-R6-recall-diagnostic/README.md](option-R6-recall-diagnostic/README.md).
- **R8 (new, replaces R7) — within-session fragment ranking.** R6 shows
  all 5 non-reader-side broken rows have the right evidence session in
  top-50 but the specific gold-naming turn below the cut-off. The
  original R7 scope (new recall channel for proper nouns globally) is
  no longer motivated — the right *session* already surfaces for every
  row. The actual constraint is turn-level ranking inside a session.
  Candidate mechanisms: (a) BM25 within top-ranked sessions (not the
  global BM25 that landed as a null-op in R2); (b) fact-density
  prioritization — surface turns containing proper nouns, dates, or
  numeric literals ahead of chatty filler; (c) two-stage rerank by
  session then by turn. **Upper-bound lift: +0.156 F1.**

## R6 outcome — 7-row ceiling reclassified (2026-10-06)

| Bucket | Count | Rows | Lever |
|---|---|---|---|
| `gold_in_pool` (reader-precision) | 2 | q7 Single; q11 Sweden | **R5** |
| `session_present_gold_absent` (within-session ranking) | 5 | q34 Mentoring; q40 "2"; q71 Becoming Nicole; q75 "3"; q76 "19 October 2023" | **R8** |
| `recall_missing` | 0 | — | — |

**Shift in Phase 2 priority order.**

- The 5-row bucket dominates the F1 upside (+0.156 vs R5's +0.0625). R8
  belongs at the top of the queue if the F1 ceiling is the goal.
- R5 is still worth running — the two reader-precision failures are
  unambiguous and the fix is zero-infra (one prompt diff).
- R7 as previously framed (new recall channel for proper nouns) is
  retired. The diagnosis it was designed for ("gold fragment never
  reaches top-50") is now empirically false.
- R2 (BM25 global hybrid) remains null — but R8 may revive a scoped
  variant of the same mechanism applied within session, not across.

## Methodology correction (2026-10-06)

Earlier revisions of this file marked R1 and R2 as "killed". That verdict is
retracted. Three problems with the original kill criterion:

1. **Serial-bottleneck confounding.** 4 of 10 R2-incorrect rows have
   `candidate="not stated"` — the reader refused on top-50 pools that
   likely contained gold. Upstream recall improvements (BM25, wider pool)
   cannot show up in end-to-end F1 while the downstream reader absorbs them
   as refusals. R2 might have been doing its job while F1 stayed flat.
2. **Statistical power.** 32 questions × single instance has a judge-mode
   noise floor around 0.03 F1. R2 at 0.422 vs P3 at 0.469 is 1.5 questions
   of noise, not a decisive regression.
3. **Design under-tested.** R2 README listed k=30 and k=100 as RRF fallbacks
   if the primary k=60 just-missed. We did not run them. We also did not
   measure recall@50 for Bucket A gold under R2 — the direct test of what
   BM25 was supposed to accomplish.

**Corrected methodology.**

- Build R3 (reader prompt redesign) with component-level instrumentation:
  recall@50 for gold fragments per row, reader refusal rate per row,
  pool composition deltas between variants.
- Run a 2×2 ablation against R3-on baseline:
  `{R2 off, R3 off}` (current P3) ×
  `{R2 off, R3 on}` ×
  `{R2 on, R3 off}` (current R2) ×
  **`{R2 on, R3 on}` ← the real test**
- Same ablation for R1. Expand to ≥2 LoCoMo instances before any further
  kill verdict so single-instance noise doesn't dominate.
- Only after the stacked tests does a "killed" verdict land, and only when
  the component is net-negative *within the stacked architecture*, not
  alone.

R1 and R2 shipped flags remain in place (default off) — the correction is
to the verdict framing, not to the code.

## R1 verdict — killed

**F1=0.391 under judge-mode, worse than P3 by 0.078.** The pool-widening
hypothesis is falsified: the top-150 cosine prefilter is quality-critical, not
a cost gate. Cross-encoder discrimination degrades when pool size exceeds
~150; broadening to all 419 fragments pulled distractors into high rerank
positions and pushed previously-correct answers out of the final top-50.

**Row-level breakdown** (P3 → R1 incorrect sets):
- **Broke** (P3 correct → R1 wrong): q11 Sweden, q37 sunset, q40 "2" — all
  three were P3 wins from the rerank-at-150 path.
- **Fixed** (P3 wrong → R1 correct): *zero*. Not a single Bucket A failure
  R1 was supposed to catch (q3 Adoption, q7 Single, q34 Mentoring,
  q76 "19 October 2023") was recovered.
- **Still broken on both**: 8 rows shared across P3 and R1 (q3, q7, q34,
  q48, q66, q71, q75, q76).

**Why rerank got worse, not better.** Cross-encoders score (query, doc)
pairs independently and output a continuous relevance score. The sort over
*all* 419 scored pairs mixes marginally-relevant distractors (fragments that
share topical cues like "family", "home", "transition") into positions that
cosine would have filtered out. The reader then sees a diluted DIALOGUE
block where the gold fragment competes with higher-reranked noise. On
Bucket A questions (proper-noun cues), the gold fragment is already
cosine-ranked past 150 — rerank *does* see it in the wider pool, but it
also sees 268 new distractors scoring in the same band, and the gold loses.

**What this means for Phase 2.** The ceiling-breaker budget shifts to R2
(BM25 hybrid) immediately. BM25 is literal-token recall — proper nouns like
"Sweden", "Adoption", "Single" that cosine buries will rank at the top of
BM25 regardless of paraphrase cue. Rank-fusion with the top-150 cosine
preserves P3's rerank discrimination while adding a parallel channel for
exact-match recall. This is the standard dense+sparse hybrid.

R1 shipped state: `cross_encoder_rerank_all` flag retained on
`CompanionSystem` + `run_locomo_gate.py` as a reproducible negative result.
Default = off. Not deleted — the kill is a learning, not a bug.

## R2 verdict — killed

**F1=0.422 under judge-mode, worse than P3 by 0.047.** The sparse+dense
fusion hypothesis is falsified for this corpus: adding BM25 as a parallel
recall channel did not surface any Bucket A proper-noun gold the reader
could extract.

**Row-level breakdown** (vs P3):
- **Broke** (P3 correct → R2 wrong): q37 sunset (R2 wrote "Melanie painted
  an abstract painting recently"), q40 "2" ("not stated"). RRF reshuffle
  swapped these previously-correct fragments out of the final top-50.
- **Fixed** (P3 wrong → R2 correct): *zero*. Not one Bucket A target
  (q3 Adoption, q7 Single, q34 Mentoring, q76 "19 October 2023") was
  recovered. BM25 did work mechanically — q11 Sweden stayed correct, which
  R1 had broken — but Bucket A gold didn't rise far enough through RRF+rerank
  to influence the final answer.
- **Both broken**: 8 rows shared with P3.

**Reading the "not stated" pattern — the real diagnosis.** Four of ten
R2-incorrect rows (q3, q7, q34, q40, q76) have `candidate="not stated"`.
That's not a recall miss — that's the reader *refusing to answer* on a
block where the gold fragment is likely already present. Both R1 and R2
attacked recall (pool width, recall channel) and both killed, which
converges on a different conclusion:

> The ceiling is the **reader's refusal gate**, not the recall layer.
> On proper-noun questions ("where did you move from?", "how do you want
> to adopt?") the reader sees the gold line in dialogue but still outputs
> "not stated" — likely because the current prompt's refusal clause fires
> conservatively on short-cue single-hop extraction.

**What this means for Phase 2.** The budget shifts to **R3 (reader
prompt redesign)** immediately — it was previously mid-priority but two
consecutive recall-channel kills promote it. Specific R3 move: loosen the
"not stated" gate for cue-matched extraction (if the cue mentions a token
and dialogue names that token, prefer the dialogue line over refusal),
while keeping the gate active for truly absent information.

R2 shipped state: `bm25_hybrid` flag retained on `CompanionSystem` +
`run_locomo_gate.py` as a reproducible negative result. Default = off.
Not deleted. R2 may become relevant again if R3 lifts the reader floor —
a less-refusing reader may then benefit from the wider-channel pool it
could not exploit under the current strict refusal.



## Per-option artefacts

- **R1 rerank-all** — [option-R1-rerank-all/README.md](option-R1-rerank-all/README.md); [e1-rag-conv26-rerank-all-judge.json](option-R1-rerank-all/e1-rag-conv26-rerank-all-judge.json)
- **R2 BM25 hybrid** — [option-R2-bm25-hybrid/README.md](option-R2-bm25-hybrid/README.md); [e1-rag-conv26-bm25-rerank-judge.json](option-R2-bm25-hybrid/e1-rag-conv26-bm25-rerank-judge.json)
- **R3 reader prompt** — [option-R3-reader-prompt/](option-R3-reader-prompt/) (empty — next)
- **R4 extractor swap** — [option-R4-extractor-swap/](option-R4-extractor-swap/) (empty)

## Reading the deltas

- **R1 lift source** — compare the per-row incorrect list against P3's known 5
  Bucket A failures (q4 Fluffy breakfast / q7 Single parent / q14 Adoption /
  q33 Hiking Charlotte's Web / q66 Pepper's Charlotte's Web). If R1 lifts F1 by
  fixing those 5 specifically, the diagnosis "rerank was cosine-pool-starved"
  is confirmed; stack with R2–R4. If R1 lifts F1 by fixing *different* rows,
  cosine pool width is not the dominant failure mode and R2–R4 ordering may
  need to change.
- **R2 lift source** — BM25's job is proper nouns (literal token match). If
  R2 fixes Bucket A rows R1 missed, BM25 complements rerank. If BM25 fixes
  rows R1 also fixed, the two are redundant and only the cheaper lever ships.
  *Observed:* R2 fixed zero Bucket A rows; the reader refused ("not stated")
  on 4/10 incorrect rows despite literal-token recall landing proper nouns
  in the pool. The lesson: with the current reader prompt, improved recall
  of proper-noun fragments is wasted — refusal dominates. Reader prompt
  must be unblocked first before any recall-side lever can be measured.
- **R3** — expect two-sided tradeoff: looser "not stated" gate recovers rows
  the reader currently refuses but may introduce hallucinations on cold-cue
  rows. Score the regression surface, not just the lift.
- **R4** — fact-store quality floor. If R4 lifts F1 even when R1/R2/R3 are
  off, extraction quality was a dominant limiter and the SIMPLE-tier
  consolidation default needs to change.
