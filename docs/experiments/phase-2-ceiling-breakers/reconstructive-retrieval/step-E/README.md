# Step E — Localize-filter LLM stage before reader

## Hypothesis

Three retrieval-side steps (A/B/F) landed at regress / null-op /
null-op-with-partial-bucket-signal. The residual `gold_in_pool`
failures (4/4 after Step F ADDITIVE: conv-30-q9 Rome,
conv-41-q18 Rob, conv-41-q49 Cakes, conv-41-q62 "two") share a
pattern: gold is in the pool (substring+coverage=1.0+session-in-top-K)
but the reader refuses ("not stated") or over-generalizes
("a colleague" instead of "Rob").

Step E inserts a **recall-biased SIMPLE-tier LLM filter** between
cross-encoder rerank and the reader with the prompt "which of
these COULD support answering Q — err on inclusion." Target: the
re-primed pool composition reduces reader refusals /
over-generalizations without widening the pool itself. EMem
reports ~4.7 LoCoMo points for this stage alone in the EMem
ablation.

## Config

- `--localize-filter` flag added to `CompanionSystem` + gate CLI.
- Position: between cross-encoder rerank and reader prep. Final
  pipeline order (EMem widen-then-filter): rerank → full-session
  union (Step F widens) → **localize filter (Step E recall-biased
  precision gate)** → fact-value substring aug → Step A fact-to-turn
  lift → reader.
- SIMPLE tier (qwen-turbo) through the consolidation router.
- Prompt: numbered-index response schema (compact, parse-stable),
  content truncated to 400 chars/fragment.
- Safety floor: `localize_filter_min_keep=5` — if LLM returns
  fewer indexes, top-N by rerank fills to the floor.
- Max input: `localize_filter_max_input=80` — tail past 80 passes
  through unfiltered (filter can't filter what it didn't see).
- Response parser tolerant of brackets, markdown, prose, dupes,
  out-of-range indexes.
- Fallback: provider error → pass-through (never drop evidence
  due to filter failure).
- Stacked on R5 shipped baseline (consolidation + cross-encoder
  rerank + reader prompt v3 + judge-mode). Measured in two
  configurations: Step E alone (F OFF) and Step F ADDITIVE + E
  stack (widen-then-filter).

## Result

### Step E isolation (F OFF, E only) — ∅ null-op (regress-leaning)

- **F1 = 0.537** (−0.007 vs R5 0.544, within noise band 0.03)
- 3-instance (74 Q, wall ~370 s): conv-26 0.469, conv-30 0.636,
  conv-41 0.517
- Reader refusal rate **9.5% (7/74)** — higher than R5 and higher
  than any Phase 2b variant to date
- Diagnostic: 18 incorrect (13 session_present_gold_absent, 5
  gold_in_pool, 0 recall_missing) vs R5 17 incorrect
- **New refusals introduced vs R5**: conv-26-q3 Adoption and
  conv-41-q40 Kyle Sara — both previously correct under judge-mode
  R5, both dropped to "not stated" / wrong set after filter
- Root cause: Step E alone applies a precision gate on top of
  cross-encoder's already-reranked top-50 — stacking two precision
  gates removes signal faster than it removes noise. The filter
  works on EMem's wider post-widen pool (see stack below), not on
  a pre-filtered reranked pool.

### Step F ADDITIVE + E stack (widen → filter) — ∅ null-op within noise

- **F1 = 0.551** (+0.007 vs R5, within noise band 0.03)
- 3-instance (74 Q, wall 357.1 s): conv-26 0.500, conv-30 0.455,
  conv-41 0.637
- Reader refusal rate **6.8% (5/74)** — the widen pass restored
  the fragments E alone dropped (q3 Adoption back to correct);
  refusal rate below R5 (R5 was 8.1%)
- Diagnostic: 15 incorrect (11 session_present_gold_absent, 4
  gold_in_pool, 0 recall_missing)
- Target buckets (vs R5 17 incorrect): session_present_gold_absent
  12→11 (−1), gold_in_pool 5→4 (−1), total incorrect 17→15 (−2)
- Same 4 residual gold_in_pool failures as Step F alone: Rome, Rob,
  Cakes, "two" — the filter does not help these reader-side cases
- **Stack vs F alone**: F ADDITIVE isolated was F1=0.557, 14
  incorrect — the stack sits ~0.006 below F alone (noise); E does
  not add net lift on top of F even in the EMem-correct
  widen-then-filter order

### Per-instance table

| Instance | R5 shipped | F ADDITIVE | E isolation | F+E stack |
|---|---|---|---|---|
| conv-26 | 0.469 | 0.484 | 0.469 | 0.500 |
| conv-30 | 0.545 | 0.545 | 0.636 | 0.455 |
| conv-41 | 0.621 | 0.637 | 0.517 | 0.637 |
| **aggregate** | **0.544** | **0.557** | **0.537** | **0.551** |

### Recall diagnostic

| Variant | incorrect | session_absent | gold_in_pool | refusals |
|---|---|---|---|---|
| R5 | 17 | 12 | 5 | 6/74 (8.1%) |
| F ADDITIVE | 14 | 10 | 4 | 4/74 (5.4%) |
| E isolation | 18 | 13 | 5 | 7/74 (9.5%) |
| F+E stack | 15 | 11 | 4 | 5/74 (6.8%) |

## Verdict

**∅ null-op, both configurations retained off-by-default.**

Four sequential Phase 2b retrieval-side levers (A fact-to-turn lift,
B hybrid fact retrieval, F full-session reader, E localize-filter)
have now each landed as regression or null-op. The 4 residual
`gold_in_pool` failures (Rome / Rob / Cakes / "two") are **identical**
in both Step F alone and the Step F+E stack — meaning neither
widening the pool nor LLM-filtering the pool resolves them. These
are reader-side pathologies: the reader has substring-exact gold
visible with coverage=1.0 and session-in-top-K, and still refuses
or over-generalizes.

**Shipped floor remains R5** (F1=0.544). Both `--full-session-reader`
and `--localize-filter` kept off-by-default. Step F ADDITIVE remains
the single-variant best observed (F1=0.557, +0.013 within noise).

**Next lever: Step A' (per-fact `source_turn_hint` via extraction
prompt v2) or a reader-side intervention (prompt v4, few-shot
grounding, or forced-citation decode).** The structural ceiling
appears to be reader-side on these 4 instances. Step A' is the
foundational schema change required by Steps G/H regardless.

## Shipped state

- `localize_filter` flag on `CompanionSystem` + gate CLI, default
  `False`.
- `localize_filter_min_keep=5` safety floor.
- `localize_filter_max_input=80` input cap.
- Pipeline order: rerank → full_session_union → localize_filter →
  fact-value aug → Step A lift → reader (EMem widen-then-filter).
- 18 unit tests in [test_companion_localize_filter.py](../../../../../tests/test_companion_localize_filter.py)
  cover: parser (7), init validation (3), behavior (6), fallback
  (2) — including malformed-output tolerance, provider-error
  fallback, max-input tail passthrough, safety-floor fill.
- Requires `cross_encoder_rerank=True` + non-None
  `consolidation_router`.
- R5 remains the shipped floor.
