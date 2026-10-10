# Reader-prompt v4 — few-shot grounding (Phase 2b reader-side attack)

## Hypothesis

Four sequential Phase 2b retrieval-side levers (Step A fact-to-turn
lift, Step B hybrid fact retrieval, Step F ADDITIVE full-session
reader, Step E localize-filter) all landed as regression or null-op.
The 4 residual `gold_in_pool` failures from the Step F+E stack
(conv-30-q9 Rome, conv-41-q18 Rob, conv-41-q49 Cakes, conv-41-q62
"two") share a pattern: gold is in top-K with substring+coverage=1.0
and session-in-top-K, but the reader refuses ("not stated") or
over-generalizes ("a colleague" instead of "Rob"). V3's rule block
states the correct behavior ("extract specific proper noun";
"anti-refusal when gold is present") but the LLM isn't applying it.

Few-shot worked examples are a well-studied technique for closing
the gap between stated rules and applied behavior. V4 appends three
examples targeting the three residual failure modes:

- **Example 1** (anti-refusal despite gold-in-pool) → q9 Rome, q49 Cakes
- **Example 2** (specific name over generic descriptor) → q18 Rob
- **Example 3** (integer count over paraphrase) → q62 "two"

V3's rule block preserved verbatim; v4 appends examples only.

## Config

- `--reader-prompt-variant v4` option added to the gate CLI.
- Prompt body: v3 full text + 3 examples, each with (question,
  fragment-present, correct-answer, wrong-answer, reason).
- Stacked on R5 shipped baseline (consolidation + cross-encoder
  rerank + judge-mode). F/E flags OFF for isolation measurement.

## Result — ∅ null-op swap

- **F1 = 0.551** (+0.007 vs R5 0.544, within noise band 0.03)
- 3-instance (74 Q, wall 318.8 s): conv-26 0.484, conv-30 0.500,
  conv-41 0.637
- Reader refusal rate **10.8% (8/74)** — highest of any Phase 2b
  variant, +2.7 pts vs R5 (8.1%)
- Diagnostic: 16 incorrect (13 session_present_gold_absent, 3
  gold_in_pool, 0 recall_missing)
- Target bucket: `gold_in_pool` 5→3 (−2) — few-shot DID help on 2
  of the 4 residual reader-side failures
- But: `session_present_gold_absent` 12→13 (+1) and overall
  incorrect 17→16 (−1) — same wash-trade pattern as R3→R5 iteration

### Row-level swap

| Row | V3/stack said | V4 said | Verdict |
|---|---|---|---|
| conv-41-q18 Rob | "a colleague" | **"Rob"** | **FIXED** (Example 2 worked) |
| conv-26-q48 bowls, cup | "pots" | **"bowls, cup"** | **FIXED** (specificity transfer) |
| conv-26-q40 "2" | "2" (correct) | "once or twice a year" | **BROKE** (reverted to v2-style frequency phrase despite Example 3 covering counts) |
| conv-41-q40 Kyle, Sara | correct | "not stated" | **BROKE** (new refusal despite Example 1 covering refusal) |
| conv-41-q52 Cindy, Laura | correct | "not stated" | **BROKE** (new refusal, same pattern as q40) |

### The three prompt-invariant failures

Rome, Cakes, and "two" produced **identical** wrong outputs under
v3, v3+E, v3+F, v3+F+E, and v4. The reader output "not stated" or
"1" every time despite gold with substring=True and coverage=1.0.
These are not rule-knowledge gaps — they are grounding failures
deeper than any prompt variant has reached.

| Row | Gold | All reader variants |
|---|---|---|
| conv-30-q9 | Rome | "not stated" (×5 variants) |
| conv-41-q49 | Cakes | "not stated" (×5 variants) |
| conv-41-q62 | two | "1" (×5 variants) |

## Verdict

**∅ null-op swap. Prompt-engineering on the current reader has
plateaued.** V4 fixed exactly the two rows its Examples 2 directly
addressed (Rob, bowls/cup) but regressed three unrelated rows into
v2-style frequency/refusal failures. Three failures (Rome, Cakes,
"two") are prompt-invariant across every variant tested — grounding
is failing at a layer below the system prompt (either fragment-order
truncation, context-length distraction, or the gold token being
present-but-unactivated in the context block).

Shipped floor remains **R5 reader-prompt v3**. V4 retained as
reproducible negative result; `--reader-prompt-variant v4` kept
available for future comparison.

**Five sequential Phase 2b levers (A/B/F/E/v4) have each landed as
regression or null-op. The ceiling is now visibly structural across
both retrieval and reader layers on the current substrate.** Next
lever is **Step A'** (per-fact `source_turn_hint` via extraction
prompt v2 + fact-cache regeneration) — Step A's mechanism was correct
but provenance granularity was wrong (window-wide, not per-fact).
Step A' is also the foundational substrate required by Steps G
(reconsolidation hook) and H (reconstructive assembly). Alternative
next-lever candidates: forced-citation decode (reader must quote the
span before answering), context-ordering experiments (promote gold
turn to position 0), or Step C (entity graph) targeting the
session_present_gold_absent bucket.

## Shipped state

- V4 prompt in [reader.py](../../../src/psychology_base/eval/benchmarks/reader.py)
  as `_SYSTEM_PROMPT_V4`.
- `--reader-prompt-variant` choices extended with `v4` in
  [run_locomo_gate.py](../phase-1-benchmark-adapters/option-gamma/run_locomo_gate.py).
- Existing reader-prompt tests cover v4 variant selection.
- R5 (v3) remains the shipped floor.
