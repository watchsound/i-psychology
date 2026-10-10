# R3 — reader prompt redesign (refusal-rebalanced)

## Hypothesis

The R1 and R2 standalone results converged on a reader-refusal diagnosis:
4 of 10 R2-incorrect rows have `candidate="not stated"` on cue-matched
Bucket A proper nouns, with the gold-naming dialogue line almost certainly
in the top-50 fragment block. Both recall-side levers (R1 pool widening,
R2 BM25 fusion) landed below P3; neither test is interpretable until the
downstream refusal gate is unblocked.

**Observation.** The v1 reader system prompt contains two strong "if not
covered, say not stated" clauses:

> "If the context does not contain the information needed, say exactly
> 'not stated' — do not guess, do not use outside knowledge."
> "If the question asks about something not covered by the context,
> answer exactly 'not stated'."

On short-cue single-hop ("where did you move from?", "which agency?",
"when did you move?"), the model interprets "not covered" strictly — a
fragment that *mentions* the topic but doesn't provide a tight canonical
answer gets scored as "not covered" and the refusal clause fires. Gold
is in the pool; the extraction stage refuses.

**Fix.** Rewrite the reader system prompt (variant v2) to rebalance toward
positive extraction:
1. Explicit "scan ALL fragments" instruction so the model doesn't shortcut.
2. Question-type-specific extraction templates (when → date, where → place,
   what → name, how many → number, list → comma-separated).
3. Tighten the "not stated" clause: fires ONLY when no fragment mentions
   the topic at all. Imprecision ≠ silence.
4. Keep rules against outside knowledge and against guessing.

## Mechanism

- `psychology_base.eval.benchmarks.reader.LLMReader` gains a
  `prompt_variant: str = "v1"` kwarg. Validation at `__post_init__` fails
  loud on a typo.
- Two prompts live in the module: `_SYSTEM_PROMPT` (v1, unchanged) and
  `_SYSTEM_PROMPT_V2` (R3). `get_reader_prompt(variant)` is the lookup.
- CLI flag `--reader-prompt-variant {v1,v2}` on `run_locomo_gate.py`
  threads through to the reader factory. Default = v1 so the Phase 1.5
  baseline is preserved.
- Component-level instrumentation added to the gate report:
  `reader_refusal_count` (rows where output matches `/^not stated\.?$/i`)
  and `reader_refusal_rate` per instance. Lets the stacked-ablation
  comparison show *refusals dropping*, not just F1 moving.

## Pass bar

- **Minimum lift** F1 > 0.47 under judge-mode on conv-26 single-hop with
  R3 standalone (reader-v2 on, R1/R2 off, same P3 recall).
- **Target** F1 ≥ 0.60 — opens the 2×2 ablation against R1 and R2.
- **Kill criterion** F1 ≤ 0.42 *and* refusal count unchanged (≥4 "not
  stated" rows) → prompt rewrite had no behavioral effect on the model;
  move to R4 (extractor swap) or a reader-model swap (qwen-max → opus).
  Note: a prompt rewrite that drops refusal count but also drops F1 is
  NOT a kill — it means the rebalance over-rotated to hallucination and
  needs tightening, not retiring.

## Risk

- **Hallucination surface widens.** Loosening the "not stated" gate may
  let the model extract a wrong-but-confident answer on rows where
  fragments mention the topic imprecisely. Mitigation: refusal count +
  per-row judge comparison on known Bucket B rows (q37 sunset, q40 "2"
  that R1/R2 broke) will catch this directly.
- **Prompt-engineering ceiling.** If the model (qwen-max) has a trained
  conservative extraction bias from RLHF, prompt-only rebalancing may
  not shift behavior. In that case the refusal count stays ≥4 and R4
  (model swap) becomes the next lever.
- **No stopword scrubbing on fragment text.** Fragments are passed
  verbatim; prompt relies on model to find the signal. Independent of R3,
  not a new risk introduced here.

## Outcome (2026-10-06)

Both standalone (R3 only) and stacked (R2 + R3) cells of the 2×2 ablation
ran on conv-26 single-hop, judge mode, 32 Q.

| Cell | R2 | R3 | F1 | Refusals | Δ F1 vs P3 (0.469) |
|---|---|---|---|---|---|
| B (R3 only) | off | on | **0.438** | 2/32 (6.2%) | −0.031 |
| D (R2 + R3) | on | on | **0.438** | 2/32 (6.2%) | −0.031 |

- Artefacts: [e1-rag-conv26-reader-v2-rerank-judge.json](e1-rag-conv26-reader-v2-rerank-judge.json)
  (Cell B); [e2-rag-conv26-r2-r3-stacked.json](e2-rag-conv26-r2-r3-stacked.json)
  (Cell D).
- **R3 did its structural job.** Refusal count 4 → 2 (−50%). The three
  rows where the lift landed: q3 (Adoption agencies) correctly extracted,
  q48 (bowls / cup) and q66 (marshmallows) extracted on imprecise-match
  pools where v1 had refused.
- **R3 traded refusals for hallucinations.** Two previously-correct rows
  regressed: q11 Sweden → "home country" (reader picked an over-generic
  place answer when the fragment named the country) and q40 "2" → "once
  or twice a year" (reader output a frequency instead of a count on a
  "how many" cue).
- **Net F1 Δ = −0.031.** Below the 0.47 minimum lift bar in the pass bar
  above. Per the pass bar's own clause: *"a prompt rewrite that drops
  refusal count but also drops F1 is NOT a kill — it means the rebalance
  over-rotated to hallucination and needs tightening, not retiring."*
  Status: **tighten, not kill**. Next lever = R5 (reader prompt v3
  precision tightening — bias "where" cues toward proper nouns, bias
  "how many" cues toward integers).
- **Stacked cell D = Cell B byte-for-byte.** Same F1, same refusal count,
  same seven incorrect rows, same candidate strings. Documented in
  [../comparison.md](../comparison.md) as the decisive R2 null-op
  verdict: with the reader unblocked, BM25 produces zero behavioral
  change on this corpus. The RRF-k=60 fused top-50 is cosine-dominated.

**Persistent broken rows after R3 (both B and D):** q7 Single, q11
Sweden, q34 Mentoring, q40 "2", q71 "Becoming Nicole", q75 "3", q76
"19 October 2023". These split into:
- *Reader-precision failures* (gold is in the pool, reader mis-extracts
  under paraphrase pressure): q11, q40, q71, q75. Target of R5.
- *Suspected recall-channel failures* (gold fragment may never reach
  top-50 under any current config): q7, q34, q76. Need
  recall@50-for-gold instrumentation (R6) before investing further.

**Shipped state.** `--reader-prompt-variant {v1,v2}` CLI flag on
`run_locomo_gate.py` and `prompt_variant: str = "v1"` kwarg on
`LLMReader`. Default = v1 (Phase 1.5-prime baseline preserved). v2 lives
as a reproducible positive-but-insufficient result alongside v1. Not
deleted.