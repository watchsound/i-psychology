# R1 — rerank over all fragments

## Hypothesis

Phase 1.5-prime P3 (cross-encoder rerank at K=150 → 50) fixed 2/7 Bucket A
proper-noun failures (q11 Sweden, q55 Sunsets) but left 5 others untouched.
Per-row diagnosis in
[../../phase-1.5-prime-structural-pivot/comparison.md](../../phase-1.5-prime-structural-pivot/comparison.md)
showed the gold fragment is cosine-ranked **past 150** on those 5 — rerank
never sees it.

If the ceiling is cosine prefilter width (not cross-encoder quality), raising
the pool to all 419 fragments should fix additional Bucket A failures and
push F1 past 0.47.

## Mechanism

New flag `CompanionSystem(cross_encoder_rerank_all=True)` (requires
`cross_encoder_rerank=True`). Changes a single cosine-pool knob in the recall
path: `cos_top_k = len(fragments)` instead of `HIERARCHICAL_RERANK_POOL_SIZE`.
Rerank API trims to the final `HIERARCHICAL_FRAGMENT_TOP_K` (50) as before.

Cost model: DashScope `gte-rerank-v2` accepts up to 500 documents per call.
conv-26 has 419 fragments — fits in a single call. Per-query rerank input
scales linearly with pool size; 419/150 ≈ 2.8× more rerank tokens than P3,
no latency hit (same single API call).

CLI: `python run_locomo_gate.py ... --cross-encoder-rerank --cross-encoder-rerank-all`.

## Pass bar

- **Minimum lift** F1 > 0.47 under judge-mode (strict improvement over P3).
- **Target** F1 ≥ 0.60 to warrant stacking with R2–R4; below that, suspect
  cosine recall itself is losing proper nouns before rerank can fire.
- **Kill criterion** F1 ≤ 0.47 → rerank-at-150 is already saturated; R1 is
  dead and the ceiling-breaker budget shifts to R2 (sparse recall).

## Risk

- 419-document rerank input may exceed DashScope's per-document token budget
  if any single fragment is unusually long. Mitigation: dialogue fragments
  average ~50 tokens; worst-case ~500. Should fit.
- Rerank cost per query ~2.8× P3. 32-question gate on conv-26 is still cheap
  (~$0.05 total at DashScope rerank pricing).

## Standalone result 2026-10-06 — F1=0.391, pending stacked ablation

**F1=0.391 under judge-mode on conv-26 single-hop, Δ=−0.078 vs P3.**
Preliminary read was "killed — regressed", but that verdict is **retracted**
pending stacked ablation. Three reasons:

1. **Serial-bottleneck confound.** The downstream reader refuses on
   `"not stated"` for cue-matched rows. Three of R1's incorrect rows (q3,
   q7, q34 Bucket A) have the gold-containing fragment in context yet
   produce `"not stated"`. Widening the rerank pool cannot *not* fail if
   the reader absorbs the surfaced gold as a refusal.
2. **Noise band.** 32-question single-instance has judge-mode noise ~0.03;
   Δ=−0.078 is 2.5× noise — meaningfully negative *as a standalone test*,
   but may still net-contribute inside a stacked architecture once the
   reader is unblocked.
3. **No component-level measurement.** We measured end-to-end F1 only.
   Rerank-pool widening may have surfaced Bucket A gold into the top-50
   even while F1 fell — the direct test (recall@50 for gold fragments)
   was never logged.

**Corrected test plan (next):**
- Instrument `run_locomo_gate.py` to log recall@50 for gold-containing
  fragments and reader refusal rate per row.
- Build R3 (reader prompt redesign) to unblock the downstream refusal
  gate.
- Run 2×2 ablation vs R3-on baseline:
  `{R1 off, R3 off}` (P3) / `{R1 off, R3 on}` / `{R1 on, R3 off}` (today's
  R1) / `{R1 on, R3 on}` ← the real test.
- Expand to ≥2 LoCoMo instances before any final verdict.

Shipped state: `cross_encoder_rerank_all` flag retained in
[src/psychology_base/companion/system.py](../../../../src/psychology_base/companion/system.py)
+ `run_locomo_gate.py`. Default off. Flag + artefact both retained for the
upcoming stacked ablation.
