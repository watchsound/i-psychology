# Phase 1.5-prime — structural pivot sweep (2026-10-06)

Phase 1.5 Option 5.C hit an architectural ceiling at **F1=0.288 (v3)** on
`conv-26` single-hop against a 0.40 gate (see
[../phase-1.5-scale-recall/comparison.md](../phase-1.5-scale-recall/comparison.md)).
Per the Phase 1.5 decision rule in [../../roadmap.md](../../roadmap.md),
E1 failure after one iteration triggers an option re-brainstorm. Three
structural pivots named at the time of the ceiling verdict:

| ID | Pivot | Hypothesis |
|---|---|---|
| **P1** | Judge-mode F1 redefinition | v3's "Bucket B" paraphrase answers are factually correct per the retrieved context; token F1 artefactually zeros them. A judge-based scorer may clear 0.70 on the same v3 recall + reader. If so, the ceiling was a scorer bug, not a substrate bug. |
| **P2** | Embedder ensemble `max(mxbai, dashscope)` per candidate | v4 DashScope swaps per-query failures rather than eliminating them (Sweden rank 75→8 but Single 18→104). An ensemble that takes the max cos per candidate across both embedders should capture each embedder's strengths without inheriting its weaknesses. |
| **P3** | Cross-encoder rerank (DashScope `gte-rerank-v2`) | mxbai is a bi-encoder; its paraphrase-cue failures are a known bi-encoder weakness. A cross-encoder reranker scores (cue, candidate) jointly and can lift top-K fragments past the cosine-rank wall. Known to deliver 10-15 F1 points on QA benchmarks. |

**Running order:** per the user directive "you need to try all of them"
(2026-10-06), each pivot runs on `conv-26` single-hop under the P1 judge
gate so Bucket B artefacts don't poison the comparison. Target: ≥ 0.70
(softened Phase 1.5 E2 target). If a pivot clears this on E1, promote to
E2 full 10-instance gate.

## Headline table

| Config | Scorer | F1 (conv-26 single-hop, 32 Q) | Pass vs ≥ 0.70? | Wall | Notes |
|---|---|---|---|---|---|
| v3 champion (reference) | token F1 | **0.288** | ✗ | 33s | from `../phase-1.5-scale-recall/v3` |
| **P1** v3 config under judge | LLM judge (qwen-max) | **0.406** | ✗ | 100s | +0.118 lift confirms Bucket B artefact exists; failure split = 6 Bucket A + 5 Bucket C close-wrong |
| **P2** embedder ensemble `max(mxbai, dashscope)` | LLM judge | **0.438** | ✗ | 165s | +0.03 over P1 (+1 Q). Fixes q40 (Bucket C), regresses on q61 (new "not stated"); 6 original Bucket A proper nouns persist. Ensemble hypothesis invalidated — mxbai + DashScope share the paraphrase-cue weakness. |
| **P3** cross-encoder rerank over cosine top-150 | LLM judge | **0.469** | ✗ | 154s | +0.06 over P1 (+2 Q). Fixes q11 Sweden + q55 Sunsets (Bucket A proper nouns rerank pulls past cosine wall). Still misses 5 other Bucket A failures — gold fragment isn't in the top-150 cosine pool for rerank to see. |

## Verdict (2026-10-06)

**All three pivots fail the 0.70 gate.** The architectural ceiling under
Phase 1.5 Option 5.C with the current extraction/recall/reader stack is
F1 ≈ 0.47 on `conv-26` single-hop. This is a 63% improvement over v3
(0.288 → 0.469) once scoring + rerank are stacked, but still 23 points
below the 0.70 Mem0-parity target.

**P3 is the strongest single pivot.** +0.063 over P1 at the cost of 32
cross-encoder rerank calls (~$0.03 per gate). P2 is marginal (+0.032)
and regresses one question (q61) — the ensemble hypothesis that mxbai
and DashScope errors are complementary is **falsified** on this test
set.

**What P3 doesn't fix tells us the next architectural lever.** The 5
Bucket A failures that persist under P3 — q3 Adoption, q7 Single, q34
Mentoring, q61 Summer Sounds, q76 date — all score 0.0 because the
reader returns "not stated". The gold fragment is either (a) not in
the top-150 cosine pool rerank sees, or (b) in the pool but the
reader can't bind the paraphrased cue to the matched token. (a) is a
retrieval problem solvable by widening the pool to the whole raw
store (419 fragments on conv-26 — well within rerank API limits) or
by adding BM25 as a parallel recall channel. (b) is a reader-side
problem no retrieval change can fix.

**Phase 1.5-prime does not clear E2.** E2 (10-instance gate at 0.70)
is not launched — predictable fail would burn ~$30 in reader + judge
tokens for no new signal. Phase 1.5-prime shipping state: P3 code
preserved, defaults restored to v3 (consolidation on, ensemble off,
rerank off). The three structural levers are retained as opt-in
flags so Phase 2 can compose them with additional mechanisms.

**Phase 2 structural levers that break the ceiling** (candidate list,
not yet prioritised):
- Rerank over *all* fragments (not cosine top-150). conv-26 has 419
  raw turns — one rerank call per cue stays within DashScope limits.
- BM25 + semantic hybrid recall. Proper nouns are keyword problems;
  cosine is a terrible first filter for them. BM25 as a parallel
  channel catches "Sweden" / "Adoption" / "Single" cleanly.
- Reader instruction redesign — current reader is overly conservative
  with "not stated" when evidence is adjacent but not exact. A reader
  tuned to *attempt* an answer from in-pool evidence (and the judge
  catches wrong attempts anyway) may lift Bucket C.
- Fact extraction redo — the Mem0 reference tier uses gpt-4o-mini;
  qwen-turbo on SIMPLE drops specific proper nouns at session scale.
  Switching extractor to qwen-max (ADVANCED) adds cost but may fix
  Bucket A at source rather than retrieval.

## P1 — Judge-mode F1 redefinition (shipped, F1=0.406)

Code change: `--judge-single-hop` flag on
`run_locomo_gate.py`; threads `force_judge_categories={"single-hop"}` into
`run_rag_mode`'s `_score` dispatch so single-hop routes to `LLMJudge`
instead of `token_f1`. Judge model = qwen-max (ADVANCED tier); judge
prompt v1-2026-10-05 (SHA-pinned, cache-safe).

Reader cache hits on identical prompts; only judge calls are new cost
(~32 ADVANCED calls per instance, cached by `(benchmark, instance_id,
question_id, gold, candidate, judge_id, judge_prompt_sha)`).

**Result:** F1 = 0.40625 (13/32 PASS). Lifted +0.118 vs token F1 (0.288)
on the same recall + reader, confirming Bucket B (paraphrase-correct)
answers were present but artefactually zeroed by token overlap.
Remaining 11 incorrect split by failure mode:

| Mode | Count | Examples (question_id → gold vs candidate) |
|---|---|---|
| Bucket A — reader says "not stated" | 6 | q3 Adoption agencies, q7 Single, q11 Sweden, q34 Mentoring, q55 Sunsets, q76 date |
| Bucket C — close but wrong fact | 5 | q40 "2" vs "once or twice", q48 bowls vs pots, q66 marshmallows vs nature, q71 Becoming Nicole vs Charlotte's Web, q75 "3" vs "2" |

Bucket A is the P2/P3 target. Bucket C is harder — it's a reader-side
reasoning + grounding problem, not a retrieval problem. If P2/P3 fix
Bucket A to zero while leaving Bucket C untouched, F1 ceiling is
(32 - 5) / 32 = **0.844** — well clear of 0.70.

Artefact: [option-P1-judge-mode/e1-rag-conv26-v3-under-judge.json](option-P1-judge-mode/e1-rag-conv26-v3-under-judge.json)

### Does P1 alone clear 0.70?
No. 0.406 < 0.70. Judge-mode is necessary (closes the scorer artefact)
but not sufficient (substrate still misses 6 proper-noun facts). Move
to P2 + P3.

## P2 — Embedder ensemble (code shipped, gate pending)

**Code state (2026-10-06):**
- `LLMRouter.embed(..., backend="local"|"cloud"|"auto")` — new kwarg.
  "local" hard-fails if Ollama isn't up; "cloud" hard-fails if the
  active provider doesn't implement `embed()`.
- `LLMRouter.embed_batch(..., backend=...)` — same kwarg threaded through
  the thread pool.
- `CompanionSystem(embedder_ensemble=True)` — new flag. When on:
  - Cue is embedded under both backends.
  - Facts: each fact's local vector (`Fact.embedding`) is paired with a
    cloud vector held in a per-system `_cloud_fact_embeddings` side-car
    (populated lazily on first ensemble recall; disk-cached by text).
    Per-fact score is `max(cos_local, cos_cloud)`.
  - Fragments: `_cosine_rank_fragments` is re-implemented in the
    `_ensemble_recall` method as a dual-backend ranker that embeds the
    corpus under both backends and takes `max` per fragment.
- `run_locomo_gate.py --embedder-ensemble` CLI flag.

**Cost shape per gate run (conv-26, 419 fragments, 99 facts):**
- First recall: ~518 cloud embedding calls (fragments + facts). After
  that the router disk cache serves everything (99 cue embeddings × 2
  = 198 incremental cloud calls for all remaining questions).
- Local calls: identical to v3 (mxbai hot on Ollama).

**Gate pending:** blocked on `ALIBABA_TONGYI_API_KEY` not being
available in the current shell. Reproduce with:
```
export ALIBABA_TONGYI_API_KEY=sk-...   # or add to .env (now auto-loaded)
python docs/experiments/phase-1-benchmark-adapters/option-gamma/run_locomo_gate.py \
  --mode rag --consolidation --instance-id conv-26 \
  --categories single-hop --gate-target 0.70 \
  --judge-single-hop --embedder-ensemble \
  --output docs/experiments/phase-1.5-prime-structural-pivot/option-P2-embedder-ensemble/e1-rag-conv26-ensemble-judge.json
```

## P3 — Cross-encoder rerank (code shipped, gate pending)

Model/endpoint confirmed via the DashScope EN docs (2026-10-06): the
legacy `https://dashscope.aliyuncs.com/api/v1/services/rerank/
text-rerank/text-rerank` endpoint with `model: "gte-rerank-v2"` and
the nested `input/parameters` request shape is live and documented.
`DASHSCOPE_RERANK_MODEL` + `DASHSCOPE_RERANK_URL` env overrides are
retained as a future-proofing hook (DashScope has signalled
qwen3-rerank as the long-term successor) but no override is needed
today.

**Code state (2026-10-06):**
- `AlibabaTongyiProvider.rerank(query, documents, top_n)` — new method.
  POST `https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank`
  with `{model: "gte-rerank-v2", input: {query, documents}, parameters:
  {top_n, return_documents: false}}`. Returns `[(index, score)]`
  sorted by score desc.
- `LLMRouter.rerank(query, documents, top_n)` — pass-through to active
  provider. Hard-fails if provider doesn't implement rerank.
- `CompanionSystem(cross_encoder_rerank=True)` — new flag. When on:
  - Cosine recall widens from top-50 to top-150 fragments (new constant
    `HIERARCHICAL_RERANK_POOL_SIZE = 150`).
  - Those 150 fragments are reranked jointly against the cue via
    `router.rerank`, trimmed to final top-50 by cross-encoder score.
  - Composable with `embedder_ensemble=True` — rerank runs after
    ensemble produces its 50-fragment list, in which case the pool to
    rerank is 50 (not 150). For isolated P3 measurement, run P3 alone.
- `run_locomo_gate.py --cross-encoder-rerank` CLI flag.

**Cost shape per gate run:**
- 32 rerank calls per gate (one per cue). Each rerank POSTs 150 docs ×
  ~50 tokens + a 20-token query ≈ 7.5K tokens request body. DashScope
  rerank pricing ≈ $0.01 per 1K input tokens → ~$2.4 incremental.
- No embedding call delta vs v3 (same mxbai cosine pool, just wider).

**Gate pending:** same `ALIBABA_TONGYI_API_KEY` requirement. Reproduce:
```
python docs/experiments/phase-1-benchmark-adapters/option-gamma/run_locomo_gate.py \
  --mode rag --consolidation --instance-id conv-26 \
  --categories single-hop --gate-target 0.70 \
  --judge-single-hop --cross-encoder-rerank \
  --output docs/experiments/phase-1.5-prime-structural-pivot/option-P3-crossencoder-rerank/e1-rag-conv26-rerank-judge.json
```

## Decision log

- 2026-10-06: sweep opened. P1 (judge-mode) ran first (cheapest) and
  returned F1=0.406, confirming the Bucket B scorer artefact without
  clearing 0.70.
- 2026-10-06: user directive "you need to try all of them" — P2 + P3
  code shipped; gate runs blocked on API key access.
- 2026-10-06: lightweight `.env` auto-loader added to `llm/config.py`
  so the key can be persisted locally without re-exporting per shell.
- Pending: P2 + P3 gate runs on `conv-26` single-hop under judge-mode.
  Final verdict table updates after both complete.
