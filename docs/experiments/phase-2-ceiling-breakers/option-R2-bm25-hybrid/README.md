# R2 — BM25 + semantic hybrid recall (rank fusion)

## Hypothesis

R1 killed the pool-widening approach: cross-encoder discrimination degrades
past ~150 candidates. The remaining failure mode is Bucket A proper nouns
(Adoption / Single / Mentoring / 19 October 2023) where the gold fragment
is cosine-ranked past 150 — rerank *can* fix them if they reach the pool,
but widening to 419 pulls in too many distractors.

**Dense recall buries proper nouns under paraphrase pressure. Sparse recall
does the opposite.** BM25 scores by literal token overlap: a cue mentioning
"Sweden" ranks any fragment containing "Sweden" at the top regardless of
semantic paraphrase. If we merge BM25's top-K with cosine's top-K via
Reciprocal Rank Fusion (RRF), proper-noun gold rises into the top-150 pool
*before* the cross-encoder sees it — the cross-encoder then does the
discrimination it does well, over a mixed-channel pool that already contains
gold.

## Mechanism

1. **BM25 index** built once per instance over tokenized fragment contents.
   Tokenizer: lowercase, strip non-word chars, split on whitespace. No
   stopword removal in v1 — stopwords in dialogue are rare and the gold
   phrases we care about (proper nouns, dates) survive any sane filter.
2. **Hybrid rank fusion** via RRF with k=60 (standard):
   `rrf_score(frag) = 1/(60 + bm25_rank) + 1/(60 + cosine_rank)`
   Fragments appearing in only one channel get a single term; fragments in
   both channels sum. Returns top-N by RRF score.
3. **Pool sizing** preserves P3's top-150 cap. The hybrid returns top-150;
   the cross-encoder reranks that pool to the final top-50. All downstream
   substring-augmentation and reader paths are unchanged.
4. New kwargs: `CompanionSystem(bm25_hybrid=True)`. CLI flag
   `--bm25-hybrid` on `run_locomo_gate.py`. Works with or without
   `--cross-encoder-rerank`; recommended combo is both.

## Pass bar

- **Minimum lift** F1 > 0.47 under judge-mode (strict improvement over P3).
- **Target** F1 ≥ 0.60 to warrant stacking R3/R4; below that, suspect
  reader-prompt refusal or fact-store quality is the binding constraint.
- **Kill criterion** F1 ≤ 0.47 → sparse+dense fusion saturates at the same
  ceiling as dense alone; move to R3 (reader prompt).

## Risk

- **BM25 index build cost** scales with corpus size. 419 fragments ×
  ~50 tokens avg = ~21K tokens of indexing per instance. One-time cost per
  gate run; negligible.
- **Tokenization mismatch** — if cue tokens drop punctuation differently
  from fragment tokens, "19 October 2023" could tokenize to
  ["19", "october", "2023"] vs ["october", "19"] in a differently-formatted
  fragment. Mitigation: identical tokenizer for both sides, no stemming.
- **RRF constant k=60** is standard but not tuned. If R2 just-misses the
  gate, retry with k=30 (favors top-rank evidence) or k=100 (flattens
  late-rank differences).

## Standalone result — F1=0.422, pending stacked ablation (verdict retracted)

**F1=0.422 under judge-mode on conv-26 single-hop, Δ=−0.047 vs P3.**
Preliminary read was "killed — regressed", but that verdict is **retracted**
pending stacked ablation. Three reasons:

1. **Serial-bottleneck confound.** 4 of 10 R2-incorrect rows have
   `candidate="not stated"` — the reader refused on top-50 pools that
   likely contained BM25-surfaced gold. BM25 could have been doing its
   job while the downstream reader absorbed the lift as refusals. Testing
   R2 in isolation with the reader refusal uncorrected cannot distinguish
   "BM25 doesn't help" from "BM25 helps but refusal eats it".
2. **Noise band.** 32-question single-instance has judge-mode noise
   ~0.03 F1. Δ=−0.047 is 1.5× noise — not decisive, and well within range
   where a stacked variant could flip the sign.
3. **Design under-tested.** The R2 README listed k=30 and k=100 as RRF
   fallbacks if the primary k=60 just-missed. We did not run them. We
   also never measured whether BM25 was surfacing Bucket A gold into the
   top-50 — the direct test of BM25's job (recall@50 for gold) was not
   logged. We measured end-to-end F1 only, which conflates every stage.

**What the standalone result does rule out:** k=60 RRF alone with the
current strict reader does not raise F1 on conv-26 single-hop. It does
*not* rule out BM25 as a contributing component in a stacked
architecture.

Row-level deltas vs P3:
- **Fixed**: *zero*. Not one of the Bucket A targets (q3 Adoption, q7
  Single, q34 Mentoring, q76 "19 October 2023") was recovered. BM25 did
  surface literal-token matches — q11 Sweden stayed correct across both
  P3 and R2 — but the Bucket A proper-noun golds were still either
  (a) not surfaced into the final top-50 after RRF+rerank, or (b)
  surfaced but passed over by the reader with `candidate="not stated"`.
- **Broke**: q37 sunset (P3 correct → R2 wrote "Melanie painted an
  abstract painting recently") and q40 "2" (P3 correct → R2 "not stated").
  RRF reshuffled the top-50 and swapped out these previously-correct
  fragments.
- **Both broken**: 8 rows shared with P3 — the same core failure set.

Reading the "not stated" pattern. Four of the ten R2-incorrect rows
(q3, q7, q34, q40, q76) have `candidate="not stated"`. That's a *reader
refusal*, not a recall miss — the reader saw the DIALOGUE block and
declined to extract. This flips the diagnosis:

> The ceiling is not recall channel (R1 pool width ✗, R2 sparse+dense ✗).
> The ceiling is the reader's refusal gate — the "if not stated, say so"
> instruction fires on Bucket A proper nouns even when the gold fragment
> is in the top-50.

**Corrected next move.** The budget shifts to **R3 (reader prompt
redesign)** — not as a replacement for R2, but as the downstream fix
that unblocks measurement of upstream components. Specific R3 move:
loosen the "not stated" gate for cue-matched extraction (if the cue
mentions "Sweden" and a dialogue line names "Sweden", prefer that line
over refusal), while keeping the gate active for truly absent info.

After R3 lands, re-run R2 in a 2×2 ablation against the R3-on baseline:
`{R2 off, R3 off}` (today's P3), `{R2 off, R3 on}`, `{R2 on, R3 off}`
(today's R2), `{R2 on, R3 on}`. The last cell is the real R2 test.
Expand to ≥2 LoCoMo instances before any kill verdict. Also run the
k=30 and k=100 fallbacks as part of the stacked sweep if the k=60
stacked variant still falls short.

R2 shipped state: `bm25_hybrid` flag retained on `CompanionSystem` +
`run_locomo_gate.py`. Default = off. Flag + artefact retained for the
upcoming stacked ablation.
