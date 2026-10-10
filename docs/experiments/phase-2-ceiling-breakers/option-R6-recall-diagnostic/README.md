# R6 — recall@K diagnostic (classify every incorrect row)

## Hypothesis

After the 2×2 (R2 × R3) ablation landed, the Phase 2 verdict table left
seven persistent incorrect rows on conv-26 single-hop (q7 Single, q11
Sweden, q34 Mentoring, q40 "2", q71 "Becoming Nicole", q75 "3", q76 "19
October 2023"). The `comparison.md` diagnosis split these informally into
"reader-precision failures" and "suspected recall-channel failures" but
without a per-row signal to confirm the split. R5 (reader prompt v3
precision tightening) only helps the reader-precision subset. A new
recall channel (R7) only helps the recall-missing subset. **Without R6
we are prescribing medicine for a diagnosis we have not actually made.**

## Mechanism

For every question the runner computes an auxiliary dict alongside the
F1 score. The dict carries three independent signals extracted from the
fragments the system actually surfaced to the reader:

1. **`evidence_sessions`** — tuple of LoCoMo session IDs the authors
   tagged as the gold source (from `question.metadata['evidence']`).
2. **`evidence_session_in_top_k`** — True iff any top-K fragment comes
   from one of those sessions. Fragment session tags are set by the
   LoCoMo adapter in `_ingest_history` (`tags=(turn.session_id,)`), so
   the lookup is lossless. This is session-level recall@K.
3. **`gold_substring_in_top_k`** — True iff the normalized gold string
   appears as a substring of the normalized concatenation of top-K
   fragment texts. SQuAD-style normalization (same path as the
   single-hop F1 scorer), so a positive signal here is a direct test of
   "the reader literally had the correct string in-context".
4. **`gold_token_coverage`** — fraction of unique normalized gold
   tokens that appear anywhere in the pool. Softer than the substring
   check — catches multi-token golds like "19 October 2023" where "19"
   or "2023" alone may surface without the full phrase.

## Classification buckets

Every row is placed in exactly one of four buckets by the gate
aggregator:

| Bucket | Trigger | Interpretation |
|---|---|---|
| `gold_in_pool` | `gold_substring_in_top_k=True` | Reader saw the answer. Incorrect rows here are **reader-precision** failures. Fix on the reader-prompt / model side (R5). |
| `session_present_gold_absent` | evidence session hit, gold string absent | Right session surfaced, wrong turn ranked. **Fragment-ranking** is the binding constraint. BM25 / cross-encoder tweaks live here. |
| `recall_missing` | neither evidence session nor gold string in top-K | **Recall-channel** failure. New mechanism needed — proper-noun entity index, dedicated recall channel for persistent gold (R7). |
| `no_evidence_metadata` | question did not ship evidence IDs | Signal only partially usable; falls back to lexical check alone. |

## Pass bar

R6 is **not an F1 lever** — it's instrumentation. "Pass" is defined by
the diagnostic it produces, not by any lift it generates.

- **Minimum**: all 32 conv-26 single-hop rows carry a non-null
  `recall_diagnostics` dict after a Cell B rerun; the aggregator reports
  non-zero counts in at least two of the three real buckets (not
  `no_evidence_metadata` only).
- **Target**: the 7 persistent broken rows classify into at least 2
  distinct buckets. If all 7 land in `gold_in_pool`, the entire remaining
  ceiling is reader-precision (R5 is the only next lever that can help).
  If some land in `recall_missing`, R7 is justified.
- **Kill criterion**: R6 cannot be "killed" — it's a measurement, not a
  mechanism. If the LoCoMo evidence metadata is too sparse or
  misaligned with the fragment-session tagging to produce useful signal
  (e.g., ≥50% rows land in `no_evidence_metadata`), the instrument
  needs rework before R5/R7 priority decisions can lean on it.

## Risk

- **Session-level overlap is coarse.** A conv-26 session contains tens
  of turns; a fragment from "the right session" doesn't guarantee the
  specific gold-naming turn is in-pool. Mitigation: the gold-substring
  check is the stronger signal — the session check only tells us whether
  the right time window surfaced at all.
- **SQuAD normalization is lossy.** Numeric golds like "2" collide with
  any "2" token anywhere in the pool, so `gold_substring_in_top_k` may
  false-positive on short numeric golds. Mitigation: the aggregator
  reports token coverage alongside substring presence; triage any
  surprising `gold_in_pool` reading on numeric-gold rows manually.
- **Fragment-session tagging contract.** The diagnostic depends on
  `Fragment.tags` carrying the session_id. If a future Phase-2 recall
  channel strips or rewrites tags, the instrument will silently degrade
  to lexical-only. The unit tests in `tests/test_recall_diagnostics.py`
  pin the current contract so a regression surfaces as a test failure.

## Outcome (2026-10-06)

Reran Cell B (R3-only: consolidation + cross-encoder rerank + judge
single-hop + reader-v2) with R6 diagnostics attached. F1=0.422, refusals
2/32 — same shape as the prior Cell B rerun; the small F1 drift from
0.438 to 0.422 is one judge roll on a borderline row (judge-cache 30/32
hits), well inside the ≈0.03 single-instance noise band.

Artefacts: [e1-r3-with-r6-conv26.json](e1-r3-with-r6-conv26.json)
(first rerun, pre-normalization) and
[e2-r3-with-r6-normalized-conv26.json](e2-r3-with-r6-normalized-conv26.json)
(post-normalization, final numbers).

**Mid-run finding — LoCoMo evidence format.** The first rerun classified
5/7 broken rows as `recall_missing`. On inspection the LoCoMo native
evidence field ships per-turn addresses (`D9:2`, `D3:3`, …) rather than
`session_<N>` IDs. Our fragment tags carry `session_<N>` from the
adapter's ingest contract, so the raw overlap check was format-mismatched
and spuriously False. Added `_normalize_evidence_sessions` to map
`D<N>:<T>` → `session_<N>` before the overlap test. The second rerun
(`e2-…json`) is on the normalized diagnostic and is the one to cite.

### Verdict table on the 7 persistent broken rows

| Bucket | Count | Rows |
|---|---|---|
| `gold_in_pool` (reader-precision) | **2** | q7 Single (cand "not stated"); q11 Sweden (cand "home country") |
| `session_present_gold_absent` (within-session ranking) | **5** | q34 Mentoring program school speech; q40 "2"; q71 "Becoming Nicole"; q75 "3"; q76 "19 October 2023" |
| `recall_missing` (new-channel) | **0** | — |
| `no_evidence_metadata` | **0** | — |

### Diagnosis — the ceiling is within-session ranking, not recall-channel

- **Reader-precision upper bound = +2/32 F1 (+0.0625).** R5 (reader
  prompt v3 tightening — bias "where" cues toward proper nouns, bias
  "how many" cues toward integers) targets q7 and q11. Even if both fix,
  F1 climbs from 0.438 → 0.500.
- **Within-session ranking is the real ceiling.** For 5 of 7 broken rows
  the evidence session reaches top-50 but the specific gold-naming turn
  does not. The gold token coverage is 0.0 on q40, q71, q75, q76 and
  only 0.25 on q34 — the specific turn with the gold answer is being
  ranked below the top-50 cut-off within an already-widened session
  pool. This is a brand-new structural lever, not R5 and not R7.
- **R7 (new recall channel for proper nouns) is no longer motivated**
  as originally scoped. With 0 recall-missing rows, there is no row the
  current pipeline fails to surface *any* evidence for. The proper-noun
  recall problem is not "the right session doesn't surface"; it is "the
  right turn-within-session doesn't rank high enough".

### Implied next moves

1. **R5** — reader prompt v3 (cheap). Upper bound +0.0625 F1. Still worth
   shipping to clear the two unambiguous reader-precision rows.
2. **R8 (new) — within-session fragment ranking.** Candidate mechanisms:
   (a) BM25 applied *within each top-ranked session* instead of globally
   (BM25-R2 was global, and the normalized R6 shows that's not the right
   grain); (b) a "fact-density" ranking signal that surfaces turns
   containing proper nouns, dates, or numeric content ahead of chatty
   filler turns; (c) a two-stage rerank — first by session, then by turn
   within session, instead of flat top-50 cosine. Any of these targets
   the 5-row bucket and has upper bound +0.156 F1 (0.438 → 0.594).
3. **R6 is a measurement, not a mechanism** — it stays shipped as
   instrumentation on every future gate run so the next ablation can be
   classified without further rework.

### Residual caveats on the diagnostic itself

- The `D<N>:<T>` → `session_<N>` normalization is a mapping *assumption*
  on the LoCoMo native schema; validated by the data (evidence tokens
  observed in conv-26: D3, D6, D7, D9, D10, D17, D18 — all within the
  session range of the instance). If a future LoCoMo revision breaks
  this convention, the normalization will degrade silently. The lexical
  check is independent and still provides a lower-bound signal.
- `gold_substring_in_top_k` false-negatives on short numeric golds that
  are spelled out in dialogue ("two" / "2"). The 0.0 token coverage on
  q40 and q75 may undercount — the gold *concept* may be in-pool even
  if the gold *string* is not. For those two rows the honest reading is
  "the specific numeric literal isn't present"; a reader forced to count
  explicit list items may still answer correctly. R5 should include a
  "how many" counting template.
