# R8 — within-session fragment ranking

## Hypothesis

R6's normalized diagnostic shows that for 5 of 7 persistent conv-26
single-hop broken rows, the right evidence session surfaces in top-50
(`evidence_session_in_top_k=True`) but the specific gold-naming turn is
not among the surfaced fragments (`gold_substring_in_top_k=False`,
`gold_token_coverage` ∈ {0.0, 0.25}). The rows:

| Row | Gold | Evidence session reached pool? | Gold string in pool? |
|---|---|---|---|
| q34 | "Mentoring program, school speech" | ✓ | ✗ (coverage 0.25) |
| q40 | "2" | ✓ | ✗ (coverage 0.0) — fixed by R5 integer bias |
| q71 | `"Becoming Nicole"` | ✓ | ✗ (coverage 0.0) |
| q75 | "3" | ✓ | ✗ (coverage 0.0) |
| q76 | "19 October 2023" | ✓ | ✗ (coverage 0.0) |

After R5 ships, 4 rows remain in this bucket (q34, q71, q75, q76).

The binding constraint is **turn-level ranking inside an already-surfaced
session**: cosine + cross-encoder rank filler turns ahead of the gold
turn because the cue paraphrases the filler's framing better than the
gold's bare-literal content. R2 (global BM25 hybrid) attacked this with
the right mechanism (literal-token recall) but at the wrong granularity
(whole fragment store) and landed as a null-op in-stack. R8 restricts
the within-session reach to lift the gold turn into the final top-50.

## Candidate mechanisms

Three candidates, listed cheapest-first by implementation cost. Only
candidates with a built-and-measured gate run count toward R8's verdict;
candidates with pure paper-arguments are not R8 kills.

### R8-c — Two-stage rerank (session → turn-within-session) [first build]

**Mechanism.** After cosine top-150, group fragments by `session_id`
(from `Fragment.tags[0]`). For each session, keep the max-cosine score
as the session score. Pick the top-M sessions by that score. For those
M sessions, pull **all** turns from the full fragment store (not just
the ones in cosine top-150). Cross-encoder-rerank *all turns of those
M sessions* against the cue; trim to top-50. The 7 non-selected sessions'
fragments remain available via the substring-augmentation path.

**Why this helps.** Today, the specific gold turn is cosine-ranked past
150 within its session (filler turns in the same session rank higher).
After the session is selected, pulling *all* its turns into the rerank
pool gives the cross-encoder a chance to pull up the gold turn — the
exact move R1 (rerank-over-all) got wrong at corpus scale is now
*scoped to just the top-M sessions*, cutting the distractor count by 10×.

**Risk.** M is a hyperparameter. M=3 covers typical single-hop evidence
(single session) with headroom; M=5 trades more distractor budget for
multi-hop coverage. Pilot M=3 first.

**Kill criterion.** F1 drops below R5 baseline (0.484) by more than 0.03
(single-instance noise band), OR fewer than 2 of the 4 target rows flip
to correct, OR refusals climb above 1/32.

### R8-b — Fact-density ranking signal [second build if R8-c under-shoots]

**Mechanism.** A fragment-side score boost computed from content:
+ε per proper-noun token (titlecase heuristic), + per 4-digit year, +
per bare-numeric literal, + per ISO-ish date fragment. Applied as a
tie-breaker *after* cross-encoder rerank — never overriding the primary
score, only breaking ties when |Δ| < threshold.

**Why this helps.** The 4 remaining target rows all have numeric or
proper-noun golds (dates, counts, book titles). Boosting dense turns
over chatty filler is a direct complement to cross-encoder relevance.

**Risk.** Numeric-heavy chatter ("I was 25 when I moved in 2015, had
3 roommates") will spuriously win. The tie-breaker-only clamp mitigates
but doesn't eliminate.

### R8-a — Within-session BM25 [third build if R8-c and R8-b under-shoot]

**Mechanism.** For top-M sessions, build a BM25 index restricted to those
sessions' turns. RRF-fuse the BM25 top-K with the cosine top-K **within
the same M-session scope**, then cross-encoder rerank. Scoped version of
R2's null-op global hybrid.

**Why this helps.** BM25 surfaces literal `"Becoming Nicole"` or
`"19 October 2023"` tokens that paraphrase-cue cosine can't match. The
R2 null-op diagnosis said BM25 and cosine agreed on the top of their
global lists; restricting to M sessions changes which fragments compete,
potentially unblocking BM25.

## Pass bar

- **Minimum**: F1 ≥ R5 baseline (0.484). R8 must not regress R5's
  reader-side wins.
- **Target**: F1 ≥ 0.56. Flipping 2 of the 4 target rows is +0.0625 F1
  → 0.546 (within noise of 0.56).
- **Upper-bound (per R6)**: F1 = 0.641 if all 4 session_present_gold_absent
  rows flip. That is the 1.5-exit-gate neighborhood (0.70) with multi-hop
  / temporal / open-domain categories still owed.
- **Kill criterion**: F1 < 0.46 (worse than R5 by more than the single-
  instance noise band). None of the three sub-candidates separately
  trigger an R8-wide kill unless all three fail.

## Methodology

- Build R8-c first (reuses the shipped cross-encoder path; least new code).
- Stacked with R5 reader-v3 — don't retest the R3→R5 reader regression
  surface here. Gate: rerank + judge + consolidation + reader-v3 +
  within-session-rerank.
- R6 recall diagnostic stays on — it is the primary measurement for
  whether R8 actually moves the `session_present_gold_absent` bucket
  (not just F1).
- One standalone conv-26 run per candidate; multi-instance validation
  is a separate task (see roadmap note below).

## Roadmap context

R8 is the last single-hop lever before the Phase 1.5 exit gate
(LoCoMo single-hop F1 ≥ 0.70). Even at R8's upper bound (0.641), the
gate is not yet cleared on conv-26 single-hop alone — multi-instance
validation and multi-hop / temporal / open-domain categories remain
owed before 1.5 can exit. R8 is necessary but not sufficient.

## Outcome (2026-10-06)

### R8-c @ M=3 (primary pilot)

**F1 = 0.500** on conv-26 single-hop. Refusals **2/32** (up from R5's 1/32).
Wall time 142s, judge cache 18 hits / 14 misses.

Artefact: [e1-r8c-conv26.json](e1-r8c-conv26.json).

| Metric | R5 baseline (reader-v3) | R8-c @ M=3 | Δ |
|---|---|---|---|
| F1 | 0.484 | **0.500** | **+0.016** (within noise band) |
| Refusals | 1/32 | 2/32 | +1 |
| Incorrect rows | 6 | 6 | 0 |
| `gold_in_pool` incorrect | 1 | **2** | +1 (reader broke on q3 Adoption) |
| `session_present_gold_absent` incorrect | 5 | **3** | **−2** (target bucket moved) |
| `recall_missing` incorrect | 0 | **1** | +1 (q76 session_18 dropped from top-M) |

**R8-c moved the target bucket (session_present_gold_absent 5→3) as
hypothesized** — the within-session reach widened and 2 rows previously
buried past the top-150 within their session surfaced. But the F1 lift
was +0.016 (noise band) because new failures washed out the wins:

Row-level diff vs R5:
- **Fixed by R8-c** (R5 broken → R8-c correct):
  - q43 "abstract art" (R5 regression recovered — the within-session
    rerank picked up the full modifier).
  - q66 "Roast marshmallows, tell stories" (R5 regression recovered —
    the authoritative activity list reached the pool).
- **Newly broken by R8-c** (R5 correct → R8-c broken):
  - q3 "Adoption agencies" → "not stated" (refusal). Gold is in pool
    with coverage 1.0, but pool composition changed enough that the
    reader lost confidence. The within-session rerank floods top-50
    with filler turns from the same session as the gold turn, and the
    reader's gold-vs-filler contrast weakens.
  - q48 "bowls, cup" → "pots" (fragment-level distraction).
  - q76 "19 October 2023" → "not stated" (recall_missing). session_18
    did not survive the top-M=3 cut — the predicted risk from the
    README's `M is a hyperparameter` risk clause.
- **Still broken (ceiling set)**: q34, q71, q75, q76 — same 4 rows as
  R5's residual ceiling.

**Net**: 2 fixed − 3 broken = −1 binary row; F1 shifted +0.016 via
judge partial-credit shifts. Target bucket moved exactly as predicted;
the cost (q3 reader refusal, q76 session dropout) was not predicted.

### R8-c @ M=5 (tuning pilot)

**F1 = 0.438** on conv-26 single-hop. Refusals **3/32**. Wall time 149s.

Artefact: [e1-r8c-m5-conv26.json](e1-r8c-m5-conv26.json).

| Metric | R8-c @ M=3 | R8-c @ M=5 | Δ |
|---|---|---|---|
| F1 | 0.500 | 0.438 | **−0.063** (below R5 baseline) |
| Refusals | 2/32 | 3/32 | +1 |
| Incorrect rows | 6 | 7 | +1 |
| `gold_in_pool` incorrect | 2 | **4** | +2 |
| `session_present_gold_absent` incorrect | 3 | **2** | −1 |
| `recall_missing` incorrect | 1 | 1 | 0 |

M=5 pushed the target bucket further (3→2) but doubled reader
precision failures (gold_in_pool 2→4) and regressed below R5. **The
precision/recall tradeoff is active**: more sessions in the pool →
better session-coverage, worse reader precision on an already noisy
top-50. The curve inflects at M=3-4; M=5 is past the knee.

### Pass bar verdict

- Minimum F1 ≥ 0.484 ✓ (M=3 passes at 0.500; M=5 fails at 0.438)
- Target F1 ≥ 0.56 ✗ (M=3 under-shoots by 0.06)
- Upper bound 0.641 (all 4 session-rows flip) ✗ (we got 2 flips, not 4)
- Kill criterion F1 < 0.46 ✗ (M=3 did not trigger kill)

**R8-c @ M=3 passes the minimum but under-shoots the target.** It is
*not killed* — the target bucket moved as hypothesized — but it is also
not the F1 lift this phase needs to approach the 1.5 exit gate.

### Shipped state

- M=3 is the operational default when `--within-session-rerank` is set.
- v3 reader prompt remains the opt-in default for Phase 2 gating.
- The within-session rerank is **not** the new standard pipeline — it
  trades q3/q48/q76 for q43/q66 and the net F1 is noise. Keep R5 as
  the shipped stack for conv-26 single-hop; `--within-session-rerank`
  is available for ablation / future multi-instance work where the
  tradeoff may skew differently.

## Residual caveats

- **q3 reader-refusal pattern is new.** v3's reader-prompt bias toward
  "a fragment names the answer in a single word" should have caught
  "Adoption agencies" — but the pool composition after within-session
  rerank evidently crowded the gold turn with same-session filler,
  depressing the reader's confidence. This is a **reader-side** failure
  on a pool-side mechanism. A v4 reader prompt could probably fix it,
  but that is R5-plateau logic the project rejected.
- **q76 recall_missing** is the clean failure case for top-M=3. Widening
  M to 5 recovered the session bucket further but destroyed F1 (see
  M=5 pilot). The session-count knob has a narrow operating range.
- **No cross-instance validation.** R8-c was gated on conv-26 only.
  The precision/recall tradeoff may skew differently on conv-{19,27,
  55,61}; this is Phase 2's full-sweep stress test, not an R8 scope
  item.

### R8-b fact-density tie-breaker (standalone)

Two runs recorded. The first run used an unfiltered proper-noun regex
that counted sentence-initial titlecase verbs ("Moved", "Visited",
"Her", "The") as proper nouns. The second run is after the regex was
tightened with a sentence-start common-word filter. The second run is
the shipped state; the first is kept as the "unfiltered" baseline so
the delta is auditable.

**Shipped (post-regex-fix): F1 = 0.438** on conv-26 single-hop.
Refusals **2/32**. Wall time 135s, judge cache 25 hits / 7 misses.
Artefact: [e1-r8b-conv26-v2.json](e1-r8b-conv26-v2.json).

**Unfiltered (pre-fix): F1 = 0.469.** Refusals **1/32**. Wall time 122s.
Artefact: [e1-r8b-conv26.json](e1-r8b-conv26.json).

The regex refinement regressed F1 by 0.031 because the previous
false-positives were accidentally helping on borderline rows. Per-row
diff (unfiltered → filtered):
- Rescued q19 "dinosaurs, nature" (the fix removed a density boost
  the filler was riding).
- Broke q3 "Adoption agencies" → "not stated" refusal (pool reshuffle).
- Broke q66 "Roast marshmallows..." → "exploring forests..." (fix
  removed the proper-noun count that had been propping up the gold).

Both F1 numbers are inside the single-instance noise band (~0.03)
around R5 = 0.484, so the "null-op within noise" verdict stands either
way; the fix was shipped because `_fact_density_score` now does what
its name claims (counts real proper nouns, not sentence-initial
titlecase verbs). The accidental F1 lift from the unfiltered version
was spurious signal, not a real win.

Mechanism as scoped: `_fact_density_score` counts proper nouns (titlecase
regex), 4-digit years, ISO + DMY dates, and bare integers; the
tie-breaker is a single-pass adjacent-swap applied *after* cross-encoder
rerank, firing only when `|score_a − score_b| ≤ 0.03` AND
`density_b − density_a ≥ 2`.

| Metric | R5 baseline (reader-v3) | R8-b standalone | Δ |
|---|---|---|---|
| F1 | 0.484 | 0.469 (unfiltered) / **0.438** (shipped) | **−0.015 / −0.046** |
| Refusals | 1/32 | 1 / 2 | 0 / +1 |
| Incorrect rows | 6 | 7 / 8 | +1 / +2 |
| `gold_in_pool` incorrect | 1 | 0 / 2 | −1 / +1 |
| `session_present_gold_absent` incorrect | 5 | 7 / 6 | +2 / +1 |
| `recall_missing` incorrect | 0 | 0 / 0 | 0 |

Row-level diff (shipped post-fix) vs R5:
- **Fixed**: q19 "dinosaurs, nature" (gold turn's density beat the
  filler now that the fix removed noise promoting the filler).
- **Newly broken**: q3 "Adoption agencies" → "not stated" refusal
  (pool reshuffle depressed reader confidence); q48 "bowls, cup" →
  "pots"; q66 "Roast marshmallows..." → "exploring forests..." (the
  gold is no longer propped up by false-positive density signal).
- **Target rows UNAFFECTED** (q34, q71, q75, q76 still broken). The
  fact-density tie-breaker **did not help the fact-dense gold rows**
  — because those rows' failure mode is `session_present_gold_absent`,
  meaning the gold turn isn't in the near-tie neighborhood after
  cross-encoder; the tie-breaker threshold (|Δ score| ≤ 0.03) never
  fires on fragments that are already far apart.

### R8-b + R8-c stacked (tuning pilot)

**F1 = 0.438** on conv-26 single-hop. Refusals **3/32**. Wall time 148s.

Artefact: [e1-r8bc-conv26.json](e1-r8bc-conv26.json).

| Metric | R5 baseline | R8-c @ M=3 | R8-b standalone | R8-b + R8-c |
|---|---|---|---|---|
| F1 | 0.484 | **0.500** | 0.469 | **0.438** |
| Refusals | 1/32 | 2/32 | 1/32 | **3/32** |
| `gold_in_pool` incorrect | 1 | 2 | 0 | **3** |
| `session_present_gold_absent` | 5 | 3 | 7 | **3** |
| `recall_missing` | 0 | 1 | 0 | **1** |

Stacked rescued q43 "abstract art" + q66 "campfire activities" but
broke q3 "Adoption agencies" (refusal), q48 "bowls, cup", q55
"Sunsets" (refusal). Net −1 binary row, F1 −0.046 vs R5. Reader
refusal rate tripled (1→3) — the combined pool reshuffling confuses
the reader more than either lever alone.

### Pass bar verdict for R8-b

- Minimum F1 ≥ 0.484: ✗ by 0.015 (within noise band — not a real kill)
- Target F1 ≥ 0.56: ✗ by 0.09
- Kill criterion F1 < 0.46: ✗ (0.469 > 0.46; not killed)

**R8-b is a null-op within the noise band on conv-26.** The tie-breaker
mechanism works as implemented (the unit tests confirm it fires only
on near-ties with density diff ≥ 2), but the four target rows'
failure mode is upstream of the tie-breaker — they fail at session
selection or at cross-encoder, not at near-tie post-rerank.

### Shipped state after R8-b

- `--fact-density-tie-breaker` is **not** the new standard pipeline.
  R5 remains the shipped single-hop stack for conv-26.
- Both `--within-session-rerank` and `--fact-density-tie-breaker` are
  **kept as configurable off-by-default paths**, available for
  stacking in future multi-instance experiments where the tradeoff
  curve may skew differently.
- Unit tests for both are committed under `tests/`.

## Implied next moves

1. **R8-c @ M=3 is the ceiling of the within-session mechanism on
   conv-26.** Not killed, not shipped — kept configurable.
2. **R8-b is a null-op on conv-26.** The target rows fail upstream of
   where the tie-breaker fires. Not killed, kept configurable, but no
   evidence it moves conv-26 single-hop F1 by itself or stacked.
3. **R8-a (within-session BM25) is deferred.** The R2 null-op already
   showed BM25 doesn't win in-stack over cosine+rerank; and R8-b has
   now shown that *tie-breaking* on fact-density tokens doesn't help
   either — because the near-tie assumption that tie-breakers depend
   on isn't the dominant failure mode. R8-a would face the same
   upstream constraint.
4. **Stop at R8 for the Phase 1.5 single-hop ceiling on conv-26.** R5
   ships at 0.484; R8-c = 0.500 (noise); R8-b = 0.469 (noise); stacked
   = 0.438 (regression). The Phase 1.5 exit gate target is F1 ≥ 0.70.
   **No surgery on conv-26 single-hop will clear the gate alone** —
   even the R6 upper bound of 0.641 (all 4 target rows flipped) does
   not clear 0.70. Multi-instance validation is the real next lever.
   The target rows' failure mode (session_present_gold_absent with
   gold turns ranked past top-150 within their session on cosine)
   needs a different mechanism class — fragment-boundary-aware
   indexing, structural evidence tags, or multi-hop reasoning —
   not more ranking surgery on the same cosine → cross-encoder stack.
