# R5 — reader prompt v3 (precision tightening on top of R3's v2)

## Hypothesis

R6's normalized diagnostic classified 2 of the 7 persistent conv-26
single-hop broken rows as `gold_in_pool` — the gold string literally
appeared in the top-50 but the reader generalized or refused:

- **q7 "Single"** → reader output "not stated" (refusal; single-word
  label was treated as too-short-to-be-an-answer).
- **q11 "Sweden"** → reader output "home country" (generalized a
  proper-noun place to a descriptor).

The R6 verdict table identifies these as **reader-precision** failures
with upper bound **+2/32 F1 = +0.0625** absolute. R5 is the cheap
zero-infra prompt-only fix for exactly this bucket. R8 (within-session
fragment ranking) targets the orthogonal 5-row `session_present_gold_absent`
bucket and is tracked separately.

## Mechanism

[`src/psychology_base/eval/benchmarks/reader.py`](../../../../src/psychology_base/eval/benchmarks/reader.py)
gains a third variant `_SYSTEM_PROMPT_V3` that inherits v2's
imprecise-match win ("pick the closest literal mention"; "scan ALL
fragments") and layers three precision rules on top of the question-type
extraction templates:

1. **'where / from where' → specific proper-noun place name.** The clause
   names the failure mode explicitly: *"Sweden, not 'home country'; Boston,
   not 'a northern city'"*. Addresses q11.
2. **'how many / how often' → integer or exact count, not a frequency
   phrase.** Example: *"'2', not 'twice a year'; '3', not 'a few'. Only
   use a frequency phrase when the question explicitly asks 'how often'"*.
   Targets numeric-count rows like q40 and q75.
3. **Single-word labels are valid.** The refusal clause gains an explicit
   carve-out: *"If a fragment names the answer in a single word or short
   phrase, that is a valid answer — do not refuse just because the answer
   is short."* Addresses q7's refusal mode.

A tie-breaker rule is appended for the two-candidate case: *"prefer the
one most specifically matched by the question's proper-noun or numeric
cue"*.

The CLI gains `--reader-prompt-variant v3` as a new choice in
[`run_locomo_gate.py`](../../phase-1-benchmark-adapters/option-gamma/run_locomo_gate.py).
No recall-channel or substrate changes.

## Pass bar

- **Minimum**: F1 ≥ R3 Cell B baseline (0.422) — R5 must not regress
  v2's imprecise-match wins.
- **Target**: F1 ≥ 0.47. R6 predicted upper bound 0.422 + 0.0625 = 0.484
  if both reader-precision rows (q7, q11) flip to correct and nothing
  else moves.
- **Kill criterion**: F1 < 0.40 (worse than Cell B by more than the
  single-instance ≈0.03 noise band), or refusals climb above the 2/32
  baseline. Either indicates v3's extra specificity rules over-fit at the
  cost of v2's recovered precision.

## Outcome (2026-10-06)

**F1 = 0.484** on conv-26 single-hop (32 questions, rerank + judge-single-hop
+ consolidation + reader-v3). Refusals **1/32** (down from 2/32).
Wall time 129s, judge cache 24 hits / 8 misses.

Artefact: [e1-r5-conv26.json](e1-r5-conv26.json).

### Delta vs R3 baseline (reader-v2)

| Metric | R3 baseline (reader-v2) | R5 (reader-v3) | Δ |
|---|---|---|---|
| F1 | 0.422 | **0.484** | **+0.063** (+15% rel) |
| Refusals | 2/32 | **1/32** | −1 |
| Incorrect rows | 7 | **6** | −1 |
| `gold_in_pool` incorrect | 2 | **1** | −1 |
| `session_present_gold_absent` incorrect | 5 | 5 | 0 (unchanged — R5 doesn't target this) |
| `recall_missing` incorrect | 0 | 0 | 0 |

**R5 landed on target.** The +0.063 F1 lift matches R6's +0.0625 upper-bound
prediction (one judge-score roll on a partial-credit row). The
`session_present_gold_absent` bucket is unmoved, as expected — this bucket
is R8's territory.

### Row-level verdict

Fixed vs R3 baseline:
- **q7 "Single"**: refusal → correct literal (single-word carve-out
  worked).
- **q11 "Sweden"**: "home country" → specific proper-noun place
  (proper-noun 'where' bias worked).
- **q40 "2"**: previously `session_present_gold_absent` and incorrect
  under reader-v2's frequency-phrase default; v3's integer bias
  surfaced the exact count from the pool (surprise win — R6 did not
  predict this because q40's gold string wasn't in top-50 as a literal,
  but the integer bias let the reader synthesize the correct count from
  listed items).

Broken by R5:
- **q43 "abstract art" → "painting"**: v3 pushed toward *shortest literal
  mention*, missing the modifier that made the gold answer specific.
  R3's reader-v2 picked the full phrase; v3 over-generalized downward.
- **q66 "Roast marshmallows, tell stories" → "exploring forests, hiking"**:
  v3's "match specificity of the question" rule combined with the
  shortest-literal bias caused the reader to pick a different activity
  mention than the authoritative one. R3 had just fixed this row;
  v3 un-fixed it.

**Net**: 3 fixed − 2 broken = +1 binary question; the extra +0.03 F1
beyond that comes from judge-mode partial-credit shifts on borderline
rows (one row moved from, e.g., 0.5 to 1.0 under judge rescoring).

### Shipped state

- Default reader prompt variant remains `v1` (conservative). Opt-in via
  `--reader-prompt-variant v3`.
- v3 passes the pass bar (F1 ≥ 0.47 ✓, refusals ≤ 2 ✓, no 7-row ceiling
  regression on the recall-side).
- Combined with R3's v2 trail, the reader-side ceiling of the current
  conv-26 single-hop pipeline is now exercised. The remaining 6-row
  incorrect set has 5 `session_present_gold_absent` + 1 `gold_in_pool`
  (q76 "19 October 2023" — persistent refusal on a date not literally
  in the pool as the full phrase; the fragments carry "October" and
  "2023" separately but not the joined date string).

## Residual caveats

- **q43 / q66 regressions are a real cost.** v3 trades two R3 wins for
  the two R6-predicted gains. The lift is still positive, but the
  trade suggests reader-prompt engineering has nearly saturated —
  further gains likely require either (a) few-shot exemplars for the
  activity-list and modifier-preserving cases (slot v4 if needed) or
  (b) accepting the trade as the reader-prompt plateau and moving to R8.
- **q40 "2" fix is a happy accident.** R5 was not designed to touch
  `session_present_gold_absent` rows; the integer bias clause surfaced
  the count from an item-list the fragments did carry. Do not generalize
  this to "R5 moves `session_present_gold_absent`" — one row out of 5
  is noise-order, and the mechanism (literal integer in a list the
  reader can count) is not present on q34 / q71 / q75 / q76.
- **No cross-instance validation.** R5 was gated on conv-26 only, same
  as R2/R3/R6. The reader-prompt tradeoff (q43 / q66 cost vs q7 / q11
  gain) may skew differently on conv-{19,27,55,61} — this is Phase 2's
  next full-sweep stress test, not an R5 scope item.

## Implied next moves

1. **R5 ships as the new reader-prompt default for conv-26 single-hop
   gating** (overriding v2 for Phase 2 ceiling-breaker runs). v2 remains
   available for regression comparison; v1 remains the long-term CLI
   default.
2. **R8 (within-session fragment ranking) is now the only remaining
   lever on the single-hop ceiling.** Upper bound +0.156 F1
   (0.484 → 0.641) if all 5 `session_present_gold_absent` rows flip.
   Mechanism design is a separate scoping task — candidates listed in
   `../option-R6-recall-diagnostic/README.md`:
   (a) within-session BM25, (b) fact-density ranking signal,
   (c) two-stage rerank (session → turn-within-session).
3. **Reader-prompt engineering is near plateau.** A v4 attempt should be
   scoped only if R8 is blocked or produces a specific class of
   reader-side failure that v4 can target. Do not iterate v3 → v4 → v5
   on the same two-row swap-bucket.
