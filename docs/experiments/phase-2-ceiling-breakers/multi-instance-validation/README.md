# Multi-instance validation — shipped R5 across the LoCoMo set

## Hypothesis

conv-26 single-hop F1 plateaued at ~0.48–0.50 across R3/R5/R8-c/R8-b.
The R6 recall diagnostic confirmed the failure mode on conv-26 is
`session_present_gold_absent` (gold turn not in top-50 within an
otherwise-correct session). Open question: **is this ceiling specific
to conv-26, or does it generalize?** If it generalizes, the Phase 1.5
exit gate (F1 ≥ 0.70 single-hop) is structurally out of reach without
a new mechanism class. If it is conv-26-specific, the gate may already
be clearable on the aggregate.

## First-3 staged pass (2026-10-06)

Shipped R5 config: `--consolidation --cross-encoder-rerank
--reader-prompt-variant v3 --judge-single-hop --mode rag`.
Instances picked in dataset order: `conv-26`, `conv-30`, `conv-41`.

Artefact: [r5-first3.json](r5-first3.json). Wall time **97 min**
(5819 s); 74 questions across the three instances.

### Per-instance

| Instance | Questions | F1 | Wall (s) | Incorrect |
|---|---|---|---|---|
| conv-26 | 32 | 0.469 | 60 | 7 |
| conv-30 | 11 | 0.545 | 2154 | 3 |
| conv-41 | 31 | 0.621 | 3605 | 7 |
| **Aggregate** | **74** | **0.544** | 5819 | 17 |

- conv-26 result (0.469) matches prior single-instance runs within a
  single judge roll — confirms the gate's instrumentation is stable.
- conv-41's **0.621 is already close to the 0.70 gate** on a single
  instance, with no mechanism changes vs R5.
- Spread per-instance is **0.152 F1** (0.469 → 0.621) — larger than
  the single-instance judge-mode noise band (~0.03), so the
  per-instance F1 is a real property of the corpus, not noise.
- Fresh-ingestion wall time dominates: conv-26 completed in 60 s
  (fact store pre-cached from prior runs), while conv-30 and conv-41
  each took 35–60 min of embed + consolidation at session boundaries
  from scratch.

### Aggregate recall diagnostics

| Bucket | Total | Incorrect |
|---|---|---|
| `gold_in_pool` (reader-precision failures) | 16 | **5** |
| `session_present_gold_absent` (within-session ranking) | 58 | **12** |
| `recall_missing` | 0 | 0 |
| `no_evidence_metadata` | 0 | 0 |

**Dominant failure mode generalizes.** `session_present_gold_absent`
is still the ceiling bucket (12/17 = 71% of incorrect) across all
three instances — the same mode that dominated conv-26. This is a
*structural* ceiling, not an instance-specific artefact of conv-26.

**Reader-precision failures also generalize.** 5 `gold_in_pool`
incorrect rows include:
- `conv-26-q66` marshmallows → "exploring forests" (same regression
  R5 traded q66 for q11 Sweden; still broken here)
- `conv-30-q9` "Rome" → "not stated" refusal (gold in pool, reader
  refused despite coverage=1.0)
- `conv-41-q18` "Rob" → "a colleague" (reader over-generalized
  despite gold in pool)
- `conv-41-q49` "Cakes" → "not stated" refusal
- `conv-41-q62` "two" → "1" (count mismatch — reader counted wrong
  items despite correct fragment in pool)

The reader-side failures aren't v3-specific; they're structural
precision gaps the v3 prompt doesn't cover.

## Gate verdict

- **Target**: F1 ≥ 0.70 single-hop (Phase 1.5 exit gate).
- **Measured (74 Q, 3 instances)**: F1 = 0.544.
- **Delta to gate**: −0.156.
- **Verdict**: ✗ FAIL at the current scope.

Even conv-41's best single-instance F1 (0.621) does not clear 0.70.
A full 10-instance run **could** land in the 0.55–0.60 aggregate
range if remaining instances cluster around conv-30/conv-41 and
conv-26 is the outlier — but it would not clear 0.70 without a new
mechanism class.

## What this does and doesn't tell us

**Does tell us:**
- conv-26 is a hard outlier, not representative. Our ceiling work
  through R1–R8 was tuned against the hardest instance.
- The `session_present_gold_absent` ceiling is **structural**: 71% of
  incorrect rows across 3 instances share this mode.
- R5's reader precision gaps generalize: `gold_in_pool` reader-side
  failures appear on every instance.
- The current cosine → cross-encoder stack is the **wrong tool class**
  for the dominant failure mode — not a tuning issue.

**Doesn't tell us:**
- The real 10-instance aggregate F1 under R5. 3 instances is a
  staged signal, not a gate test.
- Whether R8-c would help on conv-30/conv-41 (R8 variants were
  measured on conv-26 only; the "null-op" verdict is conv-26-local).
- Whether multi-hop / temporal / open-domain categories (not run
  here) regress or improve under R5.

## Implied next moves

1. **Do not immediately run full 10 instances under R5.** The signal
   from 3 is clear: aggregate F1 will not clear 0.70 under this
   mechanism. 5–6 hours of additional wall time for a confirmed-fail
   gate is not worth the budget.
2. **Attack `session_present_gold_absent` structurally.** The
   dominant failure mode is: gold turn ranked past top-150 within an
   otherwise-correct session. Candidate mechanisms (none measured
   yet):
   - **Fragment-boundary-aware indexing** — concatenate consecutive
     turns from the same speaker/topic into variable-length
     "paragraph" fragments so a single embedding carries more
     context, raising the within-session rank of gold-containing
     paragraphs.
   - **Structural evidence tags** — if the fact store carries
     back-pointers to `source_fragment_ids`, we can lift fragments
     *because a related fact matched the cue*, bypassing within-
     session cosine ranking entirely for cue-matched-fact rows.
   - **Multi-stage reasoning** — explicit "which session covers X"
     then "which turn of that session names X" as separate LLM
     calls, scoping the second to full session content, not top-50
     turns. (Expensive but likely effective.)
3. **Run R8-c/R8-b on conv-30/conv-41** standalone before structural
   work — cheap, could reveal that the "null-op" R8 verdict is
   conv-26-specific. If R8-c lifts conv-41 by +0.05, the mechanism
   is viable and the earlier "null-op" verdict was over-generalized.
4. **Reader-precision follow-on**: the 5 `gold_in_pool` incorrect
   rows share a tight failure pattern (over-generalization or
   refusal when the gold is present with full coverage). A v4
   reader prompt tuned on this cross-instance pattern might be more
   productive than earlier conv-26-only prompt iterations.

## Scope boundary

This experiment measures R5 (shipped) across 3 instances. It does
**not** attempt to clear the Phase 1.5 exit gate. The gate verdict
requires a mechanism change (structural ceiling-breaker), not more
tuning on the current stack.
