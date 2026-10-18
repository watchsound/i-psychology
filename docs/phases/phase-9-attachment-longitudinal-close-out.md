# Phase 9+ longitudinal attachment harness — close-out

**Date:** 2026-10-18
**Status:** Historical v1 software run; superseded by the [code review and corrected results](phase-9-attachment-longitudinal-review.md). The original continuity metric was too permissive.
**Plan:** [Multi-conversation attachment gate](phase-9-multi-conversation-attachment-gate-plan.md).
**Evidence:** [Cohort summary](../experiments/phase-9-attachment-longitudinal/summary.json), [reproducibility check](../experiments/phase-9-attachment-longitudinal/reproducibility.json).

> **Review correction — 2026-10-18:** The 4/14 continuity result below counted marker-bearing questions, without requiring the original user fact. It is retracted as evidence of original-fact recall. V2 records actual recalled fragment identities and passes 0/15 completed trajectories at every boundary. Fixed responses ignore recall, so these runs are software controls, not validated Companion outcomes. Original artifacts below remain unchanged for the audit trail.

## 1. Outcome

The long-term companion now has a reproducible trajectory evaluator spanning 16 sessions and 28 simulated days, with EN/ZH controls and same-persona synthetic extensions of three MSC seeds. It uses the real CompanionAgent/CompanionSystem path with injected scripts, attached safety components, and in-memory handoff/review sinks. No model or network calls were needed.

The evaluator separates software checks, bond-proxy thresholds, and actual prior-memory appearance in response context. A positive lexical attachment score is not evidence that recall worked or that a human formed an attachment.

| Measure | Result |
|---|---:|
| Selected trajectories | 16: 13 controls + MSC seeds `msc_2`, `msc_5`, `msc_7` |
| Planned / attempted / completed sessions | 255 / 248 / 246 |
| Excluded sessions | 2: one empty session, one crisis handoff |
| Unattempted sessions | 7 after the handoff |
| Execution/setup failures in the evidence run | 0 |
| Software/export/isolation checks | PASS; 16 identities, no foreign-marker leaks |
| Session-8 pause/fresh-process attachment comparison | 16/16 exact on the declared comparison fields |
| Duration-qualified proxy threshold true | 9/16 selected, including two neutral controls |
| Every block-boundary continuity probe true | 4/14 completed trajectories; 4/16 selected |
| Full regression | **1,848 passed in 45.94 s**, including 51 new trajectory tests |

Baseline at implementation start: 1,797 passed in 31.86 s. Existing Chinese attachment changes and calibration work were preserved. Production weights, EMA, cue banks, retrieval behavior, and the legacy gate property were not modified by this task. No commits were created.

## 2. Fixed-score controls

The new duration qualification requires the existing gate, 28 simulated days between the first and last completed session starts, and complete execution without handoff/error. It is a sibling evaluation field, not a replacement production gate.

| Trajectory | Completed | Days | Final trust EMA | Legacy threshold | Duration-qualified threshold |
|---|---:|---:|---:|---|---|
| Supportive EN / ZH | 16 each | 28 | 5.499999 | true | true |
| Neutral EN / ZH | 16 each | 28 | 4.000000 | true | true |
| Sustained rupture EN / ZH | 16 each | 28 | 3.000000 | false | false |
| Sustained sycophancy EN | 16 | 28 | 3.000000 | false | false |
| Rupture then repair EN / ZH | 16 each | 28 | 6.497707 | true | true |
| Short-duration supportive EN | 16 | 7 | 5.499999 | true | false |
| Fifteen-session supportive EN | 15 | 28 | 5.499998 | false | false |
| Empty-session control | 15 of 16 | 28 | 5.499998 | false | false |
| Crisis-handoff control | 8 of 16 | 13.066667 | 5.499017 | false | false |
| Three MSC-seeded continuations | 16 each | 28 | 5.499999 | true | true |

**Neutral false-positive limitation:** zero cue hits keep EMA at 4.0, which satisfies both thresholds once session-count and timing conditions are met. A threshold PASS therefore does not establish increasing trust, sustained helpfulness, or clinically valid attachment. This expected control result remains explicit in each neutral report.

## 3. Historical v1 marker matching — superseded

Only the four completed ZH controls (supportive, neutral, rupture, repair) have the marker from an earlier session present in actual response context at every block boundary. The other completed controls and all three MSC-seeded trajectories miss at least one boundary. The three MSC-seeded trajectories miss all three boundaries while passing the bond-proxy threshold.

Inspection of the supportive EN run showed repeated Companion reply fragments occupying the warm-context recall slots instead of the marker-bearing user fragments. This suggests feedback from repeated replies/activation in this specific scripted setting; it is not a demonstrated cause across natural dialogue. Cold recall after PAM import finds the marker in all 16 runs, emphasizing that stored-content availability and warm-context selection are separate properties.

The software verdict remains independent: `continuity_probe_pass` is false on missed/unfinished trajectories. The harness is not declared a memory-continuity success merely because counters, exports, or lexical bond scores pass. A negative test also proves that a scripted callback phrase cannot substitute for a missing marker.

## 4. Provenance, isolation, and checkpoint guarantees

- Each seed retains exactly one source identity, its persona, original source openers, session IDs, and original time gaps. Remaining seeker turns, all Companion replies, and continuation sessions are explicitly scripted. No unrelated MSC users are spliced together.
- Manifests pin source-file and fixture hashes; global session IDs, ordered indices, source timing, user consistency, and non-overlap are validated before execution.
- Checkpoints contain a versioned attachment/ledger sidecar and a content-addressed PAM artifact. Manifest/configuration checksums, exact transcript-prefix scoring, counters, history, next index, simulated clock, and memory content are validated before resuming.
- Checkpoint writes are atomic at accepted boundaries. Tests cover interruption and write failure, corrupt/truncated/cross-user checkpoints, altered configuration, duplicate ledger entries, invalid paths, and no double-counting after resume.
- Fresh-process continuation exactly matches fixed-transcript attachment state, per-session scores, counts, timing, and verdicts. Learned activation/edges, complete safety runtime state, and future model-generated responses are outside this equivalence guarantee.
- PAM checks cover fragment/content-address integrity and parent-chain preservation. A separate test exercises a genuine reconsolidation parent chain, rather than claiming chain preservation from parent-free fixtures alone.

The evidence reports also retain the plan's unmodified single-turn over-disclosure debt. No weight search or model calibration was performed on these trajectories. Author-intention calibration targets remain synthetic, not clinical ground truth.

## 5. Reproduction

```powershell
# Full control matrix plus three local MSC seeds; use a fresh output directory.
python benchmarks/scripts/phase9_attachment_longitudinal_run.py --verify-resume --output-dir <new-directory>

# Fixtures only; no downloaded dataset required.
python benchmarks/scripts/phase9_attachment_longitudinal_run.py --msc-limit 0 --verify-resume --output-dir <new-directory>

# Focused integration checks.
python -m pytest tests/test_attachment_longitudinal.py tests/test_attachment_state.py tests/test_attachment_weight_calibration.py -o addopts= -q

# Full regression.
python -m pytest tests -o addopts= -q
```

Every trajectory directory includes its manifest, report, checkpoint sidecar, and referenced memory artifacts. `resume-verification/` contains the independently resumed runs. The cohort summary preserves all selected trajectories and setup failures in its denominators. Independent repeated-run report hashes are recorded in `reproducibility.json`; generated wall-clock timestamps are excluded from the deterministic schema.

## 6. Next decision

The new gate is now exercisable. The next useful research task is a focused diagnostic of warm-context recall: compare varied versus repetitive replies, early versus later markers, and uninterrupted versus restored state under the same fixed prompts. Keep application outcomes separate from storage integrity.

Separately, a meaningful positive-bond criterion needs a declared methodology that discriminates neutral from helpful trajectories. The current threshold should not be promoted to clinical or real-user evidence. Weight adoption, scoring-composition changes, model-generated continuation budgets, full-runtime snapshots, and the over-disclosure algorithm residual remain separate decisions. Phase 8's historical rollup is unchanged.
