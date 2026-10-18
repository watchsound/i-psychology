# Phase 9+ attachment changes — code review and corrections

**Date:** 2026-10-18
**Scope:** Changes introduced in this conversation: the longitudinal harness,
CLI, fixtures, tests, generated evidence, plan/close-out/roadmap, and minimal
AGENTS.md. The earlier Chinese cue migration and weight-calibration tooling were
already present; this review preserves them and tests their integration.

## Findings, ordered by impact

| Priority | Finding in v1 | Correction / remaining limitation |
|---|---|---|
| High | Preselected replies ignore user input and recalled context. Their high bond scores and exact resumed scores cannot validate AI response quality. Calling this application-outcome evidence overstated what was measured. | Reports now say `scripted_software_control`, `response_uses_recalled_context=false`, `model_response_quality_measured=false`, and `attachment_validity_established=false`. Resume comparison is explicitly fixed-transcript bookkeeping. A real model-quality evaluation remains unimplemented in this utility. |
| High | Continuity accepted any prior line containing the marker. A repeated question could pass after the original user fact was lost. The initial 4/14 result was therefore too weak. | V2 audits actual recall output without changing selection. The probe requires the original user fragment's content-addressed identity, content, speaker and session tags. A regression test loses just that original fact while ordinary questions and recall still run; the probe must fail. The corrected all-boundary result is 0/15 completed trajectories. |
| High | Every session supplied `(0.0, 0.0)` to the extreme-emotion watcher. Attaching a watcher did not test extreme-emotion detection and fabricated neutral observations. | Affect is unknown unless explicitly provided. Unknown/empty observations break a measured streak. An added extreme-affect control exercises the real watcher/sink; accepted affect history is rebuilt after resume. Tests verify threshold crossing, missing observations, and rollback of flags from rejected sessions. This does not add affect inference. |
| Medium | Repeating a CLI run could replace a previous valid cohort summary with a failure-only summary. Resuming one user could overwrite the earlier cohort summary. | Fresh runs refuse existing result directories before mutation. Resume summaries are separate when a prior cohort summary exists. Tests verify exact preservation of prior bytes. |
| Medium | Resume verification always paused at eight sessions, so shorter valid trajectories could not exercise it. A subprocess timeout discarded the already-computed reference report. | The boundary is reduced when needed; runs without a prefix/suffix cannot claim a comparison. Timeout/crash failures remain in the trajectory report. Tests use an actual short child-process resume and deliberately corrupt a real child's report to ensure comparison failure is detected. |
| Medium | Checkpoints trusted context strings as long as scoring/state matched. A relabeled or future memory could be presented as recall evidence. | Context snapshots must match known prior transcript fragments, valid content addresses, exact rendered context and the six-fragment response slice. Missing, duplicate, future, altered and falsely rendered snapshots are rejected. V2 schemas reject v1 checkpoints explicitly. |
| Medium | Source-file read/parse failures could abort accounting before a useful summary; manifest-only runs unnecessarily needed the fixture catalog. | Requested source counts remain in failure accounting with no dummy replacements. Expanded manifests resume without the catalog. Tests exercise missing/malformed/insufficient source input, genuine input preservation and duplicate identity rejection. |
| Medium | Isolation checked known marker strings but could miss foreign ownership tags without those markers. | Fragment ownership tags are checked in addition to marker contamination, with a negative regression test. |
| Scope | The fixture set was called held-out, although it was hand-authored with knowledge of lexical cue rules. Calendar gaps did not exercise time-dependent aging, and the capacity was raised to one megabyte. | Removed blind-evaluation implications. Reports/docs explicitly identify software controls, simulated timing and untested capacity pressure. None establishes attachment validity or production-scale retention. |

## Review of each changed area

| Area | Result |
|---|---|
| `attachment_longitudinal.py` | Real CompanionAgent, memory, age/crisis gates, export/import, and explicit-affect watcher are exercised. Fixed replies are not AI-generated. V2 fixes provenance, watcher input/resume, ownership checks and rejected-session flags. Existing production retrieval/scoring modules were not changed. |
| `phase9_attachment_longitudinal_run.py` | Corrected output preservation, failure accounting, self-contained resume, short-trajectory comparison and timeout retention. No cloud mode is advertised or silently mocked. |
| Fixture catalog / README | Fourteen hand-authored software controls, plus three independently isolated MSC seeds in the evidence run. Only seed openers/persona/timing are dataset-derived; all Companion replies and continuations remain scripted. No clinical ground truth is claimed. |
| Tests | 128 targeted tests: original 51 plus 77 review regressions. Fault injection is restricted to testing failures such as disk replacement, missing memory and child-process failure; successful model behavior is not faked. |
| Generated reports | Original v1 reports are retained with a supersede notice. New v2 results live in a sibling reviewed directory and carry explicit evidence scope. |
| Plan / close-out / roadmap | Original 4/14 memory interpretation is retracted. Current results and unfinished model-validation work are stated explicitly. |
| `AGENTS.md` | Complete minimal reference to existing `CLAUDE.md`; no duplicated instructions. CLAUDE.md's pre-existing greenfield status remains stale, so dated roadmap/review entries describe current implementation status. |
| Pre-existing cue-bank/calibration changes | Preserved. The complete regression suite includes their tests. Synthetic calibration labels remain author-intention labels; recommended weights were not applied. |

There are no TODO/FIXME/NotImplemented/pass-stub implementation paths in the new
harness, CLI or tests. There is substantial unfinished research: model-generated
responses, independent attachment labels, meaningful neutral-versus-helpful
discrimination, and dependable original-fact recall. Those are not implemented
by fixed scripts and must not be presented as achieved.

## Verification

- **Full regression: 1,925 passed** (1,848 before review + 77 new tests).
- **Targeted: 128 passed**, with no excluded coverage lines.
- Harness: **473/475 statements (99.6%)**, **206/210 branches (98.1%)**.
- CLI: **130/133 statements (97.7%)**, **48/50 branches (96.0%)**.
- Coverage is not a proof of absence of bugs. Remaining unexecuted paths are
  recorded by coverage; no blanket 100% claim is made.
- New evidence run: **17 selected/reported software controls**, **15 complete**;
  the empty-session and handoff runs remain incomplete as expected. All 17
  bounded fresh-process comparisons pass. The explicit extreme-affect control
  produces one real in-memory review flag, including after resume.
- **Original-fact continuity: 0/15 completed trajectories pass every boundary.**
  Ten duration-qualified thresholds are true, including neutral controls;
  this is not an accuracy or model-quality success rate.

See the [reviewed cohort summary](../experiments/phase-9-attachment-longitudinal-reviewed/summary.json)
and [independent repeat comparison](../experiments/phase-9-attachment-longitudinal-reviewed/reproducibility.json).
No commits were created and no production weights were changed.
