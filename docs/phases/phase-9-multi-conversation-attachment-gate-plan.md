# Phase 9+ multi-conversation attachment gate — implementation plan

**Date:** 2026-10-18
**Status:** Software harness implemented; its initial evidence claims were corrected by the 2026-10-18 [code review](phase-9-attachment-longitudinal-review.md). Model-quality and attachment-outcome validation remain incomplete.
**Predecessors:** [Attachment overlay close-out](phase-9-attachment-overlay-close-out.md), [perspective substrate close-out](phase-9-perspective-substrate-consolidation-close-out.md).

## 1. Objective

Exercise the attachment model through a reproducible, identity-consistent trajectory of at least 16 completed sessions spanning at least 28 simulated days. Verify session continuity, negative controls, checkpoint continuation, and memory export integrity. Produce an auditable report that separates software correctness from attachment-model behavior and from empirical evidence about people.

This advances the long-term emotional-companion application. It does not establish real human attachment, clinical validity, or a real-user four-week WAI-SR outcome. The project's no-recruitment constraint remains in effect.

## 2. Verified starting point

Inspection on 2026-10-18 established:

| Surface | Current behavior | Consequence |
|---|---|---|
| `AttachmentState.meets_four_week_gate` | `sessions_completed >= 16 and trust_ema >= 4.0` | No elapsed-time check; the name currently denotes a session-count proxy. |
| Existing attachment tests | EN and ZH 16-session positive scorer loops already exist | Another scorer loop would not resolve the missing end-to-end evidence. |
| Local `.data/msc/train.jsonl` | 60 conversations, exactly 4 sessions each, maximum elapsed source duration 7.25 days | Authentic four-week trajectories are unavailable in the current local data. |
| Existing overlay runner | Constructs one agent and attachment state per MSC conversation; reports behavior/LLM traces | Cannot currently carry a labeled 16-session trajectory across continuation blocks. |
| Attachment state | EMA, cumulative counters, and bond history; no session ledger or elapsed-time fields | The new harness must own identity, session uniqueness, timing, and resume bookkeeping. |
| PAM export | Exports Companion fragments, not the complete runtime or attachment state | Attachment continuation needs a separate versioned checkpoint sidecar. |
| Pending work | Chinese attachment cue migration and calibration tooling/reports are uncommitted | Preserve these changes and keep their review boundaries distinct from this task. |

The latest verified baseline from the preceding review is 1,797 passing tests. Re-establish the baseline at implementation time if the working tree has changed.

Calibration remains exploratory: the 20 author-intention fixtures are synthetic targets, not clinical ground truth. Neither the coarse nor fine recommendations have been applied. Equal best errors demonstrate agreement within the searched grids, not external validity or a globally optimal scoring model.

## 3. Scope and fixed decisions

1. Keep the production weights, EMA coefficient, lexical rules, and existing gate property unchanged. Report the existing gate explicitly as `legacy_gate_pass`.
2. Add `duration_qualified_proxy_pass` in the new evaluation report: the existing gate is true, at least 28 simulated days separate the first and last completed sessions, the trajectory is complete, and no handoff/error invalidated completion. This is an evaluation qualification, not a replacement for the production gate.
3. Keep the original overlay CLI, historical artifacts, Phase 8 verdicts, and report schemas intact. Implement a sibling runner using shared public helpers.
4. Start with offline, deterministic execution. Cloud/local-model runs are a later optional experiment, with an explicit model/call budget recorded before execution.
5. Carry the single-turn over-disclosure residual as separate algorithm debt. Do not tune the weights or change composition to make these trajectories pass.

## 4. Trajectory construction and provenance

**Default: one source identity plus explicitly synthetic continuations.** Select an MSC seed conversation deterministically, retain its four-session source block, then append same-persona continuation blocks until the trajectory has 16 sessions. The initial offline continuations are hand-authored and versioned. The manifest distinguishes source-derived openers, scripted seeker/companion turns, and synthetic continuation events; existing `run_msc_session` uses dataset openers but generates the rest of the dialogue, so the complete resulting transcript must not be labeled authentic MSC replay.

Do not concatenate unrelated MSC conversations as if they were one person's history. Multiple authentic source blocks may be used only if an explicit identity mapping establishes that they belong to the same user; the current dataset is not assumed to provide that mapping. Otherwise each MSC seed gets an isolated agent, memory system, attachment state, and checkpoint directory.

Use three deterministic seed conversations for the initial dataset-backed integration run. Add separate hand-authored EN and ZH scenarios with stable identities and facts; do not label translated or synthetic material as native MSC data.

Each manifest records:

- Schema version, trajectory/user identifiers, language, source conversation IDs, source file hash, fixture version, and seed.
- Globally unique trajectory session IDs and monotonically increasing session indices, distinct from local source session IDs.
- Original source timing plus scheduled simulated start/end times; synthetic timing adjustments are explicit.
- Provenance for every session/block and generated turn; no duplicated source session is counted as new evidence.
- Scorer weights/EMA configuration, tokenizer configuration, execution mode, and model routing if applicable.

Retain source gaps in the seed block. Schedule continuation sessions so the final completed session is at least 28 simulated days after the first, without overlapping prior session ends. A fast 16-session trajectory is a deliberate counterexample, not a four-week result. Attempted, empty, failed, and completed sessions have separate counts.

## 5. Implementation sequence

| Step | Deliverable | Acceptance condition |
|---|---|---|
| 1 — Baseline and contracts | Record current Git/test baseline; define manifest and report dataclasses/validation | Reject mixed-user identities, duplicate IDs, invalid ordering/timing, and inconsistent resume metadata before execution. |
| 2 — Trajectory builder | Deterministic MSC-seeded and EN/ZH scripted schedules | Produce 16+ unique sessions over 28+ simulated days with explicit source/synthetic provenance; no cross-user state reuse. |
| 3 — Runner integration | Run sessions through the real CompanionAgent/CompanionSystem path and existing attachment scorer | Carry memory and attachment state across block boundaries; record per-session source IDs, counters, bond, EMA, timestamps, and completion status. |
| 4 — Controls and diagnostics | Positive, neutral, rupture, repair, short-duration, interruption, and isolation scenarios | Expected behavior holds under frozen weights; failures remain visible instead of being dropped from denominators. |
| 5 — Checkpoint and export | Versioned attachment/ledger sidecar plus existing PAM memory artifact | Resume fixed transcript scoring exactly; validate memory identities, content addressing, and the declared recall probes separately. |
| 6 — Report and close-out | Offline CLI, deterministic artifacts, focused tests, full regression, close-out memo | All structural checks pass; proxy outcomes and limitations are reported separately; no historical gate is silently promoted. |

### Proposed file boundaries

- `src/psychology_base/eval/benchmarks/attachment_longitudinal.py`: manifest, schedule validation, execution, checkpoint sidecar, and aggregation.
- `benchmarks/scripts/phase9_attachment_longitudinal_run.py`: thin CLI with scripted default, input manifest, output directory, and resume support.
- `benchmarks/fixtures/attachment_longitudinal/`: held-out EN/ZH trajectories and source-provenance manifests, distinct from the 20 weight-calibration fixtures.
- `tests/test_attachment_longitudinal.py`: end-to-end trajectory, isolation, boundary, and failure tests.
- `docs/experiments/phase-9-attachment-longitudinal/`: manifests, reports, and checkpoint/export summaries.
- `docs/phases/phase-9-attachment-longitudinal-close-out.md`: results, limitations, and the next decision.

Reuse `score_session_attachment`, `run_msc_session`, MSC adapters, and PAM export/import helpers. Construct agents with existing safety components and in-memory test sinks; the current overlay's bare-agent construction must not silently omit handoff handling in the new runner. Session-boundary hooks and identifiers must match the existing application lifecycle. No outbound notifications are needed for offline tests.

## 6. Tests and evidence required

| Case | Expected evidence |
|---|---|
| Supportive EN and ZH trajectories | At least 16 completed sessions, 28+ simulated days, coherent history/counters, expected gate true. This establishes reachability under scripted inputs. |
| Session-count boundary | Fifteen completed sessions cannot pass the existing gate even with high EMA. |
| Time boundary | Sixteen supportive sessions compressed into less than 28 days may pass the existing gate, but must fail duration qualification. |
| Neutral control | Sixteen cue-free sessions remain at EMA 4.0 and therefore pass the existing threshold. Report this as a discriminative limitation; duration qualification alone does not fix it. |
| Sustained rupture/sycophancy | Trust declines and the final threshold fails under fixture-defined negative pressure. |
| Rupture then repair | Recovery exceeds the unrepaired control under matched conditions; do not require a pass if frozen weights do not support one. |
| Block transition | An earlier identity/fact is present in memory and reaches the subsequent response context; merely emitting a callback phrase is insufficient evidence of memory use. |
| Multi-user isolation | Unique per-user sentinels never appear in another user's memory, context, checkpoint, or report. |
| Duplicate/empty/error session | Duplicates rejected; zero-response sessions do not advance scoring; execution errors mark the run incomplete and remain counted in aggregate reporting. |
| Handoff | Safety path triggers its in-memory sink, terminates the trajectory, and prevents a completed-run verdict, even if the prior EMA was high. |
| Checkpoint at session 8 | Fresh-process scoring of the same fixed transcript suffix reproduces uninterrupted attachment counters, history, EMA, and verdict exactly. |
| Checkpoint validation | Cross-user, duplicate-session, altered-manifest/configuration, truncated, or corrupt checkpoints fail explicitly. |
| PAM round-trip | Fragment IDs/content/parent chains and content-address validation survive export/import; fixed recall probes have separately declared expectations. |

**Checkpoint scope:** the sidecar saves the complete attachment dataclass, manifest/source/configuration hashes, accepted-session ledger, next index, and simulated clock. Store explicit schema versions and validated values; do not rely on permissive construction from arbitrary dictionaries. Write completed-session checkpoints atomically. An interrupted session is not accepted as completed or silently counted twice on resume.

PAM does not preserve every heartbeat, Witness, index, safety, or model state. Exact attachment replay on fixed transcripts is required; byte-identical future live-model responses or complete runtime equivalence after PAM restore are not implied. Use scripted seekers/responders with explicit deterministic fixture state for the resume test, and label any other restart semantics in the report.

## 7. Report contract and stopping conditions

Every run reports the planned, attempted, completed, excluded, and failed session totals; elapsed simulated days; per-session provenance; all six cue counts; bond/EMA trajectories; legacy and duration-qualified verdicts; handoff/error state; checkpoint checks; and memory-export checks. Preserve the denominator for every seed selected, including failed trajectories.

Record neutral-control outcomes and the single-turn over-disclosure residual prominently. Positive-signal presence and trajectory change may be diagnostics, but introducing a new positivity threshold requires a separate methodology decision. Do not call the existing midpoint threshold a validated sustained-bond measure.

LLM-WAI-SR is optional secondary reporting. Its flat-persona limitation remains documented; undefined correlation caused by a constant series must be labeled as undefined in the new report rather than treated as agreement or meaningful zero correlation. Do not fit the weights to the same trajectories used to evaluate the gate.

Stop and fix the harness if identity contamination, double-counting, skipped failures, incorrect timing, resumed-state mismatch, corrupt export, or handoff continuation occurs. A correctly measured failure of the attachment model is a reportable outcome, not permission to retune weights or move a gate.

Completion means the harness and required controls pass, artifacts reproduce, and all application-proxy results are honestly dispositioned. It does not require every scenario to produce a positive attachment verdict.

## 8. Validation and review sequence

1. At implementation start, establish the current full-suite baseline and record the pending-change boundaries.
2. Implement the smallest offline trajectory and boundary cases, then run the focused attachment/trajectory tests. Extend checks only as checkpoint, safety, and provenance behavior lands.
3. Run three MSC-seeded 16-session continuations plus the EN/ZH control matrix with no model/API dependency. Compare stable report fields on a repeat run; wall time and generation timestamps are excluded from determinism comparisons.
4. Run the complete test suite once after the integrated changes pass focused verification. Existing tests must stay green; do not use a fixed new test-count target as a quality gate.
5. Write the close-out and add a dated roadmap pointer that distinguishes legacy and duration-qualified proxy results from real-user evidence.

Suggested review/commit boundaries are the existing Chinese cue migration, the existing calibration tooling/reports, then the new longitudinal harness and evidence. Keep `AGENTS.md` separate from those feature changes. Creating commits is a later explicit action; this planning task creates only this document.

After offline acceptance, decide whether a small budgeted model-generated continuation experiment would add useful evidence. Larger cloud sweeps, weight adoption, an over-disclosure composition redesign, full runtime snapshotting, and a new clinical-validity claim are separate follow-up decisions.

## 9. Implementation finding — 2026-10-18

**Review supersede:** The initial 4/14 marker-match result in this section was too permissive: a repeated question could satisfy it. It is not evidence that the original fact was recalled. The corrected v2 audit passes 0/15 completed trajectories at every boundary. Fixed scripts are now labeled software controls; the real application outcome target remains unvalidated. See the [review](phase-9-attachment-longitudinal-review.md).

The harness now exercises 13 controls and three isolated MSC-seeded trajectories, with fresh-process checkpoint comparison. The legacy and duration-qualified thresholds are reported independently. Full regression: 1,848 passed (1,797 baseline + 51 new tests).

The first end-to-end run exposed a distinction the original block-transition criterion must retain: the memory probe itself can work correctly while the substrate fails to recall the marker. Accordingly the reports preserve an explicit `continuity_probe_pass` behavioral verdict alongside `structural_checks_pass`; a missing marker is not converted into a continuity PASS. Only 4 of 14 completed trajectories pass all block-boundary probes (4 of 16 selected). All three MSC-seeded trajectories miss those probes despite passing the duration-qualified bond threshold. This is carried as an application-level failure, not a closed memory-continuity gate.

Neutral trajectories also pass both midpoint-based thresholds, as predicted. Neither result was addressed by changing weights, thresholds, fixtures to fit a desired score, or production retrieval behavior. The software implementation is complete; meaningful attachment discrimination and reliable warm-context continuity remain open findings in the close-out.
