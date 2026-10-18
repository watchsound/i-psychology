# Attachment trajectory controls

`profiles.json` defines software controls for the Phase 9+ longitudinal harness.
These scripts are separate from the 20 weight-calibration fixtures, but were
written with knowledge of the lexical cue banks. They are not a blind or
independently labeled evaluation set and contain no clinical attachment labels.

Each recipe expands into a versioned manifest with its own identity and continuity
marker. Source-backed runs retain a single MSC conversation's original openers,
persona, session identifiers, and time gaps, then append explicitly synthetic
continuations of that same identity. Companion responses and all added seeker
turns are scripted, including during the source-derived seed sessions.

The fixed scripts exercise supportive, neutral, rupture, sycophancy, and repair
behavior in EN/ZH, plus session-count/time boundaries, an empty session, an
in-memory crisis handoff, and an explicit extreme-affect control. Fixed Companion
replies ignore memory and user input. Their scores validate the software path,
not the quality of an AI Companion's responses.

V2 records the actual recalled fragment IDs, contents, and tags. Its continuity
probe requires the designated original user turn, rather than any later question
that happens to contain the same marker. Checkpoints reject context snapshots
from unknown or future turns, and text that disagrees with those snapshots.
The original v1 marker-presence result was too permissive and is superseded.

Affect is unmeasured unless explicitly supplied as fixture input. Unknown or
empty sessions break the observed-affect streak; no neutral affect is invented.
The extreme-affect fixture exercises the actual watcher and in-memory review
sink, including rebuilding its accepted observations after resume.

Run all controls without downloaded data:

```powershell
python benchmarks/scripts/phase9_attachment_longitudinal_run.py --msc-limit 0 --output-dir <new-directory> --verify-resume
```

Run the controls plus three local MSC seeds by omitting `--msc-limit 0`. Output
directories are never silently restarted over an existing checkpoint; use a new
directory or supply the saved `--manifest` with `--resume`.

`--verify-resume` compares session-8 pause/fresh-process continuation with an
uninterrupted reference. This verifies attachment counters, history, scores,
timing, and verdicts on fixed transcripts. PAM integrity and cold recall are
reported separately; learned runtime state and future model replies are outside
that replay guarantee.

The pause boundary is reduced for shorter trajectories so that both a prefix and
suffix are exercised. A one-session trajectory cannot claim this comparison.
The check is labeled `fixed_transcript_attachment_bookkeeping`; matching
preselected replies does not validate memory-dependent response generation.

V2 manifests/checkpoints deliberately reject v1 schemas. Original v1 artifacts
are retained as historical records; reviewed reports use the sibling
`docs/experiments/phase-9-attachment-longitudinal-reviewed/` directory. A refused
rerun preserves the old cohort summary, and a resume writes a separate summary
when an earlier cohort summary already exists.

The legacy gate is 16 sessions with EMA at least 4.0. The additional evaluation
qualification requires 28 simulated days and a completed, uninterrupted chain.
Neutral input stays at 4.0 and passes both thresholds: this limitation is an
expected control finding, not evidence of increasing trust.

Calendar gaps are simulated, not a test of elapsed-time aging. The one-megabyte
memory capacity is a declared continuity-test configuration and does not exercise
production-capacity pressure. Model-generated response quality, real attachment,
and natural-language affect inference remain outside this utility's scope.
