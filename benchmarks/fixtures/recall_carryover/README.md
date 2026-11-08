# Cue-conditioned carryover regression cases

`protocol.json` freezes the candidate definitions, selection guards and source
pins. `catalog.json` contains 36 source-backed cases: 12 music/dev development
questions and 24 official film/test + music/test questions. `controls.json`
contains 10 explicitly authored software contracts; these never contribute to
public-data quality metrics.

The source is [KdConv](https://github.com/thu-coai/KdConv), Zhou et al., ACL 2020,
Apache-2.0, revision `653db76432de09a004ba708a68f8bbd5500e6bec`. Files stay in
`.data/kdconv/data/`. Download the pinned source files if absent:

- [music/dev.json](https://raw.githubusercontent.com/thu-coai/KdConv/653db76432de09a004ba708a68f8bbd5500e6bec/data/music/dev.json)
- [film/test.json](https://raw.githubusercontent.com/thu-coai/KdConv/653db76432de09a004ba708a68f8bbd5500e6bec/data/film/test.json)
- [music/test.json](https://raw.githubusercontent.com/thu-coai/KdConv/653db76432de09a004ba708a68f8bbd5500e6bec/data/music/test.json)

Every file is checked against size, SHA256 and an independently queried upstream
Git blob hash. No raw dialogue payload is vendored. Source annotations plus
literal spans define strict supporting-evidence labels. They are not independent
human relevance judgments or externally verified factual truth. Source typos and
factual mistakes are preserved.

Each case contains an eight-turn target window and two eight-turn distractor
windows from other dialogues whose annotated subjects differ from the target
window's. Source turn order and source-dialogue session IDs are retained within
each window. Cross-dialogue ordering and replay timestamps are artificial. The
missing-evidence pair removes the answer and extends the original source window
by one turn, preserving the 24-turn budget. A foreign-scoped answer copy tests
ownership. Cue text uses the original question and its two preceding turns.

Cold trials have no prior recall. Follow-up trials prime the two actual source
antecedent utterances; topic-switch trials prime two actual distractor utterances.
These are controlled replay states, not observed real-user longitudinal sessions.
All trials run actual observation, recall and Witness. Identical ingested systems
are copied before each trial to avoid repeating ingestion; no scores or contexts
are substituted. Candidate weights/gates bind per instance to installed code.

The first development comparison rejected every candidate. Protocol v2 adds a
specific cue-novelty gate, with unchanged cases, labels, controls and guards.
Only development was scored during that revision. The selected candidate and
baseline were then evaluated once on the official test cases; there was no
evaluation-based replacement. The original v1/v2 reports are preserved in the
[experiment evidence](../../../docs/experiments/phase-9-recall-carryover/README.md).

The measured gate now lives only in `companion/reconstruction.py`. The research
module references that function; it has no second production implementation.
Baseline/reset/rejected policies remain explicitly research controls. The normal
engine shares the gate between Companion and Witness. The tokenizer default,
source observations, prior graph construction and attachment weights do not
change.

Use the existing verified `.[dev,zh]` environment from the repository root:

```powershell
.\.cache\zh-validation-venv\Scripts\python.exe -m benchmarks.scripts.phase9_carryover_run development --output .cache/carryover-development-fresh.json
.\.cache\zh-validation-venv\Scripts\python.exe -m benchmarks.scripts.phase9_carryover_run evaluation --development-report .cache/carryover-development-fresh.json --output .cache/carryover-evaluation-fresh.json
.\.cache\zh-validation-venv\Scripts\python.exe -m benchmarks.scripts.phase9_carryover_run verify-production --reference-report docs/experiments/phase-9-recall-carryover/evaluation-v2.json --output .cache/carryover-production-fresh.json
```

Reports reject overwrites. Missing/mismatched files, incomplete trial matrices,
scope/budget/source violations or changed code during scoring fail the run.
Evaluation requires a development report from the same code/configuration.
Its metrics, controls, guards and selection are recomputed before scoring.
Unsupported protocol settings and inputs changed during scoring also fail.
`verify-production` checks ordinary unbound production behavior against the
frozen candidate IDs and nonresource metrics across every trial.

The [subsequent code review](../../../docs/phases/phase-9-carryover-code-review.md)
adds deterministic fragment-ID tie ordering (runtime policy v3). All 216 frozen
selected-candidate records remain unchanged. Replays of these observed cases
are regression checks, not fresh quality evidence.

The later [cold-recall diagnosis](../../../docs/phases/phase-9-cold-recall-diagnosis.md)
adds `diagnose-graph` to this runner and rejects graph-pruning candidates on
observed-case regressions. Policy v4 adds identity-safe ingestion after repeated
PAM restore exposed duplicates; the graph/ranking equations remain unchanged.
Its full diagnostic command and evidence are in the
[cold-recall artifact README](../../../docs/experiments/phase-9-cold-recall/README.md).

The authored Chinese repeat control initially used `火车站台`, which the real
backend segmented differently from the stored sentence. Preflight corrected that
cue to a natural repeat question before public-data scoring; no tokenizer rule or
source case was adjusted to make it pass. Unsupported compounds remain a known
tokenization limitation, outside this carryover change.
