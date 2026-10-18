# Reviewed v2 software-control evidence

These runs exercise real memory, CompanionAgent, explicit safety inputs and
export/import code with fixed scripted replies. They do not evaluate a language
model's response quality or establish human attachment. The 28-day schedule is
simulated; elapsed-time aging and production-capacity pressure are untested.

`summary.json` reports all 17 selected trajectories and their separate structural,
lexical-threshold and original-fact continuity results. Fifteen trajectories are
complete; none recalls the designated original user fact at every block boundary.
The neutral score-threshold false positive remains visible.

Each trajectory contains its manifest, actual response-context fragment audits,
report, and bounded checkpoint/PAM artifacts. `resume-verification/` uses real
child processes to compare fixed-transcript bookkeeping. This does not establish
equivalent learned runtime state or memory-dependent generated responses.

`verification.json` records the full/targeted test results and measured coverage.
`reproducibility.json` compares all report digests with an independent repeated
run. Historical v1 artifacts remain in the sibling original directory with a
supersede notice. See the [review](../../phases/phase-9-attachment-longitudinal-review.md)
for findings and the file-by-file audit.
