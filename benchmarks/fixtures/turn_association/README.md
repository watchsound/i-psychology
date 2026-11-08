# Portable provenance association study

The [reviewed plan](../../../docs/phases/phase-9-turn-provenance-plan.md) freezes
one undirected local co-occurrence candidate: `max(existing_prior, 1.0)` for a
validated consecutive pair. It changes no cue, residual, hop or six-slot budget.
Ordinary v4 means the unannotated baseline behavior executed by the current
runtime; exact historical source-ID replay checks its compatibility separately.

`protocol.json` and `catalog.json` were written before development scoring.
`protocol-v2.json` preserves those cases, the candidate and every threshold,
amending only resource measurement after an interrupted first run: measure one
real ingestion rather than building the same graph six times. That first run
produced no complete report; its known-case failure had already ruled out
activation. The final development run is a replay, not independent confirmation.
The catalog contains 30 development cases and 60 unscored evaluation cases,
balanced between film and music. All source roles, including distractors and
withheld answers, are disjoint across splits. Background groups are reserved
before target selection. Labels are upstream structured relations plus exact
answer spans; they are mechanical labels, not independent relevance judgments.

Provision `film/train.json` and `music/train.json` under `.data/kdconv/data/`
from `https://raw.githubusercontent.com/thu-coai/KdConv/653db76432de09a004ba708a68f8bbd5500e6bec/data/`.
The loader checks byte count, SHA-256 and Git blob identity. The independently
retrieved Git tree pins and pre-score yield audit are recorded in
[source-pins.json](../../../docs/experiments/phase-9-turn-provenance/source-pins.json).
The existing film/music dev/test source files are also needed for historical
regressions and resource measurement. Install the project's `zh` extra to use
the pinned real jieba backend.

Run through the existing CLI (choose a new output path):

```powershell
python -m benchmarks.scripts.phase9_carryover_run association-development --protocol benchmarks/fixtures/turn_association/protocol-v2.json --catalog benchmarks/fixtures/turn_association/catalog.json --controls benchmarks/fixtures/recall_carryover/controls.json --output development.json
```

`association-evaluation` additionally requires `--development-report`. It rejects
development evidence that fails a gate or differs in code, tokenizer or input
hashes. Do not replace the frozen candidate after looking at evaluation.
Reports are written exclusively; prior artifacts are never overwritten.
Each association command rebuilds the catalog through `catalog_for` and the
shared carryover selector, then compares it exactly with the frozen catalog
before scoring. This is also the catalog's reproducibility check.

The three arms use the same trial constructor, real observations, independent
Witness, state preparation and ID scorer. The metadata-only and association arms
have identical actual input IDs. Descriptors contain pinned source identity,
original dialogue index and original message index, with actor unknown. Missing
answer turns retain their original index gap. No question, answer label or target
span enters the provenance encoder or pair predicate.

Resource runs use previously exposed `film/dev`: one real measured ingestion at
24/128/512 fragments per arm, then one warmup and five independent recalls from
copies of that actual ingested state. Payload and six-slot recall budgets match
across arms. Ingestion, repeated import and recall are measured separately. Reported
traced allocations cover the measured action, excluding the already loaded jieba
dictionary and warm caches; they are not whole-process RSS. Dense ingestion is
quadratic and is intentionally measured through the ordinary implementation.
Five samples give only a coarse nearest-rank p95. Report validation checks
complete internal consistency, not authenticity of arbitrary external reports.
