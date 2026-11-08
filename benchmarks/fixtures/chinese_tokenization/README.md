# Chinese tokenization software controls

`cases.json` is a hand-authored regression catalog frozen before the runtime
repair on 2026-10-18. It contains source-backed software assertions, not clinical
labels, model responses or a representative Chinese retrieval benchmark.

Recall cases require the original stored fragment in the application's actual
response context. Shared tags, sentinel markers, custom dictionary entries and
larger context windows cannot substitute for supporting evidence. Changed facts
test provenance preservation, not semantic contradiction resolution.

The diagnostic cases preserve known granularity/filtering questions for a later
study. Their failures are reported rather than converted into correctness passes.
The experiment report records this file's SHA-256 and the real backend identity.
