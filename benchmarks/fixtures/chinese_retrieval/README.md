# Chinese retrieval representation pilot

This catalog points into [KdConv](https://github.com/thu-coai/KdConv), by Zhou,
Zheng, Huang, Huang and Zhu (ACL 2020), licensed Apache-2.0. The upstream corpus
and license stay in `.data`; no dialogue payload is vendored here. Upstream
annotations associate utterances with knowledge triples. They are not ratings
of this memory system or of attachment.

`protocol.json` fixes candidates, budgets, selection and guards before results.
`catalog.json` freezes source dialogue/turn pointers, exact supporting character
spans, the natural question and its two preceding turns, positive histories and
paired histories with supporting evidence removed. The question and all memory
utterances come from the source. Labels are mechanically derived from upstream
structured relations and literal spans; they are **not independent human
relevance judgments**. Report precision as strict supporting-span precision.

Both local splits come from `data/film/dev.json`, partitioned by dialogue hash.
Every memory distractor stays within its query's local split. They are not the
official train/test partition and entities can overlap. Film facts are a narrow
proxy for Chinese evidence recall, not personal memory or clinical dialogue.
Public downloads for the other domains were incomplete; no incomplete file is
used. The consumed file is verified by size, SHA256 and upstream Git blob SHA1.

The pool combines same-subject/other-relation and same-relation/other-subject
distractors. An answer's original local antecedent is retained. Positive and
absent pools have the same 24-turn budget, plus an identical fact in a foreign
scope. Each trial constructs actual memory, calls `observe`, marks replay session
boundaries and calls `recall` with the existing scope filter. Warm trials first
recall two observed utterances chosen without looking at labels. These replay
timestamps do not test aging or actual elapsed weeks.

The six candidates are canonical Unicode and precise jieba, then search features,
Han bigrams, a bounded word/bigram union and that union with ASCII width folding
and complete ASCII identifiers. Bounded channels keep the first 64 distinct
features each. The latter four change **Companion activation seed extraction**
only. Initial graph priors, coalescence and Witness retain precise jieba. Width
folding preserves character offsets and never edits raw observations. This is
not a full graph-policy migration or a calibrated fusion comparison.

The adapter rebinds the installed functions' globals **per instance**, executing
the actual production code objects for recall, reconstruction and spreading.
There is no copied ranking formula, alternate selection workflow, process-wide
monkeypatch or canned/mocked recall result. Tests check this boundary and an
unchanged canonical trace. Search uses the real private jieba backend owned by
`i18n`. No production source or default is changed by this study.

Latency includes Python allocation tracing and actual Companion/Witness recall;
peak bytes cover traced Python allocations during recall, not resident/native
process memory or ingestion. The study reports initial/learned graph sizes and
context diversity. A single run's timing is exploratory.

Run from the repository root with the verified Chinese environment:

```powershell
.\.cache\zh-validation-venv\Scripts\python.exe benchmarks/scripts/phase9_chinese_retrieval_run.py provision
.\.cache\zh-validation-venv\Scripts\python.exe benchmarks/scripts/phase9_chinese_retrieval_run.py run --output .cache/chinese-retrieval-results.json
```

`prepare` creates a new catalog from the fixed protocol and source; ordinary runs
require its frozen SHA256. All reports and catalogs reject overwriting. Missing
or mismatched sources/backends fail, rather than yield a partial positive result.
No generated reply, embedding, custom answer dictionary, pinned evidence,
increased context budget or attachment-weight adjustment is involved.

Production adoption is ineligible from this pilot alone. A candidate that passes
development guards is assessed once on local evaluation; a pass advances it to
independently judged, multi-domain replication and graph-policy evaluation.
