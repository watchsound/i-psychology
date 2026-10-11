# Phase 9+ i18n foundation — close-out

**Status:** SHIPPED — additive `psychology_base.i18n` module (language detection + tokenization dispatch) + `fact_language_directive` wired into the extraction prompt pipeline + reserved `i18n:` section in `configs/llm.yaml`. Zero regression surface by construction.
**Date:** 2026-10-10
**Related:** [CLAUDE.md](../../CLAUDE.md) — "learning companion (韧性学习陪伴) and long-term emotional companion (情感陪伴)" are Chinese-first applications. [docs/sota-goal.md](../sota-goal.md) — "best in-application result for the two surfaces" is the SOTA claim. [docs/roadmap.md](../roadmap.md) — this is prep infrastructure for Phase 4-style application pilots and any-language eval.

---

## 1. Verdict

The pipeline can now handle any-language user interaction behind a dispatch, without any English-pipeline behavior change. Nothing in `companion/`, `eval/`, or `llm/` reads the config section yet. The one hot-path site migrated this pass — extraction's system prompt — is bit-identical for English input and emits a language-directive for non-English input. Pre-existing Phase 1.5 / Phase 2 / Phase 7 / Phase 8 / Phase 9+ LLM-output-dependent gates are safe by construction.

- **New module:** [src/psychology_base/i18n/\_\_init\_\_.py](../../src/psychology_base/i18n/__init__.py) — `detect_lang(text) -> LangTag`, `tokenize(text, lang=None) -> list[str]`.
- **New prompt helper:** [src/psychology_base/i18n/prompts.py](../../src/psychology_base/i18n/prompts.py) — `fact_language_directive(text_or_lang, is_lang_tag=False) -> str`.
- **Wired into:** [src/psychology_base/companion/extraction.py:247-257](../../src/psychology_base/companion/extraction.py#L247-L257) — directive appended to `EXTRACTION_SYSTEM_PROMPT`; empty string for English → bit-identical composed prompt.
- **Config:** [configs/llm.yaml](../../configs/llm.yaml) `i18n:` section (`mode: en`, `tokenizer: unicode`, `embedder_multilingual: false`) — reserved knobs, no live code path yet.
- **New tests:** [tests/test_i18n.py](../../tests/test_i18n.py) (28), [tests/test_i18n_prompts.py](../../tests/test_i18n_prompts.py) (12) — **40 new tests, all green** (0.56 s wall).
- **Full suite:** **1711 green** (1671 pre + 28 i18n + 12 prompt).

This is **not** a new gate. It is the dispatch surface that lets the project migrate English-forcing sites one at a time behind a byte-identical English contract.

---

## 2. Why this slice

The project README defines two applications: **韧性学习陪伴** (resilient learning companion) and **长期情感陪伴** (long-term emotional companion). Both target Chinese-first users. The current pipeline is English-centric at four load-bearing layers:

1. **Prompts** — `EXTRACTION_SYSTEM_PROMPT`, `_JUDGE_SYSTEM_PROMPT`, consolidation prompt, and application-agent system prompts are written in English with English-only examples. A Chinese utterance fed to extraction would produce silently-anglicized facts in memory — losing affective nuance that 共情 depends on.
2. **Tokenization** — `v_metrics.py:_tokens` and `_cosine_bow`, `locomo._normalize`, `epbench.parse_answer_items` all `.split()` on whitespace. Chinese / Japanese / Thai text (no inter-word spaces) collapses to one monster token, so F1 silently returns ~0 on any non-English corpus.
3. **Benchmark normalizers** — `locomo._normalize` strips English articles (`\b(a|an|the)\b`) and ASCII punctuation; non-ASCII input bypasses normalization entirely.
4. **Embedder** — `mxbai-embed-large` (local default) and `text-embedding-v3` (cloud default) are English-leaning. A multilingual retriever (bge-m3 / multilingual-e5-large) is needed for Chinese fragment recall at scale.

Full SOTA requires closing all four. This slice closes layer 1 at the hot-path site and ships the dispatch infrastructure (helper module + config knob) that lets layers 2-4 be migrated incrementally, each with a byte-identical-English regression contract to point at.

User directive 2026-10-10 (direct quote): *"i don't care, i just align with our final goal --- a best system in the world."* — sequencing decisions deferred to execution; the task is to move toward any-language support with zero regression on existing gates. This slice is the minimum-blast-radius first move.

---

## 3. Design

### 3.1 `detect_lang(text) -> LangTag`

Unicode-block heuristic. Walks characters once, flags which script blocks appear, resolves by priority:

| Priority | Script block | Resolves to |
|---:|---|---|
| 1 | Hiragana (U+3040–U+309F) or Katakana (U+30A0–U+30FF) | `"ja"` (beats zh even when kanji is present) |
| 2 | Hangul (U+AC00–U+D7AF) | `"ko"` |
| 3 | CJK Unified / Extension A (U+4E00–U+9FFF, U+3400–U+4DBF) | `"zh"` |
| 4 | Arabic (U+0600–U+06FF) | `"ar"` |
| 5 | Hebrew (U+0590–U+05FF) | `"he"` |
| 6 | Devanagari (U+0900–U+097F) | `"hi"` |
| 7 | Thai (U+0E00–U+0E7F) | `"th"` |
| 8 | Cyrillic (U+0400–U+04FF) | `"ru"` |
| 9 | Greek (U+0370–U+03FF) | `"el"` |
| 10 | Latin (A-Z, a-z) | `"en"` |
| — | None of the above | `"unknown"` |

Not a replacement for `langdetect` / `fastText` — the heuristic is correct for mono-script utterances and imperfect on code-switched text. The common failure mode (Chinese utterance with one English technical term) resolves to `"zh"` correctly because Hiragana > CJK priority doesn't fire; CJK is detected and wins over the trailing Latin. The reverse (English utterance with one Chinese quote) misclassifies as `"zh"` — acceptable because the hot path uses this for prompt-directive dispatch where "mostly Chinese" and "mostly English with a Chinese quote" both benefit from the directive.

### 3.2 `tokenize(text, lang=None) -> list[str]`

English byte-identical contract:

```python
# locked by test_english_tokenize_byte_identical_to_split parametrized over 8 snippets
assert tokenize(text, "en") == text.split()
assert tokenize(text, "unknown") == text.split()
```

Non-English path uses a Unicode letter-run regex that matches runs of letters from all 9 supported scripts, splitting on punctuation and whitespace. For whitespace-less scripts (Chinese without spaces, Thai) the regex returns one token per clause — same as `.split()` would, so net-zero regression. Finer-grained CJK word segmentation (jieba, pythainlp) is a future opt-in behind the `configs/llm.yaml` `i18n.tokenizer` knob.

### 3.3 `fact_language_directive(text_or_lang, is_lang_tag=False) -> str`

Returns `""` for `"en"` and `"unknown"` (so append-contract is byte-identical for English). Returns a non-empty directive for 9 non-English scripts:

```
LANGUAGE: The CONVERSATION WINDOW is in <Chinese|Japanese|...>.
Write every entity, relation, and value field in <lang> as the speaker uses it —
do NOT translate to English.
Preserve names, places, affective words, and idiomatic phrases verbatim.
The op tokens (ADD / UPDATE / NOOP) and the pipe-separated grammar MUST stay
in English (parser requirement).
```

Op-token preservation is a hard parser contract: `extraction.py:_parse_line` dispatches on `op_token.upper() == "ADD"` / `"UPDATE"` / `"NOOP"`. If the model started translating tokens, every line would drop and the gate numbers would collapse.

### 3.4 Extraction wiring

```python
# src/psychology_base/companion/extraction.py:247-257
window_block = _format_window(window)
system = (
    EXTRACTION_SYSTEM_PROMPT.format(max_facts=MAX_FACTS_PER_WINDOW)
    + fact_language_directive(window_block)
)
```

For English window → directive is `""` → `system` equals the pre-i18n composition byte-for-byte. For Chinese window → directive appended → `system` is a strict superset of the pre-i18n composition. This is locked by `test_english_extraction_system_prompt_byte_identical`, which composes both ways and asserts equality.

### 3.5 Config section

```yaml
# configs/llm.yaml
i18n:
  mode: en              # "en" | "auto" | "<tag>"
  tokenizer: unicode    # "unicode" | "jieba" | ... (future)
  embedder_multilingual: false  # future: route to bge-m3 / multilingual-e5
```

Nothing reads this section yet. It's a reserved surface so future migrations (judge/reader prompt sites, metric tokenizers, embedder route) can be toggled from one place without introducing a dependency on it today.

---

## 4. Regression contract

Full suite: **1671 → 1711 green** (+28 i18n + 12 prompt, 0 failures, 0 skips introduced). All prior gates — Phase 1.5 Bucket A F1 = 0.288, Phase 2 M1/M2/M3 preservation = 1.0, Phase 3 V5 8/8 at ep1000, Phase 5 FANToM +0.791, Phase 6 month-horizon, Phase 7 Companion Personalization 5/5 hard + 2/2 soft, Phase 8 Pillars A/D/E/F PASS, Phase 8 Pillar C v4 BIG5-CHAT 0.615, Phase 8 Pillar F rate-matched sibling gate 3-of-3, Phase 9+ attachment overlay, Phase 9+ perspective substrate consolidation — are safe by construction because:

- `fact_language_directive` on an English conversation window returns `""`.
- Nothing else in the pipeline (recall, consolidation, Witness, reconstruction, scorers, judge, reader, baselines) imports `psychology_base.i18n` yet.
- The config section is reserved; no code path reads it.

---

## 5. What's still English-forcing (deliberate scope-bound)

| Site | File:line | Dispatch pattern | Priority |
|---|---|---|---|
| Judge rubric | [eval/benchmarks/judge.py:130-146](../../src/psychology_base/eval/benchmarks/judge.py#L130-L146) | append `fact_language_directive(gold_answer_text)` | low (English-only benchmarks today) |
| Reader prompt | [eval/benchmarks/reader.py](../../src/psychology_base/eval/benchmarks/reader.py) | same | low |
| Tutor system prompt | [applications/learning/tutor_agent.py](../../src/psychology_base/applications/learning/tutor_agent.py) | per-session detect + directive | medium (韧性学习 is Chinese-first) |
| Companion system prompt | [applications/emotional/companion_agent.py](../../src/psychology_base/applications/emotional/companion_agent.py) | per-session detect + directive | medium (情感陪伴 is Chinese-first) |
| `v_metrics._tokens` / `_cosine_bow` | [eval/v_metrics.py:13,25](../../src/psychology_base/eval/v_metrics.py#L13-L25) | `tokenize(text.lower(), lang)` | low (no non-English V-track today) |
| LoCoMo `_normalize` | [eval/benchmarks/locomo.py:118-134](../../src/psychology_base/eval/benchmarks/locomo.py#L118-L134) | conditional article strip on `lang=="en"` | low |
| EpBench `parse_answer_items` | [eval/benchmarks/epbench.py:49-67](../../src/psychology_base/eval/benchmarks/epbench.py#L49-L67) | conditional `" and "` splitter on `lang=="en"` | low |
| Extraction entity pre-filter | [companion/extraction.py:224-225](../../src/psychology_base/companion/extraction.py#L224-L225) | `tokenize(text.lower(), lang)` | medium (silently drops CJK entity matches today) |
| Embedder route | [llm/router.py:278-325](../../src/psychology_base/llm/router.py#L278-L325) | gated on `i18n.embedder_multilingual: true` | medium (needed for Chinese recall at scale) |

Each migration is a 1-line append or a 2-line conditional under the byte-identical English contract. None is scheduled this pass; the dispatch infrastructure exists so each can land independently when its downstream need materializes.

---

## 6. Honest limitations

- **Thai tokenization is one-token-per-clause** (letter-run regex, no word boundaries). Same behavior as `.split()` → net-zero regression, net-zero improvement. `pythainlp` / `jieba` plug in later via `configs/llm.yaml` `i18n.tokenizer: jieba|pythainlp`.
- **`detect_lang` is a block heuristic, not statistical ID.** Code-switched utterances with a dominant non-Latin script win (correct for the common 中英混合 case). Predominantly-English utterances with one Chinese quoted term misclassify as `zh`. Not yet exercised by any call site, so no gate consequence today. If this becomes load-bearing, swap in `langdetect` / `fastText-lid.176` behind the same `detect_lang` signature.
- **Embedder path is reserved only.** `embedder_multilingual: true` wires nothing in [llm/router.py](../../src/psychology_base/llm/router.py); it is a documented future knob.
- **Prompt dispatch scoped to extraction.py only.** Judge, reader, and the two application-agent prompts remain English-forcing. Chinese user interacting with the companion today still gets English scaffolding; this slice makes the extracted *memory* Chinese-faithful but not the surface response. Migrating the application prompts is the follow-up with user-visible payoff.
- **Op tokens stay English by hard parser contract.** If the user ever wants fully-localized tokens, [extraction.py:135,156,176](../../src/psychology_base/companion/extraction.py#L135-L176) parser dispatch would also need a translation table — larger surface, deliberately not touched.

---

## 7. North-star alignment

Per CLAUDE.md: *"a mechanism on tracks 1–2 that cannot be used by the learning tutor or the companion is out of scope."* The i18n foundation is explicitly scoped to the two Chinese-first applications. The extraction-site migration directly addresses the hypothesized failure mode: Chinese user → silently-anglicized facts in memory → affect loss → 共情 degradation. The remaining migrations (judge/reader/application-agent prompts, metric tokenizers, embedder) are the queue; each unblocks either a specific application behavior or a specific measurement correctness property.

Per [docs/sota-goal.md](../sota-goal.md): *"SOTA is defined by application outcomes — best in-application result for the two surfaces."* Multi-language support is a precondition for the Chinese-first applications to even measure correctly; without it, the SOTA claim would be made on an English stand-in that doesn't reflect the actual deployment surface.
