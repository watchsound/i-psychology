"""Verification contract for the γ-RAG reader and LLM-judge directives.

The reader (``eval/benchmarks/reader.py``) and the judge
(``eval/benchmarks/judge.py``) were the last two English-forcing sites
on the Phase 9+ i18n close-out migration queue. Both are migrated by
appending a ``LANGUAGE: ...`` directive to the USER content — not the
locked system prompt — so prior cached completions / cached judgments
remain valid for English runs. The invariants locked here:

  * Byte-identical EN contract. For an English fragment block or
    English gold+candidate pair, the appended directive is "" and the
    user-content bytes are the pre-i18n bytes. This is what keeps
    LongMemEval / LoCoMo / EpBench gates from drifting on migration.
  * Non-empty + language-named appendage for Chinese / Japanese /
    Korean inputs — the dispatch is actually live.
  * Parser-contract clauses carry through: the reader directive pins
    the English ``not stated`` sentinel (``locomo._normalize`` matches
    the literal); the judge directive pins the JSON envelope
    + verdict labels in English (``_parse_judge_response``).
  * Integration: for English the reader's and judge's user content
    strictly matches the pre-i18n shape; for Chinese it is a strict
    superset (same prefix + directive appended).
"""

from __future__ import annotations

import pytest

from psychology_base.eval.adapters import Fragment
from psychology_base.eval.benchmarks.judge import LLMJudge
from psychology_base.eval.benchmarks.reader import LLMReader
from psychology_base.i18n.prompts import (
    judge_language_directive,
    reader_language_directive,
)
from psychology_base.llm import (
    CacheConfig,
    LLMConfig,
    LLMRouter,
    LocalOllamaConfig,
    ProviderConfig,
)
from psychology_base.llm.providers import MockProvider


# ---------- reader_language_directive unit ----------


def test_reader_english_text_returns_empty_directive() -> None:
    assert reader_language_directive("I lived in Boston last year.") == ""


def test_reader_unknown_text_returns_empty_directive() -> None:
    assert reader_language_directive("12345") == ""
    assert reader_language_directive("") == ""
    assert reader_language_directive("!!!") == ""


def test_reader_chinese_directive_names_chinese_and_pins_sentinel() -> None:
    directive = reader_language_directive("我住在北京。")
    assert directive != ""
    assert "Chinese" in directive
    # 'not stated' is the deterministic abstention sentinel at
    # locomo._normalize — a translated form would silently flip
    # abstention metrics to false negatives, so the directive MUST
    # pin it in English.
    assert "not stated" in directive


def test_reader_japanese_directive() -> None:
    assert "Japanese" in reader_language_directive("私は東京に住んでいます。")


def test_reader_korean_directive() -> None:
    assert "Korean" in reader_language_directive("저는 서울에 살고 있습니다.")


def test_reader_arabic_directive() -> None:
    assert "Arabic" in reader_language_directive("أنا أعيش في القاهرة.")


def test_reader_explicit_lang_tag_bypasses_detection() -> None:
    assert reader_language_directive("en", is_lang_tag=True) == ""
    assert "Chinese" in reader_language_directive("zh", is_lang_tag=True)
    assert "Japanese" in reader_language_directive("ja", is_lang_tag=True)


# ---------- judge_language_directive unit ----------


def test_judge_english_text_returns_empty_directive() -> None:
    assert judge_language_directive("red\ncrimson") == ""


def test_judge_unknown_text_returns_empty_directive() -> None:
    assert judge_language_directive("12345") == ""
    assert judge_language_directive("") == ""


def test_judge_chinese_directive_names_chinese_and_pins_envelope() -> None:
    directive = judge_language_directive("红色\n猩红色")
    assert directive != ""
    assert "Chinese" in directive
    # Envelope field names + verdict labels are parser contracts.
    # If the model translated 'correct|partial|incorrect', the
    # fallback at _parse_judge_response would silently coerce the
    # judgment to 0.0 'incorrect' — same risk shape as the
    # extraction op-token clause.
    assert "score" in directive
    assert "verdict" in directive
    assert "rationale" in directive
    assert "correct" in directive
    assert "partial" in directive
    assert "incorrect" in directive


def test_judge_japanese_directive() -> None:
    # Needs hiragana/katakana to disambiguate from CJK — pure-kanji
    # strings fall through to Chinese per the detect_lang priority
    # order (same convention as test_i18n_prompts for extraction).
    assert "Japanese" in judge_language_directive("赤色です\n深紅です")


def test_judge_korean_directive() -> None:
    assert "Korean" in judge_language_directive("빨강\n주홍")


def test_judge_explicit_lang_tag_bypasses_detection() -> None:
    assert judge_language_directive("en", is_lang_tag=True) == ""
    assert "Chinese" in judge_language_directive("zh", is_lang_tag=True)


def test_judge_directive_omits_extraction_grammar() -> None:
    """Judge directive is for scoring, not fact emission.

    The extraction op-token clause (ADD / UPDATE / NOOP / pipe
    grammar) would mislead the judge into emitting pipe-separated
    output. The judge directive mentions its OWN parser contract
    (JSON envelope) only.
    """
    directive = judge_language_directive("红色\n猩红色")
    assert "ADD" not in directive
    assert "UPDATE" not in directive
    assert "NOOP" not in directive
    assert "pipe-separated" not in directive


# ---------- reader integration (byte-identical EN, append on non-EN) ----------


def _router_for_reader(tmp_path, capture: list) -> LLMRouter:
    cfg = LLMConfig(
        active_provider="mock",
        providers={
            "mock": ProviderConfig(
                name="mock",
                simple_model="mock-simple",
                advanced_model="mock-advanced",
            )
        },
        local_ollama=LocalOllamaConfig(enabled=False),
        cache=CacheConfig(enabled=False),
    )

    def responder(messages, tier):
        capture.append(messages)
        return "stub answer"

    return LLMRouter(cfg, providers={"mock": MockProvider(name="mock", responder=responder)})


def _reader_english_fragments() -> tuple[Fragment, ...]:
    from datetime import datetime

    return (
        Fragment(
            id="f0",
            content="user: I live in Boston.",
            timestamp=datetime(2026, 1, 1),
        ),
    )


def _reader_chinese_fragments() -> tuple[Fragment, ...]:
    from datetime import datetime

    return (
        Fragment(
            id="f0",
            content="user: 我住在北京。",
            timestamp=datetime(2026, 1, 1),
        ),
    )


def test_reader_english_user_content_byte_identical(tmp_path) -> None:
    capture: list = []
    router = _router_for_reader(tmp_path, capture)
    reader = LLMReader(router=router, prompt_variant="v1")
    reader.answer(
        question="where did you move from?",
        recall_content="",
        fragments=_reader_english_fragments(),
    )
    assert len(capture) == 1
    user_msg = capture[0][1]
    assert user_msg.role == "user"
    # No LANGUAGE: directive leaked into the English pipeline.
    assert "LANGUAGE:" not in user_msg.content
    # Shape is the pre-i18n template: Question / Memory / fragments /
    # Short-form answer. We check the stable sentinels so a cosmetic
    # whitespace tweak later doesn't false-fail this lock.
    assert user_msg.content.startswith("Question: where did you move from?")
    assert "Memory fragments surfaced by the system:" in user_msg.content
    assert user_msg.content.endswith("Short-form answer:")


def test_reader_chinese_user_content_appends_directive(tmp_path) -> None:
    capture: list = []
    router = _router_for_reader(tmp_path, capture)
    reader = LLMReader(router=router, prompt_variant="v1")
    reader.answer(
        question="你搬家前住在哪里?",
        recall_content="",
        fragments=_reader_chinese_fragments(),
    )
    assert len(capture) == 1
    user_msg = capture[0][1]
    assert "LANGUAGE:" in user_msg.content
    assert "Chinese" in user_msg.content
    # The 'not stated' English sentinel pin reaches the model.
    assert "not stated" in user_msg.content


def test_reader_english_pipeline_prompt_variant_sha_unchanged(tmp_path) -> None:
    """Appending the directive to USER content leaves the SYSTEM prompt
    (the sha-pinned reader-variant body) unchanged — a non-English run
    does not invalidate the cached English reader-prompt cache slot.
    """
    capture: list = []
    router = _router_for_reader(tmp_path, capture)
    reader = LLMReader(router=router, prompt_variant="v3")
    # English call, then Chinese call — the system prompt must be
    # byte-identical across both.
    reader.answer(
        question="where did you move from?",
        recall_content="",
        fragments=_reader_english_fragments(),
    )
    reader.answer(
        question="你搬家前住在哪里?",
        recall_content="",
        fragments=_reader_chinese_fragments(),
    )
    assert len(capture) == 2
    sys_en = capture[0][0].content
    sys_zh = capture[1][0].content
    assert sys_en == sys_zh


# ---------- judge integration (byte-identical EN, append on non-EN) ----------


def _router_for_judge(capture: list) -> LLMRouter:
    cfg = LLMConfig(
        active_provider="mock",
        providers={},
        local_ollama=LocalOllamaConfig(enabled=False),
        cache=CacheConfig(enabled=False),
    )

    def responder(messages, tier):
        capture.append(messages)
        return '{"score": 1.0, "verdict": "correct", "rationale": "ok"}'

    return LLMRouter(cfg, providers={"mock": MockProvider(responder=responder)})


def test_judge_english_user_content_byte_identical() -> None:
    capture: list = []
    router = _router_for_judge(capture)
    judge = LLMJudge(router=router)
    judge.judge(
        benchmark="longmemeval",
        question="color?",
        gold="red",
        candidate="crimson",
        question_type="single-session-user",
    )
    assert len(capture) == 1
    user_msg = capture[0][1]
    assert user_msg.role == "user"
    assert "LANGUAGE:" not in user_msg.content
    # Rebuild the pre-i18n content exactly, modulo the appended "" from
    # the directive, and compare bytes.
    pre_i18n_user = (
        "Benchmark: longmemeval\n"
        "Question type: single-session-user\n"
        "Question: color?\n"
        "Gold answer: red\n"
        "Candidate answer: crimson\n\n"
        "Grade the candidate against the gold using the rubric. "
        "Respond with the JSON object only."
    )
    assert user_msg.content == pre_i18n_user


def test_judge_chinese_user_content_appends_directive() -> None:
    capture: list = []
    router = _router_for_judge(capture)
    judge = LLMJudge(router=router)
    judge.judge(
        benchmark="locomo",
        question="她住在哪里?",
        gold="北京",
        candidate="北京市",
        question_type="single-session-preference",
    )
    assert len(capture) == 1
    user_msg = capture[0][1]
    assert "LANGUAGE:" in user_msg.content
    assert "Chinese" in user_msg.content
    # Parser-contract pins reach the model.
    assert "correct" in user_msg.content
    assert "partial" in user_msg.content
    assert "incorrect" in user_msg.content


def test_judge_prompt_sha_unchanged_across_languages() -> None:
    """The directive rides on USER content, not SYSTEM — ``prompt_sha``
    (which keys the judgment cache) MUST be identical for English and
    non-English calls, so prior cached English judgments stay valid
    after the migration lands.
    """
    router = _router_for_judge(capture=[])
    judge_en = LLMJudge(router=router)
    judge_zh = LLMJudge(router=router)
    # Both LLMJudge instances derive prompt_sha from the SAME locked
    # _JUDGE_SYSTEM_PROMPT. If the directive had been bolted onto the
    # system prompt instead of user content, prompt_sha would differ
    # per-call-locale and this assertion would fail.
    assert judge_en.prompt_sha == judge_zh.prompt_sha
