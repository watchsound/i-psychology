"""Verification contract for the language-conditional prompt directive.

The core invariants are:
  * English (and unknown) input returns the empty string, so any legacy
    call site appending the directive produces byte-identical bytes to
    the pre-i18n version.
  * Non-English input returns a non-empty directive that names the
    detected language.
  * The directive explicitly preserves op tokens in English (parser
    contract — if the model started translating ADD/UPDATE/NOOP, the
    parser at extraction.py:_parse_line would reject every line).
  * Integration with extract_facts: for an English window the composed
    system prompt is byte-identical to the pre-i18n composition; for a
    Chinese window the composed system prompt is a strict superset
    (legacy prompt + directive).
"""

from __future__ import annotations

import pytest

from psychology_base.companion.extraction import (
    EXTRACTION_SYSTEM_PROMPT,
    MAX_FACTS_PER_WINDOW,
)
from psychology_base.i18n.prompts import fact_language_directive


def test_english_text_returns_empty_directive() -> None:
    assert fact_language_directive("I live in Boston and I like pottery.") == ""


def test_unknown_text_returns_empty_directive() -> None:
    """Numeric-only / punct-only text has no detectable language — the
    English pipeline must stay byte-identical for these degenerate
    inputs, not accidentally get a directive appended."""
    assert fact_language_directive("12345") == ""
    assert fact_language_directive("") == ""
    assert fact_language_directive("!!! ???") == ""


def test_chinese_directive_names_chinese_and_preserves_op_tokens() -> None:
    directive = fact_language_directive("我住在北京,我喜欢陶艺。")
    assert directive != ""
    assert "Chinese" in directive
    # Op-token preservation is a hard parser contract.
    assert "ADD" in directive and "UPDATE" in directive and "NOOP" in directive


def test_japanese_directive() -> None:
    directive = fact_language_directive("私は東京に住んでいます。")
    assert "Japanese" in directive


def test_korean_directive() -> None:
    directive = fact_language_directive("저는 서울에 살고 있습니다.")
    assert "Korean" in directive


def test_arabic_directive() -> None:
    directive = fact_language_directive("أنا أعيش في القاهرة.")
    assert "Arabic" in directive


def test_russian_directive() -> None:
    directive = fact_language_directive("Я живу в Москве.")
    assert "Russian" in directive


def test_hindi_directive() -> None:
    directive = fact_language_directive("मैं दिल्ली में रहता हूँ।")
    assert "Hindi" in directive


def test_thai_directive() -> None:
    directive = fact_language_directive("ฉันอาศัยอยู่ที่กรุงเทพฯ")
    assert "Thai" in directive


def test_explicit_lang_tag_bypasses_detection() -> None:
    """Callers that already know the lang (e.g. from a benchmark label)
    can skip the detection pass and get the same result."""
    assert fact_language_directive("en", is_lang_tag=True) == ""
    assert "Chinese" in fact_language_directive("zh", is_lang_tag=True)
    assert "Japanese" in fact_language_directive("ja", is_lang_tag=True)


def test_english_extraction_system_prompt_byte_identical() -> None:
    """The composed system prompt for an English window must equal the
    pre-i18n composition (the base template .format() alone). This is
    the hard contract that locks Phase 1.5 / Phase 2 / Phase 7
    LLM-output-dependent gates against drift.
    """
    legacy = EXTRACTION_SYSTEM_PROMPT.format(max_facts=MAX_FACTS_PER_WINDOW)
    with_i18n = legacy + fact_language_directive(
        "user: I live in Boston.\nuser: I like pottery."
    )
    assert with_i18n == legacy


def test_chinese_extraction_system_prompt_is_strict_superset() -> None:
    legacy = EXTRACTION_SYSTEM_PROMPT.format(max_facts=MAX_FACTS_PER_WINDOW)
    with_i18n = legacy + fact_language_directive(
        "user: 我住在北京。\nuser: 我喜欢陶艺。"
    )
    assert with_i18n.startswith(legacy)
    assert len(with_i18n) > len(legacy)
    assert "Chinese" in with_i18n
