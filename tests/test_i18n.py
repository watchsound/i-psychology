"""Byte-identical English contract + non-English coverage for i18n helper.

These tests lock the invariant that the English code path through
``tokenize`` is bit-for-bit the same as the legacy ``str.split()`` so that
future migrations of English-forcing sites (v_metrics, locomo _normalize,
extraction prompt pre-filter) can be done piecewise with zero numeric
drift on existing gates.
"""

from __future__ import annotations

import pytest

from psychology_base.i18n import LangTag, detect_lang, tokenize


ENGLISH_CORPUS = [
    "it's a cat",
    "hello world",
    "   leading and trailing   ",
    "line1\nline2\tline3",
    "e-mail addresses like a@b.com stay whole",
    "double  spaces   between   words",
    "",
    "mixedCASE WORDS with 1234 digits and ,punct;",
]


@pytest.mark.parametrize("text", ENGLISH_CORPUS)
def test_english_tokenize_byte_identical_to_split(text: str) -> None:
    """`tokenize(text, "en")` must equal `text.split()` on English input."""
    assert tokenize(text, "en") == text.split()


@pytest.mark.parametrize("text", ENGLISH_CORPUS)
def test_english_autodetect_matches_split(text: str) -> None:
    """Auto-detected English path returns the same list as explicit en."""
    assert tokenize(text) == text.split()


def test_chinese_tokenize_non_empty() -> None:
    """CJK input returns at least one non-empty token (not one monster blob)."""
    tokens = tokenize("北京下雨了", "zh")
    assert tokens, "CJK text must not tokenize to empty"
    assert all(t for t in tokens)
    assert "".join(tokens) == "北京下雨了"


def test_chinese_autodetect() -> None:
    assert detect_lang("北京下雨了") == "zh"
    assert tokenize("北京下雨了")  # non-empty via auto-detect


def test_japanese_detect_beats_cjk() -> None:
    """Hiragana/katakana forces 'ja' even when kanji are also present."""
    assert detect_lang("こんにちは世界") == "ja"
    assert detect_lang("カタカナです") == "ja"
    assert tokenize("こんにちは世界")


def test_korean_detect() -> None:
    assert detect_lang("안녕하세요") == "ko"
    assert tokenize("안녕하세요")


def test_arabic_detect_and_tokenize() -> None:
    assert detect_lang("مرحبا بالعالم") == "ar"
    tokens = tokenize("مرحبا بالعالم")
    assert len(tokens) == 2


def test_hebrew_detect() -> None:
    assert detect_lang("שלום עולם") == "he"
    assert tokenize("שלום עולם")


def test_thai_detect_tokenizes_as_run() -> None:
    """Thai has no inter-word spaces; a letter-run tokenizer returns one
    token covering the whole clause. That is the best we can do without
    a Thai-specific tokenizer; the regression here is that today's
    `.split()` returns the same monster blob, so parity holds.
    """
    assert detect_lang("สวัสดีชาวโลก") == "th"
    assert tokenize("สวัสดีชาวโลก")


def test_russian_detect() -> None:
    assert detect_lang("Привет мир") == "ru"
    assert tokenize("Привет мир")


def test_devanagari_detect() -> None:
    assert detect_lang("नमस्ते दुनिया") == "hi"
    assert tokenize("नमस्ते दुनिया")


def test_empty_and_numeric_fall_through_to_unknown_then_split() -> None:
    """Empty / numeric-only / punct-only text is `unknown` and the
    tokenize fallback mirrors `.split()` so no caller breaks on numerics.
    """
    assert detect_lang("") == "unknown"
    assert detect_lang("12345") == "unknown"
    assert detect_lang("!!! ???") == "unknown"
    assert tokenize("12345 67890") == ["12345", "67890"]


def test_explicit_lang_overrides_detect() -> None:
    """Passing lang='en' on CJK text forces the English `.split()` path.

    This is intentional: it lets a caller lock behavior to the legacy
    tokenizer during migration validation runs, so byte-identical old
    behavior can be reproduced on any input.
    """
    text = "北京 下雨"
    assert tokenize(text, "en") == text.split()  # whitespace-split, 2 blobs
    assert tokenize(text, "zh") == ["北京", "下雨"]


def test_langtag_type_is_literal() -> None:
    """Smoke check that LangTag is importable (used as type hint by
    future call sites)."""
    from typing import get_args

    # Must at minimum cover the big-script set we test for.
    args = get_args(LangTag)
    for required in ("en", "zh", "ja", "ko", "ar", "th", "unknown"):
        assert required in args
