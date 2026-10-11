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


# ---------- Phase 9+ follow-on: jieba CJK tokenizer knob ----------


# These lock the i18n.tokenizer="jieba" opt-in path. Jieba is the
# deferred tokenizer called out in the Phase 9+ i18n close-out
# "mono-sentence CJK one-token" honest limit. The invariants:
#   - "unicode" default stays byte-identical (regression lock)
#   - "jieba" + zh produces finer word segmentation than the Unicode
#     letter-run (so recall-cue "焦虑" can now hit a stored "我最近
#     感到很焦虑" fragment through shared tokens)
#   - non-zh langs ignore the knob (jieba ships zh only; silent
#     thread-through would create a false expectation)
#   - unknown-tokenizer rejects
#   - missing-jieba falls back to unicode (graceful)


def test_default_tokenizer_is_unicode_byte_identical() -> None:
    """Explicit tokenizer='unicode' must equal the legacy no-kwarg call."""
    text = "我最近感到很焦虑,工作压力很大。"
    assert tokenize(text, "zh", tokenizer="unicode") == tokenize(text, "zh")


def test_jieba_segments_chinese_finer_than_unicode() -> None:
    """Mono-sentence CJK → unicode gives 2 clauses; jieba gives words."""
    pytest.importorskip("jieba")
    text = "我最近感到很焦虑,工作压力很大。"
    unicode_tokens = tokenize(text, "zh", tokenizer="unicode")
    jieba_tokens = tokenize(text, "zh", tokenizer="jieba")
    assert len(unicode_tokens) == 2  # ['我最近感到很焦虑', '工作压力很大']
    assert len(jieba_tokens) > len(unicode_tokens)
    # The specific word we need for the end-to-end recall round-trip.
    assert "焦虑" in jieba_tokens


def test_jieba_filters_punctuation() -> None:
    """jieba.cut yields punct as separate tokens — tokenize must drop
    them so the jieba path and the unicode path share the same
    'letters only' output contract."""
    pytest.importorskip("jieba")
    tokens = tokenize("你好,世界!", "zh", tokenizer="jieba")
    assert "," not in tokens
    assert "!" not in tokens
    assert all(t.strip() for t in tokens)


def test_jieba_knob_english_byte_identical() -> None:
    """Threading tokenizer='jieba' through an English call stays
    byte-identical. Jieba ships Chinese segmentation only; letting it
    affect English would silently drift every migrated metric."""
    text = "it's a cat and the mat"
    assert tokenize(text, "en", tokenizer="jieba") == text.split()


def test_jieba_knob_non_zh_langs_fall_through_to_unicode() -> None:
    """Japanese / Korean / Arabic ignore the jieba knob — jieba is
    zh-only. Non-zh langs route through the Unicode letter-run as if
    the knob were 'unicode'."""
    pytest.importorskip("jieba")
    for text, lang in [
        ("私は東京に住んでいます。", "ja"),
        ("저는 서울에 살고 있습니다", "ko"),
        ("أنا أعيش في القاهرة", "ar"),
    ]:
        assert tokenize(text, lang, tokenizer="jieba") == tokenize(
            text, lang, tokenizer="unicode"
        )


def test_tokenize_rejects_unknown_tokenizer() -> None:
    """Typo-protection: unknown values raise rather than silently
    falling through to unicode."""
    with pytest.raises(ValueError, match="Unknown tokenizer"):
        tokenize("hello", "en", tokenizer="mecab")


def test_jieba_fallback_when_missing(monkeypatch, caplog) -> None:
    """If jieba import fails at runtime, tokenize must degrade to the
    Unicode path (not raise) and emit a single warning. Caller code
    must not need to defensively check availability."""
    import psychology_base.i18n as i18n_mod

    # Reset the one-shot import state and force the module to look
    # missing. monkeypatch restores on test teardown.
    monkeypatch.setattr(i18n_mod, "_jieba_import_attempted", True)
    monkeypatch.setattr(i18n_mod, "_jieba_module", None)
    monkeypatch.setattr(i18n_mod, "_jieba_warned_missing", False)

    text = "我最近感到很焦虑,工作压力很大。"
    with caplog.at_level("WARNING", logger="psychology_base.i18n"):
        tokens = tokenize(text, "zh", tokenizer="jieba")
    # Fallback output matches the unicode path.
    assert tokens == tokenize(text, "zh", tokenizer="unicode")
    # Warning fired exactly once; the module-level flag is now set so
    # subsequent calls stay silent.
    warnings = [r for r in caplog.records if "jieba" in r.message]
    assert len(warnings) == 1
