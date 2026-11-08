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

    # Fault injection only: positive segmentation tests use the real backend.
    monkeypatch.setattr(i18n_mod, "_load_jieba", lambda: (None, {}, "not installed"))
    monkeypatch.setattr(i18n_mod, "_jieba_warned_missing", False)

    text = "我最近感到很焦虑,工作压力很大。"
    with caplog.at_level("WARNING", logger="psychology_base.i18n"):
        tokens = tokenize(text, "zh", tokenizer="jieba")
        assert tokenize(text, "zh", tokenizer="jieba") == tokens
    # Fallback output matches the unicode path.
    assert tokens == tokenize(text, "zh", tokenizer="unicode")
    # Warning fired exactly once; the module-level flag is now set so
    # subsequent calls stay silent.
    warnings = [r for r in caplog.records if "jieba" in r.message]
    assert len(warnings) == 1


def test_real_jieba_identity_and_single_backend():
    from hashlib import sha256
    from importlib.metadata import version
    from pathlib import Path
    import psychology_base.i18n as i18n
    from psychology_base.companion.system import CompanionSystem

    jieba = pytest.importorskip("jieba")
    first = CompanionSystem(cjk_tokenizer="jieba")
    backend, _, error = i18n._load_jieba()
    assert error is None and backend is not jieba.dt
    identity = first.inspect()["tokenizer_identity"]
    assert identity["module_version"] == identity["distribution_version"] == version("jieba")
    assert identity["HMM"] is True and identity["cut_all"] is False
    root = Path(identity["origin"]).parent
    for relative, digest in identity["resources_sha256"].items():
        assert sha256((root / relative).read_bytes()).hexdigest() == digest
    assert sha256((root / "dict.txt").read_bytes()).hexdigest() == identity["dictionary_sha256"]
    assert identity["dictionary_sha256"] in backend.cache_file
    second = CompanionSystem(cjk_tokenizer="jieba")
    assert second.inspect()["tokenizer_identity"] == identity
    assert i18n._load_jieba()[0] is backend
    identity["resources_sha256"].clear()
    assert first.inspect()["tokenizer_identity"]["resources_sha256"]


def test_backend_resolution_failure_is_cached(monkeypatch, caplog):
    import psychology_base.i18n as i18n

    attempts = []

    def unavailable(name):
        attempts.append(name)
        raise ImportError("dependency unavailable in fault injection")

    monkeypatch.setattr(i18n, "_jieba_result", None)
    monkeypatch.setattr(i18n, "_jieba_warned_missing", False)
    monkeypatch.setattr(i18n, "import_module", unavailable)
    for _ in range(2):
        with pytest.raises(ValueError, match="dependency unavailable"):
            i18n.tokenizer_identity("jieba")
    with caplog.at_level("WARNING", logger="psychology_base.i18n"):
        for _ in range(2):
            assert tokenize("中文分词", "zh", tokenizer="jieba") == ["中文分词"]
    assert attempts == ["jieba"]
    assert len([record for record in caplog.records if "falling back" in record.message]) == 1


def test_fallback_warning_is_once_when_logging_reenters_tokenization(monkeypatch, caplog):
    import logging
    import psychology_base.i18n as i18n

    monkeypatch.setattr(i18n, "_load_jieba", lambda: (None, {}, "not installed"))
    monkeypatch.setattr(i18n, "_jieba_warned_missing", False)
    text = "中文分词"

    class ReentrantHandler(logging.Handler):
        entered = False

        def emit(self, record):
            if not self.entered:
                self.entered = True
                assert tokenize(text, "zh", tokenizer="jieba") == [text]

    handler = ReentrantHandler()
    i18n.logger.addHandler(handler)
    try:
        with caplog.at_level("WARNING", logger="psychology_base.i18n"):
            assert tokenize(text, "zh", tokenizer="jieba") == [text]
    finally:
        i18n.logger.removeHandler(handler)
    assert len([record for record in caplog.records if "falling back" in record.message]) == 1


def test_backend_rejects_missing_api(monkeypatch):
    import psychology_base.i18n as i18n

    module = pytest.importorskip("jieba")
    monkeypatch.setattr(i18n, "_jieba_result", None)
    monkeypatch.delattr(module, "Tokenizer")
    with pytest.raises(ValueError, match="Tokenizer"):
        i18n.tokenizer_identity("jieba")


def test_backend_resource_failure_rejects_construction_and_preserves_fallback(monkeypatch):
    from pathlib import Path
    import psychology_base.i18n as i18n
    from psychology_base.companion.system import CompanionSystem

    pytest.importorskip("jieba")
    read_bytes = Path.read_bytes

    def unreadable_resource(path):
        if path.name == "prob_emit.py":
            raise PermissionError("HMM resource unreadable in fault injection")
        return read_bytes(path)

    monkeypatch.setattr(i18n, "_jieba_result", None)
    monkeypatch.setattr(i18n, "_jieba_warned_missing", False)
    monkeypatch.setattr(Path, "read_bytes", unreadable_resource)
    system = CompanionSystem.__new__(CompanionSystem)
    with pytest.raises(ValueError, match="HMM resource unreadable"):
        system.__init__(cjk_tokenizer="jieba")
    assert not hasattr(system, "companion")
    assert CompanionSystem().cjk_tokenizer == "unicode"
    assert tokenize("中文分词", "zh", tokenizer="jieba") == ["中文分词"]


def test_empty_backend_smoke_test_rejects_construction(monkeypatch):
    import psychology_base.i18n as i18n
    from psychology_base.companion.system import CompanionSystem

    module = pytest.importorskip("jieba")
    monkeypatch.setattr(i18n, "_jieba_result", None)
    # Fault injection only: a broken dependency must not qualify as segmentation.
    monkeypatch.setattr(module.Tokenizer, "cut", lambda *args, **kwargs: iter(()))
    system = CompanionSystem.__new__(CompanionSystem)
    with pytest.raises(ValueError, match="smoke test produced no letter tokens"):
        system.__init__(cjk_tokenizer="jieba")
    assert not hasattr(system, "companion")


def test_concurrent_first_resolution_shares_one_real_backend(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import psychology_base.i18n as i18n

    pytest.importorskip("jieba")
    monkeypatch.setattr(i18n, "_jieba_result", None)
    with ThreadPoolExecutor(max_workers=4) as pool:
        resolved = list(pool.map(lambda _: i18n._load_jieba(), range(4)))
    assert all(error is None for _, _, error in resolved)
    assert len({id(backend) for backend, _, _ in resolved}) == 1


def test_unicode_does_not_import_jieba_in_fresh_process():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", """
import sys
from psychology_base.companion.system import CompanionSystem
from psychology_base.i18n import tokenize
assert 'jieba' not in sys.modules
assert CompanionSystem().inspect()['tokenizer_identity']['backend'] == 'unicode'
assert tokenize("it's unchanged", 'en', tokenizer='jieba') == ["it's", 'unchanged']
assert 'jieba' not in sys.modules
"""], check=True, capture_output=True, text=True)


def test_private_dictionary_ignores_positive_global_word_addition():
    import subprocess
    import sys

    pytest.importorskip("jieba")
    subprocess.run([sys.executable, "-c", """
import jieba
from psychology_base.i18n import tokenize
phrase = '心理陪伴测试专用词'
before = tokenize(phrase, 'zh', tokenizer='jieba')
assert before != [phrase]
jieba.add_word(phrase, freq=1000000000)
assert list(jieba.cut(phrase, cut_all=False, HMM=True)) == [phrase]
assert tokenize(phrase, 'zh', tokenizer='jieba') == before
"""], check=True, capture_output=True, text=True, encoding="utf-8")


@pytest.mark.parametrize("tokenizer", ["unicode", "jieba"])
def test_memory_english_adapter_preserves_apostrophe_hyphen_contract(tokenizer):
    from psychology_base.companion.reconstruction import tokens

    assert tokens("It's state-of-the-art, with DOGS!", tokenizer) == [
        "it", "s", "state", "of", "the", "art", "with", "dogs",
    ]
