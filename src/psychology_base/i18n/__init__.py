"""Language-dispatch utilities for multi-language user interaction.

This module is deliberately additive: nothing in the companion/memory/eval
stack imports it yet. It exists so the four English-forcing sites
(extraction/consolidation prompts, metric tokenizers, benchmark
normalizers, embedder selection) can be migrated piecewise behind a
dispatch, with the English code path remaining byte-identical.

Design contract:
    * `tokenize(text, "en")` must return `text.split()` byte-identical.
    * `tokenize(text, lang)` for non-English must return a non-empty list
      for any text that contains at least one letter in a supported
      script (Latin, CJK, Hiragana/Katakana, Hangul, Arabic, Hebrew,
      Devanagari, Thai, Cyrillic, Greek).
    * `detect_lang` is a Unicode-block heuristic, not a replacement for
      langdetect/fastText — correct for mono-script utterances, imperfect
      on code-switched text.
"""

from __future__ import annotations

import re
from typing import Literal

LangTag = Literal[
    "en", "zh", "ja", "ko", "ar", "he", "hi", "th", "ru", "el", "unknown"
]

_UNICODE_LETTER_RUN = re.compile(
    r"[A-Za-z"
    r"\u00C0-\u024F"
    r"\u0370-\u03FF"
    r"\u0400-\u04FF"
    r"\u0590-\u05FF"
    r"\u0600-\u06FF"
    r"\u0900-\u097F"
    r"\u0E00-\u0E7F"
    r"\u3040-\u309F"
    r"\u30A0-\u30FF"
    r"\u3400-\u4DBF"
    r"\u4E00-\u9FFF"
    r"\uAC00-\uD7AF"
    r"]+"
)


def detect_lang(text: str) -> LangTag:
    """Return the dominant script tag for `text`.

    Script-priority order follows what disambiguates: Hiragana/Katakana
    beats CJK (Japanese contains kanji); Hangul beats CJK (Korean rarely
    contains Han); CJK alone -> zh. Latin -> en. ASCII-only numeric or
    punctuation -> "unknown".
    """
    if not text:
        return "unknown"
    has_cjk = has_hira = has_kata = has_hangul = False
    has_arabic = has_hebrew = has_thai = False
    has_cyrillic = has_greek = has_devanagari = False
    has_latin = False
    for ch in text:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            has_cjk = True
        elif 0x3040 <= o <= 0x309F:
            has_hira = True
        elif 0x30A0 <= o <= 0x30FF:
            has_kata = True
        elif 0xAC00 <= o <= 0xD7AF:
            has_hangul = True
        elif 0x0600 <= o <= 0x06FF:
            has_arabic = True
        elif 0x0590 <= o <= 0x05FF:
            has_hebrew = True
        elif 0x0E00 <= o <= 0x0E7F:
            has_thai = True
        elif 0x0400 <= o <= 0x04FF:
            has_cyrillic = True
        elif 0x0370 <= o <= 0x03FF:
            has_greek = True
        elif 0x0900 <= o <= 0x097F:
            has_devanagari = True
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            has_latin = True
    if has_hira or has_kata:
        return "ja"
    if has_hangul:
        return "ko"
    if has_cjk:
        return "zh"
    if has_arabic:
        return "ar"
    if has_hebrew:
        return "he"
    if has_devanagari:
        return "hi"
    if has_thai:
        return "th"
    if has_cyrillic:
        return "ru"
    if has_greek:
        return "el"
    if has_latin:
        return "en"
    return "unknown"


def tokenize(text: str, lang: str | None = None) -> list[str]:
    """Language-aware tokenization with English byte-identical dispatch.

    The English branch returns ``text.split()`` verbatim so legacy call
    sites (e.g. ``text.lower().split()``) stay bit-for-bit stable when
    migrated — the caller may still apply ``.lower()`` or any other
    pre-processing around this call. The non-English branch extracts
    runs of Unicode letters, which splits whitespace-less scripts
    (CJK/Thai/Khmer) correctly and strips punctuation uniformly.
    """
    if lang is None:
        lang = detect_lang(text)
    if lang == "en" or lang == "unknown":
        return text.split()
    return _UNICODE_LETTER_RUN.findall(text)


__all__ = ["LangTag", "detect_lang", "tokenize"]
