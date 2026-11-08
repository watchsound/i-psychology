"""Language-dispatch utilities for multi-language user interaction.

Shared dispatch for application, memory and evaluation callers. Memory keeps
its own English regex adapter; this module's English contract is str.split().

Design contract:
    * `tokenize(text, "en")` must return `text.split()` byte-identical.
    * Default `tokenize(text, lang)` for non-English returns a non-empty list
      for any text that contains at least one letter in a supported
      script (Latin, CJK, Hiragana/Katakana, Hangul, Arabic, Hebrew,
      Devanagari, Thai, Cyrillic, Greek).
    * `detect_lang` is a Unicode-block heuristic, not a replacement for
      langdetect/fastText — correct for mono-script utterances, imperfect
      on code-switched text.
    * `tokenizer="jieba"` opts into finer-grained Chinese word
      segmentation (zh only); other languages fall through to the
      Unicode letter-run path. English stays byte-identical regardless.
      Jieba is an optional soft dependency — a missing import degrades
      gracefully to the Unicode path with a one-time module-level warning
      rather than raising, so partial installs don't crash the pipeline.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib import import_module
from importlib.metadata import version
import logging
from pathlib import Path
import re
from threading import RLock
from typing import Literal
from unicodedata import category

logger = logging.getLogger(__name__)

LangTag = Literal[
    "en", "zh", "ja", "ko", "ar", "he", "hi", "th", "ru", "el", "unknown"
]

_SUPPORTED_TOKENIZERS = ("unicode", "jieba")
_TOKENIZATION_POLICY = "i18n-tokenization-v2-script-punctuation"
_jieba_result = None
_jieba_lock = RLock()
_jieba_warned_missing = False

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
    punctuation -> "unknown". Punctuation never supplies script evidence,
    including characters inside otherwise script-specific Unicode blocks.
    """
    if not text:
        return "unknown"
    has_cjk = has_hira = has_kata = has_hangul = False
    has_arabic = has_hebrew = has_thai = False
    has_cyrillic = has_greek = has_devanagari = False
    has_latin = False
    for ch in text:
        if category(ch).startswith("P"):
            continue
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


def _load_jieba():
    """Resolve one private dictionary backend, identity or failure per process.

    The instance avoids jieba.dt's vocabulary. HMM state is still shared by
    upstream jieba; runtime vocabulary mutation is outside this fixed policy.
    The lock also prevents concurrent first callers from building two backends.
    """
    global _jieba_result
    with _jieba_lock:
        if _jieba_result is not None:
            return _jieba_result
        try:
            module = import_module("jieba")
            installed_version = version("jieba")
            backend = module.Tokenizer()
            origin = Path(module.__file__).resolve()
            resources = [origin, origin.with_name("finalseg") / "__init__.py"]
            resources.extend(origin.with_name("finalseg") / name for name in (
                "prob_start.py", "prob_trans.py", "prob_emit.py",
            ))
            hashes = {str(path.relative_to(origin.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in resources}
            with backend.get_dict_file() as dictionary:
                dictionary_hash = hashlib.sha256(dictionary.read()).hexdigest()
            # Upstream's default cache filename is shared across environments.
            # Namespace ours by real code/dictionary so a legacy cache is not reused.
            backend.cache_file = f"psychology-base-jieba-{hashes['__init__.py']}-{dictionary_hash}.cache"
            backend.initialize()
            if not any(_UNICODE_LETTER_RUN.fullmatch(tok)
                       for tok in backend.cut("中文分词", cut_all=False, HMM=True)):
                raise ValueError("segmentation smoke test produced no letter tokens")
            identity = {
                "backend": "jieba", "policy": _TOKENIZATION_POLICY,
                "module_version": installed_version, "distribution_version": installed_version,
                "origin": str(origin), "resources_sha256": hashes,
                "dictionary_sha256": dictionary_hash,
                "cut_all": False, "HMM": True,
                "letter_filter": _UNICODE_LETTER_RUN.pattern, "normalization": "none",
            }
            _jieba_result = (backend, identity, None)
        except (ImportError, OSError, ValueError, AttributeError, EOFError) as exc:
            _jieba_result = (None, {}, f"{type(exc).__name__}: {exc}")
        return _jieba_result


def _validate_tokenizer(tokenizer: str) -> None:
    if tokenizer not in _SUPPORTED_TOKENIZERS:
        raise ValueError(f"Unknown tokenizer {tokenizer!r}; expected one of {_SUPPORTED_TOKENIZERS}")


def tokenizer_identity(tokenizer: str = "unicode") -> dict:
    """Validate a requested backend and return detached construction metadata.

    Explicitly configured memory systems use this strict preflight. Standalone
    tokenize() retains its warning/fallback contract through the same loader.
    This is configuration evidence, not attestation against process mutation.
    """
    _validate_tokenizer(tokenizer)
    if tokenizer == "unicode":
        return {"backend": "unicode", "policy": _TOKENIZATION_POLICY}
    backend, identity, error = _load_jieba()
    if backend is None:
        raise ValueError(
            f"i18n.tokenizer='jieba' unavailable: {error}. "
            'Install the Chinese extra with pip install "psychology-base[zh]".'
        )
    return deepcopy(identity)


def tokenize(
    text: str,
    lang: str | None = None,
    *,
    tokenizer: str = "unicode",
) -> list[str]:
    """Language-aware tokenization with English byte-identical dispatch.

    The English branch returns ``text.split()`` verbatim so legacy call
    sites (e.g. ``text.lower().split()``) stay bit-for-bit stable when
    migrated — the caller may still apply ``.lower()`` or any other
    pre-processing around this call. The default non-English branch extracts
    Unicode letter runs and strips punctuation. For scripts without spaces,
    these can be entire clauses; they are not word segmentation.

    When ``tokenizer="jieba"`` is passed AND ``lang`` resolves to ``zh``,
    the Chinese branch routes through jieba's word-segmentation instead
    of the one-token-per-clause Unicode letter-run. All other languages
    (including English) are unaffected — jieba only ships Chinese
    segmentation, so threading the flag through other langs would be a
    silent no-op. A missing jieba install degrades to the Unicode path
    with a logged warning (fires once per process) rather than raising.
    """
    global _jieba_warned_missing
    _validate_tokenizer(tokenizer)
    if lang is None:
        lang = detect_lang(text)
    if lang == "en" or lang == "unknown":
        return text.split()
    if tokenizer == "jieba" and lang == "zh":
        backend, _identity, error = _load_jieba()
        if backend is None:
            with _jieba_lock:
                warn = not _jieba_warned_missing
                _jieba_warned_missing = True
            # Publish before calling logging: handlers can re-enter tokenization.
            if warn:
                logger.warning(
                    "i18n.tokenizer='jieba' unavailable (%s); falling back to "
                    "unicode letter-run tokenization. Install psychology-base[zh].",
                    error,
                )
        else:
            # jieba.cut yields punctuation + whitespace as separate
            # tokens; filter to pure Unicode-letter runs so the jieba
            # path and the unicode path share the "letters only, no
            # punct" output contract. Comma / period / spaces drop out.
            return [
                tok for tok in backend.cut(text, cut_all=False, HMM=True)
                if _UNICODE_LETTER_RUN.fullmatch(tok)
            ]
    return _UNICODE_LETTER_RUN.findall(text)


__all__ = ["LangTag", "detect_lang", "tokenize", "tokenizer_identity"]
