"""Language-conditional prompt fragments.

The design contract is: ``fact_language_directive(text_or_lang)`` returns
the empty string for English and ``"unknown"`` inputs, so any legacy call
site that appends its return value to an existing system prompt is
byte-identical for the English pipeline (and therefore cannot drift the
Phase 1.5 / Phase 2 / Phase 7 LLM-output-dependent gates).

For non-English inputs, the returned fragment instructs the model to
write entity / relation / value fields in the user's language while
keeping op tokens (ADD / UPDATE / NOOP) and the pipe-separated grammar in
English — those are parser-internal and must not drift.
"""

from __future__ import annotations

from psychology_base.i18n import detect_lang


_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "he": "Hebrew",
    "hi": "Hindi",
    "th": "Thai",
    "ru": "Russian",
    "el": "Greek",
}


_FACT_LANGUAGE_DIRECTIVE_TEMPLATE = (
    "\n\nLANGUAGE: The CONVERSATION WINDOW is in {lang_name}. "
    "Write every entity, relation, and value field in {lang_name} as the "
    "speaker uses it — do NOT translate to English. "
    "Preserve names, places, affective words, and idiomatic phrases verbatim. "
    "The op tokens (ADD / UPDATE / NOOP) and the pipe-separated grammar MUST stay "
    "in English (parser requirement)."
)


def fact_language_directive(
    text_or_lang: str,
    *,
    is_lang_tag: bool = False,
) -> str:
    """Return a directive to append to a fact-extraction system prompt.

    Args:
        text_or_lang: Either the conversation window text (default) or a
            pre-detected language tag (when ``is_lang_tag=True``).
        is_lang_tag: Set True to pass a tag directly and skip detection.

    Returns:
        An empty string for English / unknown (so the caller's English
        prompt bytes are unchanged), or a non-empty appendable directive
        for a recognized non-English language.
    """
    if is_lang_tag:
        lang = text_or_lang
    else:
        lang = detect_lang(text_or_lang)
    if lang in ("en", "unknown"):
        return ""
    name = _LANGUAGE_NAMES.get(lang, lang)
    return _FACT_LANGUAGE_DIRECTIVE_TEMPLATE.format(lang_name=name)


__all__ = ["fact_language_directive"]
