"""Language-conditional prompt fragments.

The design contract is: every public directive in this module returns
the empty string for English and ``"unknown"`` inputs, so any legacy
call site that appends its return value to an existing system prompt
is byte-identical for the English pipeline (and therefore cannot drift
the Phase 1.5 / Phase 2 / Phase 7 / Phase 8 LLM-output-dependent gates).

Two directives live here, each with its own single context:

- ``fact_language_directive`` targets the extraction fact-emission
  pipeline. It names the entity / relation / value field slots and
  explicitly pins op tokens (ADD / UPDATE / NOOP) in English because
  those are a hard parser contract at ``extraction.py:_parse_line``.
- ``response_language_directive`` targets agent-response prompts
  (TutorAgent, CompanionAgent). It names no parser-internal grammar;
  it only tells the model to answer in the user's language and to
  preserve names / affect / idiom verbatim.

Both share a private ``_language_name_for`` helper so the language-name
lookup, detection fallback, and empty-for-en guard are not duplicated.
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


_RESPONSE_LANGUAGE_DIRECTIVE_TEMPLATE = (
    "\n\nLANGUAGE: The user is writing in {lang_name}. "
    "Respond in {lang_name} matching the user's language — do NOT switch to English. "
    "Preserve names, affective words, and idiomatic phrases verbatim as the user uses them."
)


def _language_name_for(
    text_or_lang: str, *, is_lang_tag: bool = False
) -> str | None:
    """Resolve input to a display language name, or None for en / unknown.

    Returning None (not "English") for the English/unknown path is what
    makes the empty-string-for-English contract trivial for every caller:
    one ``if name is None: return ""`` guard above the template fill.
    """
    if is_lang_tag:
        lang = text_or_lang
    else:
        lang = detect_lang(text_or_lang)
    if lang in ("en", "unknown"):
        return None
    return _LANGUAGE_NAMES.get(lang, lang)


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
    name = _language_name_for(text_or_lang, is_lang_tag=is_lang_tag)
    if name is None:
        return ""
    return _FACT_LANGUAGE_DIRECTIVE_TEMPLATE.format(lang_name=name)


def response_language_directive(
    text_or_lang: str,
    *,
    is_lang_tag: bool = False,
) -> str:
    """Return a directive to append to an agent-response system prompt.

    Intended for `TutorAgent` and `CompanionAgent` system prompts: tells
    the model to match the user's language in its reply. Does NOT
    mention any extraction-pipeline grammar (no entity/relation/value
    fields, no op tokens) — those are irrelevant to agent-response
    emission and would be misleading if appended there.

    Args:
        text_or_lang: Either the user utterance text (default) or a
            pre-detected language tag (when ``is_lang_tag=True``).
        is_lang_tag: Set True to pass a tag directly and skip detection.

    Returns:
        An empty string for English / unknown (byte-identical append
        contract preserved), or a non-empty appendable directive for a
        recognized non-English language.
    """
    name = _language_name_for(text_or_lang, is_lang_tag=is_lang_tag)
    if name is None:
        return ""
    return _RESPONSE_LANGUAGE_DIRECTIVE_TEMPLATE.format(lang_name=name)


__all__ = ["fact_language_directive", "response_language_directive"]
