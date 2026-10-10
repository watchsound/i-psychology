"""Phase 2 R8-b — fact-density tie-breaker tests.

Covers:
- ``_fact_density_score`` counts proper nouns + years + ISO/DMY dates +
  bare integers; leaves common verbs and lowercase words at zero.
- ``_apply_fact_density_tie_breaker`` only swaps adjacent near-ties
  (score delta ≤ 0.03) with density diff ≥ 2.
- Confident reranker ordering (gap > delta) is preserved.
- Density tie-breaker never promotes a density-equivalent fragment.
- ``fact_density_tie_breaker=True`` requires ``cross_encoder_rerank=True``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from psychology_base.companion import CompanionSystem
from psychology_base.companion.system import (
    _apply_fact_density_tie_breaker,
    _fact_density_score,
)
from psychology_base.eval.adapters import Fragment
from psychology_base.llm import LLMCache, LLMConfig, LLMRouter, LocalOllamaConfig
from psychology_base.llm.config import ProviderConfig
from psychology_base.llm.providers import MockProvider


def _router(tmp_path) -> LLMRouter:
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
    )
    return LLMRouter(
        cfg,
        providers={"mock": MockProvider(name="mock")},
        cache=LLMCache(cache_dir=tmp_path / "cache"),
    )


def _frag(fid: str, content: str) -> Fragment:
    return Fragment(
        id=fid,
        content=content,
        timestamp=datetime(2026, 1, 15, tzinfo=timezone.utc),
        affect={},
        tags=("session_1",),
    )


def test_fact_density_proper_nouns_counted():
    # Two proper nouns: Sweden, Charlotte.
    assert _fact_density_score("I moved from Sweden with Charlotte") == 2


def test_fact_density_year_and_proper_noun_both_counted():
    # Sweden (proper noun) + 2023 (year); 2023 is NOT a bare_int because
    # the bare_int regex uses \d{1,3} and non-word look-around, so
    # 4-digit years don't double-count. "Moved" is sentence-initial and
    # in the common-word filter, so it is dropped — only "Sweden" counts
    # as a proper noun.
    s = _fact_density_score("Moved to Sweden in 2023")
    # 1 proper noun + 1 year = 2
    assert s == 2


def test_fact_density_sentence_start_common_verb_dropped():
    # Sentence-initial "Visited" / "Moved" / "Went" are common verbs
    # that get titlecased only because they start the sentence. They
    # should NOT inflate the proper-noun count.
    assert _fact_density_score("Visited Boston yesterday") == 1  # Boston
    assert _fact_density_score("Went to Paris") == 1  # Paris
    assert _fact_density_score("The sky was blue") == 0  # The dropped
    assert _fact_density_score("She loves her cat") == 0  # She dropped


def test_fact_density_sentence_start_proper_noun_still_counted():
    # If the sentence-initial word is NOT in the common-word filter
    # (i.e., it's a plausible proper noun), it is still counted. This
    # preserves "Sweden is nice" vs "She is nice" asymmetry.
    assert _fact_density_score("Sweden is where I lived") == 1
    assert _fact_density_score("Charlotte visited yesterday") == 1


def test_fact_density_sentence_start_after_period():
    # Sentence-start detection must also fire after . ! ? + whitespace,
    # not only at the absolute start of the string.
    assert _fact_density_score("I lived there. Moved to Sweden.") == 1  # Sweden
    assert _fact_density_score("Hello. Visited Rome.") == 1  # Rome


def test_fact_density_dmy_date_counted():
    s = _fact_density_score("Her birthday was 19 October 2023")
    # "October" is a proper noun (1), "2023" is a year (1), DMY-date
    # regex also matches (+1) = 3. The double-count on October/2023 is
    # intentional — dates are extra-high-signal and the tie-breaker
    # threshold min_density_diff=2 compensates.
    assert s >= 3


def test_fact_density_bare_integer_counted():
    # "2" is a bare int.
    assert _fact_density_score("I have 2 kids") == 1


def test_fact_density_lowercase_only_scores_zero():
    assert _fact_density_score("i went to the beach yesterday") == 0


def test_fact_density_isolated_capital_letter_not_counted():
    # Single capital 'I' is not a proper-noun match (regex requires
    # at least one interior lowercase letter).
    assert _fact_density_score("I ate lunch") == 0


def test_tie_breaker_swaps_near_tie_with_denser_fragment():
    # a_score 0.80, b_score 0.79 → delta 0.01 ≤ 0.03 (threshold).
    # a density = 0 (lowercase filler); b density = 2 (Sweden + 2023).
    # Diff = 2 ≥ 2 (threshold) → swap.
    a = (_frag("a", "that was a nice trip for us"), 0.80)
    b = (_frag("b", "moved to Sweden in 2023"), 0.79)
    out = _apply_fact_density_tie_breaker([a, b])
    assert out[0][0].id == "b"
    assert out[1][0].id == "a"


def test_tie_breaker_preserves_confident_ordering():
    # Score gap 0.10 > 0.03 → DO NOT swap even though b is denser.
    a = (_frag("a", "that was a nice trip for us"), 0.80)
    b = (_frag("b", "moved to Sweden in 2023"), 0.70)
    out = _apply_fact_density_tie_breaker([a, b])
    assert out[0][0].id == "a"
    assert out[1][0].id == "b"


def test_tie_breaker_preserves_order_when_density_equal():
    # Score gap small (near-tie), but density diff < min_density_diff.
    a = (_frag("a", "moved to Sweden in 2023"), 0.80)
    b = (_frag("b", "moved to Boston in 2024"), 0.79)
    # a density = 2 (Sweden + 2023), b density = 2 (Boston + 2024).
    # Diff = 0 < 2 → no swap.
    out = _apply_fact_density_tie_breaker([a, b])
    assert out[0][0].id == "a"
    assert out[1][0].id == "b"


def test_tie_breaker_single_pass_not_full_sort():
    # 3-fragment case where a bubble-sort would promote c to position 0
    # but our single-pass only swaps adjacent pairs once. Verify
    # single-pass semantics so we don't accidentally over-promote.
    a = (_frag("a", "trip was nice"), 0.80)
    b = (_frag("b", "trip was great"), 0.79)
    c = (_frag("c", "visited Sweden in 2023 with Charlotte"), 0.78)
    # pair(a,b): delta 0.01, density 0 vs 0 → no swap.
    # pair(b,c): delta 0.01, density 0 vs 3 → swap b/c.
    # Result: [a, c, b]. c did NOT bubble past a (not a full sort).
    out = _apply_fact_density_tie_breaker([a, b, c])
    assert [t[0].id for t in out] == ["a", "c", "b"]


def test_tie_breaker_empty_and_single_fragment_pass_through():
    assert _apply_fact_density_tie_breaker([]) == []
    one = [(_frag("a", "whatever"), 0.5)]
    assert _apply_fact_density_tie_breaker(one) == one


def test_fact_density_tie_breaker_requires_cross_encoder_rerank(tmp_path):
    router = _router(tmp_path)
    with pytest.raises(ValueError, match="requires cross_encoder_rerank=True"):
        CompanionSystem(
            witness_active=False,
            consolidation_enabled=True,
            consolidation_router=router,
            cross_encoder_rerank=False,
            fact_density_tie_breaker=True,
        )


def test_fact_density_tie_breaker_defaults_off(tmp_path):
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        cross_encoder_rerank=True,
    )
    assert cs.fact_density_tie_breaker is False
