"""Phase 8 Pillar C v2 — trait_inference internals.

Covers the F2 (signed-anchor rescale) + F1 (A/O implicit detectors) +
backward-compat regressions. No LLM, no network, in-memory only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psychology_base.applications.emotional.chat_import import (
    ImportedTurn,
    ImportFormat,
)
from psychology_base.applications.emotional.trait_inference import (
    _infer_implicit_traits,
    _rescale_signed,
    infer_traits,
)
from psychology_base.applications.emotional.trait_profile import (
    LIKERT_MAX,
    LIKERT_MID,
    LIKERT_MIN,
    OceanDim,
)


_T0 = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


def _u(content: str, i: int = 0) -> tuple[int, ImportedTurn]:
    return (
        i,
        ImportedTurn(
            speaker="user",
            content=content,
            timestamp=_T0 + timedelta(minutes=i),
            source_format=ImportFormat.GENERIC_JSONL,
        ),
    )


def _score_for(candidates, dim: OceanDim) -> float | None:
    for c in candidates:
        if c.kind != "trait":
            continue
        implicit_tag = f"trait:{dim.value}:implicit"
        if implicit_tag not in c.tags:
            continue
        for t in c.tags:
            prefix = f"trait:{dim.value}:score:"
            if t.startswith(prefix):
                return float(t[len(prefix):])
    return None


# ---------------------------------------------------------------------------
# F2 — _rescale_signed
# ---------------------------------------------------------------------------


def test_rescale_signed_at_baseline_is_midpoint() -> None:
    assert _rescale_signed(0.05, baseline=0.05, half_width=0.3) == LIKERT_MID
    assert _rescale_signed(0.0, baseline=0.0, half_width=0.3) == LIKERT_MID


def test_rescale_signed_saturates_up_and_down() -> None:
    assert _rescale_signed(10.0, baseline=0.0, half_width=0.3) == LIKERT_MAX
    assert _rescale_signed(-10.0, baseline=0.0, half_width=0.3) == LIKERT_MIN


def test_rescale_signed_linear_between_endpoints() -> None:
    # At baseline + half_width → LIKERT_MAX.
    assert _rescale_signed(0.3, baseline=0.0, half_width=0.3) == LIKERT_MAX
    # At baseline - half_width → LIKERT_MIN.
    assert _rescale_signed(-0.3, baseline=0.0, half_width=0.3) == LIKERT_MIN
    # Halfway up → LIKERT_MID + 1.0 == 4.0 (since LIKERT_MAX−LIKERT_MID=2).
    assert _rescale_signed(0.15, baseline=0.0, half_width=0.3) == LIKERT_MID + 1.0


def test_rescale_signed_zero_half_width_returns_midpoint() -> None:
    assert _rescale_signed(1.0, baseline=0.0, half_width=0.0) == LIKERT_MID
    assert _rescale_signed(1.0, baseline=0.0, half_width=-0.1) == LIKERT_MID


# ---------------------------------------------------------------------------
# F2 — N/E/C implicit detectors land at MIDPOINT on empty signal
#       (regression on Pillar C Tier-2 pathology)
# ---------------------------------------------------------------------------


def test_empty_content_implicit_neuroticism_at_midpoint() -> None:
    turns = [_u("went to the grocery store today", 0)]
    cands = _infer_implicit_traits(turns, now=_T0)
    assert _score_for(cands, OceanDim.NEUROTICISM) == LIKERT_MID


def test_empty_content_implicit_extraversion_near_midpoint() -> None:
    # Short conversational utterance → below E baseline → below midpoint
    # but NOT at LIKERT_MIN. Threshold: strictly > LIKERT_MIN.
    turns = [_u("ok", 0)]
    cands = _infer_implicit_traits(turns, now=_T0)
    score = _score_for(cands, OceanDim.EXTRAVERSION)
    assert score is not None
    assert score > LIKERT_MIN


def test_empty_content_implicit_conscientiousness_at_midpoint() -> None:
    turns = [_u("sounds good", 0)]
    cands = _infer_implicit_traits(turns, now=_T0)
    assert _score_for(cands, OceanDim.CONSCIENTIOUSNESS) == LIKERT_MID


def test_conscientiousness_disfluency_pushes_low() -> None:
    turns = [
        _u("like, idk, whatever, i dunno sorta thing", i)
        for i in range(3)
    ]
    cands = _infer_implicit_traits(turns, now=_T0)
    score = _score_for(cands, OceanDim.CONSCIENTIOUSNESS)
    assert score is not None
    assert score < LIKERT_MID


def test_conscientiousness_struct_markers_push_high() -> None:
    turns = [
        _u("today:\n- morning run\n- deep work\n- journal", 0),
        _u("plan:\n- step 1\n- step 2\n- step 3", 1),
    ]
    cands = _infer_implicit_traits(turns, now=_T0)
    score = _score_for(cands, OceanDim.CONSCIENTIOUSNESS)
    assert score is not None
    assert score > LIKERT_MID


def test_high_neuroticism_lexical_still_above_midpoint() -> None:
    # Phase 7 B4 backward-compat pin.
    turns = [
        _u("i'm so anxious about everything, i panic daily", 0),
        _u("overwhelmed, scared, worried about tomorrow", 1),
    ]
    cands = _infer_implicit_traits(turns, now=_T0)
    score = _score_for(cands, OceanDim.NEUROTICISM)
    assert score is not None
    assert score >= LIKERT_MID + 0.5


def test_high_extraversion_long_exclam_above_midpoint() -> None:
    long_exclam = (
        "yes!!! that's amazing, i absolutely love it, "
        "we should hang out this weekend for sure!!! "
        "can't wait to see everyone again!!"
    )
    turns = [_u(long_exclam, i) for i in range(3)]
    cands = _infer_implicit_traits(turns, now=_T0)
    score = _score_for(cands, OceanDim.EXTRAVERSION)
    assert score is not None
    assert score >= LIKERT_MID + 0.5


# ---------------------------------------------------------------------------
# F1 — Agreeableness implicit detector
# ---------------------------------------------------------------------------


def test_agreeableness_warmth_above_midpoint() -> None:
    turns = [
        _u("thank you so much, i really appreciate your help", 0),
        _u("please take care, no worries about the delay", 1),
        _u("i understand, that makes sense to me", 2),
    ]
    cands = _infer_implicit_traits(turns, now=_T0)
    score = _score_for(cands, OceanDim.AGREEABLENESS)
    assert score is not None
    assert score >= LIKERT_MID + 0.5


def test_agreeableness_hostility_below_midpoint() -> None:
    turns = [
        _u("ugh, that is stupid and ridiculous", 0),
        _u("whatever, this is pathetic nonsense", 1),
        _u("shut up, you are an idiot", 2),
    ]
    cands = _infer_implicit_traits(turns, now=_T0)
    score = _score_for(cands, OceanDim.AGREEABLENESS)
    assert score is not None
    assert score <= LIKERT_MID - 0.5


def test_agreeableness_neutral_content_at_midpoint() -> None:
    turns = [_u("went to the grocery store today", 0)]
    cands = _infer_implicit_traits(turns, now=_T0)
    assert _score_for(cands, OceanDim.AGREEABLENESS) == LIKERT_MID


# ---------------------------------------------------------------------------
# F1 — Openness implicit detector
# ---------------------------------------------------------------------------


def test_openness_hypothetical_above_midpoint() -> None:
    turns = [
        _u("what if we tried a totally different approach to this", 0),
        _u("i wonder what would happen if we changed everything", 1),
        _u("imagine the possibilities, maybe we could explore more", 2),
    ]
    cands = _infer_implicit_traits(turns, now=_T0)
    score = _score_for(cands, OceanDim.OPENNESS)
    assert score is not None
    assert score >= LIKERT_MID + 0.3


def test_openness_novelty_ttr_fires_above_midpoint() -> None:
    # High-TTR content: diverse, abstract, varied vocabulary with no
    # explicit hypothetical cues — pure novelty-density signal.
    turns = [
        _u(
            "philosophy, poetry, cosmology, biology, chemistry, "
            "linguistics, geology, astronomy, botany, mythology",
            0,
        ),
    ]
    cands = _infer_implicit_traits(turns, now=_T0)
    score = _score_for(cands, OceanDim.OPENNESS)
    assert score is not None
    assert score >= LIKERT_MID + 0.3


def test_openness_neutral_content_at_midpoint() -> None:
    # TTR < 4 content tokens → shortsample floor returns 0.5 →
    # offset=0, no hypothetical hits → signal=0 → LIKERT_MID.
    turns = [_u("okay fine", 0)]
    cands = _infer_implicit_traits(turns, now=_T0)
    assert _score_for(cands, OceanDim.OPENNESS) == LIKERT_MID


# ---------------------------------------------------------------------------
# Discovered Issue #1 (2026-10-08) — O routine-cue anti-openness signal
# ---------------------------------------------------------------------------


def test_openness_routine_cues_push_below_midpoint() -> None:
    turns = [
        _u("i'll stick to the usual, no need to change", 0),
        _u("same old plan, keep it simple", 1),
        _u("nothing fancy, nothing new to try", 2),
    ]
    cands = _infer_implicit_traits(turns, now=_T0)
    score = _score_for(cands, OceanDim.OPENNESS)
    assert score is not None
    assert score <= LIKERT_MID - 0.5


def test_openness_routine_cue_dominates_moderate_ttr() -> None:
    # Even with varied vocabulary (modest TTR offset above 0.5), a
    # single-turn routine cue should still pull the signal to or below
    # midpoint. Guards against TTR-only ceiling drift on low-O dialogs
    # that happen to use diverse vocabulary while expressing routine.
    turns = [
        _u(
            "philosophy aside, i stick to my usual breakfast routine daily",
            0,
        ),
    ]
    cands = _infer_implicit_traits(turns, now=_T0)
    score = _score_for(cands, OceanDim.OPENNESS)
    assert score is not None
    assert score < LIKERT_MID


def test_openness_cjk_routine_cue_fires() -> None:
    turns = [_u("按部就班就好，没什么新想法", 0)]
    cands = _infer_implicit_traits(turns, now=_T0)
    score = _score_for(cands, OceanDim.OPENNESS)
    assert score is not None
    assert score < LIKERT_MID


# ---------------------------------------------------------------------------
# Discovered Issue #2 (2026-10-08) — E exclamation-rate dominant signal
# ---------------------------------------------------------------------------


def test_extraversion_no_exclaim_lands_near_midpoint() -> None:
    # Long measured prose with ZERO exclamations should NOT saturate
    # high (prior bug). Expected: below midpoint but within half Likert.
    long_measured = (
        "I spent the morning reading quietly. The weather was fair, "
        "nothing remarkable. I made tea and watched the garden for "
        "a while before heading in to work on the usual projects."
    )
    turns = [_u(long_measured, i) for i in range(3)]
    cands = _infer_implicit_traits(turns, now=_T0)
    score = _score_for(cands, OceanDim.EXTRAVERSION)
    assert score is not None
    assert score < LIKERT_MID
    assert score >= LIKERT_MID - 1.0  # not pathologically low either


def test_extraversion_length_without_exclaim_does_not_saturate_high() -> None:
    # Pre-fix regression pin: 240-char measured prose, 0 exclaims
    # (matches BIG5-CHAT low-E empirical profile) should land BELOW
    # midpoint, not at LIKERT_MAX.
    content = "I " + ("spent some time quietly reflecting today. " * 5)
    turns = [_u(content, 0)]
    cands = _infer_implicit_traits(turns, now=_T0)
    score = _score_for(cands, OceanDim.EXTRAVERSION)
    assert score is not None
    assert score < LIKERT_MID


# ---------------------------------------------------------------------------
# Cross-cutting: full pipeline through infer_traits emits all five dims
# ---------------------------------------------------------------------------


def test_infer_traits_emits_candidates_for_all_five_dims() -> None:
    turns = [
        ImportedTurn(
            speaker="user",
            content="what if we explored a creative new direction here",
            timestamp=_T0,
            source_format=ImportFormat.GENERIC_JSONL,
        ),
    ]
    cands = infer_traits(turns)
    dims_seen = {
        tag.split(":")[1]
        for c in cands
        if c.kind == "trait"
        for tag in c.tags
        if tag.startswith("trait:") and tag.count(":") == 1
    }
    assert "openness" in dims_seen
    assert "conscientiousness" in dims_seen
    assert "extraversion" in dims_seen
    assert "agreeableness" in dims_seen
    assert "neuroticism" in dims_seen
