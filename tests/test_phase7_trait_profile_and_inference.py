"""Phase 7 §2.2 — TraitProfile reconstruction + trait_inference passive mode.

Three sections:
- TraitProfile: reconstruction (not field) over tagged fragments; covenant
  weighting; confidence floor; insight-gap detection.
- trait_inference: passive pass over ImportedTurn[]; one-way boundary
  (does not call observe); confirmation gate; covenant self-disclosure
  routed to Tier-2 handoff path.
- ingest_confirmed_candidates: refuses unconfirmed; stamps
  source:inferred-confirmed tag.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from psychology_base.applications.emotional.chat_import import (
    ImportFormat,
    ImportedTurn,
)
from psychology_base.applications.emotional.trait_inference import (
    SOURCE_INFERRED_CONFIRMED,
    SOURCE_INFERRED_UNCONFIRMED,
    InferenceCandidate,
    confirm_candidates,
    infer_traits,
    ingest_confirmed_candidates,
    summarize_meta,
)
from psychology_base.applications.emotional.trait_profile import (
    LIKERT_MAX,
    LIKERT_MID,
    LIKERT_MIN,
    OceanDim,
    TraitEstimate,
    TraitProfile,
)
from psychology_base.companion.fragment import COVENANT_TAG
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation


TS = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


def _mk_turn(speaker: str, content: str, *, offset_min: int = 0) -> ImportedTurn:
    return ImportedTurn(
        speaker=speaker,
        content=content,
        timestamp=TS + timedelta(minutes=offset_min),
        source_format=ImportFormat.GENERIC_JSONL,
    )


def _observe_trait(
    system: CompanionSystem,
    *,
    dim: OceanDim,
    score: float,
    layer: str,
    content: str = "evidence",
    offset_min: int = 0,
    covenant: bool = False,
) -> None:
    tags = (
        f"trait:{dim.value}",
        f"trait:{dim.value}:{layer}",
        f"trait:{dim.value}:score:{score:.1f}",
        SOURCE_INFERRED_CONFIRMED,
    )
    if covenant:
        tags = tags + (COVENANT_TAG,)
    system.observe(
        Observation(
            timestamp=TS + timedelta(minutes=offset_min),
            speaker="user",
            content=content,
            tags=tags,
            subject_id="user",
        )
    )


# ---------------------------------------------------------------------------
# TraitProfile — reconstruction over tagged fragments
# ---------------------------------------------------------------------------


def test_trait_profile_empty_store_returns_mid_likert() -> None:
    system = CompanionSystem(top_k=4)
    profile = TraitProfile(system=system, now=TS)
    est = profile.get(OceanDim.OPENNESS)
    assert est.value == LIKERT_MID
    assert est.confidence == 0.0
    assert est.n_explicit == 0 and est.n_implicit == 0


def test_trait_profile_single_explicit_high_openness() -> None:
    system = CompanionSystem(top_k=4)
    _observe_trait(system, dim=OceanDim.OPENNESS, score=5.0, layer="explicit")
    profile = TraitProfile(system=system, now=TS)
    est = profile.get(OceanDim.OPENNESS)
    assert est.value == pytest.approx(5.0, abs=0.01)
    assert est.n_explicit == 1
    assert est.n_implicit == 0
    assert 0.0 < est.confidence < 1.0


def test_trait_profile_explicit_implicit_agreement_boosts_confidence() -> None:
    system = CompanionSystem(top_k=4)
    for i in range(5):
        _observe_trait(
            system, dim=OceanDim.NEUROTICISM, score=5.0, layer="explicit",
            content=f"explicit {i}", offset_min=i,
        )
        _observe_trait(
            system, dim=OceanDim.NEUROTICISM, score=5.0, layer="implicit",
            content=f"implicit {i}", offset_min=i,
        )
    profile = TraitProfile(system=system, now=TS + timedelta(minutes=5))
    est = profile.get(OceanDim.NEUROTICISM)
    assert est.value == pytest.approx(5.0, abs=0.1)
    assert est.confidence >= 0.5


def test_trait_profile_disagreement_lowers_confidence() -> None:
    system = CompanionSystem(top_k=4)
    _observe_trait(system, dim=OceanDim.EXTRAVERSION, score=5.0, layer="explicit",
                   content="says high", offset_min=0)
    _observe_trait(system, dim=OceanDim.EXTRAVERSION, score=1.0, layer="implicit",
                   content="shows low", offset_min=1)
    profile = TraitProfile(system=system, now=TS + timedelta(minutes=1))
    est = profile.get(OceanDim.EXTRAVERSION)
    # (5 + 1) / 2 = 3 combined; confidence docked by |5-1|/4 = 1 → 0 agreement.
    assert est.value == pytest.approx(3.0, abs=0.01)
    assert est.confidence == pytest.approx(0.0, abs=0.01)
    assert est.insight_gap == pytest.approx(4.0, abs=0.01)


def test_trait_profile_covenant_floor_lifts_confidence() -> None:
    system = CompanionSystem(top_k=4)
    _observe_trait(
        system, dim=OceanDim.NEUROTICISM, score=5.0, layer="explicit",
        content="diagnosed", covenant=True,
    )
    profile = TraitProfile(system=system, now=TS)
    est = profile.get(OceanDim.NEUROTICISM)
    assert est.n_covenant == 1
    assert est.confidence >= 0.6


def test_trait_profile_vector_returns_all_five_dims() -> None:
    system = CompanionSystem(top_k=4)
    profile = TraitProfile(system=system, now=TS)
    vec = profile.vector()
    assert set(vec.keys()) == set(OceanDim)
    assert all(isinstance(v, TraitEstimate) for v in vec.values())


def test_trait_profile_insight_gap_detection() -> None:
    system = CompanionSystem(top_k=4)
    _observe_trait(system, dim=OceanDim.AGREEABLENESS, score=5.0, layer="explicit")
    _observe_trait(system, dim=OceanDim.AGREEABLENESS, score=2.0, layer="implicit",
                   offset_min=1)
    profile = TraitProfile(system=system, now=TS + timedelta(minutes=1))
    gaps = profile.insight_gaps(min_gap=0.5)
    assert OceanDim.AGREEABLENESS in gaps


def test_trait_profile_scope_bound_to_subject_user() -> None:
    system = CompanionSystem(top_k=4)
    # Observe a trait-tagged fragment on a different subject.
    system.observe(
        Observation(
            timestamp=TS,
            speaker="other",
            content="not the user",
            tags=(
                f"trait:{OceanDim.OPENNESS.value}",
                f"trait:{OceanDim.OPENNESS.value}:explicit",
                f"trait:{OceanDim.OPENNESS.value}:score:5.0",
                SOURCE_INFERRED_CONFIRMED,
            ),
            subject_id="other-person",
        )
    )
    profile = TraitProfile(system=system, user_id="user", now=TS)
    est = profile.get(OceanDim.OPENNESS)
    # Fragment is tagged subject:other-person, not subject:user, so it
    # must NOT contribute.
    assert est.n_explicit == 0
    assert est.value == LIKERT_MID


def test_trait_profile_confident_dims_threshold() -> None:
    system = CompanionSystem(top_k=4)
    for i in range(10):
        _observe_trait(
            system, dim=OceanDim.OPENNESS, score=5.0, layer="explicit",
            offset_min=i, content=f"ev {i}",
        )
        _observe_trait(
            system, dim=OceanDim.OPENNESS, score=5.0, layer="implicit",
            offset_min=i, content=f"im {i}",
        )
    profile = TraitProfile(system=system, now=TS + timedelta(minutes=10))
    confident = profile.confident_dims(threshold=0.5)
    assert OceanDim.OPENNESS in confident


# ---------------------------------------------------------------------------
# trait_inference — passive pass
# ---------------------------------------------------------------------------


def test_infer_traits_passive_does_not_observe() -> None:
    system = CompanionSystem(top_k=4)
    turns = [
        _mk_turn("user", "i'm anxious and overthink everything", offset_min=0),
        _mk_turn("user", "my mom called again", offset_min=10),
    ]
    cands = infer_traits(turns)
    assert cands  # non-empty
    # Fragment store unchanged — the pass is passive.
    assert len(system.companion.store.all()) == 0


def test_infer_traits_explicit_high_neuroticism_cue() -> None:
    turns = [_mk_turn("user", "i'm anxious every single day")]
    cands = infer_traits(turns)
    neurotic = [
        c for c in cands
        if c.kind == "trait"
        and any(f"trait:{OceanDim.NEUROTICISM.value}:explicit" in t for t in c.tags)
    ]
    assert neurotic, f"expected explicit N candidate, got {[c.summary for c in cands]}"
    assert any(f"trait:{OceanDim.NEUROTICISM.value}:score:5.0" in t for t in neurotic[0].tags)


def test_infer_traits_implicit_neuroticism_from_pad2() -> None:
    turns = [
        _mk_turn("user", "I'm scared and lonely", offset_min=0),
        _mk_turn("user", "feeling panic and anxious today", offset_min=10),
        _mk_turn("user", "hopeless and afraid", offset_min=20),
    ]
    cands = infer_traits(turns)
    implicit = [
        c for c in cands
        if c.kind == "trait"
        and any(f"trait:{OceanDim.NEUROTICISM.value}:implicit" in t for t in c.tags)
    ]
    assert implicit
    # Score encoded in a tag; parse it to confirm it's elevated.
    score_tag = next(
        t for t in implicit[0].tags
        if f"trait:{OceanDim.NEUROTICISM.value}:score:" in t
    )
    score = float(score_tag.rsplit(":", 1)[1])
    assert score > 3.0


def test_infer_traits_entity_mention_mother() -> None:
    turns = [
        _mk_turn("user", "my mom called again", offset_min=0),
        _mk_turn("user", "my mother never listens", offset_min=10),
    ]
    cands = infer_traits(turns)
    entities = [c for c in cands if c.kind == "entity"]
    mother = [c for c in entities if c.entity_id == "mother"]
    assert mother
    assert mother[0].entity_id == "mother"


def test_infer_traits_affective_baseline_emitted() -> None:
    turns = [
        _mk_turn("user", "I had a wonderful happy day", offset_min=0),
        _mk_turn("user", "feeling grateful and calm", offset_min=10),
    ]
    cands = infer_traits(turns)
    baseline = [c for c in cands if c.kind == "affective_baseline"]
    assert baseline
    assert baseline[0].affect["valence"] > 0.3


def test_infer_traits_only_scans_user_turns() -> None:
    turns = [
        _mk_turn("assistant", "you seem anxious", offset_min=0),
        _mk_turn("other:Mei", "we're both overthinking", offset_min=10),
        _mk_turn("user", "yeah", offset_min=20),
    ]
    cands = infer_traits(turns)
    # Assistant / other turns MUST NOT drive explicit-trait inference
    # about the user — the "I'm anxious" cue appears in assistant text.
    explicit_neurotic = [
        c for c in cands
        if c.kind == "trait" and "explicit" in " ".join(c.tags)
        and OceanDim.NEUROTICISM.value in " ".join(c.tags)
    ]
    assert explicit_neurotic == []


def test_infer_traits_covenant_self_disclosure_routes_to_handoff() -> None:
    turns = [_mk_turn("user", "I was diagnosed with major depression two years ago")]
    cands = infer_traits(turns)
    covenant = [c for c in cands if COVENANT_TAG in c.tags]
    assert covenant
    assert "requires:tier2-handoff" in covenant[0].tags


# ---------------------------------------------------------------------------
# Confirmation gate + ingest
# ---------------------------------------------------------------------------


def test_confirm_candidates_filters_by_confidence_floor() -> None:
    cands = (
        InferenceCandidate(
            kind="trait",
            summary="low conf",
            content="x",
            tags=(f"trait:{OceanDim.OPENNESS.value}", SOURCE_INFERRED_UNCONFIRMED),
            timestamp=TS,
            confidence=0.2,
        ),
        InferenceCandidate(
            kind="trait",
            summary="high conf",
            content="y",
            tags=(f"trait:{OceanDim.OPENNESS.value}", SOURCE_INFERRED_UNCONFIRMED),
            timestamp=TS,
            confidence=0.9,
        ),
    )
    confirmed = confirm_candidates(cands, confidence_floor=0.5)
    assert len(confirmed) == 1
    assert confirmed[0].summary == "high conf"
    assert confirmed[0].confirmed is True
    assert SOURCE_INFERRED_CONFIRMED in confirmed[0].tags
    assert SOURCE_INFERRED_UNCONFIRMED not in confirmed[0].tags


def test_confirm_candidates_excludes_covenant_candidates() -> None:
    cand = InferenceCandidate(
        kind="self_model_statement",
        summary="covenant",
        content="diagnosed",
        tags=(
            COVENANT_TAG, SOURCE_INFERRED_UNCONFIRMED, "requires:tier2-handoff",
        ),
        timestamp=TS,
        confidence=0.95,
    )
    confirmed = confirm_candidates([cand])
    assert confirmed == ()


def test_ingest_confirmed_candidates_refuses_unconfirmed() -> None:
    system = CompanionSystem(top_k=4)
    cand = InferenceCandidate(
        kind="trait",
        summary="unconfirmed",
        content="x",
        tags=(f"trait:{OceanDim.OPENNESS.value}", SOURCE_INFERRED_UNCONFIRMED),
        timestamp=TS,
        confidence=0.9,
        confirmed=False,
    )
    ingested = ingest_confirmed_candidates(system, [cand])
    assert ingested == 0
    assert len(system.companion.store.all()) == 0


def test_ingest_confirmed_candidates_requires_source_confirmed_tag() -> None:
    system = CompanionSystem(top_k=4)
    cand = InferenceCandidate(
        kind="trait",
        summary="confirmed but missing tag",
        content="x",
        tags=(f"trait:{OceanDim.OPENNESS.value}",),  # no source tag
        timestamp=TS,
        confidence=0.9,
        confirmed=True,
    )
    with pytest.raises(ValueError, match="source:inferred-confirmed"):
        ingest_confirmed_candidates(system, [cand])
    assert len(system.companion.store.all()) == 0


def test_ingest_confirmed_candidates_round_trip_populates_profile() -> None:
    system = CompanionSystem(top_k=4)
    turns = [
        _mk_turn("user", "i'm anxious and i overthink", offset_min=0),
        _mk_turn("user", "my mom called", offset_min=5),
    ]
    cands = infer_traits(turns)
    confirmed = confirm_candidates(cands, confidence_floor=0.0)
    ingested = ingest_confirmed_candidates(system, confirmed)
    assert ingested > 0
    assert len(system.companion.store.all()) == ingested

    profile = TraitProfile(system=system, now=TS + timedelta(hours=1))
    est = profile.get(OceanDim.NEUROTICISM)
    assert est.n_explicit + est.n_implicit > 0
    assert est.value > LIKERT_MID


def test_summarize_meta_pulls_entities_and_baseline() -> None:
    turns = [
        _mk_turn("user", "i'm happy and grateful today", offset_min=0),
        _mk_turn("user", "my mother is proud", offset_min=10),
    ]
    cands = infer_traits(turns)
    meta = summarize_meta(cands)
    assert "mother" in meta.significant_others
    assert meta.affective_baseline[0] > 0.3


def test_infer_traits_timestamps_preserve_source_time() -> None:
    old_ts = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    turn = ImportedTurn(
        speaker="user",
        content="i'm anxious",
        timestamp=old_ts,
        source_format=ImportFormat.GENERIC_JSONL,
    )
    cands = infer_traits([turn])
    trait_cands = [c for c in cands if c.kind == "trait"]
    assert any(c.timestamp == old_ts for c in trait_cands), (
        "Explicit trait timestamps must preserve source turn time, not reset to now."
    )
