"""Phase 7 §6 item 6 — CompanionAgent trait-conditioning hook tests.

Three sections:
- ``build_conditioning_addendum`` — reads a reconstructed TraitProfile
  and emits a scaffolding addendum, or empty string if no confident
  extreme dim exists.
- ``apply_trait_conditioning`` — the byte-identity guarantees:
  flag=None, flag.enabled=False, profile=None, or no-confident-dim
  → base prompt unchanged, byte-for-byte.
- CompanionAgent wiring — the LLM responder is called with the
  conditioned prompt only when the flag is enabled; the hard
  constraints (crisis handoff) remain untouched regardless.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from psychology_base.applications.emotional.companion_agent import (
    CompanionAgent,
    CompanionSystemPrompt,
    HANDOFF_SCRIPT,
    TraitConditioningFlag,
)
from psychology_base.applications.emotional.trait_conditioning import (
    apply_trait_conditioning,
    build_conditioning_addendum,
)
from psychology_base.applications.emotional.trait_inference import (
    SOURCE_INFERRED_CONFIRMED,
)
from psychology_base.applications.emotional.trait_profile import (
    OceanDim,
    TraitProfile,
)
from psychology_base.applications.safety.crisis_escalation import CrisisEscalation
from psychology_base.companion.fragment import COVENANT_TAG
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation


TS = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


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


def _populate_high_neuroticism(system: CompanionSystem, n: int = 6) -> None:
    for i in range(n):
        _observe_trait(
            system, dim=OceanDim.NEUROTICISM, score=5.0, layer="explicit",
            content=f"anxious evidence {i}", offset_min=i,
        )
        _observe_trait(
            system, dim=OceanDim.NEUROTICISM, score=5.0, layer="implicit",
            content=f"anxious trace {i}", offset_min=i,
        )


# ---------------------------------------------------------------------------
# build_conditioning_addendum
# ---------------------------------------------------------------------------


def test_addendum_empty_when_no_confident_dim() -> None:
    system = CompanionSystem(top_k=4)
    profile = TraitProfile(system=system, now=TS)
    assert build_conditioning_addendum(profile) == ""


def test_addendum_names_high_neuroticism_dim() -> None:
    system = CompanionSystem(top_k=4)
    _populate_high_neuroticism(system)
    profile = TraitProfile(system=system, now=TS)
    addendum = build_conditioning_addendum(profile)
    assert addendum != ""
    assert "high Neuroticism" in addendum
    # Hard-constraint reminder is present verbatim.
    assert "HARD CONSTRAINTS" in addendum
    assert "Rule 2" in addendum


def test_addendum_respects_confidence_floor() -> None:
    system = CompanionSystem(top_k=4)
    # One observation = low confidence.
    _observe_trait(system, dim=OceanDim.OPENNESS, score=5.0, layer="explicit")
    profile = TraitProfile(system=system, now=TS)
    est = profile.get(OceanDim.OPENNESS)
    assert est.confidence < 0.5
    assert build_conditioning_addendum(profile, confidence_floor=0.5) == ""


def test_addendum_includes_low_dim_clauses() -> None:
    system = CompanionSystem(top_k=4)
    for i in range(6):
        _observe_trait(
            system, dim=OceanDim.AGREEABLENESS, score=1.0, layer="explicit",
            content=f"direct evidence {i}", offset_min=i,
        )
        _observe_trait(
            system, dim=OceanDim.AGREEABLENESS, score=1.0, layer="implicit",
            content=f"direct trace {i}", offset_min=i,
        )
    profile = TraitProfile(system=system, now=TS)
    addendum = build_conditioning_addendum(profile)
    assert "low Agreeableness" in addendum
    # Low-A clause raises Rule 2's floor, must say so explicitly.
    assert "sharpen the Rule 2 move" in addendum


def test_addendum_covenant_lifts_single_statement_above_floor() -> None:
    """A single covenant-tagged self-disclosure clears the floor via

    the covenant confidence floor of 0.6 — a user who explicitly
    attests a diagnosis (e.g. 'I have major depressive disorder')
    gets scaffolding immediately, without a thousand turns of
    implicit evidence.
    """
    system = CompanionSystem(top_k=4)
    _observe_trait(
        system, dim=OceanDim.NEUROTICISM, score=5.0, layer="explicit",
        covenant=True,
    )
    profile = TraitProfile(system=system, now=TS)
    est = profile.get(OceanDim.NEUROTICISM)
    assert est.confidence >= 0.6  # covenant floor
    addendum = build_conditioning_addendum(profile, confidence_floor=0.5)
    assert "high Neuroticism" in addendum


# ---------------------------------------------------------------------------
# apply_trait_conditioning — byte-identity guarantees
# ---------------------------------------------------------------------------


def test_apply_byte_identical_when_flag_none() -> None:
    base = CompanionSystemPrompt.SYSTEM_PROMPT
    assert apply_trait_conditioning(base, profile=None, flag=None) == base


def test_apply_byte_identical_when_flag_disabled() -> None:
    base = CompanionSystemPrompt.SYSTEM_PROMPT
    system = CompanionSystem(top_k=4)
    _populate_high_neuroticism(system)
    profile = TraitProfile(system=system, now=TS)
    flag = TraitConditioningFlag(enabled=False)
    assert apply_trait_conditioning(base, profile, flag) == base


def test_apply_byte_identical_when_profile_none() -> None:
    base = CompanionSystemPrompt.SYSTEM_PROMPT
    flag = TraitConditioningFlag(enabled=True)
    assert apply_trait_conditioning(base, profile=None, flag=flag) == base


def test_apply_byte_identical_when_no_confident_dim() -> None:
    base = CompanionSystemPrompt.SYSTEM_PROMPT
    system = CompanionSystem(top_k=4)
    _observe_trait(system, dim=OceanDim.OPENNESS, score=5.0, layer="explicit")
    profile = TraitProfile(system=system, now=TS)
    flag = TraitConditioningFlag(enabled=True, confidence_floor=0.5)
    assert apply_trait_conditioning(base, profile, flag) == base


def test_apply_appends_addendum_when_flag_on_and_confident_dim() -> None:
    base = CompanionSystemPrompt.SYSTEM_PROMPT
    system = CompanionSystem(top_k=4)
    _populate_high_neuroticism(system)
    profile = TraitProfile(system=system, now=TS)
    flag = TraitConditioningFlag(enabled=True)
    conditioned = apply_trait_conditioning(base, profile, flag)
    assert conditioned.startswith(base)
    assert len(conditioned) > len(base)
    assert "high Neuroticism" in conditioned


# ---------------------------------------------------------------------------
# CompanionAgent wiring
# ---------------------------------------------------------------------------


class _RecordingResponder:
    """Captures (system_prompt, context, user) triples per call."""

    def __init__(self, reply: str = "ok.") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, system_prompt: str, context_block: str, user: str) -> str:
        self.calls.append((system_prompt, context_block, user))
        return self.reply


def _agent(
    system: CompanionSystem,
    responder: _RecordingResponder,
    *,
    trait_profile=None,
    flag=None,
) -> CompanionAgent:
    return CompanionAgent(
        user_id="u1",
        session_id="s1",
        companion_system=system,
        llm_responder=responder,
        trait_profile=trait_profile,
        trait_conditioning_flag=flag,
    )


def test_agent_uses_base_prompt_when_flag_absent() -> None:
    system = CompanionSystem(top_k=4)
    _populate_high_neuroticism(system)
    profile = TraitProfile(system=system, now=TS)
    responder = _RecordingResponder()
    agent = _agent(system, responder, trait_profile=profile, flag=None)
    agent.respond("hi", timestamp=TS)
    sent_prompt = responder.calls[0][0]
    assert sent_prompt == CompanionSystemPrompt.SYSTEM_PROMPT


def test_agent_uses_base_prompt_when_flag_disabled() -> None:
    system = CompanionSystem(top_k=4)
    _populate_high_neuroticism(system)
    profile = TraitProfile(system=system, now=TS)
    responder = _RecordingResponder()
    agent = _agent(
        system,
        responder,
        trait_profile=profile,
        flag=TraitConditioningFlag(enabled=False),
    )
    agent.respond("hi", timestamp=TS)
    assert responder.calls[0][0] == CompanionSystemPrompt.SYSTEM_PROMPT


def test_agent_uses_conditioned_prompt_when_flag_enabled() -> None:
    system = CompanionSystem(top_k=4)
    _populate_high_neuroticism(system)
    profile = TraitProfile(system=system, now=TS)
    responder = _RecordingResponder()
    agent = _agent(
        system,
        responder,
        trait_profile=profile,
        flag=TraitConditioningFlag(enabled=True),
    )
    agent.respond("hi", timestamp=TS)
    sent_prompt = responder.calls[0][0]
    assert sent_prompt.startswith(CompanionSystemPrompt.SYSTEM_PROMPT)
    assert "high Neuroticism" in sent_prompt


def test_agent_crisis_handoff_untouched_by_conditioning() -> None:
    """Crisis short-circuit precedes the LLM call. The conditioned

    prompt must NOT be constructed or applied — the handoff script
    is returned verbatim regardless of trait flag state. This is the
    hard-constraint boundary.
    """
    system = CompanionSystem(top_k=4)
    _populate_high_neuroticism(system)
    profile = TraitProfile(system=system, now=TS)
    responder = _RecordingResponder(reply="should not be called")
    agent = CompanionAgent(
        user_id="u1",
        session_id="s1",
        companion_system=system,
        llm_responder=responder,
        crisis_escalation=CrisisEscalation(handoff_sink=lambda _incident: None),
        trait_profile=profile,
        trait_conditioning_flag=TraitConditioningFlag(enabled=True),
    )
    resp = agent.respond("I want to kill myself", timestamp=TS)
    assert resp.handoff_triggered is True
    assert resp.response_text == HANDOFF_SCRIPT
    assert responder.calls == []  # LLM never called → conditioning never ran


def test_agent_default_fields_are_none() -> None:
    """Byte-identity on construction: a bare CompanionAgent has no

    trait profile and no conditioning flag. This is the Phase 4a/4b
    gate-regression guarantee — pre-Phase-7 callers see no change.
    """
    system = CompanionSystem(top_k=4)
    responder = _RecordingResponder()
    agent = CompanionAgent(
        user_id="u1",
        session_id="s1",
        companion_system=system,
        llm_responder=responder,
    )
    assert agent.trait_profile is None
    assert agent.trait_conditioning_flag is None
    agent.respond("hi", timestamp=TS)
    assert responder.calls[0][0] == CompanionSystemPrompt.SYSTEM_PROMPT
