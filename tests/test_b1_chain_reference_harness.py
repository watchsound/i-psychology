"""B1 scripted chain-reference harness tests.

Covers:

- :func:`has_chain_reference_cue` lexical cue vocabulary EN + ZH
- :func:`auto_label_chain_reference` four-case fold
- :func:`run_chain_reference_script` end-to-end: a scripted stream
  through CompanionAgent + stub responder → :class:`B1ChainReferenceReport`
  with the pivot-gate check
"""

from __future__ import annotations

from datetime import datetime, timezone

from psychology_base.applications.emotional import CompanionAgent
from psychology_base.applications.measurement import (
    ChainReferenceLabel,
    ReinterpretationAct,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks.b1_chain_reference_harness import (
    B1ChainReferenceReport,
    ChainReferenceScript,
    ChainReferenceScriptStep,
    auto_label_chain_reference,
    has_chain_reference_cue,
    run_chain_reference_script,
)


# ---------------------------------------------------------------------------
# Cue detector
# ---------------------------------------------------------------------------


def test_cue_fires_on_canonical_english_cue() -> None:
    assert has_chain_reference_cue(
        "You used to describe this as neglect — what shifted?"
    )


def test_cue_fires_on_what_shifted_alone() -> None:
    assert has_chain_reference_cue("What shifted for you between then and now?")


def test_cue_fires_on_chinese_cue() -> None:
    assert has_chain_reference_cue("你之前说的和今天说的不太一样，有什么变了吗？")


def test_cue_absent_on_generic_support() -> None:
    assert not has_chain_reference_cue(
        "That sounds painful. I'm here. Tell me more about today."
    )


def test_cue_absent_on_empty_string() -> None:
    assert not has_chain_reference_cue("")


def test_cue_is_case_insensitive() -> None:
    assert has_chain_reference_cue("YOU USED TO call it that way.")


# ---------------------------------------------------------------------------
# auto_label_chain_reference fold
# ---------------------------------------------------------------------------


def test_fold_reinterp_with_cue_is_appropriate() -> None:
    label = auto_label_chain_reference(cue_fired=True, is_reinterpretation=True)
    assert label is ChainReferenceLabel.APPROPRIATE_CITE


def test_fold_reinterp_without_cue_is_missed() -> None:
    label = auto_label_chain_reference(cue_fired=False, is_reinterpretation=True)
    assert label is ChainReferenceLabel.MISSED_OPPORTUNITY


def test_fold_non_reinterp_with_cue_is_inappropriate() -> None:
    label = auto_label_chain_reference(cue_fired=True, is_reinterpretation=False)
    assert label is ChainReferenceLabel.INAPPROPRIATE_CITE


def test_fold_non_reinterp_without_cue_is_na() -> None:
    label = auto_label_chain_reference(cue_fired=False, is_reinterpretation=False)
    assert label is ChainReferenceLabel.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# ChainReferenceScript.from_turns
# ---------------------------------------------------------------------------


def test_from_turns_builds_expected_steps() -> None:
    script = ChainReferenceScript.from_turns(
        user_id="user42",
        turns=[
            ("my dad was never there", False),
            ("i think he was overwhelmed", True),
            ("the weather today was fine", False),
        ],
    )
    assert script.user_id == "user42"
    assert len(script.steps) == 3
    assert script.steps[1].is_reinterpretation is True
    assert script.steps[1].event_label is None


# ---------------------------------------------------------------------------
# run_chain_reference_script — end-to-end
# ---------------------------------------------------------------------------


def _responder_sequence(replies: list[str]):
    """Build a responder that returns replies[i] on turn i."""
    counter = {"i": 0}

    def _r(system_prompt: str, context_block: str, user_utterance: str) -> str:
        reply = replies[counter["i"]]
        counter["i"] += 1
        return reply

    return _r


def _build_agent(responder) -> CompanionAgent:
    cs = CompanionSystem(top_k=4)
    return CompanionAgent(
        user_id="u0",
        session_id="s0",
        companion_system=cs,
        llm_responder=responder,
    )


def test_run_script_perfect_companion_passes_pivot_gate() -> None:
    """6 reinterpretation steps, companion cites all → 100% appropriate."""
    script = ChainReferenceScript.from_turns(
        user_id="perfect_user",
        turns=[
            ("i was waiting at the window", False),           # anchor
            ("but maybe he was overwhelmed", True),            # reinterp 1
            ("he was grieving his mother too", True),          # reinterp 2
            ("saturday pancakes were a ritual", True),         # reinterp 3
            ("the waiting was not about me", True),            # reinterp 4
            ("both can be true", True),                        # reinterp 5
            ("i can release the waiting", True),               # reinterp 6
        ],
    )
    responder = _responder_sequence(
        [
            "I hear you. That image is vivid.",
            "You used to describe this as pure absence — what shifted?",
            "That's a shift from how you first talked about him.",
            "Earlier you called it a wound; today it sounds like a ritual.",
            "You used to carry this as self-blame — what changed?",
            "You said earlier it was either-or; now it sounds like both.",
            "The way you described this months ago was heavier.",
        ]
    )
    agent = _build_agent(responder)

    result = run_chain_reference_script(agent, script)

    assert isinstance(result, B1ChainReferenceReport)
    assert result.chain_reference_report.reinterpretation_count == 6
    assert result.chain_reference_report.appropriate_cite_count == 6
    assert result.chain_reference_report.chain_reference_rate() == 1.0
    assert result.meets_pivot_gate()


def test_run_script_sycophantic_companion_fails_pivot_gate() -> None:
    """6 reinterpretation steps, companion never cites → 0% appropriate."""
    script = ChainReferenceScript.from_turns(
        user_id="sycophant_victim",
        turns=[
            ("my dad was never there", False),
            ("maybe he was overwhelmed", True),
            ("he was grieving", True),
            ("saturday mornings were different", True),
            ("the waiting was not about me", True),
            ("both can be true", True),
            ("i can release this", True),
        ],
    )
    responder = _responder_sequence(
        [
            "That sounds really hard.",
            "I'm so glad you're seeing it that way.",
            "You're doing great work.",
            "That's beautiful.",
            "Yes, you're so right.",
            "What an insight.",
            "I'm proud of you.",
        ]
    )
    agent = _build_agent(responder)

    result = run_chain_reference_script(agent, script)
    assert result.chain_reference_report.appropriate_cite_count == 0
    assert result.chain_reference_report.missed_opportunity_count == 6
    assert result.chain_reference_report.chain_reference_rate() == 0.0
    assert not result.meets_pivot_gate()


def test_run_script_inappropriate_cite_counted_separately() -> None:
    """Companion cites when the user did NOT reinterpret — INAPPROPRIATE_CITE."""
    script = ChainReferenceScript.from_turns(
        user_id="chatty",
        turns=[
            ("the weather today was rainy", False),
            ("my coffee is cold", False),
        ],
    )
    responder = _responder_sequence(
        [
            "You used to call rainy days peaceful — what shifted?",
            "Earlier you said cold coffee was fine. What changed?",
        ]
    )
    agent = _build_agent(responder)

    result = run_chain_reference_script(agent, script)
    assert result.chain_reference_report.reinterpretation_count == 0
    assert result.chain_reference_report.inappropriate_cite_count == 2
    # Zero reinterpretations → rate = 0.0 by convention; min_reinterpretations
    # guard in the pivot gate also fires.
    assert result.chain_reference_report.chain_reference_rate() == 0.0
    assert not result.meets_pivot_gate()


def test_run_script_cue_firings_align_with_responses() -> None:
    """Per-step cue firings should match whether responses contained cues."""
    script = ChainReferenceScript.from_turns(
        user_id="mixed",
        turns=[
            ("i was lonely", False),
            ("maybe i was never lonely, just unseen", True),
            ("the sun is bright today", False),
            ("i think my loneliness was a shape i gave to not being seen", True),
        ],
    )
    responder = _responder_sequence(
        [
            "I hear that.",                                              # no cue
            "You used to call this loneliness — what shifted?",          # cue
            "The sun is kind today.",                                    # no cue
            "That's a different frame than you had earlier.",            # cue
        ]
    )
    agent = _build_agent(responder)

    result = run_chain_reference_script(agent, script)
    assert result.cue_firings == (False, True, False, True)
    labels = [t.chain_reference_label for t in result.turn_log]
    assert labels == [
        ChainReferenceLabel.NOT_APPLICABLE,
        ChainReferenceLabel.APPROPRIATE_CITE,
        ChainReferenceLabel.NOT_APPLICABLE,
        ChainReferenceLabel.APPROPRIATE_CITE,
    ]


def test_run_script_mixed_cohort_hits_80pct_gate_exactly() -> None:
    """5 reinterp steps, 4 cited → 80% hits the floor exactly."""
    script = ChainReferenceScript.from_turns(
        user_id="borderline",
        turns=[
            ("initial framing", False),
            ("reframe 1", True),
            ("reframe 2", True),
            ("reframe 3", True),
            ("reframe 4", True),
            ("reframe 5", True),
        ],
    )
    responder = _responder_sequence(
        [
            "ok.",
            "What shifted for you?",
            "You used to carry this differently.",
            "Something's changed — tell me.",
            "That's a shift from last time.",
            "I hear you.",  # missed — total 4/5
        ]
    )
    agent = _build_agent(responder)
    result = run_chain_reference_script(agent, script)
    assert result.chain_reference_report.appropriate_cite_count == 4
    assert result.chain_reference_report.reinterpretation_count == 5
    assert result.chain_reference_report.chain_reference_rate() == 0.8
    assert result.meets_pivot_gate()


def test_run_script_rotates_session_on_step_boundary() -> None:
    """A step with session_id != current rotates the agent + emits a boundary."""
    cs = CompanionSystem(top_k=4)
    agent = CompanionAgent(
        user_id="u0",
        session_id="s0",
        companion_system=cs,
        llm_responder=_responder_sequence(["r1", "r2", "r3"]),
    )
    boundary_count = {"calls": 0}
    original_mark = agent.mark_session_boundary

    def counting_mark() -> None:
        boundary_count["calls"] += 1
        original_mark()

    agent.mark_session_boundary = counting_mark  # type: ignore[method-assign]

    script = ChainReferenceScript(
        user_id="u0",
        steps=(
            ChainReferenceScriptStep(user_utterance="turn1", is_reinterpretation=False),
            ChainReferenceScriptStep(
                user_utterance="turn2",
                is_reinterpretation=True,
                session_id="s1",
            ),
            ChainReferenceScriptStep(
                user_utterance="turn3",
                is_reinterpretation=True,
                session_id="s2",
            ),
        ),
    )
    run_chain_reference_script(agent, script)

    # Two session rotations → two mark_session_boundary() calls.
    assert boundary_count["calls"] == 2
    assert agent.session_id == "s2"


def test_run_script_overrides_agent_user_id() -> None:
    cs = CompanionSystem(top_k=4)
    agent = CompanionAgent(
        user_id="original_u",
        session_id="s0",
        companion_system=cs,
        llm_responder=_responder_sequence(["ok"]),
    )
    script = ChainReferenceScript.from_turns(
        user_id="script_u",
        turns=[("hi", False)],
    )
    run_chain_reference_script(agent, script)
    assert agent.user_id == "script_u"


def test_run_script_report_user_id_matches_script() -> None:
    cs = CompanionSystem(top_k=4)
    agent = CompanionAgent(
        user_id="original",
        session_id="s0",
        companion_system=cs,
        llm_responder=_responder_sequence(["ok", "ok", "ok"]),
    )
    script = ChainReferenceScript.from_turns(
        user_id="script_pinned",
        turns=[("a", False), ("b", True), ("c", True)],
    )
    result = run_chain_reference_script(agent, script)
    assert result.chain_reference_report.user_id == "script_pinned"


def test_min_reinterpretations_floor_fires_before_rate() -> None:
    """Even at 100% appropriate rate, fewer than 5 reinterps fails gate."""
    script = ChainReferenceScript.from_turns(
        user_id="too_few",
        turns=[
            ("init", False),
            ("reframe 1", True),
            ("reframe 2", True),
            ("reframe 3", True),
        ],
    )
    responder = _responder_sequence(
        [
            "ok.",
            "You used to say otherwise.",
            "What shifted?",
            "Earlier you had a different frame.",
        ]
    )
    agent = _build_agent(responder)
    result = run_chain_reference_script(agent, script)
    assert result.chain_reference_report.chain_reference_rate() == 1.0
    assert result.chain_reference_report.reinterpretation_count == 3
    assert not result.meets_pivot_gate()
