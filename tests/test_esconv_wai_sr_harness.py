"""Phase 4b benchmark pivot — ESConv × WAI-SR harness shape tests.

In-memory scripted seekers only. No live LLM calls — tests cover the
plumbing (session loop, Likert parse, cohort aggregation + gate) so a
live-mode run can be a thin wrapper on the shaped surface.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from psychology_base.applications.emotional import CompanionAgent
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks.public_datasets.esconv import (
    ESConvConversation,
    ESConvTurn,
)
from psychology_base.eval.benchmarks.public_datasets.esconv_wai_sr_harness import (
    CohortBondReport,
    WAISRParseError,
    administer_wai_sr_simulated,
    aggregate_cohort_bond,
    run_esconv_session,
)
from psychology_base.applications.measurement import score_wai_sr


TS = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)


def _convo(cid: str = "esc_t", seeker_opener: str = "I feel lost.") -> ESConvConversation:
    return ESConvConversation(
        conversation_id=cid,
        emotion_type="anxiety",
        problem_type="job crisis",
        situation="I lost my job last week and I don't know what to do.",
        turns=(ESConvTurn(role="seeker", utterance=seeker_opener),),
    )


def _stub_responder(reply: str = "I hear you — tell me more about that."):
    def _r(system_prompt: str, context_block: str, user_utterance: str) -> str:
        return reply
    return _r


def _companion_agent() -> CompanionAgent:
    cs = CompanionSystem(top_k=4)
    return CompanionAgent(
        user_id="u1",
        session_id="s1",
        companion_system=cs,
        llm_responder=_stub_responder(),
    )


class ScriptedSeeker:
    """Callable seeker returning scripted replies, in order, per invocation."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.received: list[str] = []

    def __call__(self, previous_agent_reply: str) -> str:
        self.received.append(previous_agent_reply)
        if not self._replies:
            return ""
        return self._replies.pop(0)


# ---------------------------------------------------------------------------
# run_esconv_session
# ---------------------------------------------------------------------------


def test_session_runs_alternating_turns_up_to_max() -> None:
    convo = _convo()
    agent = _companion_agent()
    seeker = ScriptedSeeker(
        ["It's really overwhelming.", "I don't know where to begin.", "That helps a bit."]
    )
    transcript = run_esconv_session(
        convo, agent, seeker, max_turns=3, starting_timestamp=TS
    )
    # Alternating: seeker-opener, companion, seeker, companion, seeker, companion
    assert len(transcript.turns) == 6
    assert transcript.turns[0].speaker == "seeker"
    assert transcript.turns[1].speaker == "companion"
    assert transcript.turns[-1].speaker == "companion"
    assert transcript.handoff_triggered is False
    assert transcript.conversation_id == "esc_t"


def test_session_opener_sourced_from_convo() -> None:
    convo = _convo(seeker_opener="I feel like I'm failing at everything.")
    agent = _companion_agent()
    seeker = ScriptedSeeker(["mm"])
    transcript = run_esconv_session(convo, agent, seeker, max_turns=1, starting_timestamp=TS)
    assert transcript.turns[0].utterance == "I feel like I'm failing at everything."


def test_session_falls_back_to_situation_when_no_seeker_turn() -> None:
    convo = ESConvConversation(
        conversation_id="esc_no_opener",
        emotion_type="anxiety",
        problem_type="x",
        situation="I keep waking up at 3am.",
        turns=(),  # no seeker turn
    )
    agent = _companion_agent()
    seeker = ScriptedSeeker(["mm"])
    transcript = run_esconv_session(convo, agent, seeker, max_turns=1, starting_timestamp=TS)
    assert transcript.turns[0].utterance == "I keep waking up at 3am."


def test_session_breaks_early_when_seeker_falls_silent() -> None:
    convo = _convo()
    agent = _companion_agent()
    seeker = ScriptedSeeker([""])  # seeker emits empty on first follow-up
    transcript = run_esconv_session(convo, agent, seeker, max_turns=4, starting_timestamp=TS)
    # Opener + first companion turn only; seeker returned "" so we exit.
    assert len(transcript.turns) == 2
    assert transcript.turns[0].speaker == "seeker"
    assert transcript.turns[1].speaker == "companion"


def test_session_rejects_zero_max_turns() -> None:
    convo = _convo()
    agent = _companion_agent()
    seeker = ScriptedSeeker([])
    with pytest.raises(ValueError, match="max_turns"):
        run_esconv_session(convo, agent, seeker, max_turns=0)


# ---------------------------------------------------------------------------
# administer_wai_sr_simulated
# ---------------------------------------------------------------------------


def test_administer_parses_bare_integer_replies() -> None:
    seeker = ScriptedSeeker(["5", "6", "4", "5", "6"])
    rs = administer_wai_sr_simulated(
        seeker,
        user_id="u1",
        administered_at_seconds=TS.timestamp(),
    )
    assert len(rs.responses) == 5
    assert rs.source == "llm_simulated"
    assert rs.subject_noun == "Companion"
    assert [r.score for r in rs.responses] == [5, 6, 4, 5, 6]


def test_administer_extracts_integer_from_prose_reply() -> None:
    seeker = ScriptedSeeker(
        [
            "I'd say about a 6, because they listen.",
            "Maybe 5 — not sure.",
            "4 — somewhere in the middle.",
            "I think 7.",
            "Probably 6.",
        ]
    )
    rs = administer_wai_sr_simulated(
        seeker, user_id="u1", administered_at_seconds=TS.timestamp()
    )
    assert [r.score for r in rs.responses] == [6, 5, 4, 7, 6]


def test_administer_raises_when_reply_has_no_valid_likert() -> None:
    seeker = ScriptedSeeker(["I'm not sure how to answer that."])
    with pytest.raises(WAISRParseError):
        administer_wai_sr_simulated(
            seeker, user_id="u1", administered_at_seconds=TS.timestamp()
        )


def test_administer_rejects_out_of_range_integer() -> None:
    # "15" is out of [1,7] and no in-range integer appears → parse fails
    seeker = ScriptedSeeker(["15"])
    with pytest.raises(WAISRParseError):
        administer_wai_sr_simulated(
            seeker, user_id="u1", administered_at_seconds=TS.timestamp()
        )


def test_administer_subject_noun_passes_through() -> None:
    seeker = ScriptedSeeker(["6"] * 5)
    rs = administer_wai_sr_simulated(
        seeker,
        user_id="u1",
        administered_at_seconds=TS.timestamp(),
        subject_noun="AI",
    )
    assert rs.subject_noun == "AI"


def test_administer_default_prompt_has_no_scope_hint() -> None:
    seeker = ScriptedSeeker(["5"] * 5)
    administer_wai_sr_simulated(
        seeker,
        user_id="u1",
        administered_at_seconds=TS.timestamp(),
    )
    assert len(seeker.received) == 5
    for prompt in seeker.received:
        assert prompt.startswith("Please rate this statement on a scale")


def test_administer_prepends_scope_hint_when_provided() -> None:
    seeker = ScriptedSeeker(["5"] * 5)
    hint = "Only rate THIS session."
    administer_wai_sr_simulated(
        seeker,
        user_id="u1",
        administered_at_seconds=TS.timestamp(),
        scope_hint=hint,
    )
    assert len(seeker.received) == 5
    for prompt in seeker.received:
        assert prompt.startswith(hint)
        assert "Please rate this statement on a scale" in prompt


def test_administer_empty_scope_hint_falls_back_to_default_prompt() -> None:
    seeker = ScriptedSeeker(["5"] * 5)
    administer_wai_sr_simulated(
        seeker,
        user_id="u1",
        administered_at_seconds=TS.timestamp(),
        scope_hint="",
    )
    for prompt in seeker.received:
        assert prompt.startswith("Please rate this statement on a scale")


# ---------------------------------------------------------------------------
# aggregate_cohort_bond
# ---------------------------------------------------------------------------


def _rs_with_bond(bond_score: int, user_id: str = "u") -> "object":
    """Build a response set whose Bond subscale mean is ``bond_score``."""
    # Bond is item 5 only — set items 1–4 to any valid score, item 5 to target.
    seeker = ScriptedSeeker(["4", "4", "4", "4", str(bond_score)])
    return administer_wai_sr_simulated(
        seeker, user_id=user_id, administered_at_seconds=TS.timestamp()
    )


def test_cohort_aggregation_scores_bond_per_user() -> None:
    rsets = [_rs_with_bond(b, user_id=f"u{i}") for i, b in enumerate([5, 6, 4])]
    report = aggregate_cohort_bond(rsets)
    assert report.n == 3
    assert report.mean_bond == pytest.approx(5.0)
    assert report.ci95_half_width > 0.0


def test_cohort_aggregation_counts_emotion_types() -> None:
    rsets = [_rs_with_bond(5, user_id=f"u{i}") for i in range(4)]
    report = aggregate_cohort_bond(
        rsets, emotion_types=["anxiety", "anxiety", "sadness", "anger"]
    )
    assert report.emotion_type_counts == {"anxiety": 2, "sadness": 1, "anger": 1}


def test_cohort_aggregation_mismatched_emotion_types_raises() -> None:
    rsets = [_rs_with_bond(5)]
    with pytest.raises(ValueError, match="emotion_types length"):
        aggregate_cohort_bond(rsets, emotion_types=["anxiety", "sadness"])


def test_cohort_aggregation_empty_cohort_is_benign() -> None:
    report = aggregate_cohort_bond([])
    assert report.n == 0
    assert report.mean_bond == 0.0
    assert report.stderr == 0.0
    assert report.meets_pivot_gate() is False


# ---------------------------------------------------------------------------
# Pivot gate
# ---------------------------------------------------------------------------


def test_meets_pivot_gate_requires_n_and_floor() -> None:
    # 50 users all Bond=4 → passes defaults (3.8, 50).
    rsets = [_rs_with_bond(4, user_id=f"u{i}") for i in range(50)]
    report = aggregate_cohort_bond(rsets)
    assert report.n == 50
    assert report.mean_bond == pytest.approx(4.0)
    assert report.meets_pivot_gate() is True
    # Raise N floor → fails on denominator.
    assert report.meets_pivot_gate(min_users=51) is False
    # Raise Bond floor → fails on mean.
    assert report.meets_pivot_gate(bond_floor=4.5) is False


def test_meets_pivot_gate_fails_below_bond_floor() -> None:
    rsets = [_rs_with_bond(3, user_id=f"u{i}") for i in range(50)]
    report = aggregate_cohort_bond(rsets)
    assert report.mean_bond == pytest.approx(3.0)
    assert report.meets_pivot_gate() is False


def test_cohort_report_tracks_handoff_exclusions() -> None:
    rsets = [_rs_with_bond(5, user_id=f"u{i}") for i in range(3)]
    report = aggregate_cohort_bond(rsets, excluded_handoffs=2)
    assert report.excluded_handoffs == 2
    assert report.n == 3  # excluded users don't inflate denominator


# ---------------------------------------------------------------------------
# score_wai_sr integration (sanity — Bond subscale math unchanged)
# ---------------------------------------------------------------------------


def test_wai_sr_score_computation_matches_direct_scoring() -> None:
    rs = _rs_with_bond(6)
    scores = score_wai_sr(rs)
    assert scores.bond == pytest.approx(6.0)
    assert scores.goal == pytest.approx(4.0)
    assert scores.task == pytest.approx(4.0)
