"""Phase 8 Pillar A — MSC × WAI-SR longitudinal Bond harness shape tests.

In-memory scripted seekers only. No live LLM calls — tests cover the
plumbing (session loop, cross-session memory passthrough, dual WAI-SR
administration, longitudinal aggregation + gate) so a live-mode run can
be a thin wrapper on the shaped surface.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from psychology_base.applications.emotional import CompanionAgent
from psychology_base.applications.measurement import score_wai_sr
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks.public_datasets.esconv_wai_sr_harness import (
    administer_wai_sr_simulated,
)
from psychology_base.eval.benchmarks.public_datasets.msc import (
    MSCConversation,
    MSCSession,
    MSCTurn,
    OTHER_SPEAKER_IDX,
    SELF_SPEAKER_IDX,
)
from psychology_base.eval.benchmarks.public_datasets.msc_bond_longitudinal import (
    LongitudinalBondReport,
    MSCUserRun,
    aggregate_longitudinal_bond,
    run_msc_conversation,
    run_msc_session,
)


TS = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    """Callable seeker returning scripted replies in order."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.received: list[str] = []

    def __call__(self, previous_agent_reply: str) -> str:
        self.received.append(previous_agent_reply)
        if not self._replies:
            return ""
        return self._replies.pop(0)


def _two_session_convo(opener_1: str = "Hi, how are you?", opener_5: str = "Hey — been a while.") -> MSCConversation:
    """A minimal 2-session MSC convo stand-in — session ids are 1 and 5."""
    return MSCConversation(
        conversation_id="msc_test",
        personas=(("I love dogs.",), ("I live in Kansas.",)),
        sessions=(
            MSCSession(
                session_id=1,
                time_interval_days=0.0,
                turns=(
                    MSCTurn(speaker_idx=SELF_SPEAKER_IDX, utterance=opener_1),
                    MSCTurn(speaker_idx=OTHER_SPEAKER_IDX, utterance="I'm good!"),
                ),
            ),
            MSCSession(
                session_id=5,
                time_interval_days=14.0,
                turns=(
                    MSCTurn(speaker_idx=SELF_SPEAKER_IDX, utterance=opener_5),
                    MSCTurn(speaker_idx=OTHER_SPEAKER_IDX, utterance="Hey!"),
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# run_msc_session
# ---------------------------------------------------------------------------


def test_session_runs_alternating_turns_up_to_max() -> None:
    convo = _two_session_convo()
    agent = _companion_agent()
    seeker = ScriptedSeeker(["Yeah.", "I guess.", "Okay."])
    transcript = run_msc_session(
        convo.sessions[0], agent, seeker, max_turns=3, starting_timestamp=TS
    )
    assert len(transcript.turns) == 6  # opener + 3×(companion + seeker) less tail-seeker
    assert transcript.turns[0].speaker == "seeker"
    assert transcript.turns[-1].speaker == "companion"
    assert transcript.handoff_triggered is False


def test_session_opener_sourced_from_session() -> None:
    convo = _two_session_convo(opener_1="I feel like I'm failing at everything.")
    agent = _companion_agent()
    seeker = ScriptedSeeker(["mm"])
    transcript = run_msc_session(
        convo.sessions[0], agent, seeker, max_turns=1, starting_timestamp=TS
    )
    assert transcript.turns[0].utterance == "I feel like I'm failing at everything."


def test_session_opener_override_wins() -> None:
    convo = _two_session_convo()
    agent = _companion_agent()
    seeker = ScriptedSeeker(["mm"])
    transcript = run_msc_session(
        convo.sessions[0],
        agent,
        seeker,
        max_turns=1,
        starting_timestamp=TS,
        opening_utterance="Different opener.",
    )
    assert transcript.turns[0].utterance == "Different opener."


def test_session_falls_back_to_hi_when_no_self_turn() -> None:
    session = MSCSession(
        session_id=1,
        time_interval_days=0.0,
        turns=(MSCTurn(speaker_idx=OTHER_SPEAKER_IDX, utterance="?"),),
    )
    agent = _companion_agent()
    seeker = ScriptedSeeker(["mm"])
    transcript = run_msc_session(session, agent, seeker, max_turns=1, starting_timestamp=TS)
    assert transcript.turns[0].utterance == "Hi."


def test_session_rejects_zero_max_turns() -> None:
    convo = _two_session_convo()
    agent = _companion_agent()
    seeker = ScriptedSeeker([])
    with pytest.raises(ValueError, match="max_turns"):
        run_msc_session(convo.sessions[0], agent, seeker, max_turns=0)


# ---------------------------------------------------------------------------
# run_msc_conversation — multi-session end-to-end
# ---------------------------------------------------------------------------


def test_conversation_runs_all_sessions_and_administers_wai_sr_twice() -> None:
    convo = _two_session_convo()
    agent = _companion_agent()
    # Per session at max_turns=2: 1 chatter seeker call + 5 WAI-SR calls = 6.
    # Two sessions → 12 seeker calls total; order matters (FIFO queue).
    seeker_replies = (
        ["Yeah."]  # session 1 chatter
        + ["5", "6", "4", "5", "6"]  # session 1 WAI-SR (bond item 5 = 6)
        + ["Right."]  # session 5 chatter
        + ["6", "6", "5", "5", "7"]  # session 5 WAI-SR (bond item 5 = 7)
    )
    seeker = ScriptedSeeker(seeker_replies)

    run = run_msc_conversation(
        convo,
        companion_agent=agent,
        seeker=seeker,
        max_turns_per_session=2,
        base_timestamp=TS,
        wai_sr_at_sessions=(1, 5),
    )
    assert run.conversation_id == "msc_test"
    assert len(run.per_session_transcripts) == 2
    assert set(run.wai_sr_by_session_id.keys()) == {1, 5}
    assert run.handoff_triggered is False
    s1_bond = score_wai_sr(run.wai_sr_by_session_id[1]).bond
    s5_bond = score_wai_sr(run.wai_sr_by_session_id[5]).bond
    # Bond is item 5 only: 6 → bond=6, 7 → bond=7.
    assert s1_bond == pytest.approx(6.0)
    assert s5_bond == pytest.approx(7.0)


def test_conversation_defaults_wai_sr_to_first_and_last() -> None:
    convo = _two_session_convo()
    agent = _companion_agent()
    seeker = ScriptedSeeker(
        ["Yeah."]
        + ["5", "5", "5", "5", "5"]
        + ["Yeah."]
        + ["6", "6", "6", "6", "6"]
    )
    run = run_msc_conversation(
        convo,
        companion_agent=agent,
        seeker=seeker,
        max_turns_per_session=2,
        base_timestamp=TS,
    )
    assert set(run.wai_sr_by_session_id.keys()) == {1, 5}


def test_conversation_rejects_empty_sessions_list() -> None:
    convo = MSCConversation(
        conversation_id="empty",
        personas=(("a",), ("b",)),
        sessions=(),
    )
    agent = _companion_agent()
    seeker = ScriptedSeeker([])
    with pytest.raises(ValueError, match="no sessions"):
        run_msc_conversation(
            convo, companion_agent=agent, seeker=seeker, base_timestamp=TS
        )


def test_conversation_advances_time_across_sessions() -> None:
    """Session 2's starting_timestamp is session 1's + time_interval_days.

    This is the signal that makes multi-week in-world gaps visible to the
    memory substrate's aging policy. We verify via the companion-side
    system's internal clock exposing turn timestamps.
    """
    convo = _two_session_convo()
    cs = CompanionSystem(top_k=4)
    agent = CompanionAgent(
        user_id="u1",
        session_id="s1",
        companion_system=cs,
        llm_responder=_stub_responder(),
    )
    seeker = ScriptedSeeker(
        ["Yeah."] + ["5"] * 5 + ["Yeah."] + ["6"] * 5
    )
    run_msc_conversation(
        convo,
        companion_agent=agent,
        seeker=seeker,
        max_turns_per_session=2,
        base_timestamp=TS,
    )
    # Pull all observations from the companion system — their timestamps
    # should span at least session 2's 14-day in-world gap.
    all_fragments = cs.companion.store.all()
    assert all_fragments, "Companion should have observed turns across sessions"
    timestamps = [f.timestamp for f in all_fragments]
    span_s = (max(timestamps) - min(timestamps)).total_seconds()
    assert span_s >= 14 * 86400 - 60, (
        f"Expected session span ≥ 14 days; got {span_s / 86400:.1f} days"
    )


# ---------------------------------------------------------------------------
# aggregate_longitudinal_bond
# ---------------------------------------------------------------------------


def _rs_with_bond(bond_score: int, user_id: str = "u"):
    """Build a response set whose Bond subscale mean is ``bond_score``.

    Bond is item 5 only in the chatbot-adapted 5-item WAI-SR.
    """
    seeker = ScriptedSeeker(["4", "4", "4", "4", str(bond_score)])
    return administer_wai_sr_simulated(
        seeker, user_id=user_id, administered_at_seconds=TS.timestamp()
    )


def _user_run(
    bond_s1: int | None,
    bond_s5: int | None,
    user_id: str = "u",
    handoff: bool = False,
) -> MSCUserRun:
    wai_sr = {}
    if bond_s1 is not None:
        wai_sr[1] = _rs_with_bond(bond_s1, user_id)
    if bond_s5 is not None:
        wai_sr[5] = _rs_with_bond(bond_s5, user_id)
    return MSCUserRun(
        conversation_id=user_id,
        per_session_transcripts=(),
        wai_sr_by_session_id=wai_sr,
        handoff_triggered=handoff,
    )


def test_aggregation_scores_session_1_and_session_5_cohorts() -> None:
    runs = [
        _user_run(bond_s1=5, bond_s5=6, user_id=f"u{i}") for i in range(3)
    ]
    report = aggregate_longitudinal_bond(runs)
    assert report.n_session_1 == 3
    assert report.n_session_5 == 3
    assert report.n_paired == 3
    assert report.mean_session_1_bond == pytest.approx(5.0)
    assert report.mean_session_5_bond == pytest.approx(6.0)
    assert report.mean_paired_delta == pytest.approx(1.0)


def test_aggregation_handles_missing_session_5() -> None:
    """A user who hits handoff mid-chain contributes to session_1 only."""
    runs = [
        _user_run(bond_s1=5, bond_s5=6, user_id="u1"),
        _user_run(bond_s1=4, bond_s5=None, user_id="u2", handoff=True),
    ]
    report = aggregate_longitudinal_bond(runs)
    assert report.n_session_1 == 2
    assert report.n_session_5 == 1
    assert report.n_paired == 1
    assert report.excluded_users == 1
    assert report.excluded_handoffs == 1


def test_aggregation_persona_counts_track_distribution() -> None:
    runs = [_user_run(bond_s1=5, bond_s5=6, user_id=f"u{i}") for i in range(3)]
    report = aggregate_longitudinal_bond(
        runs, persona_labels=["teacher", "teacher", "collector"]
    )
    assert report.persona_counts == {"teacher": 2, "collector": 1}


def test_aggregation_mismatched_persona_labels_raises() -> None:
    runs = [_user_run(bond_s1=5, bond_s5=6)]
    with pytest.raises(ValueError, match="persona_labels"):
        aggregate_longitudinal_bond(runs, persona_labels=["a", "b"])


def test_aggregation_empty_user_runs_is_benign() -> None:
    report = aggregate_longitudinal_bond([])
    assert report.n_session_1 == 0
    assert report.n_session_5 == 0
    assert report.mean_session_5_bond == 0.0
    assert report.meets_pillar_gate() is False


# ---------------------------------------------------------------------------
# Pillar A gate
# ---------------------------------------------------------------------------


def test_pillar_gate_requires_n_and_session_5_floor() -> None:
    # 50 users all with session-5 Bond=5 → passes defaults (3.89, 50).
    runs = [_user_run(bond_s1=4, bond_s5=5, user_id=f"u{i}") for i in range(50)]
    report = aggregate_longitudinal_bond(runs)
    assert report.n_session_5 == 50
    assert report.mean_session_5_bond == pytest.approx(5.0)
    assert report.meets_pillar_gate() is True
    # Raise N floor → fails on denominator.
    assert report.meets_pillar_gate(min_users=51) is False
    # Raise Bond floor → fails on mean.
    assert report.meets_pillar_gate(bond_floor=5.5) is False


def test_pillar_gate_fails_below_woebot_floor() -> None:
    runs = [_user_run(bond_s1=3, bond_s5=3, user_id=f"u{i}") for i in range(50)]
    report = aggregate_longitudinal_bond(runs)
    assert report.mean_session_5_bond == pytest.approx(3.0)
    assert report.meets_pillar_gate() is False


def test_pillar_gate_rejects_small_cohort_even_if_mean_passes() -> None:
    runs = [_user_run(bond_s1=5, bond_s5=6, user_id=f"u{i}") for i in range(10)]
    report = aggregate_longitudinal_bond(runs)
    assert report.mean_session_5_bond == pytest.approx(6.0)
    assert report.meets_pillar_gate() is False
    assert report.n_session_5 == 10


def test_session_5_ci95_is_zero_for_single_point() -> None:
    runs = [_user_run(bond_s1=5, bond_s5=6)]
    report = aggregate_longitudinal_bond(runs)
    assert report.session_5_stderr == 0.0
    assert report.session_5_ci95_half_width == 0.0


def test_session_5_ci95_grows_with_variance() -> None:
    # Spread: bonds {3, 7} → more variance than {5, 5}.
    runs_spread = [_user_run(bond_s1=5, bond_s5=3, user_id="u1"), _user_run(bond_s1=5, bond_s5=7, user_id="u2")]
    runs_flat = [_user_run(bond_s1=5, bond_s5=5, user_id="u1"), _user_run(bond_s1=5, bond_s5=5, user_id="u2")]
    assert (
        aggregate_longitudinal_bond(runs_spread).session_5_ci95_half_width
        > aggregate_longitudinal_bond(runs_flat).session_5_ci95_half_width
    )
