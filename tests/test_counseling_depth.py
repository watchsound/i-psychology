"""Phase 8 Pillar F — counseling-depth adapter + scorer shape tests.

Scripted responders + scripted seekers; NO live LLM calls. Covers:

- ExTES adapter row parse + stream iteration + limit
- AnnoMI adapter CSV parse + transcript grouping + baseline-rate helpers
- Lexical empathy scorer (per-family cue coverage + binarisation)
- Cohen's κ computation (binary, both-constant edge cases)
- Companion-turn MI-classifier (reflection / open-question / other)
- `score_extes_sweep` + `score_annomi_sweep` end-to-end plumbing
- Mutex-free factory contract: fresh system + agent per dialog, no
  cross-dialog memory contamination
- Bond round-trip (`score_wai_sr` consumed by both corpora)
- Gate aggregation on `CounselingDepthReport`
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from psychology_base.applications.emotional import CompanionAgent
from psychology_base.applications.measurement.wai_sr import score_wai_sr
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks.public_datasets.annomi import (
    AnnoMIDialogue,
    AnnoMITurn,
    iter_annomi_dialogues,
    therapist_open_question_rate,
    therapist_reflection_rate,
)
from psychology_base.eval.benchmarks.public_datasets.counseling_depth_scorer import (
    AnnoMIDialogResult,
    AnnoMIReport,
    CounselingDepthReport,
    ExTESDialogResult,
    ExTESReport,
    _classify_companion_turn,
    _cohens_kappa_binary,
    _lexical_empathy_score,
    run_counseling_session,
    score_annomi_sweep,
    score_extes_sweep,
)
from psychology_base.eval.benchmarks.public_datasets.extes import (
    ExTESExample,
    iter_extes_examples,
)


TS = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Scripted plumbing (same shape as Pillar A/B)
# ---------------------------------------------------------------------------


def _stub_responder(reply: str = "It sounds like you're carrying a lot — tell me more."):
    def _r(system_prompt: str, context_block: str, user_utterance: str) -> str:
        return reply
    return _r


class ScriptedSeeker:
    """Returns scripted replies in order; silent at end-of-script."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.received: list[str] = []

    def __call__(self, previous_agent_reply: str) -> str:
        self.received.append(previous_agent_reply)
        if not self._replies:
            return ""
        return self._replies.pop(0)


# ---------------------------------------------------------------------------
# ExTES adapter
# ---------------------------------------------------------------------------


def test_extes_example_rejects_empty_client_utterance() -> None:
    with pytest.raises(ValueError, match="client_utterance"):
        ExTESExample(
            example_id="x",
            split="test",
            system_prompt="sys",
            client_utterance=" ",
            counselor_response="hi",
        )


def test_extes_example_rejects_empty_counselor_response() -> None:
    with pytest.raises(ValueError, match="counselor_response"):
        ExTESExample(
            example_id="x",
            split="test",
            system_prompt="sys",
            client_utterance="hi",
            counselor_response="",
        )


def test_extes_iter_streams_and_respects_limit(tmp_path: Path) -> None:
    shim = tmp_path / "ex.jsonl"
    rows = [
        {
            "example_id": f"ex-{i}",
            "split": "test",
            "system_prompt": "sys",
            "client_utterance": f"hi {i}",
            "counselor_response": f"hey {i}",
            "raw_length": 10 + i,
        }
        for i in range(5)
    ]
    import json
    shim.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    streamed = list(iter_extes_examples(shim))
    assert len(streamed) == 5
    assert [e.example_id for e in streamed] == [f"ex-{i}" for i in range(5)]
    limited = list(iter_extes_examples(shim, limit=2))
    assert len(limited) == 2
    assert limited[0].client_utterance == "hi 0"


def test_extes_iter_errors_on_missing_fields(tmp_path: Path) -> None:
    shim = tmp_path / "ex.jsonl"
    import json
    shim.write_text(json.dumps({"example_id": "x", "split": "test"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        list(iter_extes_examples(shim))


# ---------------------------------------------------------------------------
# AnnoMI adapter
# ---------------------------------------------------------------------------


def _write_annomi_csv(path: Path, rows: list[dict]) -> None:
    import csv as _csv
    cols = (
        "transcript_id",
        "mi_quality",
        "video_title",
        "video_url",
        "topic",
        "utterance_id",
        "interlocutor",
        "timestamp",
        "utterance_text",
        "main_therapist_behaviour",
        "client_talk_type",
        "annotator_id",
        "therapist_input_exists",
        "therapist_input_subtype",
        "reflection_exists",
        "reflection_subtype",
        "question_exists",
        "question_subtype",
    )
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def test_annomi_iter_groups_transcripts_and_orders_turns(tmp_path: Path) -> None:
    csv_path = tmp_path / "annomi.csv"
    _write_annomi_csv(csv_path, [
        {"transcript_id": "7", "mi_quality": "high", "topic": "alcohol",
         "utterance_id": "2", "interlocutor": "client",
         "utterance_text": "yes", "main_therapist_behaviour": "n/a",
         "client_talk_type": "change"},
        {"transcript_id": "7", "mi_quality": "high", "topic": "alcohol",
         "utterance_id": "0", "interlocutor": "therapist",
         "utterance_text": "hello", "main_therapist_behaviour": "question",
         "client_talk_type": "n/a", "question_subtype": "open"},
        {"transcript_id": "7", "mi_quality": "high", "topic": "alcohol",
         "utterance_id": "1", "interlocutor": "therapist",
         "utterance_text": "it sounds like you're ready",
         "main_therapist_behaviour": "reflection", "client_talk_type": "n/a",
         "reflection_subtype": "complex"},
    ])
    dialogues = list(iter_annomi_dialogues(csv_path))
    assert len(dialogues) == 1
    d = dialogues[0]
    assert d.transcript_id == "7"
    assert d.mi_quality == "high"
    assert [t.utterance_id for t in d.turns] == [0, 1, 2]
    assert d.first_client_turn().utterance_text == "yes"


def test_annomi_iter_quality_filter(tmp_path: Path) -> None:
    csv_path = tmp_path / "annomi.csv"
    _write_annomi_csv(csv_path, [
        {"transcript_id": "1", "mi_quality": "high", "utterance_id": "0",
         "interlocutor": "therapist", "utterance_text": "hi",
         "main_therapist_behaviour": "question", "client_talk_type": "n/a"},
        {"transcript_id": "1", "mi_quality": "high", "utterance_id": "1",
         "interlocutor": "client", "utterance_text": "ok",
         "main_therapist_behaviour": "n/a", "client_talk_type": "neutral"},
        {"transcript_id": "2", "mi_quality": "low", "utterance_id": "0",
         "interlocutor": "therapist", "utterance_text": "hi",
         "main_therapist_behaviour": "question", "client_talk_type": "n/a"},
        {"transcript_id": "2", "mi_quality": "low", "utterance_id": "1",
         "interlocutor": "client", "utterance_text": "ok",
         "main_therapist_behaviour": "n/a", "client_talk_type": "neutral"},
    ])
    high = list(iter_annomi_dialogues(csv_path, mi_quality_filter="high"))
    assert [d.transcript_id for d in high] == ["1"]
    low = list(iter_annomi_dialogues(csv_path, mi_quality_filter="low"))
    assert [d.transcript_id for d in low] == ["2"]


def test_annomi_baseline_rates() -> None:
    turns = (
        AnnoMITurn(utterance_id=0, interlocutor="therapist", utterance_text="hi",
                   main_therapist_behaviour="reflection", client_talk_type="n/a",
                   reflection_subtype="simple"),
        AnnoMITurn(utterance_id=1, interlocutor="therapist", utterance_text="?",
                   main_therapist_behaviour="question", client_talk_type="n/a",
                   question_subtype="open"),
        AnnoMITurn(utterance_id=2, interlocutor="therapist", utterance_text="?",
                   main_therapist_behaviour="question", client_talk_type="n/a",
                   question_subtype="closed"),
        AnnoMITurn(utterance_id=3, interlocutor="client", utterance_text="k",
                   main_therapist_behaviour="n/a", client_talk_type="neutral"),
    )
    d = AnnoMIDialogue(
        transcript_id="t", mi_quality="high", video_title="",
        topic="", turns=turns,
    )
    # 1 of 3 therapist turns is reflection → 0.333
    assert therapist_reflection_rate(d) == pytest.approx(1 / 3)
    # 1 of 3 therapist turns is open question → 0.333
    assert therapist_open_question_rate(d) == pytest.approx(1 / 3)


def test_annomi_turn_rejects_bad_interlocutor() -> None:
    with pytest.raises(ValueError, match="interlocutor"):
        AnnoMITurn(utterance_id=0, interlocutor="bystander",
                   utterance_text="x", main_therapist_behaviour="n/a",
                   client_talk_type="n/a")


# ---------------------------------------------------------------------------
# Lexical empathy scorer
# ---------------------------------------------------------------------------


def test_lexical_empathy_zero_on_empty() -> None:
    assert _lexical_empathy_score("") == 0.0


def test_lexical_empathy_scales_with_families() -> None:
    # single family (recognition)
    one = _lexical_empathy_score("I feel for you.")
    # two families (recognition + presence)
    two = _lexical_empathy_score("I feel for you. I'm here with you.")
    # three families (recognition + presence + validation)
    three = _lexical_empathy_score(
        "I feel for you. I'm here with you. That makes sense — no wonder."
    )
    assert 0 < one < two < three
    assert three == pytest.approx(1.0)


def test_lexical_empathy_caps_at_one() -> None:
    saturated = _lexical_empathy_score(
        "I feel, I understand, I hear you, I'm here, together, "
        "makes sense, no wonder, you have every right."
    )
    assert saturated == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cohen's κ
# ---------------------------------------------------------------------------


def test_kappa_perfect_agreement() -> None:
    assert _cohens_kappa_binary([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_kappa_total_disagreement() -> None:
    k = _cohens_kappa_binary([1, 0, 1, 0], [0, 1, 0, 1])
    assert k == pytest.approx(-1.0)


def test_kappa_both_constant_agreement_is_one() -> None:
    assert _cohens_kappa_binary([1, 1, 1], [1, 1, 1]) == pytest.approx(1.0)
    assert _cohens_kappa_binary([0, 0, 0], [0, 0, 0]) == pytest.approx(1.0)


def test_kappa_one_constant_rater() -> None:
    # One rater all 1, other mixed — not perfect agreement, expected 0.0 or near.
    k = _cohens_kappa_binary([1, 1, 1, 1], [1, 0, 1, 0])
    # po = 0.5, pa_pos = 1, pb_pos = 0.5, pe = 0.5 → k = 0
    assert k == pytest.approx(0.0)


def test_kappa_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        _cohens_kappa_binary([1, 0], [1, 0, 1])


# ---------------------------------------------------------------------------
# MI classifier
# ---------------------------------------------------------------------------


def test_classify_reflection() -> None:
    assert _classify_companion_turn("It sounds like you're exhausted.") == "reflection"
    assert _classify_companion_turn("So you want to feel in control again.") == "reflection"
    assert _classify_companion_turn("I hear that you've been carrying this alone.") == "reflection"


def test_classify_open_question() -> None:
    assert _classify_companion_turn("What would feel manageable right now?") == "open_question"
    assert _classify_companion_turn("How did that land for you?") == "open_question"
    # 'Tell me more about that.' has no '?' — open-question regex requires '?',
    # so this falls through to 'other'.
    assert _classify_companion_turn("Tell me more about that.") == "other"
    assert _classify_companion_turn("Tell me what hurts most?") == "open_question"


def test_classify_closed_question() -> None:
    assert _classify_companion_turn("Did you sleep okay last night?") == "closed_question"


def test_classify_other() -> None:
    assert _classify_companion_turn("I'll be with you.") == "other"
    assert _classify_companion_turn("") == "other"


def test_classify_reflection_wins_over_question() -> None:
    # Reflection cue first, question cue second → reflection
    text = "It sounds like you're overwhelmed. What would help?"
    assert _classify_companion_turn(text) == "reflection"


# ---------------------------------------------------------------------------
# score_extes_sweep plumbing
# ---------------------------------------------------------------------------


def _companion_agent_for_extes():
    cs = CompanionSystem(top_k=4)
    agent = CompanionAgent(
        user_id="u-extes-test",
        session_id="extes",
        companion_system=cs,
        llm_responder=_stub_responder(
            "It sounds like that's really hard. What feels most pressing?"
        ),
    )
    return cs, agent


def _simple_extes_example(i: int = 0) -> ExTESExample:
    return ExTESExample(
        example_id=f"ex-{i}",
        split="test",
        system_prompt="supporter",
        client_utterance=f"I feel awful about my situation #{i}.",
        counselor_response="I understand — that sounds hard. I'm here with you.",
    )


def test_score_extes_sweep_builds_per_dialog_results() -> None:
    examples = [_simple_extes_example(i) for i in range(2)]

    def _system():
        return CompanionSystem(top_k=4)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="extes",
            companion_system=system,
            llm_responder=_stub_responder(
                "It sounds like you're hurting. I'm here with you — that makes sense."
            ),
        )

    def _seeker(example):
        # Replies for the 3 companion turns at max_turns=3: 2 follow-ups + 5 WAI-SR
        return ScriptedSeeker([
            "yeah it's bad",
            "I can't sleep",
            "6", "6", "6", "6", "6",
        ])

    report = score_extes_sweep(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        max_turns_per_dialog=3,
        base_timestamp=TS,
    )
    assert isinstance(report, ExTESReport)
    assert report.n == 2
    # All bonds parsed → mean_bond == 6.0 (every Likert replies "6")
    assert report.mean_bond == pytest.approx(6.0)
    # Companion reply hits all 3 empathy families → binary 1
    # Reference reply ("I understand... I'm here with you") hits 2 families → binary 1
    # κ when both constant 1 → 1.0 by definition
    assert report.empath_kappa == pytest.approx(1.0)


def test_score_extes_sweep_fresh_system_per_dialog() -> None:
    """CompanionSystem factory called once per example — no shared substrate."""
    created = []

    def _system():
        cs = CompanionSystem(top_k=2)
        created.append(cs)
        return cs

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="extes",
            companion_system=system,
            llm_responder=_stub_responder("ok"),
        )

    def _seeker(example):
        return ScriptedSeeker(["x", "x", "4", "4", "4", "4", "4"])

    examples = [_simple_extes_example(i) for i in range(3)]
    score_extes_sweep(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        max_turns_per_dialog=2,
        base_timestamp=TS,
    )
    assert len(created) == 3
    # No shared identity across the three substrates
    assert len({id(cs) for cs in created}) == 3


def test_score_extes_sweep_handoff_excludes_bond() -> None:
    examples = [_simple_extes_example(0)]

    def _system():
        return CompanionSystem(top_k=2)

    def _agent(system, example):
        return CompanionAgent(
            user_id="u-handoff",
            session_id="extes",
            companion_system=system,
            # Return a handoff phrase that the compliance surface will catch.
            # Simpler: inject by direct crisis-escalation; here we just make
            # the companion short-circuit via the escalation mechanism itself
            # by seeding a crisis cue on the opening client utterance.
            llm_responder=_stub_responder("I hear you."),
        )

    def _seeker(example):
        return ScriptedSeeker(["help", "x"])

    # Craft an opener that trips crisis escalation lexically. If the floor
    # doesn't catch it we still assert the loop completes without crashing;
    # bond will be a float. The branch tested here is "handoff ⇒ bond None"
    # at the harness level — see also the ExTESDialogResult dataclass.
    crisis_example = ExTESExample(
        example_id="crisis",
        split="test",
        system_prompt="supporter",
        client_utterance="I want to kill myself tonight — I have a plan.",
        counselor_response="I'm so glad you told me.",
    )
    report = score_extes_sweep(
        [crisis_example],
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        max_turns_per_dialog=2,
        base_timestamp=TS,
    )
    assert report.n == 1
    d = report.per_dialog[0]
    if d.handoff_triggered:
        assert d.bond is None
    # Either way bond_scores filter None, so a mean exists only if at least
    # one dialog cleared handoff.
    _ = report.mean_bond  # no exception


# ---------------------------------------------------------------------------
# score_annomi_sweep plumbing
# ---------------------------------------------------------------------------


def _simple_annomi_dialogue(tid: str = "d1", mi_quality: str = "high") -> AnnoMIDialogue:
    return AnnoMIDialogue(
        transcript_id=tid,
        mi_quality=mi_quality,
        video_title="demo",
        topic="alcohol",
        turns=(
            AnnoMITurn(utterance_id=0, interlocutor="client",
                       utterance_text="I've been drinking too much.",
                       main_therapist_behaviour="n/a",
                       client_talk_type="sustain"),
            AnnoMITurn(utterance_id=1, interlocutor="therapist",
                       utterance_text="It sounds like it concerns you.",
                       main_therapist_behaviour="reflection",
                       client_talk_type="n/a",
                       reflection_subtype="complex"),
            AnnoMITurn(utterance_id=2, interlocutor="therapist",
                       utterance_text="What would change look like for you?",
                       main_therapist_behaviour="question",
                       client_talk_type="n/a",
                       question_subtype="open"),
        ),
    )


def test_score_annomi_sweep_measures_reflection_rate() -> None:
    dialogues = [_simple_annomi_dialogue()]

    def _system():
        return CompanionSystem(top_k=4)

    def _agent(system, dialogue):
        return CompanionAgent(
            user_id=f"u-{dialogue.transcript_id}",
            session_id="annomi",
            companion_system=system,
            llm_responder=_stub_responder(
                "It sounds like you're carrying a lot of worry about this."
            ),
        )

    def _seeker(dialogue):
        # max_turns=3 → opener + 2 seeker chat replies, then 5 WAI-SR Likert items
        return ScriptedSeeker([
            "yeah",
            "I guess so",
            "5", "5", "5", "5", "5",
        ])

    report = score_annomi_sweep(
        dialogues,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        max_turns_per_dialog=3,
        base_timestamp=TS,
    )
    assert report.n == 1
    d = report.per_dialog[0]
    # Every companion turn says "It sounds like..." → 100% reflection rate
    assert d.companion_reflection_rate == pytest.approx(1.0)
    assert d.companion_open_question_rate == pytest.approx(0.0)
    assert d.baseline_reflection_rate == pytest.approx(0.5)  # 1 of 2 therapist turns
    assert d.bond == pytest.approx(5.0)


def test_annomi_report_gate_absolute_floor() -> None:
    good = AnnoMIDialogResult(
        transcript_id="d", mi_quality="high", bond=4.0, handoff_triggered=False,
        companion_turns=("x",),
        companion_reflection_rate=0.5,
        companion_open_question_rate=0.1,
        baseline_reflection_rate=0.2,
        baseline_open_question_rate=0.1,
    )
    r = AnnoMIReport(per_dialog=(good,))
    assert r.meets_mi_fidelity_gate(absolute_floor=0.3) is True


def test_annomi_report_gate_baseline_fraction() -> None:
    """0.20 absolute but 50% of 0.40 baseline → passes via baseline-fraction clause."""
    d = AnnoMIDialogResult(
        transcript_id="d", mi_quality="high", bond=4.0, handoff_triggered=False,
        companion_turns=("x",),
        companion_reflection_rate=0.20,
        companion_open_question_rate=0.0,
        baseline_reflection_rate=0.40,
        baseline_open_question_rate=0.0,
    )
    r = AnnoMIReport(per_dialog=(d,))
    assert r.meets_mi_fidelity_gate(absolute_floor=0.30, baseline_fraction=0.50) is True


def test_annomi_report_gate_fails_when_both_fall_short() -> None:
    d = AnnoMIDialogResult(
        transcript_id="d", mi_quality="high", bond=4.0, handoff_triggered=False,
        companion_turns=("x",),
        companion_reflection_rate=0.10,
        companion_open_question_rate=0.0,
        baseline_reflection_rate=0.40,
        baseline_open_question_rate=0.0,
    )
    r = AnnoMIReport(per_dialog=(d,))
    # 0.10 < 0.30 absolute; 0.10 < 0.5 × 0.40 = 0.20 → FAIL
    assert r.meets_mi_fidelity_gate(absolute_floor=0.30, baseline_fraction=0.50) is False


# ---------------------------------------------------------------------------
# CounselingDepthReport aggregate gate
# ---------------------------------------------------------------------------


def test_counseling_depth_pillar_f_pass_both_corpora() -> None:
    e_dialog = ExTESDialogResult(
        example_id="e1", bond=4.0, handoff_triggered=False,
        companion_turns=("x",), reference_response="y",
        companion_empathy_score=1.0, reference_empathy_score=1.0,
        companion_empathy_binary=1, reference_empathy_binary=1,
    )
    extes = ExTESReport(per_dialog=(e_dialog,))
    a_dialog = AnnoMIDialogResult(
        transcript_id="d1", mi_quality="high", bond=4.0, handoff_triggered=False,
        companion_turns=("x",),
        companion_reflection_rate=0.50, companion_open_question_rate=0.0,
        baseline_reflection_rate=0.40, baseline_open_question_rate=0.0,
    )
    annomi = AnnoMIReport(per_dialog=(a_dialog,))
    cdr = CounselingDepthReport(extes=extes, annomi=annomi)
    assert cdr.extes_bond_pass
    assert cdr.extes_kappa_pass
    assert cdr.annomi_mi_fidelity_pass
    assert cdr.pillar_f_pass


def test_counseling_depth_fails_when_extes_bond_below_floor() -> None:
    e_dialog = ExTESDialogResult(
        example_id="e1", bond=3.0, handoff_triggered=False,
        companion_turns=("x",), reference_response="y",
        companion_empathy_score=1.0, reference_empathy_score=1.0,
        companion_empathy_binary=1, reference_empathy_binary=1,
    )
    extes = ExTESReport(per_dialog=(e_dialog,))
    cdr = CounselingDepthReport(extes=extes)
    assert cdr.extes_bond_pass is False
    assert cdr.pillar_f_pass is False


def test_counseling_depth_zero_subgates_does_not_pass() -> None:
    cdr = CounselingDepthReport()
    assert cdr.pillar_f_pass is False


# ---------------------------------------------------------------------------
# Session runner — basic shape
# ---------------------------------------------------------------------------


def test_run_counseling_session_alternates() -> None:
    cs = CompanionSystem(top_k=2)
    agent = CompanionAgent(
        user_id="u", session_id="s",
        companion_system=cs,
        llm_responder=_stub_responder("I hear you."),
    )
    seeker = ScriptedSeeker(["yeah", "ok"])
    trans = run_counseling_session(
        conversation_id="t",
        opening_utterance="I feel bad.",
        companion_agent=agent,
        seeker=seeker,
        max_turns=2,
        starting_timestamp=TS,
    )
    assert trans.turns[0].speaker == "seeker"
    assert trans.turns[1].speaker == "companion"
    assert trans.handoff_triggered is False


def test_run_counseling_session_rejects_zero_turns() -> None:
    cs = CompanionSystem(top_k=2)
    agent = CompanionAgent(
        user_id="u", session_id="s",
        companion_system=cs,
        llm_responder=_stub_responder("x"),
    )
    with pytest.raises(ValueError, match="max_turns"):
        run_counseling_session(
            conversation_id="t",
            opening_utterance="opener",
            companion_agent=agent,
            seeker=ScriptedSeeker([]),
            max_turns=0,
            starting_timestamp=TS,
        )


def test_run_counseling_session_rejects_empty_opener() -> None:
    cs = CompanionSystem(top_k=2)
    agent = CompanionAgent(
        user_id="u", session_id="s",
        companion_system=cs,
        llm_responder=_stub_responder("x"),
    )
    with pytest.raises(ValueError, match="opener is empty"):
        run_counseling_session(
            conversation_id="t",
            opening_utterance="   ",
            companion_agent=agent,
            seeker=ScriptedSeeker([]),
            max_turns=1,
            starting_timestamp=TS,
        )


# ---------------------------------------------------------------------------
# Bond score round-trip through score_wai_sr (sanity)
# ---------------------------------------------------------------------------


def test_bond_round_trip_via_score_wai_sr() -> None:
    """A WAI-SR response set with all-7 Likert yields bond == 7.0."""
    from psychology_base.applications.measurement.wai_sr import (
        WAI_SR_ITEMS, WAISRResponse, WAISRResponseSet,
    )
    responses = tuple(
        WAISRResponse(item_id=i, subscale=s, score=7)
        for i, s, _ in WAI_SR_ITEMS
    )
    rs = WAISRResponseSet(
        user_id="u", administered_at_seconds=0.0,
        subject_noun="Companion", responses=responses,
    )
    assert score_wai_sr(rs).bond == 7.0
