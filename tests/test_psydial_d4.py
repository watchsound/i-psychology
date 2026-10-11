"""Phase 8 Pillar F — PsyDial-D4 adapter + CN-register LLM-judge scorer tests.

In-memory fixtures only. No network, no real LLM router. Covers:

- adapter: ``PsyDialD4Example`` validation + ``iter_psydial_d4_examples`` streaming
- end-to-end sweep: κ-across-conversants under scripted CN judge
- fresh-substrate-per-dialog contract
- judge-vs-lexical κ diagnostic completes without raising on CN inputs
- loader round-trip for a representative PsyDial-D4 shim row
- shared CN prompt-builder reused byte-identically from PsyQA scorer
  (one sanity-check — detailed CN prompt tests live in test_psyqa.py)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pytest

from psychology_base.applications.emotional import CompanionAgent
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks.public_datasets.counseling_depth_llm_judge import (
    EmpathyJudge,
    ParsedEmpathyJudgment,
)
from psychology_base.eval.benchmarks.public_datasets.psydial_d4 import (
    PsyDialD4Example,
    iter_psydial_d4_examples,
)
from psychology_base.eval.benchmarks.public_datasets.psydial_d4_counseling_depth import (
    LLMPsyDialD4DialogResult,
    LLMPsyDialD4Report,
    judge_vs_lexical_kappa_psydial_d4,
    score_psydial_d4_sweep_llm,
)
from psychology_base.eval.benchmarks.public_datasets.psyqa_counseling_depth import (
    build_empathy_prompt_zh,
)


TS = datetime(2026, 10, 10, 14, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Scripted plumbing
# ---------------------------------------------------------------------------


def _stub_responder(reply: str = "听起来你心里真的很难受——我在这里陪着你。"):
    def _r(system_prompt: str, context_block: str, user_utterance: str) -> str:
        return reply
    return _r


class ScriptedSeeker:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.received: list[str] = []

    def __call__(self, previous_agent_reply: str) -> str:
        self.received.append(previous_agent_reply)
        if not self._replies:
            return ""
        return self._replies.pop(0)


def _ex(
    i: int = 0,
    *,
    seeker_opening: str = "我最近总是很难过，睡不着。",
    counselor: str = "我理解你的感受。",
) -> PsyDialD4Example:
    return PsyDialD4Example(
        example_id=f"psydial-d4-ex-{i}",
        split="train",
        seeker_opening=seeker_opening,
        counselor_response=counselor,
        n_messages=10,
        n_user_messages=5,
        n_assistant_messages=5,
        seeker_opening_turn_index=3,
        counselor_response_turn_index=4,
    )


def _scripted_judge(
    *,
    companion_label: str = "yes",
    companion_conf: float = 0.9,
    reference_label: str = "yes",
    reference_conf: float = 0.9,
) -> EmpathyJudge:
    def _judge(turn: str, context: Sequence[str]) -> ParsedEmpathyJudgment:
        if turn.startswith("听起来"):
            return ParsedEmpathyJudgment(
                label=companion_label, confidence=companion_conf, raw_response="stub"
            )
        return ParsedEmpathyJudgment(
            label=reference_label, confidence=reference_conf, raw_response="stub"
        )
    return _judge


# ---------------------------------------------------------------------------
# Adapter: PsyDialD4Example dataclass validation
# ---------------------------------------------------------------------------


def test_psydial_d4_example_requires_example_id() -> None:
    with pytest.raises(ValueError):
        PsyDialD4Example(
            example_id="",
            split="train",
            seeker_opening="开头",
            counselor_response="回应",
        )


def test_psydial_d4_example_requires_nonempty_seeker_opening() -> None:
    with pytest.raises(ValueError):
        PsyDialD4Example(
            example_id="x",
            split="train",
            seeker_opening="   ",
            counselor_response="回应",
        )


def test_psydial_d4_example_requires_nonempty_counselor_response() -> None:
    with pytest.raises(ValueError):
        PsyDialD4Example(
            example_id="x",
            split="train",
            seeker_opening="开头",
            counselor_response="",
        )


def test_psydial_d4_example_keywords_default_empty() -> None:
    """PsyDial-D4 has no topic tags — keywords defaults to empty string."""
    ex = _ex(0)
    assert ex.keywords == ""


# ---------------------------------------------------------------------------
# Adapter: iter_psydial_d4_examples streaming
# ---------------------------------------------------------------------------


def test_iter_psydial_d4_examples_round_trip(tmp_path: Path) -> None:
    shim = tmp_path / "train.jsonl"
    rows = [
        {
            "example_id": "psydial-d4-00000",
            "split": "train",
            "n_messages": 42,
            "n_user_messages": 21,
            "n_assistant_messages": 21,
            "seeker_opening": "我想知道为什么自己现在会这么迷茫。",
            "counselor_response": "我明白了。你刚才提到了迷茫，这些感觉让你很困扰吗？",
            "seeker_opening_turn_index": 5,
            "counselor_response_turn_index": 6,
            "n_alt_responses": 20,
            "alt_responses": ["听起来你...", "你能描述一下..."],
        },
        {
            "example_id": "psydial-d4-00001",
            "split": "train",
            "n_messages": 55,
            "n_user_messages": 27,
            "n_assistant_messages": 27,
            "seeker_opening": "最近心情很差。",
            "counselor_response": "能和我多说说吗？",
            "seeker_opening_turn_index": 1,
            "counselor_response_turn_index": 2,
            "n_alt_responses": 25,
            "alt_responses": [],
        },
    ]
    shim.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )
    out = list(iter_psydial_d4_examples(shim))
    assert len(out) == 2
    assert out[0].seeker_opening.startswith("我想知道")
    assert out[0].n_messages == 42
    assert out[1].alt_responses == ()


def test_iter_psydial_d4_examples_respects_limit(tmp_path: Path) -> None:
    shim = tmp_path / "train.jsonl"
    rows = [
        {
            "example_id": f"psydial-d4-{i:05d}",
            "split": "train",
            "seeker_opening": f"开头{i}很长很长很长很长很长很长",
            "counselor_response": f"回应{i}",
        }
        for i in range(5)
    ]
    shim.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )
    out = list(iter_psydial_d4_examples(shim, limit=2))
    assert len(out) == 2
    assert out[0].example_id == "psydial-d4-00000"


def test_iter_psydial_d4_examples_missing_required_field_raises(tmp_path: Path) -> None:
    shim = tmp_path / "bad.jsonl"
    shim.write_text(
        json.dumps({"example_id": "x", "split": "train"}, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required fields"):
        list(iter_psydial_d4_examples(shim))


# ---------------------------------------------------------------------------
# Shared CN prompt-builder is byte-identical to PsyQA's — sanity-check
# (detailed CN prompt shape tests live in test_psyqa.py)
# ---------------------------------------------------------------------------


def test_psydial_d4_reuses_psyqa_cn_prompt_builder() -> None:
    msgs = build_empathy_prompt_zh(
        turn="听起来你压力真的很大。",
        context=["我最近情绪很糟糕。"],
    )
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"
    assert "听起来你压力真的很大。" in msgs[1].content
    assert "我最近情绪很糟糕。" in msgs[1].content
    assert "EMPATHIC=<yes|no|ambiguous>" in msgs[1].content
    assert "CONFIDENCE=" in msgs[1].content


# ---------------------------------------------------------------------------
# score_psydial_d4_sweep_llm — end-to-end plumbing
# ---------------------------------------------------------------------------


def test_score_psydial_d4_sweep_llm_builds_per_dialog_results() -> None:
    examples = [_ex(i) for i in range(2)]

    def _system():
        return CompanionSystem(top_k=4)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="psydial_d4",
            companion_system=system,
            llm_responder=_stub_responder(
                "听起来你现在真的在扛着很重的东西，我陪着你。"
            ),
        )

    def _seeker(example):
        return ScriptedSeeker(["嗯", "是的", "6", "6", "6", "6", "6"])

    judge = _scripted_judge(companion_label="yes", reference_label="yes")

    report = score_psydial_d4_sweep_llm(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        judge=judge,
        max_turns_per_dialog=3,
        base_timestamp=TS,
    )
    assert isinstance(report, LLMPsyDialD4Report)
    assert report.n == 2
    assert report.n_unresolved == 0
    assert report.empath_kappa == pytest.approx(1.0)
    assert report.mean_llm_judge_companion_empathy_binary_rate == pytest.approx(1.0)
    assert report.mean_llm_judge_reference_empathy_binary_rate == pytest.approx(1.0)


def test_score_psydial_d4_sweep_llm_fresh_system_per_dialog() -> None:
    created: list = []

    def _system():
        cs = CompanionSystem(top_k=2)
        created.append(cs)
        return cs

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="psydial_d4",
            companion_system=system,
            llm_responder=_stub_responder("听起来你很担心。"),
        )

    def _seeker(example):
        return ScriptedSeeker(["嗯", "嗯", "4", "4", "4", "4", "4"])

    examples = [_ex(i) for i in range(3)]
    judge = _scripted_judge()
    score_psydial_d4_sweep_llm(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        judge=judge,
        max_turns_per_dialog=2,
        base_timestamp=TS,
    )
    assert len(created) == 3
    assert len({id(cs) for cs in created}) == 3


def test_score_psydial_d4_sweep_llm_disagreement_lowers_kappa() -> None:
    """Companion all yes / reference all no → constant different → κ=0."""
    examples = [_ex(i) for i in range(3)]

    def _system():
        return CompanionSystem(top_k=2)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="psydial_d4",
            companion_system=system,
            llm_responder=_stub_responder("听起来……"),
        )

    def _seeker(example):
        return ScriptedSeeker(["嗯", "4", "4", "4", "4", "4"])

    judge = _scripted_judge(
        companion_label="yes", companion_conf=0.9,
        reference_label="no", reference_conf=0.9,
    )
    report = score_psydial_d4_sweep_llm(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        judge=judge,
        max_turns_per_dialog=2,
        base_timestamp=TS,
    )
    assert report.empath_kappa == pytest.approx(0.0)


def test_score_psydial_d4_sweep_llm_routing_tag_preserved() -> None:
    examples = [_ex(0)]

    def _system():
        return CompanionSystem(top_k=2)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="psydial_d4",
            companion_system=system,
            llm_responder=_stub_responder("听起来……"),
        )

    def _seeker(example):
        return ScriptedSeeker(["嗯", "4", "4", "4", "4", "4"])

    report = score_psydial_d4_sweep_llm(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        judge=_scripted_judge(),
        max_turns_per_dialog=2,
        base_timestamp=TS,
        judge_routing="tongyi / qwen-turbo (reason=ollama_unavailable_fallback_to_cloud)",
    )
    assert "tongyi" in report.judge_routing


def test_score_psydial_d4_sweep_llm_unresolved_dialog_dropped_from_kappa() -> None:
    examples = [_ex(i) for i in range(2)]

    def _system():
        return CompanionSystem(top_k=2)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="psydial_d4",
            companion_system=system,
            llm_responder=_stub_responder("听起来……"),
        )

    def _seeker(example):
        return ScriptedSeeker(["嗯", "4", "4", "4", "4", "4"])

    class StateJudge:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self, turn, context):
            self.n += 1
            if self.n <= 3:
                return ParsedEmpathyJudgment(
                    label="yes", confidence=0.9, raw_response=""
                )
            return ParsedEmpathyJudgment(
                label="ambiguous", confidence=0.5, raw_response="", unresolved=True
            )

    report = score_psydial_d4_sweep_llm(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        judge=StateJudge(),
        max_turns_per_dialog=2,
        base_timestamp=TS,
    )
    assert report.n == 2
    assert report.n_unresolved == 1
    assert "psydial-d4-ex-1" in report.unresolved_dialog_ids


def test_score_psydial_d4_sweep_llm_handoff_zeros_bond() -> None:
    """If CrisisEscalation handoff fires, bond must be None (per fresh-stack contract)."""
    examples = [_ex(0, seeker_opening="我想结束一切，活不下去了。")]

    def _system():
        return CompanionSystem(top_k=2)

    from psychology_base.applications.safety import (
        AIIdentityReminder,
        CrisisEscalation,
        ExtremeEmotionWatcher,
    )
    crisis_sink: list = []

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="psydial_d4",
            companion_system=system,
            llm_responder=_stub_responder("听起来你现在非常痛苦，我在这里陪着你。"),
            ai_identity_reminder=AIIdentityReminder(),
            crisis_escalation=CrisisEscalation(handoff_sink=crisis_sink.append),
            extreme_emotion_watcher=ExtremeEmotionWatcher(
                review_sink=list().append,
                user_id=f"u-{example.example_id}",
            ),
        )

    def _seeker(example):
        return ScriptedSeeker(["4"])

    report = score_psydial_d4_sweep_llm(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        judge=_scripted_judge(),
        max_turns_per_dialog=2,
        base_timestamp=TS,
    )
    # Depending on the specific CrisisEscalation phrasing-match, handoff
    # may or may not fire on this short-input case. The contract we're
    # testing is: IF handoff triggered → bond is None.
    for d in report.per_dialog:
        if d.handoff_triggered:
            assert d.bond is None


# ---------------------------------------------------------------------------
# judge-vs-lexical κ diagnostic — doesn't raise on CN content
# ---------------------------------------------------------------------------


def test_judge_vs_lexical_kappa_psydial_d4_completes_without_raising() -> None:
    examples = [_ex(i) for i in range(2)]

    def _system():
        return CompanionSystem(top_k=2)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="psydial_d4",
            companion_system=system,
            llm_responder=_stub_responder("听起来你很担心——我在这里陪着你。"),
        )

    def _seeker(example):
        return ScriptedSeeker(["嗯", "4", "4", "4", "4", "4"])

    report = score_psydial_d4_sweep_llm(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        judge=_scripted_judge(companion_label="yes"),
        max_turns_per_dialog=2,
        base_timestamp=TS,
    )
    diag = judge_vs_lexical_kappa_psydial_d4(report, examples)
    assert isinstance(diag, float)


def test_judge_vs_lexical_kappa_psydial_d4_doesnt_raise_on_empty() -> None:
    report = LLMPsyDialD4Report(per_dialog=())
    assert judge_vs_lexical_kappa_psydial_d4(report, []) == 0.0


# ---------------------------------------------------------------------------
# Pillar F rate-matched sibling gate — PsyDial-D4 (CN AI-training-target)
# ---------------------------------------------------------------------------


def _mk_psydial_d4_dialog(
    example_id: str,
    *,
    companion_rate: float,
    reference_binary: int,
) -> LLMPsyDialD4DialogResult:
    return LLMPsyDialD4DialogResult(
        example_id=example_id,
        bond=4.0,
        handoff_triggered=False,
        companion_turns=("x",),
        reference_response="y",
        companion_empathy_scores=(companion_rate,),
        companion_empathy_binary_rate=companion_rate,
        reference_empathy_score=float(reference_binary),
        reference_empathy_binary=reference_binary,
        companion_empathy_binary=1 if companion_rate >= 0.5 else 0,
        unresolved=False,
    )


def test_psydial_d4_rate_matched_gate_empty_report_fails_closed() -> None:
    report = LLMPsyDialD4Report(per_dialog=())
    assert report.meets_rate_matched_empathy_gate() is False


def test_psydial_d4_rate_matched_gate_mirrors_live_result() -> None:
    # Mirrors PsyDial-D4 κ-ceiling live-fire finding: Companion rate ~0.78,
    # reference rate ~0.36 (CN AI-counselor-training register — intermediate
    # between PsyQA 0.22 advice-heavy and ExTES 0.48 role-play). Companion
    # clears absolute floor by wide margin.
    dialogs = tuple(
        _mk_psydial_d4_dialog(
            f"d{i}",
            companion_rate=0.78,
            reference_binary=1 if i < 4 else 0,  # ~0.40 reference rate
        )
        for i in range(10)
    )
    report = LLMPsyDialD4Report(per_dialog=dialogs)
    assert report.meets_rate_matched_empathy_gate() is True


def test_psydial_d4_rate_matched_gate_fails_when_both_clauses_miss() -> None:
    dialogs = tuple(
        _mk_psydial_d4_dialog(f"d{i}", companion_rate=0.10, reference_binary=1)
        for i in range(10)
    )
    report = LLMPsyDialD4Report(per_dialog=dialogs)
    assert report.meets_rate_matched_empathy_gate() is False
