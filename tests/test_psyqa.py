"""Phase 8 Pillar F — PsyQA adapter + Chinese-register LLM-judge scorer tests.

In-memory fixtures only. No network, no real LLM router. Covers:

- adapter: ``PsyQAExample`` validation + ``iter_psyqa_examples`` streaming
- seeker_opening composition (question + description)
- prompt builder: Chinese system+user shape, context+reply+few-shot preserved
- Chinese register few-shot anchor presence (抱抱 / 陪着 / 听起来)
- end-to-end sweep: κ-across-conversants under scripted Chinese judge
- fresh-substrate-per-dialog contract
- judge-vs-lexical κ diagnostic completes without raising
- loader round-trip for a representative PsyQA shim row
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
from psychology_base.eval.benchmarks.public_datasets.psyqa import (
    PsyQAExample,
    iter_psyqa_examples,
)
from psychology_base.eval.benchmarks.public_datasets.psyqa_counseling_depth import (
    LLMPsyQADialogResult,
    LLMPsyQAReport,
    build_empathy_prompt_zh,
    judge_vs_lexical_kappa_psyqa,
    score_psyqa_sweep_llm,
)


TS = datetime(2026, 10, 10, 12, 0, 0, tzinfo=timezone.utc)


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


def _ex(i: int = 0, *, question: str = "我最近总是很难过。", description: str = "睡不着觉。", counselor: str = "我理解你的感受。") -> PsyQAExample:
    return PsyQAExample(
        example_id=f"psyqa-ex-{i}",
        split="test",
        question=question,
        description=description,
        keywords="情绪",
        counselor_response=counselor,
        question_id=1000 + i,
    )


def _scripted_judge(
    *,
    companion_label: str = "yes",
    companion_conf: float = 0.9,
    reference_label: str = "yes",
    reference_conf: float = 0.9,
) -> EmpathyJudge:
    """Judge distinguishing companion-side (prefix 听起来) vs reference-side."""

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
# Adapter: PsyQAExample dataclass validation
# ---------------------------------------------------------------------------


def test_psyqa_example_requires_example_id() -> None:
    with pytest.raises(ValueError):
        PsyQAExample(
            example_id="",
            split="test",
            question="q",
            description="d",
            keywords="",
            counselor_response="r",
        )


def test_psyqa_example_requires_nonempty_question() -> None:
    with pytest.raises(ValueError):
        PsyQAExample(
            example_id="x",
            split="test",
            question="   ",
            description="d",
            keywords="",
            counselor_response="r",
        )


def test_psyqa_example_requires_nonempty_counselor_response() -> None:
    with pytest.raises(ValueError):
        PsyQAExample(
            example_id="x",
            split="test",
            question="q",
            description="d",
            keywords="",
            counselor_response="",
        )


def test_psyqa_example_seeker_opening_joins_question_and_description() -> None:
    ex = _ex(0, question="头痛失眠怎么办？", description="最近连续三周睡不着。")
    opener = ex.seeker_opening
    assert "头痛失眠怎么办？" in opener
    assert "最近连续三周睡不着。" in opener
    assert "\n" in opener


def test_psyqa_example_seeker_opening_without_description_falls_back_to_question() -> None:
    ex = _ex(
        0,
        question="怎么办？",
        description="",
    )
    assert ex.seeker_opening == "怎么办？"


# ---------------------------------------------------------------------------
# Adapter: iter_psyqa_examples streaming
# ---------------------------------------------------------------------------


def test_iter_psyqa_examples_round_trip(tmp_path: Path) -> None:
    shim = tmp_path / "test.jsonl"
    rows = [
        {
            "example_id": "psyqa-test-00000",
            "split": "test",
            "question_id": 1,
            "question": "问题一",
            "description": "描述一",
            "keywords": "情绪",
            "counselor_response": "回复一",
            "has_label": False,
            "n_alt_responses": 0,
            "alt_responses": [],
        },
        {
            "example_id": "psyqa-test-00001",
            "split": "test",
            "question_id": 2,
            "question": "问题二",
            "description": "描述二",
            "keywords": "工作",
            "counselor_response": "回复二",
            "has_label": True,
            "n_alt_responses": 1,
            "alt_responses": ["备选回复"],
        },
    ]
    shim.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )
    out = list(iter_psyqa_examples(shim))
    assert len(out) == 2
    assert out[0].question == "问题一"
    assert out[1].has_label is True
    assert out[1].alt_responses == ("备选回复",)


def test_iter_psyqa_examples_respects_limit(tmp_path: Path) -> None:
    shim = tmp_path / "test.jsonl"
    rows = [
        {
            "example_id": f"psyqa-test-{i:05d}",
            "split": "test",
            "question_id": i,
            "question": f"q{i}",
            "description": f"d{i}",
            "keywords": "",
            "counselor_response": f"r{i}",
            "has_label": False,
            "n_alt_responses": 0,
            "alt_responses": [],
        }
        for i in range(5)
    ]
    shim.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )
    out = list(iter_psyqa_examples(shim, limit=2))
    assert len(out) == 2
    assert out[0].example_id == "psyqa-test-00000"


def test_iter_psyqa_examples_missing_required_field_raises(tmp_path: Path) -> None:
    shim = tmp_path / "bad.jsonl"
    shim.write_text(
        json.dumps({"example_id": "x", "split": "test"}, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required fields"):
        list(iter_psyqa_examples(shim))


# ---------------------------------------------------------------------------
# Chinese prompt builder
# ---------------------------------------------------------------------------


def test_prompt_builder_zh_emits_system_plus_user_pair() -> None:
    msgs = build_empathy_prompt_zh(
        turn="我在这里陪着你。",
        context=["我控制不住一直哭。"],
    )
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"
    # system prompt declares task in Chinese
    assert "心理咨询督导" in msgs[0].content or "共情" in msgs[0].content


def test_prompt_builder_zh_injects_turn_and_context() -> None:
    msgs = build_empathy_prompt_zh(
        turn="听起来你压力真的很大。",
        context=["今天一整天都觉得很糟糕。"],
    )
    user = msgs[1].content
    assert "听起来你压力真的很大。" in user
    assert "今天一整天都觉得很糟糕。" in user
    # ASCII response contract preserved for shared parser.
    assert "EMPATHIC=<yes|no|ambiguous>" in user
    assert "CONFIDENCE=" in user


def test_prompt_builder_zh_includes_chinese_fewshot_anchors() -> None:
    msgs = build_empathy_prompt_zh(turn="hi", context=["hi"])
    user = msgs[1].content
    # Anchor labels present (ASCII).
    assert "EMPATHIC=yes" in user
    assert "EMPATHIC=no" in user
    assert "EMPATHIC=ambiguous" in user
    # Chinese-register anchor phrases present (at least one).
    assert any(
        phrase in user
        for phrase in ("抱抱", "陪着", "听起来", "我在这里")
    )


def test_prompt_builder_zh_empty_context_falls_back_to_sentinel() -> None:
    msgs = build_empathy_prompt_zh(turn="hi", context=[])
    user = msgs[1].content
    assert "无前置话语" in user or "独立评价" in user


# ---------------------------------------------------------------------------
# score_psyqa_sweep_llm — end-to-end plumbing
# ---------------------------------------------------------------------------


def test_score_psyqa_sweep_llm_builds_per_dialog_results() -> None:
    examples = [_ex(i) for i in range(2)]

    def _system():
        return CompanionSystem(top_k=4)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="psyqa",
            companion_system=system,
            llm_responder=_stub_responder(
                "听起来你现在真的在扛着很重的东西，我陪着你。"
            ),
        )

    def _seeker(example):
        return ScriptedSeeker(["嗯", "是的", "6", "6", "6", "6", "6"])

    judge = _scripted_judge(companion_label="yes", reference_label="yes")

    report = score_psyqa_sweep_llm(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        judge=judge,
        max_turns_per_dialog=3,
        base_timestamp=TS,
    )

    assert isinstance(report, LLMPsyQAReport)
    assert report.n == 2
    assert report.n_unresolved == 0
    # Both raters all yes → both constant equal → κ fallback 1.0.
    assert report.empath_kappa == pytest.approx(1.0)
    assert report.mean_llm_judge_companion_empathy_binary_rate == pytest.approx(1.0)
    assert report.mean_llm_judge_reference_empathy_binary_rate == pytest.approx(1.0)


def test_score_psyqa_sweep_llm_fresh_system_per_dialog() -> None:
    created: list = []

    def _system():
        cs = CompanionSystem(top_k=2)
        created.append(cs)
        return cs

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="psyqa",
            companion_system=system,
            llm_responder=_stub_responder("听起来你很担心。"),
        )

    def _seeker(example):
        return ScriptedSeeker(["嗯", "嗯", "4", "4", "4", "4", "4"])

    examples = [_ex(i) for i in range(3)]
    judge = _scripted_judge()
    score_psyqa_sweep_llm(
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


def test_score_psyqa_sweep_llm_disagreement_lowers_kappa() -> None:
    """Companion all yes / reference all no → both constant different → κ=0."""
    examples = [_ex(i) for i in range(3)]

    def _system():
        return CompanionSystem(top_k=2)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="psyqa",
            companion_system=system,
            llm_responder=_stub_responder("听起来……"),
        )

    def _seeker(example):
        return ScriptedSeeker(["嗯", "4", "4", "4", "4", "4"])

    judge = _scripted_judge(
        companion_label="yes", companion_conf=0.9,
        reference_label="no", reference_conf=0.9,
    )
    report = score_psyqa_sweep_llm(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        judge=judge,
        max_turns_per_dialog=2,
        base_timestamp=TS,
    )
    assert report.empath_kappa == pytest.approx(0.0)


def test_score_psyqa_sweep_llm_routing_tag_preserved() -> None:
    examples = [_ex(0)]

    def _system():
        return CompanionSystem(top_k=2)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="psyqa",
            companion_system=system,
            llm_responder=_stub_responder("听起来……"),
        )

    def _seeker(example):
        return ScriptedSeeker(["嗯", "4", "4", "4", "4", "4"])

    report = score_psyqa_sweep_llm(
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


def test_score_psyqa_sweep_llm_unresolved_dialog_dropped_from_kappa() -> None:
    """A dialog where ALL companion turns + reference are unresolved → logged
    as unresolved, dropped from κ denominator."""
    examples = [_ex(i) for i in range(2)]

    def _system():
        return CompanionSystem(top_k=2)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="psyqa",
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
            # First dialog (ex-0): 2 companion turns + 1 reference → 3 resolved.
            # Second dialog (ex-1): 2 companion turns + 1 reference → unresolved.
            if self.n <= 3:
                return ParsedEmpathyJudgment(
                    label="yes", confidence=0.9, raw_response=""
                )
            return ParsedEmpathyJudgment(
                label="ambiguous", confidence=0.5, raw_response="", unresolved=True
            )

    report = score_psyqa_sweep_llm(
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
    assert "psyqa-ex-1" in report.unresolved_dialog_ids


# ---------------------------------------------------------------------------
# judge-vs-lexical κ diagnostic (should complete without raising even on
# Chinese content — lexical scorer is English-tuned, will likely return 0
# but must not error)
# ---------------------------------------------------------------------------


def test_judge_vs_lexical_kappa_psyqa_completes_without_raising() -> None:
    examples = [_ex(i) for i in range(2)]

    def _system():
        return CompanionSystem(top_k=2)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="psyqa",
            companion_system=system,
            llm_responder=_stub_responder("听起来你很担心——我在这里陪着你。"),
        )

    def _seeker(example):
        return ScriptedSeeker(["嗯", "4", "4", "4", "4", "4"])

    report = score_psyqa_sweep_llm(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        judge=_scripted_judge(companion_label="yes"),
        max_turns_per_dialog=2,
        base_timestamp=TS,
    )
    # Returns a float — no raise even though lexical scorer is English-tuned
    # on Chinese companion turns.
    diag = judge_vs_lexical_kappa_psyqa(report, examples)
    assert isinstance(diag, float)


def test_judge_vs_lexical_kappa_psyqa_doesnt_raise_on_empty() -> None:
    report = LLMPsyQAReport(per_dialog=())
    assert judge_vs_lexical_kappa_psyqa(report, []) == 0.0
