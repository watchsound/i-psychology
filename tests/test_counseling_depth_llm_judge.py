"""Phase 8 Pillar F — LLM semantic judge scorer tests (ExTES empathy).

In-memory fixtures only. No network, no real LLM router. Covers:

- prompt builder: context + reply + few-shot structure, response contract
- parser: canonical two-line format, case-insensitive, whitespace
  variants, markdown wrappers, free-text fallbacks, bare-numeric
  fallback, unparseable → ambiguous+unresolved (no raise)
- ``score_turn_empathy_llm`` / ``judge_turn_empathy_binary``: continuous
  mapping (yes→conf, no→1-conf, ambiguous→0.5)
- end-to-end sweep with a scripted judge returning deterministic
  verdicts per (turn-prefix) pattern
- κ-across-conversants under same judge; unresolved items excluded
  from κ denominator
- fresh-substrate-per-dialog contract preserved (companion system +
  agent factory called once per example)
- judge-vs-lexical κ diagnostic computes without raising
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import pytest

from psychology_base.applications.emotional import CompanionAgent
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks.public_datasets.counseling_depth_llm_judge import (
    EmpathyJudge,
    LLMExTESDialogResult,
    LLMExTESReport,
    ParsedEmpathyJudgment,
    build_empathy_prompt,
    judge_turn_empathy_binary,
    judge_vs_lexical_kappa,
    parse_empathy_judgment,
    score_extes_sweep_llm,
    score_turn_empathy_llm,
)
from psychology_base.eval.benchmarks.public_datasets.extes import ExTESExample


TS = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Scripted plumbing (same shape as the lexical-scorer test file)
# ---------------------------------------------------------------------------


def _stub_responder(reply: str = "It sounds like you're carrying a lot — I'm here with you."):
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


def _ex(i: int = 0, *, client: str = "I feel awful.", ref: str = "I understand.") -> ExTESExample:
    return ExTESExample(
        example_id=f"ex-{i}",
        split="test",
        system_prompt="supporter",
        client_utterance=client,
        counselor_response=ref,
    )


def _scripted_judge(
    *,
    companion_label: str = "yes",
    companion_conf: float = 0.9,
    reference_label: str = "yes",
    reference_conf: float = 0.9,
) -> EmpathyJudge:
    """Judge that returns distinct verdicts for companion vs reference based on
    whether the turn-text matches the stub-companion reply or the reference.
    """

    def _judge(turn: str, context: Sequence[str]) -> ParsedEmpathyJudgment:
        # Heuristic to split companion vs reference: context contains the
        # dataset client_utterance for both calls; distinguish by matching
        # the raw turn text.
        if turn.startswith("It sounds like"):
            return ParsedEmpathyJudgment(
                label=companion_label, confidence=companion_conf, raw_response="stub"
            )
        return ParsedEmpathyJudgment(
            label=reference_label, confidence=reference_conf, raw_response="stub"
        )

    return _judge


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def test_prompt_builder_emits_system_plus_user_pair() -> None:
    msgs = build_empathy_prompt(
        turn="I'm here with you.",
        context=["I can't stop crying."],
    )
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"
    assert "counseling" in msgs[0].content.lower() or "counselor" in msgs[0].content.lower()


def test_prompt_builder_injects_turn_and_context() -> None:
    msgs = build_empathy_prompt(
        turn="That sounds very hard.",
        context=["Everything feels wrong today."],
    )
    user = msgs[1].content
    assert "That sounds very hard." in user
    assert "Everything feels wrong today." in user
    # Response contract literal survives.
    assert "EMPATHIC=<yes|no|ambiguous>" in user
    assert "CONFIDENCE=" in user


def test_prompt_builder_includes_fewshot_anchors() -> None:
    msgs = build_empathy_prompt(turn="hi", context=["hi"])
    user = msgs[1].content.lower()
    # At least one empathic yes anchor, one no anchor, one ambiguous.
    assert "empathic=yes" in user
    assert "empathic=no" in user
    assert "empathic=ambiguous" in user


def test_prompt_builder_empty_context_falls_back_to_sentinel() -> None:
    msgs = build_empathy_prompt(turn="hi", context=[])
    user = msgs[1].content
    assert "no prior context" in user.lower() or "stand-alone" in user.lower()


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def test_parse_canonical_two_line() -> None:
    out = parse_empathy_judgment("EMPATHIC=yes\nCONFIDENCE=0.9")
    assert out.label == "yes"
    assert out.confidence == pytest.approx(0.9)
    assert out.unresolved is False


def test_parse_case_insensitive() -> None:
    out = parse_empathy_judgment("EMPATHIC=NO\nCONFIDENCE=0.8")
    assert out.label == "no"
    assert out.confidence == pytest.approx(0.8)


def test_parse_markdown_wrapper_tolerated() -> None:
    out = parse_empathy_judgment("```\nEMPATHIC=ambiguous\nCONFIDENCE=0.5\n```")
    assert out.label == "ambiguous"
    assert out.confidence == pytest.approx(0.5)


def test_parse_free_text_empathic() -> None:
    out = parse_empathy_judgment("This reply is empathic — strong recognition.")
    assert out.label == "yes"
    assert out.confidence > 0.5


def test_parse_free_text_not_empathic() -> None:
    out = parse_empathy_judgment("Not empathic. Deflecting into advice.")
    assert out.label == "no"
    assert out.confidence > 0.5


def test_parse_free_text_ambiguous() -> None:
    out = parse_empathy_judgment("This is ambiguous — polite but shallow.")
    assert out.label == "ambiguous"


def test_parse_bare_numeric_above_half_is_yes() -> None:
    out = parse_empathy_judgment("0.8")
    assert out.label == "yes"
    assert out.confidence == pytest.approx(0.8)


def test_parse_bare_numeric_below_half_is_no() -> None:
    out = parse_empathy_judgment("0.2")
    assert out.label == "no"
    assert out.confidence == pytest.approx(0.2)


def test_parse_unparseable_falls_back_to_ambiguous_unresolved() -> None:
    out = parse_empathy_judgment("I'm not sure how to answer this.")
    assert out.label == "ambiguous"
    assert out.confidence == pytest.approx(0.5)
    assert out.unresolved is True


def test_parse_empty_text_is_unresolved() -> None:
    out = parse_empathy_judgment("")
    assert out.unresolved is True
    assert out.label == "ambiguous"


def test_parse_confidence_clipped_to_unit_interval() -> None:
    high = parse_empathy_judgment("EMPATHIC=yes\nCONFIDENCE=1.5")
    assert high.confidence == pytest.approx(1.0)
    low = parse_empathy_judgment("EMPATHIC=no\nCONFIDENCE=-0.2")
    assert low.confidence == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Scoring wrappers (score_turn_empathy_llm, judge_turn_empathy_binary)
# ---------------------------------------------------------------------------


def test_score_turn_without_router_falls_back_to_half() -> None:
    assert score_turn_empathy_llm("hi", ["hi"]) == pytest.approx(0.5)


def test_judge_turn_binary_without_router_falls_back_to_midpoint() -> None:
    # No router → score=0.5; default threshold 0.5 → 0.5 >= 0.5 → binary=1.
    assert judge_turn_empathy_binary("hi", ["hi"], threshold=0.5) == 1
    # Raising threshold above 0.5 flips it to 0.
    assert judge_turn_empathy_binary("hi", ["hi"], threshold=0.6) == 0


class _StubRouter:
    def __init__(self, response_content: str) -> None:
        self.response_content = response_content
        self.calls: list[list] = []

    def chat(self, messages, *, tier=None, max_tokens=None, temperature=None, use_cache=True):
        self.calls.append(list(messages))
        # Return a dummy Completion-shaped object.
        class _C:
            pass
        c = _C()
        c.content = self.response_content
        return c


def test_score_turn_with_mocked_router_yes_returns_confidence() -> None:
    r = _StubRouter("EMPATHIC=yes\nCONFIDENCE=0.85")
    out = score_turn_empathy_llm("reply", ["ctx"], router=r)
    assert out == pytest.approx(0.85)


def test_score_turn_with_mocked_router_no_returns_one_minus() -> None:
    r = _StubRouter("EMPATHIC=no\nCONFIDENCE=0.9")
    out = score_turn_empathy_llm("reply", ["ctx"], router=r)
    assert out == pytest.approx(1.0 - 0.9)


def test_score_turn_with_mocked_router_ambiguous_returns_half() -> None:
    r = _StubRouter("EMPATHIC=ambiguous\nCONFIDENCE=0.5")
    out = score_turn_empathy_llm("reply", ["ctx"], router=r)
    assert out == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# score_extes_sweep_llm — end-to-end plumbing
# ---------------------------------------------------------------------------


def test_score_extes_sweep_llm_builds_per_dialog_results() -> None:
    examples = [_ex(i) for i in range(2)]

    def _system():
        return CompanionSystem(top_k=4)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="extes",
            companion_system=system,
            llm_responder=_stub_responder(
                "It sounds like you're hurting right now. I'm here with you."
            ),
        )

    def _seeker(example):
        return ScriptedSeeker(
            ["ok", "yeah", "6", "6", "6", "6", "6"]
        )

    judge = _scripted_judge(
        companion_label="yes", companion_conf=0.9,
        reference_label="yes", reference_conf=0.9,
    )

    report = score_extes_sweep_llm(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        judge=judge,
        max_turns_per_dialog=3,
        base_timestamp=TS,
    )

    assert isinstance(report, LLMExTESReport)
    assert report.n == 2
    assert report.n_unresolved == 0
    # All raters say yes → both companion+reference dialog-binary = 1 →
    # κ = 1.0 by both-constant-equal fallback in _cohens_kappa_binary.
    assert report.empath_kappa == pytest.approx(1.0)
    assert report.mean_llm_judge_companion_empathy_binary_rate == pytest.approx(1.0)
    assert report.mean_llm_judge_reference_empathy_binary_rate == pytest.approx(1.0)


def test_score_extes_sweep_llm_fresh_system_per_dialog() -> None:
    created: list = []

    def _system():
        cs = CompanionSystem(top_k=2)
        created.append(cs)
        return cs

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="extes",
            companion_system=system,
            llm_responder=_stub_responder("It sounds like you're worried."),
        )

    def _seeker(example):
        return ScriptedSeeker(["x", "x", "4", "4", "4", "4", "4"])

    examples = [_ex(i) for i in range(3)]
    judge = _scripted_judge()
    score_extes_sweep_llm(
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


def test_score_extes_sweep_llm_disagreement_lowers_kappa() -> None:
    """Companion labelled empathic / reference not-empathic across all dialogs
    → both constant but DIFFERENT → κ falls to 0 (not 1) per the
    _cohens_kappa_binary both-constant policy."""
    examples = [_ex(i) for i in range(3)]

    def _system():
        return CompanionSystem(top_k=2)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="extes",
            companion_system=system,
            llm_responder=_stub_responder("It sounds like..."),
        )

    def _seeker(example):
        return ScriptedSeeker(["x", "4", "4", "4", "4", "4"])

    judge = _scripted_judge(
        companion_label="yes", companion_conf=0.9,
        reference_label="no", reference_conf=0.9,
    )
    report = score_extes_sweep_llm(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        judge=judge,
        max_turns_per_dialog=2,
        base_timestamp=TS,
    )
    # Companion binaries all 1, reference binaries all 0.
    # Both are constant → po = 0, pe = 0 → by both-constant-and-not-equal
    # policy κ = 0.0 (po ≠ 1.0 branch).
    assert report.empath_kappa == pytest.approx(0.0)


def test_score_extes_sweep_llm_mixed_dialogs_produce_signed_kappa() -> None:
    """Mix of agreeing + disagreeing dialogs so both raters vary → real κ."""
    examples = [_ex(i) for i in range(4)]

    def _system():
        return CompanionSystem(top_k=2)

    # Alternate companion responders to produce varied dialog labels.
    responders = [
        _stub_responder("It sounds like you're hurting."),   # ex-0: empathic cue
        _stub_responder("just move on from it."),            # ex-1: no cue
        _stub_responder("It sounds like you're frustrated."),  # ex-2
        _stub_responder("fix it yourself."),                 # ex-3
    ]

    def _agent(system, example):
        idx = int(example.example_id.split("-")[1])
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="extes",
            companion_system=system,
            llm_responder=responders[idx],
        )

    def _seeker(example):
        return ScriptedSeeker(["x", "4", "4", "4", "4", "4"])

    def _judge(turn: str, context):
        # Mirror lexical sense: "It sounds like" → empathic yes;
        # advice-giving → no. Reference is always empathic.
        if turn.startswith("It sounds like"):
            return ParsedEmpathyJudgment(label="yes", confidence=0.9, raw_response="")
        if turn.startswith("I understand"):
            return ParsedEmpathyJudgment(label="yes", confidence=0.9, raw_response="")
        return ParsedEmpathyJudgment(label="no", confidence=0.9, raw_response="")

    report = score_extes_sweep_llm(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        judge=_judge,
        max_turns_per_dialog=2,
        base_timestamp=TS,
    )
    # Companion: [1, 0, 1, 0]; reference: [1, 1, 1, 1] → 2 agree, 2 disagree.
    # po = 0.5; pa_pos = 0.5; pb_pos = 1.0; pe = 0.5*1.0 + 0.5*0 = 0.5.
    # κ = (0.5 - 0.5)/(1 - 0.5) = 0.
    assert report.empath_kappa == pytest.approx(0.0, abs=1e-9)


def test_score_extes_sweep_llm_unresolved_dialog_dropped_from_kappa() -> None:
    """A dialog where ALL companion turns + reference are unresolved → logged
    as unresolved, dropped from κ denominator."""
    examples = [
        _ex(0, client="help", ref="I understand."),
        _ex(1, client="help", ref="I understand."),
    ]

    def _system():
        return CompanionSystem(top_k=2)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="extes",
            companion_system=system,
            llm_responder=_stub_responder("It sounds like..."),
        )

    def _seeker(example):
        return ScriptedSeeker(["x", "4", "4", "4", "4", "4"])

    def _judge(turn, context):
        # ex-1 → everything unresolved; ex-0 → resolved yes.
        if "frustrated-no-such-pattern" in turn:  # never matches
            return ParsedEmpathyJudgment(label="no", confidence=0.9, raw_response="")
        # Default: ex-0 resolves yes, ex-1 unresolved via trigger ...
        return ParsedEmpathyJudgment(label="yes", confidence=0.9, raw_response="")

    # Can't easily differentiate without poking state. Use a stateful mock:
    class StateJudge:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self, turn, context):
            self.n += 1
            # First dialog (ex-0): 2 companion turns + 1 reference, resolved.
            # Second dialog (ex-1): 2 companion turns + 1 reference, unresolved.
            if self.n <= 3:
                return ParsedEmpathyJudgment(
                    label="yes", confidence=0.9, raw_response=""
                )
            return ParsedEmpathyJudgment(
                label="ambiguous", confidence=0.5, raw_response="", unresolved=True
            )

    report = score_extes_sweep_llm(
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
    assert "ex-1" in report.unresolved_dialog_ids
    # κ computed over 1 resolved dialog only — both=1 → fallback 1.0.
    assert report.empath_kappa == pytest.approx(1.0)


def test_score_extes_sweep_llm_routing_tag_preserved() -> None:
    examples = [_ex(0)]

    def _system():
        return CompanionSystem(top_k=2)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="extes",
            companion_system=system,
            llm_responder=_stub_responder("It sounds like..."),
        )

    def _seeker(example):
        return ScriptedSeeker(["x", "4", "4", "4", "4", "4"])

    report = score_extes_sweep_llm(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        judge=_scripted_judge(),
        max_turns_per_dialog=2,
        base_timestamp=TS,
        judge_routing="ollama / qwen2.5:7b (reason=ollama_available)",
    )
    assert "ollama" in report.judge_routing


# ---------------------------------------------------------------------------
# judge-vs-lexical κ diagnostic
# ---------------------------------------------------------------------------


def test_judge_vs_lexical_kappa_agreement_is_one() -> None:
    """When judge and lexical both flag the same dialogs as empathic, κ=1."""
    examples = [_ex(i) for i in range(2)]

    def _system():
        return CompanionSystem(top_k=2)

    def _agent(system, example):
        return CompanionAgent(
            user_id=f"u-{example.example_id}",
            session_id="extes",
            companion_system=system,
            # This reply hits all 3 lexical empathy families.
            llm_responder=_stub_responder(
                "It sounds like that feels overwhelming — I'm here with you. "
                "That makes sense."
            ),
        )

    def _seeker(example):
        return ScriptedSeeker(["x", "4", "4", "4", "4", "4"])

    report = score_extes_sweep_llm(
        examples,
        system_factory=_system,
        agent_factory=_agent,
        seeker_factory=_seeker,
        judge=_scripted_judge(companion_label="yes"),
        max_turns_per_dialog=2,
        base_timestamp=TS,
    )
    kappa = judge_vs_lexical_kappa(report, examples)
    # Both judge and lexical say "empathic" on both dialogs → both-constant
    # fallback yields κ=1.0.
    assert kappa == pytest.approx(1.0)


def test_judge_vs_lexical_kappa_doesnt_raise_on_empty() -> None:
    report = LLMExTESReport(per_dialog=())
    assert judge_vs_lexical_kappa(report, []) == 0.0


# ---------------------------------------------------------------------------
# Pillar F rate-matched sibling gate (parallel to κ-across-conversants)
# ---------------------------------------------------------------------------


def _mk_dialog(
    example_id: str,
    *,
    companion_rate: float,
    reference_binary: int,
) -> LLMExTESDialogResult:
    """Minimal dialog fixture — only the fields the rate-matched gate reads."""
    return LLMExTESDialogResult(
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


def test_rate_matched_gate_empty_report_fails_closed() -> None:
    report = LLMExTESReport(per_dialog=())
    assert report.meets_rate_matched_empathy_gate() is False


def test_rate_matched_gate_passes_on_absolute_floor() -> None:
    # Companion rate well above 0.30 absolute floor — passes regardless of
    # reference rate. This is the ExTES/PsyQA/PsyDial-D4 κ-ceiling case:
    # Companion ~0.78 clears the absolute floor by a wide margin.
    dialogs = tuple(
        _mk_dialog(f"d{i}", companion_rate=0.78, reference_binary=i % 2)
        for i in range(10)
    )
    report = LLMExTESReport(per_dialog=dialogs)
    assert report.meets_rate_matched_empathy_gate() is True


def test_rate_matched_gate_fails_below_absolute_and_below_half_baseline() -> None:
    # Companion 0.10 < 0.30 absolute; also < 0.5 * 0.80 = 0.40.
    dialogs = tuple(
        _mk_dialog(f"d{i}", companion_rate=0.10, reference_binary=1)
        for i in range(10)
    )
    report = LLMExTESReport(per_dialog=dialogs)
    assert report.meets_rate_matched_empathy_gate() is False


def test_rate_matched_gate_passes_below_absolute_via_half_baseline() -> None:
    # Companion 0.20 < 0.30 absolute, but ≥ 0.5 * 0.30 = 0.15 → OR clause fires.
    # Reference mean = 0.30 (3-of-10 dialogs reference-empathic).
    dialogs = tuple(
        _mk_dialog(f"d{i}", companion_rate=0.20, reference_binary=1 if i < 3 else 0)
        for i in range(10)
    )
    report = LLMExTESReport(per_dialog=dialogs)
    # Sanity: this is the "below absolute, above half-baseline" lane.
    assert report.mean_llm_judge_companion_empathy_binary_rate < 0.30
    assert report.meets_rate_matched_empathy_gate() is True


def test_rate_matched_gate_absolute_floor_wins_when_reference_zero() -> None:
    # Reference-rate zero → half-baseline clause can't fire; only absolute
    # clause can pass. 0.35 ≥ 0.30 → PASS.
    dialogs = tuple(
        _mk_dialog(f"d{i}", companion_rate=0.35, reference_binary=0)
        for i in range(10)
    )
    report = LLMExTESReport(per_dialog=dialogs)
    assert report.meets_rate_matched_empathy_gate() is True


def test_rate_matched_gate_custom_floors_respected() -> None:
    # Companion 0.25 fails default (0.30 absolute, 0.5 * reference), but
    # passes with absolute_floor=0.20.
    dialogs = tuple(
        _mk_dialog(f"d{i}", companion_rate=0.25, reference_binary=0)
        for i in range(10)
    )
    report = LLMExTESReport(per_dialog=dialogs)
    assert report.meets_rate_matched_empathy_gate() is False
    assert report.meets_rate_matched_empathy_gate(absolute_floor=0.20) is True
