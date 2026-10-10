"""Phase 8 Pillar F — PsyQA Chinese-register counseling-depth sub-gate.

**Why this file (not reuse counseling_depth_llm_judge).** The ExTES LLM
judge in :mod:`counseling_depth_llm_judge` is authored in English register
— system prompt, few-shot anchors, response format, and the simulated-
seeker persona are all English-trained. PsyQA records are Chinese; a
semantic rater trained to anchor on English counseling vocabulary (``I
hear you``, ``that sounds hard``) will misread Chinese empathic
constructions (``抱抱题主``, ``很理解你的心情``, ``陪着你``). Rather
than muddy the ExTES scorer with code-switched anchors, this file ships
a parallel scorer with Chinese anchors + Chinese seeker persona and
reuses the structural machinery (:func:`run_counseling_session`,
:func:`_cohens_kappa_binary`, :func:`administer_wai_sr_simulated`,
Bond administration) byte-identically.

**Gate contract (per phase-8-dataset-replay-expansion-plan.md §Pillar F).**

- PsyQA Bond ≥ 3.89 (Woebot aggregate floor, same as ExTES)
- PsyQA EMPATH-κ ≥ 0.50 (κ-across-conversants under the LLM judge,
  companion-side binary vs reference-side binary on the ``counselor_response``)

**Honest architectural reading.** ExTES ran κ=0.026 under this same
pattern — reason documented at ``docs/phases/phase-8-f-close-out.md``
§10.3: Companion emits ~76% empathic turns vs reference at ~48%, and
κ hits a hard arithmetic ceiling near 0 by independence (po≈pe). The
PsyQA sweep is therefore an *architectural-signature re-test*: if the
same κ-ceiling shows under Chinese register, it confirms the pattern
is a property of rater-distribution-independence (Companion >> reference
on empathy density regardless of corpus language), NOT an artefact of
English lexical tuning. If PsyQA κ clears 0.50 while ExTES fails, the
architectural reading was wrong and the gap was corpus-specific.

**NOT A κ-vs-lexical GATE.** Same contract as ExTES LLM judge — this
file carries no "if κ < 0.3 vs lexical STOP" hard constraint. The gate
is κ-across-conversants under one judge; it either clears 0.50 or
reports an honest architectural miss.

**Shared machinery (imported, not re-implemented).**
- Session loop :func:`run_counseling_session`
- Bond administration :func:`administer_wai_sr_simulated`
- κ helper :func:`_cohens_kappa_binary`
- Simulated-user factory :func:`make_llm_simulated_user`

**Nothing modified in:**
- :mod:`counseling_depth_scorer` (ExTES/AnnoMI lexical scorer — audit baseline)
- :mod:`counseling_depth_llm_judge` (ExTES LLM scorer — English register)
- :mod:`psyqa` (PsyQA adapter — flat loader, no LLM dependency)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Callable, Iterable, Optional, Sequence

from psychology_base.applications.emotional.companion_agent import CompanionAgent
from psychology_base.applications.llm_responders import (
    SimulatedUser,
    make_llm_simulated_user,
)
from psychology_base.applications.measurement.wai_sr import score_wai_sr
from psychology_base.eval.benchmarks.public_datasets.counseling_depth_scorer import (
    _cohens_kappa_binary,
    _lexical_empathy_score,
    run_counseling_session,
)
from psychology_base.eval.benchmarks.public_datasets.counseling_depth_llm_judge import (
    EmpathyJudge,
    ParsedEmpathyJudgment,
    parse_empathy_judgment,
)
from psychology_base.eval.benchmarks.public_datasets.esconv_wai_sr_harness import (
    administer_wai_sr_simulated,
)
from psychology_base.eval.benchmarks.public_datasets.psyqa import PsyQAExample
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, TaskTier


# ---------------------------------------------------------------------------
# Gate floors (same as ExTES — plan contract)
# ---------------------------------------------------------------------------


_WOEBOT_BOND_FLOOR: float = 3.89
_EMPATH_KAPPA_FLOOR: float = 0.50


# ---------------------------------------------------------------------------
# Chinese-register prompt scaffolding
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT_ZH = (
    "你是一位资深的心理咨询督导，对单条咨询回复的共情质量做分级评价。"
    "共情回复能够回应来访者当下的感受、传递'陪伴/在场'或'接纳/认可'，"
    "或是以让人感到'被看见'的方式回映来访者的处境。非共情回复可能是"
    "事实性的、指导性的、回避性的或答非所问——礼貌≠共情。请严格依据"
    "回复本身与来访者前一轮话语的衔接来评价这一条回复。"
)


# Four Chinese-register few-shot anchors — authored in PsyQA counseling
# voice but NOT drawn from the test shim (zero data leakage). Covers
# clearly empathic (reflection + presence), clearly non-empathic
# (directive advice without acknowledgment), ambiguous (polite but
# generic), and empathic-via-reflection-only.
_FEWSHOT_ANCHORS_ZH: tuple[tuple[str, str, str, float], ...] = (
    (
        "我最近每天晚上都会哭，但在公司我不能告诉任何人。",
        "听到这里心里特别酸——夜里独自哭、白天还要装作没事，这种"
        "两面的日子真的太孤单了。我在这里陪着你。愿意多说说这段时间"
        "一个人扛着的感觉是什么样子的吗？",
        "yes",
        0.95,
    ),
    (
        "我上周被公司裁员了，不知道该怎么办。",
        "建议你尽快更新简历，立刻开始投简历。现在招聘机会其实不少，"
        "多投几家就有希望。",
        "no",
        0.90,
    ),
    (
        "我妈妈的阿尔茨海默症越来越严重了，感觉我在一点点失去她。",
        "我明白你的感受。谢谢你愿意告诉我。可以再多说一点你现在的情况吗？",
        "ambiguous",
        0.70,
    ),
    (
        "我总是因为一些小事对伴侣发脾气，然后又特别讨厌自己这样。",
        "听起来你其实已经很累了——那些'小事'背后似乎还压着别的什么。"
        "愿意说说最近还有什么让你喘不过气的事吗？",
        "yes",
        0.90,
    ),
)


_USER_PROMPT_TEMPLATE_ZH = (
    "来访者前一轮话语（作为上下文）：\n"
    "{context}\n\n"
    "示例：\n"
    "{examples}\n\n"
    "需要评价的回复：\n"
    "{reply}\n\n"
    "请严格按以下两行格式作答，不要输出任何其他内容：\n"
    "EMPATHIC=<yes|no|ambiguous>\n"
    "CONFIDENCE=<0.0..1.0>\n"
)


def _format_examples_zh() -> str:
    """Render the Chinese few-shot block (same order every call — deterministic)."""
    rendered: list[str] = []
    for client, reply, label, conf in _FEWSHOT_ANCHORS_ZH:
        rendered.append(
            f"- 来访者：\"{client}\"\n"
            f"  咨询师回复：\"{reply}\"\n"
            f"  EMPATHIC={label}\n"
            f"  CONFIDENCE={conf}\n"
        )
    return "\n".join(rendered)


def build_empathy_prompt_zh(
    *,
    turn: str,
    context: Sequence[str],
) -> list[ChatMessage]:
    """Assemble the Chinese two-message prompt scoring one reply turn."""
    ctx = "\n".join(c.strip() for c in context if c and c.strip())
    if not ctx:
        ctx = "（无前置话语——请根据回复本身独立评价）"
    user = _USER_PROMPT_TEMPLATE_ZH.format(
        context=ctx,
        examples=_format_examples_zh(),
        reply=turn.strip(),
    )
    return [
        ChatMessage(role="system", content=_SYSTEM_PROMPT_ZH),
        ChatMessage(role="user", content=user),
    ]


# ---------------------------------------------------------------------------
# Judge wrappers (reuse ParsedEmpathyJudgment + parse_empathy_judgment from
# the ExTES judge — the parsing contract is language-agnostic because the
# response format EMPATHIC=... CONFIDENCE=... is ASCII)
# ---------------------------------------------------------------------------


def make_llm_empathy_judge_zh(
    router: LLMRouter,
    *,
    tier: TaskTier = TaskTier.SIMPLE,
    use_cache: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 32,
) -> EmpathyJudge:
    """Build a Chinese-register :class:`EmpathyJudge` closure routed through ``router``.

    Mirror of :func:`make_llm_empathy_judge` with the Chinese prompt
    scaffolding substituted. The response format contract (``EMPATHIC=...
    CONFIDENCE=...``) is ASCII, so the shared parser handles both.
    """

    def _judge(turn: str, context: Sequence[str]) -> ParsedEmpathyJudgment:
        prompt = build_empathy_prompt_zh(turn=turn, context=context)
        try:
            completion = router.chat(
                prompt,
                tier=tier,
                max_tokens=max_tokens,
                temperature=temperature,
                use_cache=use_cache,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return ParsedEmpathyJudgment(
                label="ambiguous",
                confidence=0.5,
                raw_response=f"[judge_error: {exc}]",
                unresolved=True,
            )
        return parse_empathy_judgment(completion.content)

    return _judge


# ---------------------------------------------------------------------------
# Simulated PsyQA help-seeker (Chinese register)
# ---------------------------------------------------------------------------


_PSYQA_SEEKER_PERSONA_ZH: str = (
    "你是一个来向心理陪伴者（陪伴AI）求助的真实用户。你最近正在经历一些困扰，"
    "愿意开口说出心里的话，但也会观察对方是否真的在听。请用自然、短句"
    "的中文表达（一句或两句即可），不要切换成咨询师或治疗师的口吻。"
    "如果对方的话让你感到被看见，你可以稍微打开自己；如果对方的话让你"
    "觉得被忽视或被评判，你可以直接表达不舒服。"
)


def make_psyqa_seeker(
    router: LLMRouter,
    example: PsyQAExample,
    *,
    persona_prompt: str = _PSYQA_SEEKER_PERSONA_ZH,
    tier: TaskTier = TaskTier.SIMPLE,
    max_tokens: Optional[int] = 200,
    temperature: Optional[float] = 0.7,
) -> SimulatedUser:
    """Build a :class:`SimulatedUser` whose first line is the PsyQA question + description."""
    opener = example.seeker_opening.strip()
    if not opener:
        raise ValueError(
            f"PsyQA example {example.example_id!r} has no seeker opening"
        )
    keywords = (example.keywords or "").strip()
    scenario_bits = [
        "你正在用中文向陪伴AI说出你的困扰。保持第一人称、保持当事人视角，"
        "不要切换到咨询师角度。"
    ]
    if keywords:
        scenario_bits.append(f"你目前的主要困扰方向：{keywords}。")
    scenario = " ".join(scenario_bits)
    return make_llm_simulated_user(
        router,
        persona_prompt=persona_prompt,
        scenario_prompt=scenario,
        opening_user_utterance=opener,
        tier=tier,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Per-dialog + aggregate report shapes (mirror LLMExTESReport structure
# so JSON consumers can read both outputs with the same schema)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMPsyQADialogResult:
    """One PsyQA example scored end-to-end via LLM semantic judge."""

    example_id: str
    bond: Optional[float]
    handoff_triggered: bool
    companion_turns: tuple[str, ...]
    reference_response: str
    companion_empathy_scores: tuple[float, ...]
    companion_empathy_binary_rate: float  # mean across companion turns
    reference_empathy_score: float
    reference_empathy_binary: int
    companion_empathy_binary: int  # dialog-level: 1 iff mean rate >= 0.5
    unresolved: bool  # true iff ALL companion turns + reference were unresolved


@dataclass(frozen=True)
class LLMPsyQAReport:
    """Aggregate report over the PsyQA LLM-judge scoring sweep."""

    per_dialog: tuple[LLMPsyQADialogResult, ...]
    judge_routing: str = ""
    n_unresolved: int = 0
    unresolved_dialog_ids: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return len(self.per_dialog)

    @property
    def bond_scores(self) -> tuple[float, ...]:
        return tuple(d.bond for d in self.per_dialog if d.bond is not None)

    @property
    def mean_bond(self) -> float:
        scores = self.bond_scores
        return mean(scores) if scores else 0.0

    @property
    def resolved_dialogs(self) -> tuple[LLMPsyQADialogResult, ...]:
        return tuple(d for d in self.per_dialog if not d.unresolved)

    @property
    def empath_kappa(self) -> float:
        """κ-across-conversants under the Chinese LLM judge, resolved dialogs only."""
        resolved = self.resolved_dialogs
        if not resolved:
            return 0.0
        companion_labels = [d.companion_empathy_binary for d in resolved]
        reference_labels = [d.reference_empathy_binary for d in resolved]
        return _cohens_kappa_binary(companion_labels, reference_labels)

    @property
    def mean_llm_judge_companion_empathy_binary_rate(self) -> float:
        if not self.per_dialog:
            return 0.0
        return mean(d.companion_empathy_binary_rate for d in self.per_dialog)

    @property
    def mean_llm_judge_reference_empathy_binary_rate(self) -> float:
        if not self.per_dialog:
            return 0.0
        return mean(d.reference_empathy_binary for d in self.per_dialog)

    @property
    def excluded_handoffs(self) -> int:
        return sum(1 for d in self.per_dialog if d.handoff_triggered)

    def meets_bond_gate(self, floor: float = _WOEBOT_BOND_FLOOR) -> bool:
        return self.n > 0 and self.mean_bond >= floor

    def meets_kappa_gate(self, floor: float = _EMPATH_KAPPA_FLOOR) -> bool:
        return self.n > 0 and self.empath_kappa >= floor


# ---------------------------------------------------------------------------
# Scorer entry-point (parallel surface to score_extes_sweep_llm)
# ---------------------------------------------------------------------------


def score_psyqa_sweep_llm(
    examples: Iterable[PsyQAExample],
    *,
    system_factory: Callable[[], object],
    agent_factory: Callable[[object, PsyQAExample], CompanionAgent],
    seeker_factory: Callable[[PsyQAExample], SimulatedUser],
    judge: EmpathyJudge,
    max_turns_per_dialog: int = 4,
    base_timestamp: Optional[datetime] = None,
    seconds_per_turn: float = 30.0,
    judge_routing: str = "",
    progress_sink: Optional[Callable[[int, LLMPsyQADialogResult], None]] = None,
) -> LLMPsyQAReport:
    """Score a PsyQA sweep: Bond + Chinese LLM-judge empathy across Companion+Reference.

    Fresh ``CompanionSystem`` + fresh ``CompanionAgent`` per example —
    same contract as the ExTES scorer. The judge is called with the
    seeker's immediately-preceding utterance as context so semantic
    rating happens *in situ*.
    """
    base = base_timestamp or datetime.now(timezone.utc)
    per_dialog: list[LLMPsyQADialogResult] = []
    unresolved_ids: list[str] = []
    for idx, example in enumerate(examples):
        system = system_factory()
        agent = agent_factory(system, example)
        seeker = seeker_factory(example)
        start = base + timedelta(days=idx)
        transcript = run_counseling_session(
            conversation_id=example.example_id,
            opening_utterance=example.seeker_opening,
            companion_agent=agent,
            seeker=seeker,
            max_turns=max_turns_per_dialog,
            starting_timestamp=start,
            seconds_per_turn=seconds_per_turn,
        )

        # Walk the transcript and pair each companion turn with the
        # immediately-preceding seeker utterance for judge context.
        turns = transcript.turns
        companion_turns: list[str] = []
        companion_scores: list[float] = []
        companion_binaries: list[int] = []
        all_unresolved_turns = True
        last_seeker: Optional[str] = None
        for t in turns:
            if t.speaker == "seeker":
                last_seeker = t.utterance
                continue
            # Companion turn — rate it with the judge.
            companion_turns.append(t.utterance)
            context = [last_seeker] if last_seeker else []
            judgment = judge(t.utterance, context)
            if not judgment.unresolved:
                all_unresolved_turns = False
            if judgment.label == "yes":
                cont = judgment.confidence
            elif judgment.label == "no":
                cont = 1.0 - judgment.confidence
            else:
                cont = 0.5
            companion_scores.append(cont)
            companion_binaries.append(1 if cont >= 0.5 else 0)

        if companion_turns:
            companion_mean_rate = mean(companion_binaries)
        else:
            companion_mean_rate = 0.0
            all_unresolved_turns = True
        companion_dialog_binary = 1 if companion_mean_rate >= 0.5 else 0

        # Rate the reference counselor response — context = seeker opener.
        reference_judgment = judge(
            example.counselor_response, [example.seeker_opening]
        )
        if reference_judgment.label == "yes":
            reference_score = reference_judgment.confidence
        elif reference_judgment.label == "no":
            reference_score = 1.0 - reference_judgment.confidence
        else:
            reference_score = 0.5
        reference_binary = 1 if reference_score >= 0.5 else 0

        dialog_unresolved = all_unresolved_turns and reference_judgment.unresolved
        if dialog_unresolved:
            unresolved_ids.append(example.example_id)

        bond: Optional[float]
        if transcript.handoff_triggered or not companion_turns:
            bond = None
        else:
            admin_ts = start + timedelta(
                seconds=seconds_per_turn * max_turns_per_dialog + 60.0
            )
            try:
                rs = administer_wai_sr_simulated(
                    seeker,
                    user_id=f"psyqa-llm-{example.example_id}",
                    administered_at_seconds=admin_ts.timestamp(),
                    subject_noun="陪伴AI",
                )
                bond = score_wai_sr(rs).bond
            except Exception:
                bond = None

        result = LLMPsyQADialogResult(
            example_id=example.example_id,
            bond=bond,
            handoff_triggered=transcript.handoff_triggered,
            companion_turns=tuple(companion_turns),
            reference_response=example.counselor_response,
            companion_empathy_scores=tuple(companion_scores),
            companion_empathy_binary_rate=companion_mean_rate,
            reference_empathy_score=reference_score,
            reference_empathy_binary=reference_binary,
            companion_empathy_binary=companion_dialog_binary,
            unresolved=dialog_unresolved,
        )
        per_dialog.append(result)
        if progress_sink is not None:
            progress_sink(idx, result)

    return LLMPsyQAReport(
        per_dialog=tuple(per_dialog),
        judge_routing=judge_routing,
        n_unresolved=len(unresolved_ids),
        unresolved_dialog_ids=tuple(unresolved_ids),
    )


# ---------------------------------------------------------------------------
# Judge-vs-lexical κ diagnostic (NOT a halt gate — reported only, same
# contract as the English scorer)
# ---------------------------------------------------------------------------


def judge_vs_lexical_kappa_psyqa(
    report: LLMPsyQAReport,
    examples: Sequence[PsyQAExample],
) -> float:
    """Diagnostic κ comparing LLM-judge binary labels vs lexical binary labels.

    The lexical scorer (``_lexical_empathy_score`` from
    :mod:`counseling_depth_scorer`) is English-tuned, so on Chinese
    PsyQA companion turns it is expected to fire poorly — the diagnostic
    is kept for symmetry with the ExTES surface but is NOT the gate.
    """
    by_id = {d.example_id: d for d in report.per_dialog}
    judge_labels: list[int] = []
    lex_labels: list[int] = []
    for ex in examples:
        d = by_id.get(ex.example_id)
        if d is None or d.unresolved:
            continue
        if not d.companion_turns:
            continue
        lex_scores = [_lexical_empathy_score(t) for t in d.companion_turns]
        lex_mean = mean(lex_scores)
        lex_binary = 1 if lex_mean >= (1.0 / 3.0) else 0
        judge_labels.append(d.companion_empathy_binary)
        lex_labels.append(lex_binary)
    if not judge_labels:
        return 0.0
    return _cohens_kappa_binary(judge_labels, lex_labels)


__all__ = [
    "LLMPsyQADialogResult",
    "LLMPsyQAReport",
    "build_empathy_prompt_zh",
    "judge_vs_lexical_kappa_psyqa",
    "make_llm_empathy_judge_zh",
    "make_psyqa_seeker",
    "score_psyqa_sweep_llm",
]
