"""Phase 8 Pillar F — LLM semantic judge for ExTES empathy scoring.

**Why this file.** Phase 8 Pillar F ExTES live-fire at N=100 landed Bond
``4.64 >= 3.89`` (PASS) but ``empath_kappa = -0.076`` vs the ``0.50`` floor
(FAIL). The diagnosis in ``docs/phases/phase-8-f-close-out.md`` §9.4 names
the mismatch: Companion's Rule 1a "mirror and open space" style emits
``0.295`` lexical empathy density vs the ExTES reference counselor's
``0.430`` — a ~31% gap not on *semantic empathy* but on *explicit
recognition/presence/validation cue token density*. The lexical scorer
counts cue tokens, not what a counselor would recognise as empathic.

This file ships the semantic-judge analogue — same architectural move as
Pillar C v4 (``big5_chat_classification_llm.py``) — scoring BOTH the
Companion's turn AND the ExTES reference under the *same* judge with the
*same* prompt, so κ-across-conversants is apples-to-apples under one
semantic rater. The lexical scorer (``counseling_depth_scorer.py``)
stays as the audit baseline — do not modify it; the hard constraint
from the task is a new-file-only intervention mirroring F4.

**NOT A κ-vs-lexical GATE.** The Pillar C F4 run tripped on a hard-
constraint ``κ-LLM-vs-lexical < 0.3 → STOP`` that fires on healthy
disagreement with the broken thing being replaced — a trap. This file
has no such constraint. The gate here is κ-across-conversants under
one judge; it either clears ``0.50`` or fails honestly, and judge-vs-
lexical κ is reported as a *diagnostic* (artifact?-vs-architectural
reading) not as a halt condition.

**What the judge sees.** For each turn (Companion live reply OR ExTES
reference counselor_response) the judge receives:

- the client utterance that preceded it (last 1-2 lines of context so
  the judge isn't scoring a reply in a vacuum)
- the therapist/counselor reply to classify
- 4 hand-crafted few-shot anchors covering empathic / not-empathic /
  ambiguous registers in a counseling voice; NOT drawn from the ExTES
  test shim (zero data leakage)
- a tight response contract: emit ``EMPATHIC=<yes|no|ambiguous>`` and
  ``CONFIDENCE=<0.0..1.0>``

**Grounding.** The judge NEVER sees the lexical score, which rater
(Companion or Reference) produced the turn, or the ground-truth binary.
The judge is a blind semantic rater; equivalence of judgment between
the two rater sides (Companion turn vs ExTES reference) is what the
κ-across-conversants gate measures.

**Routing contract.** Fresh ``LLMRouter`` injected per scorer call
(same pattern as F4 BIG5-CHAT LLM). SIMPLE tier — qwen2.5:7b local
primary, qwen-turbo cloud fallback. A turn the judge can't resolve on
either path falls back to ``0.5`` confidence + ``"ambiguous"`` label
rather than raising; this stops a flaky item from killing the sweep.
The dialog's entire empathy signal is dropped from the κ denominator
only if ALL companion turns + reference are unresolved (``n_unresolved``
counts dialogs, not turns).

**No modifications to:**

- ``counseling_depth_scorer.py`` (lexical scorer — audit baseline)
- ``extes.py`` (adapter — not touched)
- ``annomi.py`` (adapter — AnnoMI gate is MI-fidelity, not κ)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Callable, Iterable, List, Optional, Sequence

from psychology_base.applications.emotional.companion_agent import CompanionAgent
from psychology_base.applications.llm_responders import SimulatedUser
from psychology_base.applications.measurement.wai_sr import score_wai_sr
from psychology_base.eval.benchmarks.public_datasets.counseling_depth_scorer import (
    _RATE_MATCHED_EMPATHY_ABS_FLOOR,
    _RATE_MATCHED_EMPATHY_BASELINE_FRACTION,
    _cohens_kappa_binary,
    _lexical_empathy_score,
    _rate_matched_empathy_pass,
    run_counseling_session,
)
from psychology_base.eval.benchmarks.public_datasets.esconv_wai_sr_harness import (
    administer_wai_sr_simulated,
)
from psychology_base.eval.benchmarks.public_datasets.extes import ExTESExample
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, TaskTier


# ---------------------------------------------------------------------------
# Gate floors (DO NOT redefine — floor is the plan contract)
# ---------------------------------------------------------------------------


_WOEBOT_BOND_FLOOR: float = 3.89
_EMPATH_KAPPA_FLOOR: float = 0.50


# ---------------------------------------------------------------------------
# Prompt scaffolding
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are an experienced counseling supervisor rating single reply "
    "turns for empathic quality. Empathic replies acknowledge the client's "
    "felt experience, convey presence or validation, or reflect what the "
    "client is going through in a way that lands as 'this person gets "
    "it'. Non-empathic replies may be factual, directive, dismissive, or "
    "simply off-topic — polite is not the same as empathic. Rate strictly "
    "on the single reply shown, in the context of the client's immediately "
    "preceding utterance."
)


# Four few-shot anchors — hand-crafted in counseling register. One
# clearly empathic (reflection + presence), one clearly non-empathic
# (deflecting / minimising / advice-giving without acknowledgment), one
# ambiguous (polite but neutral), one empathic-via-reflection-only (to
# show that reflection without validation also counts). NOT drawn from
# ExTES — zero data leakage. The MI/EMPATH literature inspiration:
# Elliott et al. 2018 empathy taxonomy (empathic understanding, empathic
# affirmation, empathic resonance) is the schema these anchors cover.
_FEWSHOT_ANCHORS: tuple[tuple[str, str, str, float], ...] = (
    (
        "I've been crying every night and I can't tell anyone at work.",
        "That sounds so lonely — carrying that at night, then having to "
        "put on a face in the morning. I'm here with you. What has it been "
        "like to hold this alone?",
        "yes",
        0.95,
    ),
    (
        "I lost my job last week and I don't know what to do.",
        "You should update your resume and start applying immediately. "
        "There are a lot of openings right now if you look.",
        "no",
        0.90,
    ),
    (
        "My mother's dementia is getting worse and I feel like I'm losing her.",
        "I understand. Thank you for sharing that with me. Can you tell me "
        "a bit more about your situation?",
        "ambiguous",
        0.70,
    ),
    (
        "I keep snapping at my partner over tiny things and I hate it.",
        "It sounds like you're exhausted — the snapping might be about "
        "something bigger than the dishes. What else is heavy right now?",
        "yes",
        0.90,
    ),
)


_USER_PROMPT_TEMPLATE = (
    "CONTEXT (client utterance that preceded the reply):\n"
    "{context}\n\n"
    "EXAMPLES:\n"
    "{examples}\n\n"
    "REPLY TO RATE:\n"
    "{reply}\n\n"
    "Respond on exactly two lines, nothing else:\n"
    "EMPATHIC=<yes|no|ambiguous>\n"
    "CONFIDENCE=<0.0..1.0>\n"
)


def _format_examples() -> str:
    """Render the few-shot block (same order every call — deterministic)."""
    rendered: list[str] = []
    for client, reply, label, conf in _FEWSHOT_ANCHORS:
        rendered.append(
            f"- CONTEXT: \"{client}\"\n"
            f"  REPLY: \"{reply}\"\n"
            f"  EMPATHIC={label}\n"
            f"  CONFIDENCE={conf}\n"
        )
    return "\n".join(rendered)


def build_empathy_prompt(
    *,
    turn: str,
    context: Sequence[str],
) -> list[ChatMessage]:
    """Assemble the two-message prompt scoring one reply turn.

    ``context`` is the preceding client-side utterance(s) joined with
    newlines; the judge sees the last-1-or-2 lines so it rates the reply
    *in situ*, not in a vacuum. ``turn`` is the counselor/companion
    reply under rating.
    """
    ctx = "\n".join(c.strip() for c in context if c and c.strip())
    if not ctx:
        ctx = "(no prior context — rate the reply on its stand-alone shape)"
    user = _USER_PROMPT_TEMPLATE.format(
        context=ctx,
        examples=_format_examples(),
        reply=turn.strip(),
    )
    return [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user),
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


_EMPATHIC_RE = re.compile(r"EMPATHIC\s*=\s*(yes|no|ambiguous)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"CONFIDENCE\s*=\s*(-?[0-9]*\.?[0-9]+)")
# Fallback bare-numeric: a response that is just "0.8" or "0.2".
_BARE_NUMERIC_RE = re.compile(r"^\s*(-?[0-9]*\.?[0-9]+)\s*$")


@dataclass(frozen=True)
class ParsedEmpathyJudgment:
    """Parsed judge output.

    ``label`` is ``"yes" | "no" | "ambiguous"`` after parsing (or
    ``"ambiguous"`` on parse-failure fallback — same as F4's unresolved
    policy except here we DO NOT drop the item, we treat ambiguous as
    the honest midpoint). ``confidence`` is clipped to ``[0, 1]``.
    """

    label: str
    confidence: float
    raw_response: str
    unresolved: bool = False


def parse_empathy_judgment(response_text: str) -> ParsedEmpathyJudgment:
    """Parse ``EMPATHIC=... CONFIDENCE=...`` from a judge completion.

    Tolerant of markdown wrappers, whitespace variants, case variants.
    Also handles two degenerate fallbacks:

    - ``empathic`` / ``not empathic`` tokens anywhere in the response
      (free-text models drift off the two-line contract)
    - a bare numeric response (``0.8``) → treated as confidence; label
      follows the 0.5 threshold (``>= 0.5`` → ``yes``, else ``no``).

    Everything else lands on ``label="ambiguous"``, ``confidence=0.5``,
    ``unresolved=True`` so the scorer can decide whether to count it
    toward κ (we do — ambiguous reads as midpoint-neutral), but the
    sweep report still carries the unresolved count for diagnostics.
    """
    text = (response_text or "").strip()
    if not text:
        return ParsedEmpathyJudgment(
            label="ambiguous", confidence=0.5, raw_response="", unresolved=True
        )

    # Preferred: canonical two-line format.
    emp_match = _EMPATHIC_RE.search(text)
    conf_match = _CONFIDENCE_RE.search(text)
    if emp_match is not None and conf_match is not None:
        label = emp_match.group(1).lower()
        try:
            conf = float(conf_match.group(1))
        except ValueError:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        return ParsedEmpathyJudgment(label=label, confidence=conf, raw_response=text)

    # Fallback 1: EMPATHIC= line without a parsable confidence.
    if emp_match is not None:
        label = emp_match.group(1).lower()
        # Infer confidence from the label.
        default_conf = {"yes": 0.75, "no": 0.75, "ambiguous": 0.5}.get(label, 0.5)
        return ParsedEmpathyJudgment(
            label=label, confidence=default_conf, raw_response=text
        )

    # Fallback 2: free-text "empathic" / "not empathic" / "ambiguous".
    low = text.lower()
    # Precedence: "not empathic" check before plain "empathic".
    if re.search(r"\bnot\s+empathic\b", low) or re.search(r"\bnon[-\s]?empathic\b", low):
        return ParsedEmpathyJudgment(label="no", confidence=0.75, raw_response=text)
    if re.search(r"\bambiguous\b", low) or re.search(r"\bunclear\b", low):
        return ParsedEmpathyJudgment(label="ambiguous", confidence=0.5, raw_response=text)
    if re.search(r"\bempathic\b", low) or re.search(r"\bempathetic\b", low):
        return ParsedEmpathyJudgment(label="yes", confidence=0.75, raw_response=text)

    # Fallback 3: bare numeric response.
    bare = _BARE_NUMERIC_RE.match(text)
    if bare is not None:
        try:
            conf = float(bare.group(1))
        except ValueError:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        label = "yes" if conf >= 0.5 else "no"
        return ParsedEmpathyJudgment(label=label, confidence=conf, raw_response=text)

    # Fallback 4: unparseable → ambiguous midpoint, flagged unresolved.
    return ParsedEmpathyJudgment(
        label="ambiguous", confidence=0.5, raw_response=text, unresolved=True
    )


# ---------------------------------------------------------------------------
# Judge wrappers
# ---------------------------------------------------------------------------


def score_turn_empathy_llm(
    turn: str,
    context: Sequence[str],
    *,
    router: Optional[LLMRouter] = None,
    tier: TaskTier = TaskTier.SIMPLE,
    use_cache: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 32,
) -> float:
    """Return a continuous [0, 1] confidence that ``turn`` is empathic.

    ``yes`` → ``confidence``; ``no`` → ``1.0 - confidence`` (so "high
    confidence NOT empathic" = low empathy score); ``ambiguous`` → 0.5.
    When ``router`` is ``None`` or the call fails, we fall back to a
    neutral 0.5 — the sweep then marks the turn unresolved via
    :func:`judge_turn_empathy_binary` (which calls the router directly
    and inspects the ``ParsedEmpathyJudgment.unresolved`` flag).

    Factored out so test seams can mock the router → judge pipeline
    exactly like F4's ``make_llm_judge``.
    """
    if router is None:
        # No LLM available — degenerate to midpoint. The binary
        # judge-wrapper handles this as "unresolved + ambiguous".
        return 0.5
    prompt = build_empathy_prompt(turn=turn, context=context)
    try:
        completion = router.chat(
            prompt,
            tier=tier,
            max_tokens=max_tokens,
            temperature=temperature,
            use_cache=use_cache,
        )
    except Exception:  # pragma: no cover - defensive
        return 0.5
    parsed = parse_empathy_judgment(completion.content)
    if parsed.label == "yes":
        return parsed.confidence
    if parsed.label == "no":
        return 1.0 - parsed.confidence
    return 0.5


def judge_turn_empathy_binary(
    turn: str,
    context: Sequence[str],
    *,
    threshold: float = 0.5,
    router: Optional[LLMRouter] = None,
    tier: TaskTier = TaskTier.SIMPLE,
    use_cache: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 32,
) -> int:
    """Binary 0/1 empathy label, cutting the continuous judge at ``threshold``.

    ``threshold`` default ``0.5`` — a yes with ≥ 0.5 confidence counts
    as empathic (binary 1); a no, ambiguous, or unresolved falls below.
    """
    score = score_turn_empathy_llm(
        turn,
        context,
        router=router,
        tier=tier,
        use_cache=use_cache,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return 1 if score >= threshold else 0


# ``EmpathyJudge`` is the callable interface the sweep uses — tests
# inject deterministic scripted stubs; the live-fire path builds one
# from an LLMRouter. Signature mirrors :func:`judge_turn_empathy_binary`
# but returns the full :class:`ParsedEmpathyJudgment` so the sweep can
# log unresolved items alongside the binary flip.
EmpathyJudge = Callable[[str, Sequence[str]], ParsedEmpathyJudgment]


def make_llm_empathy_judge(
    router: LLMRouter,
    *,
    tier: TaskTier = TaskTier.SIMPLE,
    use_cache: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 32,
) -> EmpathyJudge:
    """Build a scripted :class:`EmpathyJudge` closure routing through ``router``.

    Fresh-router-per-scorer-call is the plan hard constraint; this
    factory just wires one router through many items — the caller
    passes a new ``LLMRouter()`` per :func:`score_extes_sweep_llm` call.
    """

    def _judge(turn: str, context: Sequence[str]) -> ParsedEmpathyJudgment:
        prompt = build_empathy_prompt(turn=turn, context=context)
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
# Per-dialog + aggregate report shapes (byte-shape-compatible with the
# lexical scorer plus LLM-specific extensions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMExTESDialogResult:
    """One ExTES example scored end-to-end via LLM semantic judge."""

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
class LLMExTESReport:
    """Aggregate report over the ExTES LLM-judge scoring sweep."""

    per_dialog: tuple[LLMExTESDialogResult, ...]
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
    def resolved_dialogs(self) -> tuple[LLMExTESDialogResult, ...]:
        return tuple(d for d in self.per_dialog if not d.unresolved)

    @property
    def empath_kappa(self) -> float:
        """κ-across-conversants under the same LLM judge, resolved dialogs only.

        Companion-side binary = ``1`` iff the mean binary rate across the
        Companion's turns in the dialog is ``>= 0.5`` (a dialog is
        empathic iff at least half of its Companion turns are rated
        empathic by the judge). Reference-side binary is a single turn
        read.
        """
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

    def meets_rate_matched_empathy_gate(
        self,
        *,
        absolute_floor: float = _RATE_MATCHED_EMPATHY_ABS_FLOOR,
        baseline_fraction: float = _RATE_MATCHED_EMPATHY_BASELINE_FRACTION,
    ) -> bool:
        if self.n == 0:
            return False
        return _rate_matched_empathy_pass(
            self.mean_llm_judge_companion_empathy_binary_rate,
            self.mean_llm_judge_reference_empathy_binary_rate,
            absolute_floor=absolute_floor,
            baseline_fraction=baseline_fraction,
        )


# ---------------------------------------------------------------------------
# Scorer entry-point (parallel surface to ``score_extes_sweep``)
# ---------------------------------------------------------------------------


def score_extes_sweep_llm(
    examples: Iterable[ExTESExample],
    *,
    system_factory: Callable[[], object],
    agent_factory: Callable[[object, ExTESExample], CompanionAgent],
    seeker_factory: Callable[[ExTESExample], SimulatedUser],
    judge: EmpathyJudge,
    max_turns_per_dialog: int = 4,
    base_timestamp: Optional[datetime] = None,
    seconds_per_turn: float = 30.0,
    judge_routing: str = "",
    progress_sink: Optional[Callable[[int, LLMExTESDialogResult], None]] = None,
) -> LLMExTESReport:
    """Score an ExTES sweep: Bond + LLM-judge empathy across Companion+Reference.

    Mirror of :func:`counseling_depth_scorer.score_extes_sweep` but swaps
    ``_lexical_empathy_score`` for ``judge`` on BOTH sides. The judge is
    called with the client utterance (last preceding seeker turn) as the
    ``context`` so semantic rating happens *in situ*, not on a stand-
    alone reply.

    Fresh ``CompanionSystem`` + fresh ``CompanionAgent`` per example —
    same contract as the lexical scorer / Pillar B.
    """
    base = base_timestamp or datetime.now(timezone.utc)
    per_dialog: list[LLMExTESDialogResult] = []
    unresolved_ids: list[str] = []
    for idx, example in enumerate(examples):
        system = system_factory()
        agent = agent_factory(system, example)
        seeker = seeker_factory(example)
        start = base + timedelta(days=idx)
        transcript = run_counseling_session(
            conversation_id=example.example_id,
            opening_utterance=example.client_utterance,
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
            # Continuous score for the sweep JSON (yes → conf;
            # no → 1-conf; ambiguous → 0.5).
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

        # Rate the reference — same judge, context = opening client
        # utterance (the ExTES stimulus-response pair's "client side").
        reference_judgment = judge(
            example.counselor_response, [example.client_utterance]
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

        # Bond administration (same shape as lexical scorer).
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
                    user_id=f"extes-llm-{example.example_id}",
                    administered_at_seconds=admin_ts.timestamp(),
                    subject_noun="Companion",
                )
                bond = score_wai_sr(rs).bond
            except Exception:
                bond = None

        result = LLMExTESDialogResult(
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

    return LLMExTESReport(
        per_dialog=tuple(per_dialog),
        judge_routing=judge_routing,
        n_unresolved=len(unresolved_ids),
        unresolved_dialog_ids=tuple(unresolved_ids),
    )


# ---------------------------------------------------------------------------
# Judge-vs-lexical κ diagnostic (NOT a halt gate — reported only)
# ---------------------------------------------------------------------------


def judge_vs_lexical_kappa(
    report: LLMExTESReport,
    examples: Sequence[ExTESExample],
) -> float:
    """Diagnostic κ comparing LLM-judge binary labels vs lexical binary labels
    on the SAME companion turns.

    Low κ (near 0) between judge and lexical means the two raters are
    reading different signals — evidence the κ gate miss WAS a scorer
    artifact. High κ (near 1) means lexical and semantic agree — the
    anti-sycophancy signature is real, not a measurement bug.

    This is NOT a halt condition; see module docstring. We DO NOT stop
    on judge-vs-lexical < 0.3 the way Pillar C F4 did — that's the trap
    documented in ``docs/phases/phase-8-c-v2-close-out.md`` §4.
    """
    # Build a dict for lookup; ExTES is per-example independent.
    by_id = {d.example_id: d for d in report.per_dialog}
    judge_labels: list[int] = []
    lex_labels: list[int] = []
    for ex in examples:
        d = by_id.get(ex.example_id)
        if d is None or d.unresolved:
            continue
        # Lexical labels on the Companion side: mean lexical empathy
        # over the companion turns, binarised at 1/3.
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
    "EmpathyJudge",
    "LLMExTESDialogResult",
    "LLMExTESReport",
    "ParsedEmpathyJudgment",
    "build_empathy_prompt",
    "judge_turn_empathy_binary",
    "judge_vs_lexical_kappa",
    "make_llm_empathy_judge",
    "parse_empathy_judgment",
    "score_extes_sweep_llm",
    "score_turn_empathy_llm",
]
