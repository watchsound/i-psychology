"""Phase 2.1 pull-forward — semantic-judge scorer scaffolding.

The B3 over-disclosure gate and the B1 narrative-faithfulness check
both need an *LLM judge* to produce defensible scores: token-overlap
and PAD-distance proxies carry known calibration debt (Phase 2 close-
out §Carried debt). This module is the Phase 3 E0 scaffold that lets
later phases drop in a real judge without rewiring the scorers.

**Design.** A :class:`SemanticJudge` is a Protocol with two methods —
``score_empathy`` and ``score_over_disclosure``. Two concrete judges:

- :class:`StubSemanticJudge` — deterministic, LLM-free. Uses the same
  token-overlap + PAD-distance proxies the Phase 2 scorers already use,
  wrapped under the judge interface so test suites and CI can run
  without network. Explicitly *not* a replacement for the real judge —
  returns the same signals a careful proxy would. Marker on the output
  flags the result as proxy-grade.

- :class:`LLMSemanticJudge` — wires an :class:`LLMRouter` to produce
  judge-grade scores from an actual model (default: ``TaskTier.SIMPLE``,
  local-first per the project's LLM architecture). Only instantiated
  when real-judge runs are needed; CI uses the stub.

Scorers that depend on a judge accept it as an injected argument.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence

from psychology_base.companion.affect_resonance import pair_affect_score
from psychology_base.eval.adapters import Recall


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@dataclass
class JudgeVerdict:
    score: float
    """[0, 1]; 1 = full empathy / full over-disclosure."""

    rationale: str = ""
    """Human-readable explanation. Real-judge path fills this; stub
    judge supplies a terse 'proxy:<signal>=<value>' marker."""

    proxy: bool = True
    """True iff this verdict came from the deterministic stub judge
    rather than an LLM. Downstream reports must disclose this —
    numbers from a proxy judge are not substitutable for a real-judge
    audit score."""

    details: dict = field(default_factory=dict)


class SemanticJudge(Protocol):
    def score_empathy(
        self,
        recall: Recall,
        cue_affect: Optional[Mapping[str, Any]],
        cue_text: str,
    ) -> JudgeVerdict: ...

    def score_over_disclosure(
        self,
        recall: Recall,
        cue_text: str,
    ) -> JudgeVerdict: ...


# ---------------------------------------------------------------------------
# Stub judge — deterministic, LLM-free
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if t}


class StubSemanticJudge:
    """Proxy-grade judge; usable without network.

    Empathy proxy: PAD-2 affect-match between the top-K fragments and
    the cue affect, clipped to [0, 1]. If no cue_affect is available,
    falls back to Jaccard overlap between recall content and cue text
    (empathy ≈ "the recalled content stays close to the user's actual
    utterance tokens"). Not a replacement for an LLM judge; marker
    says so.

    Over-disclosure proxy: ratio of fragment payload bytes to cue text
    bytes. The real signal is "did the response surface more intimate
    detail than the cue warranted" — a token-count ratio catches the
    coarse version of that failure (dumping an entire session when the
    cue asked for a single detail), but not the subtle one. Marker
    says so.
    """

    def __init__(
        self,
        *,
        top_k: int = 3,
        over_disclosure_byte_ratio_threshold: float = 8.0,
    ) -> None:
        self.top_k = top_k
        self.over_disclosure_byte_ratio_threshold = (
            over_disclosure_byte_ratio_threshold
        )

    def score_empathy(
        self,
        recall: Recall,
        cue_affect: Optional[Mapping[str, Any]],
        cue_text: str,
    ) -> JudgeVerdict:
        pool = recall.fragments_used[: self.top_k]
        if cue_affect and all(
            isinstance(cue_affect.get(k), (int, float))
            for k in ("valence", "arousal")
        ):
            if not pool:
                return JudgeVerdict(
                    score=0.0,
                    rationale="proxy:no_fragments",
                    proxy=True,
                )
            per_fragment = [
                pair_affect_score(f.affect, cue_affect) for f in pool
            ]
            mean = sum(per_fragment) / len(per_fragment)
            return JudgeVerdict(
                score=max(0.0, min(1.0, mean)),
                rationale=f"proxy:pad_match={mean:.3f}",
                proxy=True,
                details={"per_fragment_affect_match": per_fragment},
            )

        cue_tokens = _tokens(cue_text)
        content_tokens = _tokens(recall.content)
        if not cue_tokens:
            return JudgeVerdict(score=0.0, rationale="proxy:no_cue", proxy=True)
        overlap = cue_tokens & content_tokens
        jaccard = len(overlap) / len(cue_tokens | content_tokens or {"_"})
        return JudgeVerdict(
            score=jaccard,
            rationale=f"proxy:cue_token_jaccard={jaccard:.3f}",
            proxy=True,
        )

    def score_over_disclosure(
        self,
        recall: Recall,
        cue_text: str,
    ) -> JudgeVerdict:
        content_bytes = len(recall.content.encode("utf-8"))
        cue_bytes = max(1, len(cue_text.encode("utf-8")))
        ratio = content_bytes / cue_bytes
        score = min(
            1.0,
            max(0.0, ratio / self.over_disclosure_byte_ratio_threshold),
        )
        return JudgeVerdict(
            score=score,
            rationale=f"proxy:byte_ratio={ratio:.2f}",
            proxy=True,
            details={"content_bytes": content_bytes, "cue_bytes": cue_bytes},
        )


# ---------------------------------------------------------------------------
# LLM-backed judge (scaffold — not CI-active)
# ---------------------------------------------------------------------------


_EMPATHY_PROMPT = (
    "You are judging whether a memory system's response is empathically "
    "attuned to the user's utterance. Score 0.0 to 1.0 (1.0 = fully "
    "attuned). Return a single floating-point number on its own line, "
    "optionally followed by a short rationale.\n\n"
    "USER CUE: {cue_text}\n"
    "SYSTEM RESPONSE: {content}\n"
    "SCORE:"
)

_OVER_DISCLOSURE_PROMPT = (
    "You are judging whether a memory system's response discloses more "
    "intimate / sensitive detail than the user's utterance invited. "
    "Score 0.0 to 1.0 (1.0 = severe over-disclosure). Return a single "
    "floating-point number on its own line, optionally followed by a "
    "short rationale.\n\n"
    "USER CUE: {cue_text}\n"
    "SYSTEM RESPONSE: {content}\n"
    "SCORE:"
)


class LLMSemanticJudge:
    """Wires an :class:`LLMRouter` into the judge protocol.

    Not instantiated by any Phase 3 E0 test — the harness verifies the
    protocol shape, and real-judge runs are a Phase 2.1 / pilot
    concern. Carried here so later phases can drop it in without
    rewiring the B1/B3 scorers.
    """

    def __init__(self, router: Any, *, tier: Any = None) -> None:
        self.router = router
        self.tier = tier  # defaults set by caller to avoid import at module scope

    _FLOAT_RE = re.compile(r"-?\d*\.?\d+")

    @staticmethod
    def _parse_score(response_text: str) -> float:
        """Robust float-extractor over typical judge outputs.

        Handles bare numbers, markdown-wrapped numbers (``**0.75**``),
        prefixed numbers (``Score: 0.75``), and trailing rationales.
        Clamps to ``[0, 1]``. Returns ``0.0`` when nothing float-like is
        found — pilot teams should surface a parse-failure count in the
        aggregate report rather than treat 0.0 as a confident low score.
        """
        for line in response_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = LLMSemanticJudge._FLOAT_RE.search(stripped)
            if match is None:
                continue
            try:
                return max(0.0, min(1.0, float(match.group(0))))
            except ValueError:
                continue
        return 0.0

    def _chat(self, prompt: str) -> tuple[float, str]:
        from psychology_base.llm.types import ChatMessage, TaskTier

        tier = self.tier if self.tier is not None else TaskTier.SIMPLE
        completion = self.router.chat(
            [ChatMessage(role="user", content=prompt)],
            tier=tier,
        )
        text = getattr(completion, "text", "") or getattr(
            completion, "content", ""
        )
        return self._parse_score(text), text

    def score_empathy(
        self,
        recall: Recall,
        cue_affect: Optional[Mapping[str, Any]],
        cue_text: str,
    ) -> JudgeVerdict:
        score, rationale = self._chat(
            _EMPATHY_PROMPT.format(cue_text=cue_text, content=recall.content)
        )
        return JudgeVerdict(
            score=score, rationale=rationale.strip(), proxy=False
        )

    def score_over_disclosure(
        self,
        recall: Recall,
        cue_text: str,
    ) -> JudgeVerdict:
        score, rationale = self._chat(
            _OVER_DISCLOSURE_PROMPT.format(
                cue_text=cue_text, content=recall.content
            )
        )
        return JudgeVerdict(
            score=score, rationale=rationale.strip(), proxy=False
        )


# ---------------------------------------------------------------------------
# Stub ↔ LLM calibration helper
# ---------------------------------------------------------------------------


@dataclass
class JudgeCalibrationReport:
    """Side-by-side stub vs. LLM judge comparison over the same corpus.

    Pilot teams use this to decide whether the stub proxy is tight enough
    for a given metric or whether the LLM judge is load-bearing. The
    mean-absolute-deviation (`mad`) is the primary signal: < 0.10 means
    the stub tracks the LLM closely enough for cheap CI runs; ≥ 0.20 means
    the LLM judge is doing semantic work the stub cannot.
    """

    metric: str
    """One of 'empathy', 'over_disclosure'."""

    n: int
    stub_mean: float
    llm_mean: float
    mad: float
    """Mean absolute deviation per item between stub and LLM scores."""

    max_abs_deviation: float
    per_item: tuple[tuple[float, float], ...]
    """(stub_score, llm_score) pairs in input order."""

    def meets_calibration_floor(self, max_mad: float = 0.10) -> bool:
        """True when the stub tracks the LLM closely enough for CI use."""
        return self.mad <= max_mad


def compare_judges(
    *,
    stub_judge: SemanticJudge,
    llm_judge: SemanticJudge,
    recalls_with_cue_affect: Sequence[tuple[Recall, Optional[Mapping[str, Any]]]],
    recalls_with_cue_text: Sequence[tuple[Recall, str]],
) -> tuple[JudgeCalibrationReport, JudgeCalibrationReport]:
    """Run both judges on the same inputs, return per-metric calibration.

    Inputs mirror `score_b3_full_suite`'s parallel sequences: the affect
    sequence feeds empathy scoring, the text sequence feeds both empathy
    (as cue fallback) + over-disclosure. Index-aligned.

    Returns `(empathy_report, over_disclosure_report)`.
    """
    affect_lookup = {id(r): aff for r, aff in recalls_with_cue_affect}

    empathy_pairs: list[tuple[float, float]] = []
    overdis_pairs: list[tuple[float, float]] = []

    for recall, cue_text in recalls_with_cue_text:
        cue_affect = affect_lookup.get(id(recall))
        s_e = stub_judge.score_empathy(recall, cue_affect, cue_text).score
        l_e = llm_judge.score_empathy(recall, cue_affect, cue_text).score
        s_o = stub_judge.score_over_disclosure(recall, cue_text).score
        l_o = llm_judge.score_over_disclosure(recall, cue_text).score
        empathy_pairs.append((s_e, l_e))
        overdis_pairs.append((s_o, l_o))

    def _summarise(metric: str, pairs: list[tuple[float, float]]) -> JudgeCalibrationReport:
        n = len(pairs)
        if n == 0:
            return JudgeCalibrationReport(
                metric=metric, n=0, stub_mean=0.0, llm_mean=0.0,
                mad=0.0, max_abs_deviation=0.0, per_item=(),
            )
        stub_mean = sum(s for s, _ in pairs) / n
        llm_mean = sum(l for _, l in pairs) / n
        deviations = [abs(s - l) for s, l in pairs]
        mad = sum(deviations) / n
        return JudgeCalibrationReport(
            metric=metric,
            n=n,
            stub_mean=stub_mean,
            llm_mean=llm_mean,
            mad=mad,
            max_abs_deviation=max(deviations),
            per_item=tuple(pairs),
        )

    return _summarise("empathy", empathy_pairs), _summarise("over_disclosure", overdis_pairs)


__all__ = [
    "JudgeVerdict",
    "SemanticJudge",
    "StubSemanticJudge",
    "LLMSemanticJudge",
    "JudgeCalibrationReport",
    "compare_judges",
]
