"""ESConv adapter — simulated-user support stream + support-strategy distribution.

**Dataset provenance.** ESConv (Liu, Zheng, Chen, et al., 2021; "Towards
Emotional Support Conversation: A Review of Benchmarks and Approaches",
ACL 2021; arXiv 2106.01144). 1,300 multi-turn conversations between a
help-seeker and a supporter across 8 emotion categories (anxiety /
depression / sadness / anger / fear / shame / disgust / jealousy — exact
set per repo README; **verify-before-cite** before any external reference).
Repo at `github.com/thu-coai/Emotional-Support-Conversation`. License —
**verify at the repo README before any derivative distribution**; the
repo README is the authoritative source.

**Pivot gate this adapter targets.** `docs/phases/phase-4b-benchmark-pivot.md`
§3 row "情感陪伴 WAI-SR Bond" — target mean WAI-SR Bond ≥ 3.8 on
simulated-user self-report across ≥50 simulated users spanning the ESConv
emotion categories. The WAI-SR instrument administration + aggregation is
wired separately (`wai_sr.py`); this adapter supplies the conversation
seeds, help-seeker ground-truth personas, and support-strategy taxonomy.

**Why no vendor-download.** Same posture as the MathDial adapter —
license-neutral and size-agnostic. Loader expects a user-provided JSONL
shim; conversion script lands separately per dataset.

**JSONL shim schema.** One conversation per line::

    {
      "conversation_id": "esc_000001",
      "emotion_type": "anxiety",
      "problem_type": "job crisis",
      "situation": "I lost my job last week and I don't know what to do.",
      "turns": [
        {"role": "seeker", "utterance": "I'm feeling really overwhelmed."},
        {"role": "supporter", "utterance": "That sounds really hard.", "strategy": "affirmation_and_reassurance"},
        ...
      ],
      "survey_score": 4.2,
      "metadata": {}
    }

The ESConv native format has a `supporter` and `seeker` role distinction
with a `strategy` tag on supporter turns. We preserve both. The optional
`survey_score` field is the help-seeker's post-conversation satisfaction
rating from the ESConv annotation protocol (not WAI-SR; this is ESConv's
own 5-point helpfulness scale — kept for calibration against our WAI-SR
simulated-user administration).

**Strategy taxonomy.** The 8-strategy ESConv taxonomy
(reflection_of_feelings / question / restatement / self_disclosure /
affirmation_and_reassurance / providing_suggestions / information /
others). Mapped to the semantic buckets our CompanionAgent could emit,
not to any CompanionAgent-side classifier — unlike MathDial we do not
have a response-move classifier for emotional support yet. Taxonomy is
exposed so a Phase 4b scorer can analyse the companion's emitted-strategy
distribution against the ESConv gold distribution (over- or
under-representation of specific strategies is a diagnostic surface).

**No LLM dependency.** Loader + taxonomy only. Simulated-user playback
and WAI-SR administration live in sibling modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Literal, Optional


class ESConvStrategy(str, Enum):
    """ESConv's 8-strategy supporter-move taxonomy."""

    REFLECTION_OF_FEELINGS = "reflection_of_feelings"
    QUESTION = "question"
    RESTATEMENT = "restatement"
    SELF_DISCLOSURE = "self_disclosure"
    AFFIRMATION_AND_REASSURANCE = "affirmation_and_reassurance"
    PROVIDING_SUGGESTIONS = "providing_suggestions"
    INFORMATION = "information"
    OTHERS = "others"


# ESConv ships eight emotion labels in the public release; we keep them as
# strings rather than an Enum because the authoritative set is in the repo
# README and the Enum maintenance cost outweighs the readability gain.
# A `verify-before-cite` note is sufficient.


@dataclass(frozen=True)
class ESConvTurn:
    """One turn in an ESConv conversation."""

    role: Literal["seeker", "supporter"]
    utterance: str
    strategy: Optional[ESConvStrategy] = None  # only on supporter turns


@dataclass(frozen=True)
class ESConvConversation:
    """One ESConv conversation = one help-seeker scenario + its support dialogue."""

    conversation_id: str
    emotion_type: str
    problem_type: str
    situation: str
    turns: tuple[ESConvTurn, ...]
    survey_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyDistributionReport:
    """Roll-up of supporter-strategy frequencies across one or more conversations.

    Used by Phase 4b scorers to compare CompanionAgent's emitted-strategy
    distribution against the ESConv gold distribution. A chi-square or KL
    divergence against this report is the measurement surface — not implemented
    here to keep the adapter dependency-light.
    """

    sampled_supporter_turns: int
    counts: dict[ESConvStrategy, int]

    def frequency(self, strategy: ESConvStrategy) -> float:
        if self.sampled_supporter_turns == 0:
            return 0.0
        return self.counts.get(strategy, 0) / self.sampled_supporter_turns


# ---------------------------------------------------------------------------
# JSONL shim loader
# ---------------------------------------------------------------------------


def iter_esconv_conversations(shim_path: Path) -> Iterator[ESConvConversation]:
    """Stream ESConv conversations from a JSONL shim file."""
    with Path(shim_path).open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            turns = tuple(_parse_turn(t, line_no, idx) for idx, t in enumerate(obj.get("turns", [])))
            survey = obj.get("survey_score")
            yield ESConvConversation(
                conversation_id=str(obj["conversation_id"]),
                emotion_type=str(obj.get("emotion_type", "")),
                problem_type=str(obj.get("problem_type", "")),
                situation=str(obj.get("situation", "")),
                turns=turns,
                survey_score=float(survey) if survey is not None else None,
                metadata=dict(obj.get("metadata", {})),
            )


def _parse_turn(raw: dict, line_no: int, idx: int) -> ESConvTurn:
    role = raw.get("role")
    if role not in ("seeker", "supporter"):
        raise ValueError(
            f"ESConv shim line {line_no} turn {idx}: role must be 'seeker' or 'supporter', got {role!r}"
        )
    strategy_raw = raw.get("strategy")
    strategy: Optional[ESConvStrategy] = None
    if strategy_raw is not None:
        if role != "supporter":
            raise ValueError(
                f"ESConv shim line {line_no} turn {idx}: 'strategy' only valid on supporter turns"
            )
        strategy = ESConvStrategy(strategy_raw)
    utterance = raw.get("utterance")
    if not isinstance(utterance, str):
        raise ValueError(
            f"ESConv shim line {line_no} turn {idx}: 'utterance' must be a string"
        )
    return ESConvTurn(role=role, utterance=utterance, strategy=strategy)


# ---------------------------------------------------------------------------
# Scorers — strategy-distribution snapshot
# ---------------------------------------------------------------------------


def summarise_strategy_distribution(
    conversations: list[ESConvConversation],
) -> StrategyDistributionReport:
    """Pool supporter-strategy frequencies across a list of conversations.

    Supporter turns without a strategy label are skipped. Student (seeker)
    turns are skipped. Suitable for both the gold-distribution snapshot (pool
    over the ESConv conversations) and the predicted-distribution snapshot
    (pool over CompanionAgent-labeled turns, if a classifier is available).
    """
    counts: dict[ESConvStrategy, int] = {}
    sampled = 0
    for convo in conversations:
        for turn in convo.turns:
            if turn.role != "supporter" or turn.strategy is None:
                continue
            counts[turn.strategy] = counts.get(turn.strategy, 0) + 1
            sampled += 1
    return StrategyDistributionReport(
        sampled_supporter_turns=sampled,
        counts=counts,
    )


def seeker_opening_utterance(convo: ESConvConversation) -> Optional[str]:
    """Return the first help-seeker utterance — the seed for a simulated-user playback.

    Returns None if no seeker turn exists (malformed shim or empty dialogue).
    Used by the simulated-user WAI-SR harness to kick off a run with an
    authentic help-seeker opener rather than a synthesised one.
    """
    for turn in convo.turns:
        if turn.role == "seeker":
            return turn.utterance
    return None
