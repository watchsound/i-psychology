"""Crisis-cue detection + human-handoff shim.

Mandated by 《办法》 + CLAUDE.md: any crisis / self-harm / severe-pathology
path must escalate to a licensed human. The AI is 能力增强器, not 专业替代者.

Scope:
- **Lexical cue bank** for v1 — fast, auditable, no false negatives on the
  obvious set. Semantic+LLM-judge replacement is a Phase 4a §6d carried-debt
  item once the LLMSemanticJudge wire-up lands.
- **Handoff shim** — writes a structured incident record to a sink callable
  the application layer provides. The sink's job is to initiate the real-world
  handoff (SMS/email the designated clinician, break-glass crisis-line route);
  the detector's job is to *force the hand-off call* synchronously on detection,
  not to deliver the message itself.

Design:
- Detection runs on every user utterance. Thresholds and the sink are fixed at
  construction time and covenant-tagged in the memory substrate — they must
  not decay out of the prompt per CLAUDE.md Principle F non-decaying surface.
- Zero-tolerance for silent false negatives on the explicit-ideation set. A
  false positive escalates unnecessarily; a false negative misses a crisis.
- The detector never *responds* to the user — that is the application layer's
  job. It only emits an incident + flips the handoff flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Optional

from psychology_base.applications.safety.compliance_floor import (
    EffectivePolicy,
    UNIVERSAL_FLOOR,
)
from psychology_base.applications.safety.crisis_lines import CrisisLine


class CrisisSeverity(str, Enum):
    NONE = "none"
    AMBIGUOUS = "ambiguous"
    EXPLICIT_IDEATION = "explicit_ideation"
    IMMINENT_DANGER = "imminent_danger"


# Lexical cue bank. v1 only — intentionally conservative on explicit-ideation
# and imminent-danger to minimise false negatives. English + Chinese parallel
# because the project targets Chinese-speaking users under 《办法》.
_IMMINENT_DANGER_CUES: tuple[str, ...] = (
    "tonight i will",
    "i have the pills",
    "i'm going to kill",
    "i am going to kill",
    "goodbye forever",
    "this is my last",
    "今晚我就",
    "我现在就要",
    "我已经准备好",
    "永别了",
    "最后一次",
)

_EXPLICIT_IDEATION_CUES: tuple[str, ...] = (
    "kill myself",
    "end my life",
    "end it all",
    "suicide",
    "want to die",
    "wish i was dead",
    "wish i were dead",
    "self-harm",
    "self harm",
    "cut myself",
    "hurt myself",
    "想死",
    "自杀",
    "结束生命",
    "活不下去",
    "想自残",
    "伤害自己",
)

_AMBIGUOUS_CUES: tuple[str, ...] = (
    "can't go on",
    "cant go on",
    "no point in living",
    "no reason to live",
    "give up on everything",
    "没意思",
    "撑不下去",
    "没有希望",
    "一了百了",
)


@dataclass(frozen=True)
class CrisisIncident:
    """One detection event — passed to the handoff sink."""

    severity: CrisisSeverity
    matched_cue: str
    utterance: str
    timestamp_seconds: float
    user_id: Optional[str] = None


@dataclass
class CrisisEscalation:
    """Scans utterances for crisis cues; routes to a human-handoff sink.

    The sink is a callable: ``sink(incident) -> None``. The detector does not
    wait for the sink; the application layer is responsible for making the
    sink's work happen (fire-and-forget vs synchronous is a sink-side policy).

    Phase 4c Option C: pass `policy=` to resolve regional crisis lines through
    `recommended_crisis_lines()`. The detection gate itself does not vary by
    region (crisis_handoff_required is True in every regime); only the user-
    facing line table does. Default policy = `UNIVERSAL_FLOOR` (all shipped
    lines surfaced) — pre-migration behavior.
    """

    handoff_sink: Callable[[CrisisIncident], None]
    policy: EffectivePolicy | None = None
    _incident_log: list[CrisisIncident] = field(default_factory=list)

    def scan(
        self,
        utterance: str,
        timestamp_seconds: float,
        user_id: Optional[str] = None,
    ) -> CrisisIncident | None:
        lowered = utterance.lower()
        severity, cue = self._match(lowered)
        if severity is CrisisSeverity.NONE:
            return None
        incident = CrisisIncident(
            severity=severity,
            matched_cue=cue,
            utterance=utterance,
            timestamp_seconds=timestamp_seconds,
            user_id=user_id,
        )
        self._incident_log.append(incident)
        self.handoff_sink(incident)
        return incident

    @staticmethod
    def _match(lowered: str) -> tuple[CrisisSeverity, str]:
        for cue in _IMMINENT_DANGER_CUES:
            if cue in lowered:
                return CrisisSeverity.IMMINENT_DANGER, cue
        for cue in _EXPLICIT_IDEATION_CUES:
            if cue in lowered:
                return CrisisSeverity.EXPLICIT_IDEATION, cue
        for cue in _AMBIGUOUS_CUES:
            if cue in lowered:
                return CrisisSeverity.AMBIGUOUS, cue
        return CrisisSeverity.NONE, ""

    def incident_log(self) -> list[CrisisIncident]:
        return list(self._incident_log)

    def recommended_crisis_lines(self) -> Mapping[str, CrisisLine]:
        """Return the crisis-line table the handoff sink should surface.

        Resolves against `self.policy` when set; otherwise the universal
        floor's full-table — all shipped lines surface. Overlay authors
        narrow the table to their region (e.g. CN serves only
        `010-82951332`; US only `988`).
        """
        if self.policy is not None:
            return self.policy.crisis_lines
        return UNIVERSAL_FLOOR.crisis_lines
