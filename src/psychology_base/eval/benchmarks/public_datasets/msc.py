"""MSC adapter — Multi-Session Chat (Xu et al. 2022) longitudinal persona dialogue.

**Dataset provenance.** Xu, J., Szlam, A., & Weston, J. (2022). *Beyond
Goldfish Memory: Long-Term Open-Domain Conversation* (ACL 2022; arXiv
2107.07567). Five-session persona-grounded dialogues where each pair of
speakers chats across simulated day-to-week gaps (``time_interval`` in
natural-language days), with each later session able to reference prior
sessions. Dataset page: https://parl.ai/projects/msc/.
License — **CC BY-NC 4.0** per the MSC paper + ParlAI distribution; the
shim is a derivative and inherits the non-commercial restriction.
``.data/`` is gitignored to keep transitive repo redistribution clean.

**Phase 8 Pillar A gate this adapter targets.** `docs/phases/phase-8-dataset-replay-expansion-plan.md`
§2 pillar A — mean WAI-SR Bond ≥ 3.89 (Woebot aggregate floor) across
≥ 50 simulated users at **session 5** (not session 1), measured per the
same simulated-user WAI-SR administration pattern as Phase 4b ESConv Bond.
The longitudinal signal — Δ(bond_session_5 − bond_session_1) — is the
pillar's own additional contribution over ESConv: Phase 4b produces a
single-session Bond; Phase 8 Pillar A produces a 5-session trajectory
with explicit between-session time gaps, so Bond growth / drift / collapse
across simulated weeks is measurable without recruiting humans.

**Why no vendor-download.** Same posture as the ESConv / MathDial / FANToM
adapters — license-neutral and size-agnostic. Loader expects a user-provided
JSONL shim; the conversion script lands separately at
``benchmarks/scripts/provision_msc.py``.

**JSONL shim schema.** One conversation per line (not per session — the
longitudinal unit is the whole 5-session chain)::

    {
      "conversation_id": "msc_000001",
      "personas": [
        ["I love dogs.", "I work as a teacher."],
        ["I grew up in Kansas.", "I collect vintage cameras."]
      ],
      "sessions": [
        {
          "session_id": 1,
          "time_interval_days": 0,
          "turns": [
            {"speaker_idx": 0, "utterance": "Hi, how are you today?"},
            {"speaker_idx": 1, "utterance": "I'm doing great, you?"},
            ...
          ]
        },
        {
          "session_id": 2,
          "time_interval_days": 7,
          "turns": [...]
        },
        ...
      ],
      "metadata": {}
    }

``speaker_idx`` 0 is the "self" speaker whose persona anchors the
longitudinal Bond measurement (the seeker-equivalent); ``speaker_idx`` 1
is the "other" (companion-equivalent surface). The shim keeps both so a
Phase 8+ scorer can flip the direction without re-provisioning.

``time_interval_days`` is **in-world gap** between this session and the
previous one (session 1 is anchor 0; later sessions carry the dataset's
natural-language gap parsed to float days — e.g. "3 days" → 3.0, "about
a week" → 7.0). The parse rule lives in ``provision_msc.py`` so loader
stays dependency-light.

**No LLM dependency.** Loader + simple accessors only. The longitudinal
Bond harness, simulated seeker factory, and runner live in sibling
modules (``msc_bond_longitudinal.py`` + ``phase8_a_msc_run.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


SELF_SPEAKER_IDX: int = 0
OTHER_SPEAKER_IDX: int = 1


@dataclass(frozen=True)
class MSCTurn:
    """One turn within one MSC session."""

    speaker_idx: int
    utterance: str

    def __post_init__(self) -> None:
        if self.speaker_idx not in (SELF_SPEAKER_IDX, OTHER_SPEAKER_IDX):
            raise ValueError(
                f"MSCTurn.speaker_idx must be 0 or 1; got {self.speaker_idx}"
            )


@dataclass(frozen=True)
class MSCSession:
    """One session within one MSC conversation chain."""

    session_id: int
    time_interval_days: float
    turns: tuple[MSCTurn, ...]

    def __post_init__(self) -> None:
        if self.session_id < 1:
            raise ValueError(
                f"MSCSession.session_id must be ≥ 1; got {self.session_id}"
            )
        if self.time_interval_days < 0:
            raise ValueError(
                f"MSCSession.time_interval_days must be ≥ 0; got {self.time_interval_days}"
            )


@dataclass(frozen=True)
class MSCConversation:
    """One MSC conversation chain = 2 personas + up to 5 sessions."""

    conversation_id: str
    personas: tuple[tuple[str, ...], tuple[str, ...]]
    sessions: tuple[MSCSession, ...]
    metadata: dict = field(default_factory=dict)

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    def session(self, session_id: int) -> Optional[MSCSession]:
        for s in self.sessions:
            if s.session_id == session_id:
                return s
        return None

    def cumulative_days_at(self, session_id: int) -> Optional[float]:
        """Days elapsed in-world from session 1 start to ``session_id`` start."""
        total = 0.0
        for s in self.sessions:
            if s.session_id == session_id:
                return total + s.time_interval_days
            total += s.time_interval_days
        return None


# ---------------------------------------------------------------------------
# JSONL shim loader
# ---------------------------------------------------------------------------


def iter_msc_conversations(shim_path: Path) -> Iterator[MSCConversation]:
    """Stream MSC conversations from a JSONL shim file."""
    with Path(shim_path).open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            yield _parse_conversation(obj, line_no)


def _parse_conversation(obj: dict, line_no: int) -> MSCConversation:
    try:
        conv_id = str(obj["conversation_id"])
    except KeyError as exc:
        raise ValueError(
            f"MSC shim line {line_no}: missing 'conversation_id'"
        ) from exc

    raw_personas = obj.get("personas", [])
    if not isinstance(raw_personas, list) or len(raw_personas) != 2:
        raise ValueError(
            f"MSC shim line {line_no} ({conv_id}): 'personas' must be a 2-element list; "
            f"got {raw_personas!r}"
        )
    personas: tuple[tuple[str, ...], tuple[str, ...]] = (
        tuple(str(p) for p in raw_personas[0] or ()),
        tuple(str(p) for p in raw_personas[1] or ()),
    )

    raw_sessions = obj.get("sessions", [])
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError(
            f"MSC shim line {line_no} ({conv_id}): 'sessions' must be a non-empty list"
        )
    sessions = tuple(
        _parse_session(s, line_no, conv_id, idx)
        for idx, s in enumerate(raw_sessions)
    )
    return MSCConversation(
        conversation_id=conv_id,
        personas=personas,
        sessions=sessions,
        metadata=dict(obj.get("metadata", {})),
    )


def _parse_session(
    raw: dict, line_no: int, conv_id: str, idx: int
) -> MSCSession:
    session_id = raw.get("session_id", idx + 1)
    try:
        session_id = int(session_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"MSC shim line {line_no} ({conv_id}) session {idx}: "
            f"'session_id' must be an int; got {raw.get('session_id')!r}"
        ) from exc
    try:
        time_interval_days = float(raw.get("time_interval_days", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"MSC shim line {line_no} ({conv_id}) session {idx}: "
            f"'time_interval_days' must be numeric; got {raw.get('time_interval_days')!r}"
        ) from exc
    raw_turns = raw.get("turns", [])
    if not isinstance(raw_turns, list):
        raise ValueError(
            f"MSC shim line {line_no} ({conv_id}) session {idx}: "
            f"'turns' must be a list; got {type(raw_turns).__name__}"
        )
    turns = tuple(
        _parse_turn(t, line_no, conv_id, idx, tidx)
        for tidx, t in enumerate(raw_turns)
    )
    return MSCSession(
        session_id=session_id,
        time_interval_days=time_interval_days,
        turns=turns,
    )


def _parse_turn(
    raw: dict, line_no: int, conv_id: str, sidx: int, tidx: int
) -> MSCTurn:
    try:
        speaker_idx = int(raw["speaker_idx"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"MSC shim line {line_no} ({conv_id}) session {sidx} turn {tidx}: "
            f"'speaker_idx' must be an int"
        ) from exc
    utterance = raw.get("utterance")
    if not isinstance(utterance, str):
        raise ValueError(
            f"MSC shim line {line_no} ({conv_id}) session {sidx} turn {tidx}: "
            f"'utterance' must be a string"
        )
    return MSCTurn(speaker_idx=speaker_idx, utterance=utterance)


# ---------------------------------------------------------------------------
# Convenience accessors used by the longitudinal Bond harness
# ---------------------------------------------------------------------------


def self_persona(convo: MSCConversation) -> tuple[str, ...]:
    """Persona bullets for the seeker-equivalent speaker (speaker_idx=0)."""
    return convo.personas[SELF_SPEAKER_IDX]


def other_persona(convo: MSCConversation) -> tuple[str, ...]:
    """Persona bullets for the companion-equivalent speaker (speaker_idx=1)."""
    return convo.personas[OTHER_SPEAKER_IDX]


def self_opening_utterance(
    convo: MSCConversation, session_id: int = 1
) -> Optional[str]:
    """First utterance from the self-speaker in ``session_id``.

    Returns None if the session doesn't exist or has no self turns.
    Used by the simulated-user harness to seed each session's playback
    with a dataset-authentic opener (same posture as ``seeker_opening_utterance``
    for ESConv).
    """
    session = convo.session(session_id)
    if session is None:
        return None
    for turn in session.turns:
        if turn.speaker_idx == SELF_SPEAKER_IDX:
            return turn.utterance
    return None


__all__ = [
    "MSCConversation",
    "MSCSession",
    "MSCTurn",
    "OTHER_SPEAKER_IDX",
    "SELF_SPEAKER_IDX",
    "iter_msc_conversations",
    "other_persona",
    "self_opening_utterance",
    "self_persona",
]
