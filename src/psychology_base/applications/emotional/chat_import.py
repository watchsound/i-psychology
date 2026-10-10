"""Chat-history import for 情感陪伴 onboarding (bootstrap-only).

Phase 7 §2.1 + §7 one-way boundary.

Imports feed trait inference only; raw turns NEVER enter the fragment
store. This module imports NOTHING from ``psychology_base.companion.*``.
Only confirmed inferences reach the companion-substrate observe path,
via a separate module (``trait_inference.py``) after the user
confirmation gate.

Three format adapters:
- ChatGPT JSON export (mapping node_id → message)
- WhatsApp text export (``[date, time] Name: message`` lines)
- Generic JSONL (one object per line: ``{speaker, content, timestamp}``)

Pre-ingest safety sweep uses ``CrisisEscalation`` on user-produced turns.
Fail-closed: on any explicit-ideation or imminent-danger match, the
caller MUST route the user through Tier-2 handoff before trait inference.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Union

from psychology_base.applications.safety.crisis_escalation import (
    CrisisEscalation,
    CrisisIncident,
    CrisisSeverity,
)
from psychology_base.applications.safety.extreme_emotion import (
    ExtremeEmotionFlag,
    ExtremeEmotionWatcher,
    SessionAffect,
)


class ImportFormat(str, Enum):
    CHATGPT_JSON = "chatgpt_json"
    WHATSAPP_TEXT = "whatsapp_text"
    GENERIC_JSONL = "generic_jsonl"


@dataclass(frozen=True)
class ImportedTurn:
    """One turn from an imported chat log.

    This is NOT a Fragment. It carries no id and never enters the
    fragment store. Trait inference reads these; nothing else does.
    """

    speaker: str
    content: str
    timestamp: datetime
    source_format: ImportFormat
    source_metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ImportConsent:
    """User attestation at import time (Phase 7 §2.1 consent gate)."""

    user_id: str
    consent_text_version: str
    attested_at: datetime
    scope: str


# ---------------------------------------------------------------------------
# ChatGPT JSON parser
# ---------------------------------------------------------------------------


def parse_chatgpt_json(data: Union[dict, list]) -> list[ImportedTurn]:
    """Parse a ChatGPT export (``conversations.json`` shape).

    Each conversation has a ``mapping`` dict of node_id → node, where
    ``node["message"]`` carries ``author.role``, ``content.parts``, and
    ``create_time``. Nodes with ``role=="system"`` are skipped. Nodes
    with no textual content are skipped.
    """
    conversations = data if isinstance(data, list) else [data]
    turns: list[ImportedTurn] = []
    for conv in conversations:
        mapping = conv.get("mapping", {})
        nodes = [n for n in mapping.values() if n.get("message") is not None]
        nodes.sort(key=lambda n: n["message"].get("create_time") or 0.0)
        for node in nodes:
            msg = node["message"]
            role = (msg.get("author") or {}).get("role", "user")
            if role == "system":
                continue
            parts = (msg.get("content") or {}).get("parts", [])
            text = "\n".join(
                p for p in parts if isinstance(p, str) and p.strip()
            )
            if not text:
                continue
            create_time = msg.get("create_time")
            if create_time is None:
                ts = datetime.fromtimestamp(0, tz=timezone.utc)
            else:
                ts = datetime.fromtimestamp(float(create_time), tz=timezone.utc)
            speaker = "user" if role == "user" else "assistant"
            turns.append(
                ImportedTurn(
                    speaker=speaker,
                    content=text,
                    timestamp=ts,
                    source_format=ImportFormat.CHATGPT_JSON,
                    source_metadata={
                        "conversation_id": conv.get("conversation_id")
                        or conv.get("id")
                        or "",
                        "role": role,
                    },
                )
            )
    return turns


# ---------------------------------------------------------------------------
# WhatsApp text parser
# ---------------------------------------------------------------------------


_WHATSAPP_LINE_RE = re.compile(
    r"^\[?\s*(?P<date>\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?)"
    r"\s*(?P<ampm>[APap][Mm])?"
    r"(?:\s*\]\s*|\s+-\s+)"
    r"(?P<speaker>[^:]+?):\s*(?P<content>.*)$"
)


def parse_whatsapp_text(text: str, *, user_name: str) -> list[ImportedTurn]:
    """Parse a WhatsApp text export.

    Accepts bracketed and dash-separated line prefixes. Continuation
    lines (lines not matching the prefix regex) are appended to the
    previous turn's content. ``user_name`` identifies the importing
    user; other speakers are tagged ``other:<name>``.
    """
    turns: list[ImportedTurn] = []
    current: Optional[dict] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _WHATSAPP_LINE_RE.match(line)
        if m:
            if current is not None:
                turns.append(_make_whatsapp_turn(current, user_name))
            current = {
                "date": m.group("date"),
                "time": m.group("time"),
                "ampm": m.group("ampm"),
                "speaker": m.group("speaker").strip(),
                "content": m.group("content"),
            }
        elif current is not None:
            current["content"] = (current["content"] + "\n" + line).strip()
    if current is not None:
        turns.append(_make_whatsapp_turn(current, user_name))
    return turns


_DATE_YMD_SLASH = re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$")
_DATE_YMD_DASH = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_DATE_DMY_SLASH = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")


def _parse_whatsapp_datetime(
    date_str: str, time_str: str, ampm: Optional[str]
) -> datetime:
    date_str = date_str.strip()
    time_str = time_str.strip()
    if _DATE_YMD_SLASH.match(date_str):
        y, m, d = (int(x) for x in date_str.split("/"))
    elif _DATE_YMD_DASH.match(date_str):
        y, m, d = (int(x) for x in date_str.split("-"))
    elif _DATE_DMY_SLASH.match(date_str):
        d, m, y = (int(x) for x in date_str.split("/"))
        if y < 100:
            y += 2000
    else:
        now = datetime.now(timezone.utc)
        y, m, d = now.year, now.month, now.day
    parts = time_str.split(":")
    hh = int(parts[0])
    mm = int(parts[1])
    ss = int(parts[2]) if len(parts) > 2 else 0
    if ampm:
        if ampm.lower() == "pm" and hh < 12:
            hh += 12
        elif ampm.lower() == "am" and hh == 12:
            hh = 0
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


def _make_whatsapp_turn(current: dict, user_name: str) -> ImportedTurn:
    ts = _parse_whatsapp_datetime(
        current["date"], current["time"], current.get("ampm")
    )
    speaker_raw = current["speaker"].strip()
    speaker = "user" if speaker_raw == user_name else f"other:{speaker_raw}"
    return ImportedTurn(
        speaker=speaker,
        content=current["content"].strip(),
        timestamp=ts,
        source_format=ImportFormat.WHATSAPP_TEXT,
        source_metadata={"raw_speaker": speaker_raw},
    )


# ---------------------------------------------------------------------------
# Generic JSONL parser
# ---------------------------------------------------------------------------


def parse_jsonl(text: str) -> list[ImportedTurn]:
    """Parse a generic JSONL export.

    Each non-blank line is a JSON object with required fields
    ``speaker``, ``content``, ``timestamp`` (ISO-8601) and optional
    ``metadata``.
    """
    turns: list[ImportedTurn] = []
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        ts = datetime.fromisoformat(obj["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        turns.append(
            ImportedTurn(
                speaker=obj["speaker"],
                content=obj["content"],
                timestamp=ts,
                source_format=ImportFormat.GENERIC_JSONL,
                source_metadata={
                    "line_number": i,
                    **(obj.get("metadata") or {}),
                },
            )
        )
    return turns


# ---------------------------------------------------------------------------
# Safety sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportSafetyIncident:
    turn_index: int
    severity: str
    matched_cue: str
    utterance: str
    timestamp: datetime


@dataclass(frozen=True)
class ImportSafetyReport:
    """Pre-inference safety sweep over imported turns.

    ``requires_handoff`` is True iff any user-produced turn matched an
    explicit-ideation or imminent-danger cue. Trait inference MUST NOT
    run until the user has been routed through Tier-2 handoff and
    affirmatively consented to continue.
    """

    total_turns: int
    user_turns_scanned: int
    incidents: tuple[ImportSafetyIncident, ...]
    requires_handoff: bool
    tier_handoff_rationale: str = ""


def safety_sweep(
    turns: Iterable[ImportedTurn],
    *,
    crisis_detector: Optional[CrisisEscalation] = None,
    user_id: Optional[str] = None,
) -> ImportSafetyReport:
    """Scan user-produced turns for crisis / self-harm cues.

    Only turns with ``speaker == "user"`` are scanned — assistant
    replies and third-party speakers in group chats are out of scope
    (we are inferring about the importing user, not screening third
    parties).
    """
    turn_list = list(turns)
    incidents: list[ImportSafetyIncident] = []
    sink_log: list[CrisisIncident] = []

    def _sink(incident: CrisisIncident) -> None:
        sink_log.append(incident)

    detector = crisis_detector or CrisisEscalation(handoff_sink=_sink)
    user_scanned = 0

    for idx, turn in enumerate(turn_list):
        if turn.speaker != "user":
            continue
        user_scanned += 1
        incident = detector.scan(
            utterance=turn.content,
            timestamp_seconds=turn.timestamp.timestamp(),
            user_id=user_id,
        )
        if incident is None:
            continue
        incidents.append(
            ImportSafetyIncident(
                turn_index=idx,
                severity=incident.severity.value,
                matched_cue=incident.matched_cue,
                utterance=incident.utterance,
                timestamp=turn.timestamp,
            )
        )

    high_set = {
        CrisisSeverity.EXPLICIT_IDEATION.value,
        CrisisSeverity.IMMINENT_DANGER.value,
    }
    requires_handoff = any(inc.severity in high_set for inc in incidents)
    rationale = ""
    if requires_handoff:
        counts: dict[str, int] = {}
        for inc in incidents:
            counts[inc.severity] = counts.get(inc.severity, 0) + 1
        rationale = (
            f"Historical crisis-adjacent content detected in import ({counts}). "
            "Route user through Tier-2 handoff before ingesting any inferences."
        )

    return ImportSafetyReport(
        total_turns=len(turn_list),
        user_turns_scanned=user_scanned,
        incidents=tuple(incidents),
        requires_handoff=requires_handoff,
        tier_handoff_rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Lexical PAD-2 scoring + session-level extreme-emotion aggregation
# ---------------------------------------------------------------------------


_POSITIVE_VALENCE_CUES: tuple[str, ...] = (
    "happy", "joy", "joyful", "love", "loved", "grateful", "thank",
    "calm", "peace", "peaceful", "excited", "proud", "hope", "hopeful",
    "wonderful", "great", "amazing",
    # Role-play / generative-dialogue register (F3). Affect-about-events
    # or self, not interpersonal praise aimed at other.
    "nice", "cool", "awesome", "sweet", "lovely", "fantastic", "fun",
    "nailed it", "crushed it",
    "高兴", "开心", "快乐", "平静", "感激", "希望",
    "好棒", "太棒了", "厉害", "牛",
)

# These are affect-about-events/self. Interpersonal dismissive tokens
# ("stupid", "ugh", "whatever", "idiot") belong to
# ``trait_inference._HOSTILITY_CUES``, NOT here, so the agreeableness
# and neuroticism signals do not double-count the same word.
_NEGATIVE_VALENCE_CUES: tuple[str, ...] = (
    "sad", "lonely", "depressed", "hopeless", "worthless", "angry",
    "afraid", "scared", "miserable", "awful", "terrible", "hate",
    "disappointed", "guilt", "guilty", "shame", "ashamed",
    # Role-play / generative-dialogue register (F3). Affect-about-events
    # or self, NOT interpersonal dismissal — see split rule above.
    "pointless", "useless", "ruin", "ruined", "waste", "wasted",
    "rip-off", "trash", "mess", "sucks", "annoying", "frustrating",
    "悲伤", "孤独", "绝望", "无望", "愤怒", "害怕", "糟糕", "讨厌",
    "失望", "内疚", "羞愧",
    "烂", "糟", "烦人",
)

_HIGH_AROUSAL_CUES: tuple[str, ...] = (
    "panic", "overwhelmed", "overwhelm", "anxious", "anxiety", "furious",
    "terrified", "screaming", "scream", "shaking", "shake", "racing",
    "frantic", "intense", "scared", "afraid", "worried",
    # Role-play / generative-dialogue register (F3). Note: "ugh" is
    # excluded — belongs to ``trait_inference._HOSTILITY_CUES`` as
    # dismissive, not arousal.
    "wow", "whoa", "yikes", "literally", "absolutely", "insane",
    "crazy", "wild", "freaking out", "losing it",
    "恐慌", "焦虑", "惊恐", "暴怒", "崩溃", "疯狂", "害怕", "紧张",
    "炸裂", "疯了", "吓死",
)

_LOW_AROUSAL_CUES: tuple[str, ...] = (
    "tired", "numb", "empty", "exhausted", "drained", "flat", "dull",
    # Role-play / generative-dialogue register (F3). Note: "whatever"
    # is excluded — belongs to ``trait_inference._HOSTILITY_CUES``.
    "meh", "bland", "boring", "nothing special",
    "疲惫", "麻木", "空虚", "没劲", "没有力气",
    "没意思", "乏味",
)


_ASCII_WORD_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_")


def _count_cues(text: str, cues: tuple[str, ...]) -> int:
    """Count cue hits with word-boundary matching for ASCII cues.

    Substring matching produces false positives — "hope" inside
    "hopeless" would read as positive valence. For ASCII cues, require
    a non-word-char boundary on each side. For non-ASCII (CJK) cues,
    fall back to substring, since \\b doesn't help with CJK (all CJK
    chars are \\w in Python's regex).
    """
    lower = text.lower()
    count = 0
    for cue in cues:
        if cue and all(ch in _ASCII_WORD_CHARS or ch == " " for ch in cue):
            start = 0
            while True:
                idx = lower.find(cue, start)
                if idx < 0:
                    break
                left_ok = idx == 0 or lower[idx - 1] not in _ASCII_WORD_CHARS
                right_idx = idx + len(cue)
                right_ok = (
                    right_idx == len(lower)
                    or lower[right_idx] not in _ASCII_WORD_CHARS
                )
                if left_ok and right_ok:
                    count += 1
                    break
                start = idx + 1
        else:
            if cue in lower:
                count += 1
    return count


def score_valence_lexical(text: str) -> float:
    """Lexical PAD valence ∈ [-1, 1]. v1 scorer — CI-safe heuristic.

    LLM-backed replacement is Phase 7+ carried debt once trait inference
    needs higher-precision affect per turn.
    """
    pos = _count_cues(text, _POSITIVE_VALENCE_CUES)
    neg = _count_cues(text, _NEGATIVE_VALENCE_CUES)
    if pos == 0 and neg == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / max(1, pos + neg)))


def score_arousal_lexical(text: str) -> float:
    """Lexical PAD arousal ∈ [-1, 1]. v1 scorer."""
    high = _count_cues(text, _HIGH_AROUSAL_CUES)
    low = _count_cues(text, _LOW_AROUSAL_CUES)
    if high == 0 and low == 0:
        return 0.0
    return max(-1.0, min(1.0, (high - low) / max(1, high + low)))


def group_turns_into_sessions(
    turns: Iterable[ImportedTurn],
    *,
    gap_minutes: int = 180,
) -> list[list[ImportedTurn]]:
    """Group chronologically-ordered turns by inter-turn time gap.

    Default: ≥ 180 min between consecutive turns opens a new session.
    """
    turn_list = sorted(turns, key=lambda t: t.timestamp)
    sessions: list[list[ImportedTurn]] = []
    for turn in turn_list:
        if not sessions:
            sessions.append([turn])
            continue
        last = sessions[-1][-1]
        gap = (turn.timestamp - last.timestamp).total_seconds() / 60.0
        if gap >= gap_minutes:
            sessions.append([turn])
        else:
            sessions[-1].append(turn)
    return sessions


def aggregate_session_affect(
    session_turns: list[ImportedTurn],
    *,
    session_id: str,
    user_only: bool = True,
) -> SessionAffect:
    """Mean-aggregate lexical valence × arousal over a session's user turns.

    Returns the ``SessionAffect`` that ``ExtremeEmotionWatcher.record``
    expects. Sessions with zero scanned turns return neutral affect.
    """
    scored = [
        t for t in session_turns if not user_only or t.speaker == "user"
    ]
    if not scored:
        last_ts = session_turns[-1].timestamp if session_turns else datetime.now(timezone.utc)
        return SessionAffect(
            session_id=session_id,
            timestamp_seconds=last_ts.timestamp(),
            valence=0.0,
            arousal=0.0,
        )
    valence = sum(score_valence_lexical(t.content) for t in scored) / len(scored)
    arousal = sum(score_arousal_lexical(t.content) for t in scored) / len(scored)
    return SessionAffect(
        session_id=session_id,
        timestamp_seconds=scored[-1].timestamp.timestamp(),
        valence=valence,
        arousal=arousal,
    )


@dataclass(frozen=True)
class ExtendedSafetyReport:
    """Union of per-turn crisis + per-session extreme-emotion findings."""

    crisis: ImportSafetyReport
    extreme_emotion_flag: Optional[ExtremeEmotionFlag]
    session_affects: tuple[SessionAffect, ...]
    requires_handoff: bool
    requires_review: bool

    @property
    def rationale(self) -> str:
        reasons: list[str] = []
        if self.crisis.requires_handoff:
            reasons.append(self.crisis.tier_handoff_rationale)
        if self.extreme_emotion_flag is not None:
            reasons.append(
                f"Prolonged negative-valence/high-arousal across "
                f"{self.extreme_emotion_flag.consecutive_sessions} "
                "consecutive imported sessions — flag for Tier-1 review."
            )
        return " ".join(reasons)


def extended_safety_sweep(
    turns: Iterable[ImportedTurn],
    *,
    user_id: Optional[str] = None,
    gap_minutes: int = 180,
    extreme_emotion_threshold_sessions: int = 3,
    crisis_detector: Optional[CrisisEscalation] = None,
) -> ExtendedSafetyReport:
    """Crisis scan + session-grouped extreme-emotion scan.

    ``requires_handoff`` is True iff crisis sweep demanded it (explicit
    ideation or imminent danger). ``requires_review`` is True iff
    either handoff OR extreme-emotion flag fired. Trait inference is
    gated on ``requires_handoff`` only; ``requires_review`` surfaces a
    softer "Tier-1 review before continuing" signal.
    """
    turn_list = list(turns)
    crisis_report = safety_sweep(
        turn_list,
        crisis_detector=crisis_detector,
        user_id=user_id,
    )

    sessions = group_turns_into_sessions(turn_list, gap_minutes=gap_minutes)
    session_affects: list[SessionAffect] = []
    flag_captured: list[ExtremeEmotionFlag] = []

    def _review_sink(flag: ExtremeEmotionFlag) -> None:
        flag_captured.append(flag)

    watcher = ExtremeEmotionWatcher(
        review_sink=_review_sink,
        threshold_sessions=extreme_emotion_threshold_sessions,
        user_id=user_id,
    )
    for i, s in enumerate(sessions):
        affect = aggregate_session_affect(s, session_id=f"imported-{i}")
        session_affects.append(affect)
        watcher.record(affect)

    flag = flag_captured[0] if flag_captured else None
    return ExtendedSafetyReport(
        crisis=crisis_report,
        extreme_emotion_flag=flag,
        session_affects=tuple(session_affects),
        requires_handoff=crisis_report.requires_handoff,
        requires_review=crisis_report.requires_handoff or flag is not None,
    )


# ---------------------------------------------------------------------------
# Raw audit log (outside fragment store)
# ---------------------------------------------------------------------------


DEFAULT_IMPORTS_ROOT = Path("imports")


def write_raw_audit_log(
    turns: Iterable[ImportedTurn],
    *,
    user_id: str,
    consent: ImportConsent,
    out_root: Path = DEFAULT_IMPORTS_ROOT,
) -> Path:
    """Write imported turns under ``<out_root>/<user_id>/raw/<stamp>.json``.

    This is the 《办法》 compliance audit log, not a memory substrate.
    Nothing in ``psychology_base.companion.*`` or the companion agent
    reads this directory. Shred-on-request: delete the file.
    """
    turn_list = list(turns)
    user_root = out_root / user_id / "raw"
    user_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_path = user_root / f"{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "user_id": user_id,
                "consent": {
                    "consent_text_version": consent.consent_text_version,
                    "attested_at": consent.attested_at.isoformat(),
                    "scope": consent.scope,
                },
                "written_at": datetime.now(timezone.utc).isoformat(),
                "turn_count": len(turn_list),
                "turns": [
                    {
                        "speaker": t.speaker,
                        "content": t.content,
                        "timestamp": t.timestamp.isoformat(),
                        "source_format": t.source_format.value,
                        "source_metadata": t.source_metadata,
                    }
                    for t in turn_list
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------------
# High-level entrypoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportResult:
    turns: tuple[ImportedTurn, ...]
    safety_report: ImportSafetyReport
    audit_log_path: Path
    consent: ImportConsent


def import_chat_history(
    *,
    source_format: ImportFormat,
    raw_text_or_data: Union[str, dict, list],
    user_id: str,
    consent: ImportConsent,
    user_name: Optional[str] = None,
    out_root: Path = DEFAULT_IMPORTS_ROOT,
    crisis_detector: Optional[CrisisEscalation] = None,
) -> ImportResult:
    """Parse → safety sweep → audit log.

    Does NOT run trait inference. Does NOT touch ``CompanionSystem``.
    Caller runs inference as a separate step iff
    ``safety_report.requires_handoff`` is ``False``.
    """
    if source_format is ImportFormat.CHATGPT_JSON:
        if not isinstance(raw_text_or_data, (dict, list)):
            raise TypeError(
                "CHATGPT_JSON requires parsed dict/list, not raw text"
            )
        turns = parse_chatgpt_json(raw_text_or_data)
    elif source_format is ImportFormat.WHATSAPP_TEXT:
        if not isinstance(raw_text_or_data, str):
            raise TypeError("WHATSAPP_TEXT requires raw text")
        if user_name is None:
            raise ValueError(
                "WHATSAPP_TEXT requires user_name to disambiguate speakers"
            )
        turns = parse_whatsapp_text(raw_text_or_data, user_name=user_name)
    elif source_format is ImportFormat.GENERIC_JSONL:
        if not isinstance(raw_text_or_data, str):
            raise TypeError("GENERIC_JSONL requires raw text")
        turns = parse_jsonl(raw_text_or_data)
    else:
        raise ValueError(f"Unknown source_format {source_format!r}")

    safety_report = safety_sweep(
        turns,
        crisis_detector=crisis_detector,
        user_id=user_id,
    )

    audit_log_path = write_raw_audit_log(
        turns,
        user_id=user_id,
        consent=consent,
        out_root=out_root,
    )

    return ImportResult(
        turns=tuple(turns),
        safety_report=safety_report,
        audit_log_path=audit_log_path,
        consent=consent,
    )


# ---------------------------------------------------------------------------
# One-way boundary self-check (also enforced via test)
# ---------------------------------------------------------------------------


def _assert_one_way_boundary() -> None:
    """Raise if this module imports a forbidden symbol.

    This runs at import time of the test suite via a dedicated test;
    it never runs at normal import time. The function exists so the
    rule is readable from the module itself.
    """
    forbidden = {
        "psychology_base.companion.system",
        "psychology_base.companion.fragment",
        "psychology_base.companion.reconstruction",
    }
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
    bad = imports & forbidden
    if bad:
        raise AssertionError(
            f"chat_import.py violates Phase 7 one-way boundary: imports {bad}"
        )


__all__ = [
    "ImportFormat",
    "ImportedTurn",
    "ImportConsent",
    "ImportSafetyIncident",
    "ImportSafetyReport",
    "ImportResult",
    "DEFAULT_IMPORTS_ROOT",
    "parse_chatgpt_json",
    "parse_whatsapp_text",
    "parse_jsonl",
    "safety_sweep",
    "write_raw_audit_log",
    "import_chat_history",
]
