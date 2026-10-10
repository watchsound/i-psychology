"""AnnoMI adapter — expert-annotated motivational interviewing dialogues.

**Dataset provenance.** Wu et al. 2022 (ICASSP) + 2023 (Future Internet),
``github.com/uccollab/AnnoMI``. 133 faithfully-transcribed and
expert-annotated motivational-interviewing (MI) dialogue demonstrations
split roughly 110 high-MI / 23 low-MI (observed ratio). Per-utterance
labels:

- ``main_therapist_behaviour`` ∈ {reflection, question, therapist_input, other}
- ``client_talk_type``        ∈ {change, neutral, sustain}

AnnoMI-full additionally splits behaviour labels into sub-types
(reflection: simple/complex; question: open/closed; therapist_input:
information/advice/negotiation/options).

**Why both CSVs are kept raw.** AnnoMI-simple suffices for high-level
behaviour-mix statistics the MI-fidelity scorer measures (reflection
rate, open-question rate — the two MITI-core counts). AnnoMI-full adds
``annotator_id`` and sub-types used by the κ computation when a
scorer needs inter-rater agreement on a subset of utterances. The
adapter exposes a single ``iter_annomi_dialogues()`` entry point that
reads whichever CSV the caller supplies.

**Shape.** The CSV is utterance-granular (one row per utterance, grouped
by ``transcript_id``, ordered by ``utterance_id``). The adapter folds
each transcript into an :class:`AnnoMIDialogue` carrying ordered
:class:`AnnoMITurn` objects.

**Pillar F scoring surface.**

- MI-fidelity (reflection rate, open-question rate) comes from counting
  the therapist utterance labels and gating the Companion's live replies
  against the same lexical/behavioural signature (same approach as
  MathDial-EMPATH on PsyQA-adjacent surfaces).

- Bond is scored by playing the Companion AS the counselor against an
  LLM-simulated client seeded by the AnnoMI first client utterance;
  the dataset's own therapist utterances are NOT fed to the Companion
  (that would leak the reference).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


_THERAPIST_LABEL = "therapist"
_CLIENT_LABEL = "client"


@dataclass(frozen=True)
class AnnoMITurn:
    """One utterance in one AnnoMI dialogue."""

    utterance_id: int
    interlocutor: str  # "therapist" | "client"
    utterance_text: str
    main_therapist_behaviour: str  # "reflection"|"question"|"therapist_input"|"other"|"n/a"
    client_talk_type: str  # "change"|"neutral"|"sustain"|"n/a"
    reflection_subtype: str = "n/a"  # only populated from AnnoMI-full
    question_subtype: str = "n/a"  # only populated from AnnoMI-full
    therapist_input_subtype: str = "n/a"  # only populated from AnnoMI-full

    def __post_init__(self) -> None:
        if self.interlocutor not in (_THERAPIST_LABEL, _CLIENT_LABEL):
            raise ValueError(
                f"AnnoMITurn.interlocutor must be {_THERAPIST_LABEL!r} or "
                f"{_CLIENT_LABEL!r}; got {self.interlocutor!r}"
            )

    @property
    def is_therapist(self) -> bool:
        return self.interlocutor == _THERAPIST_LABEL

    @property
    def is_client(self) -> bool:
        return self.interlocutor == _CLIENT_LABEL


@dataclass(frozen=True)
class AnnoMIDialogue:
    """One AnnoMI counselling dialogue = ordered utterances by transcript_id."""

    transcript_id: str
    mi_quality: str  # "high" | "low"
    video_title: str
    topic: str
    turns: tuple[AnnoMITurn, ...]
    metadata: dict = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def num_therapist_turns(self) -> int:
        return sum(1 for t in self.turns if t.is_therapist)

    @property
    def num_client_turns(self) -> int:
        return sum(1 for t in self.turns if t.is_client)

    def first_client_turn(self) -> Optional[AnnoMITurn]:
        for t in self.turns:
            if t.is_client:
                return t
        return None


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def iter_annomi_dialogues(
    csv_path: Path,
    *,
    limit: Optional[int] = None,
    mi_quality_filter: Optional[str] = None,
) -> Iterator[AnnoMIDialogue]:
    """Stream AnnoMI dialogues from a CSV (simple or full variant).

    Dialogues are yielded in ascending ``transcript_id`` order. Within a
    dialogue, turns are ordered by ``utterance_id`` (which the dataset
    guarantees is ordering-sensitive).

    ``mi_quality_filter`` restricts to one of ``{"high", "low"}``; absent
    for the full dataset.

    ``limit`` caps the dialogue count — the full dataset is 133 which is
    tractable, but smoke runs want <5.
    """
    csv_path = Path(csv_path)
    buckets: dict[str, list[dict]] = {}
    order: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tid = str(row.get("transcript_id", "")).strip()
            if not tid:
                continue
            if tid not in buckets:
                buckets[tid] = []
                order.append(tid)
            buckets[tid].append(row)

    yielded = 0
    for tid in order:
        rows = buckets[tid]
        # AnnoMI-full repeats the same (transcript_id, utterance_id) pair
        # across 3 annotator_id values (0..9 but subset per dialogue). The
        # simple variant has one row per utterance. We deduplicate on
        # (utterance_id, interlocutor) and keep the first observation —
        # labels can disagree between annotators but we're not resolving
        # inter-annotator disputes here (that's the empath-κ job).
        seen: set[int] = set()
        unique_rows: list[dict] = []
        for row in sorted(rows, key=lambda r: _safe_int(r.get("utterance_id"))):
            uid = _safe_int(row.get("utterance_id"))
            if uid in seen:
                continue
            seen.add(uid)
            unique_rows.append(row)
        anchor = unique_rows[0] if unique_rows else {}
        mi_quality = str(anchor.get("mi_quality", "")).strip().lower() or "unknown"
        if mi_quality_filter is not None and mi_quality != mi_quality_filter.lower():
            continue
        video_title = str(anchor.get("video_title", "")).strip()
        topic = str(anchor.get("topic", "")).strip()
        turns = tuple(_parse_turn(r) for r in unique_rows)
        yield AnnoMIDialogue(
            transcript_id=tid,
            mi_quality=mi_quality,
            video_title=video_title,
            topic=topic,
            turns=turns,
            metadata={"source_rows": len(rows)},
        )
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def _parse_turn(row: dict) -> AnnoMITurn:
    interlocutor = str(row.get("interlocutor", "")).strip().lower()
    if interlocutor not in (_THERAPIST_LABEL, _CLIENT_LABEL):
        raise ValueError(
            f"AnnoMI row utterance_id={row.get('utterance_id')} has unexpected "
            f"interlocutor={interlocutor!r}"
        )
    return AnnoMITurn(
        utterance_id=_safe_int(row.get("utterance_id")),
        interlocutor=interlocutor,
        utterance_text=str(row.get("utterance_text", "")).strip(),
        main_therapist_behaviour=str(row.get("main_therapist_behaviour", "n/a") or "n/a").strip().lower(),
        client_talk_type=str(row.get("client_talk_type", "n/a") or "n/a").strip().lower(),
        reflection_subtype=str(row.get("reflection_subtype", "n/a") or "n/a").strip().lower(),
        question_subtype=str(row.get("question_subtype", "n/a") or "n/a").strip().lower(),
        therapist_input_subtype=str(row.get("therapist_input_subtype", "n/a") or "n/a").strip().lower(),
    )


def _safe_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# MI-fidelity primitives (used by counseling_depth_scorer)
# ---------------------------------------------------------------------------


def therapist_reflection_rate(dialogue: AnnoMIDialogue) -> float:
    """Fraction of therapist turns labeled 'reflection' in the dataset.

    This is the AnnoMI *human-baseline* reflection rate; the scorer
    compares the Companion's live reflection rate against this per
    dialogue, same way MathDial-EMPATH compares Companion empathy to
    the labelled gold.
    """
    denom = dialogue.num_therapist_turns
    if denom == 0:
        return 0.0
    hits = sum(
        1 for t in dialogue.turns
        if t.is_therapist and t.main_therapist_behaviour == "reflection"
    )
    return hits / denom


def therapist_open_question_rate(dialogue: AnnoMIDialogue) -> float:
    """Fraction of therapist turns that are open-ended questions.

    Open-ended counts: AnnoMI-full labels every question turn with
    ``question_subtype`` ∈ {open, closed}. AnnoMI-simple only has the
    coarse ``question`` label — if sub-types are "n/a" we treat all
    question turns as the denominator and open-rate becomes undefined
    (returns 0.0 with a note the caller can inspect).
    """
    denom = dialogue.num_therapist_turns
    if denom == 0:
        return 0.0
    opens = sum(
        1 for t in dialogue.turns
        if t.is_therapist
        and t.main_therapist_behaviour == "question"
        and t.question_subtype == "open"
    )
    return opens / denom


__all__ = [
    "AnnoMIDialogue",
    "AnnoMITurn",
    "iter_annomi_dialogues",
    "therapist_open_question_rate",
    "therapist_reflection_rate",
]
