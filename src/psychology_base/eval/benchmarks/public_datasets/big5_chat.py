"""BIG5-CHAT adapter — Shao et al. 2024 trait-conditioned dialogue corpus.

**Dataset provenance.** Shao, W. et al. (2024). *BIG5-CHAT: Shaping LLM
Personalities Through Training on Human-Grounded Data* (arXiv 2410.16491,
ACL 2025). 100,000 single-turn dialogues generated via the PsychSteer
framework over 10k SODA social scenarios; each dialogue is labelled with
exactly one of the five OCEAN traits at exactly one of two levels (``high``
or ``low``). The paper reports 20k dialogues per trait, evenly split between
levels.

License — **Apache 2.0** per the ``wenkai-li/big5_chat`` HuggingFace mirror
(https://huggingface.co/datasets/wenkai-li/big5_chat). The shim is a
derivative and inherits that license.

**Phase 8 Pillar C soft gate this adapter targets.**
`docs/phases/phase-8-dataset-replay-expansion-plan.md` §2 pillar C — the
trait-condition-forward classification gate at ≥ 60% accuracy on the
labeled OCEAN dimension's polarity. Companion inference reads the
trait-conditioned response (``train_output``) as a single user turn and
reconstructs an OCEAN vector; the vector's sign on the labeled dim is
compared to the ground-truth level.

**Why single-turn.** The paper's PsychSteer generation produces one
trait-conditioned response per scenario; the trait signal is concentrated
in ``train_output``. Our Companion trait inference is designed for
longer imports, so BIG5-CHAT is a worst-case stress test for the pipeline
(one utterance per profile vs. the Phase 7 B4 corpus of 50–500).

**No LLM dependency.** Loader + simple accessors only. The scorer and
runner live in sibling modules under ``src/psychology_base/eval/benchmarks``
and ``benchmarks/scripts``.

**JSONL shim schema.** One dialog per line::

    {
      "dialog_id": "big5_000001",
      "env_idx": 0,
      "trait": "openness",
      "level": "low",
      "train_input": "Hey Karis, I'm so glad...",
      "train_output": "Thanks, Sharif. I'm fine now...",
      "narrative": "Sharif is happy because...",
      "literal": "Sharif gets Karis out of a jam...",
      "personx": "Sharif",
      "persony": "Karis",
      "metadata": {}
    }

``trait`` ∈ {``openness``, ``conscientiousness``, ``extraversion``,
``agreeableness``, ``neuroticism``}. ``level`` ∈ {``high``, ``low``}.
``train_output`` is the trait-conditioned response (person-Y responding
to person-X); it is the primary signal our scorer feeds to inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


_VALID_TRAITS: frozenset[str] = frozenset(
    {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
)
_VALID_LEVELS: frozenset[str] = frozenset({"high", "low"})


@dataclass(frozen=True)
class Big5ChatDialog:
    """One BIG5-CHAT single-turn dialogue with a trait+level label."""

    dialog_id: str
    env_idx: int
    trait: str
    level: str
    train_input: str
    train_output: str
    narrative: str = ""
    literal: str = ""
    personx: str = ""
    persony: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.trait not in _VALID_TRAITS:
            raise ValueError(
                f"Big5ChatDialog.trait must be one of {sorted(_VALID_TRAITS)}; "
                f"got {self.trait!r}"
            )
        if self.level not in _VALID_LEVELS:
            raise ValueError(
                f"Big5ChatDialog.level must be one of {sorted(_VALID_LEVELS)}; "
                f"got {self.level!r}"
            )
        if not self.train_output:
            raise ValueError(
                f"Big5ChatDialog.train_output must be non-empty "
                f"(dialog_id={self.dialog_id!r})"
            )


def iter_big5_chat_dialogs(shim_path: Path) -> Iterator[Big5ChatDialog]:
    """Stream BIG5-CHAT dialogs from a JSONL shim file."""
    with Path(shim_path).open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            yield _parse_dialog(obj, line_no)


def _parse_dialog(obj: dict, line_no: int) -> Big5ChatDialog:
    try:
        dialog_id = str(obj["dialog_id"])
        trait = str(obj["trait"]).strip().lower()
        level = str(obj["level"]).strip().lower()
        train_output = str(obj["train_output"])
    except KeyError as exc:
        raise ValueError(
            f"BIG5-CHAT shim line {line_no}: missing required field {exc!s}"
        ) from exc
    env_idx = int(obj.get("env_idx", 0))
    return Big5ChatDialog(
        dialog_id=dialog_id,
        env_idx=env_idx,
        trait=trait,
        level=level,
        train_input=str(obj.get("train_input", "")),
        train_output=train_output,
        narrative=str(obj.get("narrative", "")),
        literal=str(obj.get("literal", "")),
        personx=str(obj.get("personx", "")),
        persony=str(obj.get("persony", "")),
        metadata=dict(obj.get("metadata", {})),
    )


def speaker_name(dialog: Big5ChatDialog) -> str:
    """Return the name of the trait-conditioned speaker (person-Y / ``train_output``)."""
    return dialog.persony or "speaker"


def partner_name(dialog: Big5ChatDialog) -> str:
    """Return the name of the partner in the dialogue (person-X / ``train_input``)."""
    return dialog.personx or "partner"


__all__ = [
    "Big5ChatDialog",
    "iter_big5_chat_dialogs",
    "partner_name",
    "speaker_name",
]
