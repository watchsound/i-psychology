"""PsyQA adapter — Chinese long-form counseling Q&A stimulus-response pairs.

**Dataset provenance.** PsyQA = Chinese psychological counseling Q&A
corpus, Liu et al. 2023 (companion to ESC paper, EMNLP'21). The
upstream Tsinghua COAI release gates access behind an email form; the
``lsy641/PsyQA`` HuggingFace mirror ships the same records MIT-licensed
auth-free (observed 2026-10-10). See
``benchmarks/scripts/provision_psyqa.py`` for the shim writer.

**Shape finding (2026-10-10 pin-shakeout).** Each PsyQA record is one
Chinese help-seeking question (short ``question`` + longer
``description`` of circumstances) with one or more long-form
``counselor_response`` text(s). Shim schema flattens to one row per
question — the first answer is the ``counselor_response`` reference
and extra answers (if any) are preserved as ``alt_responses`` for
future scorers (ensembling / cross-reference agreement).

For the Pillar F counseling-depth sweep this adapter behaves identically
to the ExTES adapter: emit a one-stimulus-one-reference loader; the
scorer runs the Companion against an LLM-simulated help-seeker, scores
Bond + LLM-judge empathy, computes κ-across-conversants against the
reference. The one PsyQA-specific contract is Chinese register — the
seeker persona, judge system prompt, and few-shot anchors in the
scorer are all authored in Chinese.

**Shim schema** (one row per PsyQA record, JSONL)::

    {
      "example_id": "psyqa-test-00000",
      "split": "test",
      "question_id": 10081,
      "question": "大学生不玩网混网却遭受网络暴力...",
      "description": "被网络黑社会威胁...",
      "keywords": "人际,社交软件",
      "counselor_response": "~~给题主一个温暖的抱抱...",
      "has_label": true,
      "n_alt_responses": 0,
      "alt_responses": []
    }

**No LLM dependency.** This adapter is a flat loader; the Pillar F
scorer (``psyqa_counseling_depth.py``) handles live routing + judge
wiring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class PsyQAExample:
    """One PsyQA Chinese help-seeking record."""

    example_id: str
    split: str
    question: str
    description: str
    keywords: str
    counselor_response: str
    question_id: Optional[int] = None
    has_label: bool = False
    n_alt_responses: int = 0
    alt_responses: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("PsyQAExample.example_id must be non-empty")
        if not self.question.strip():
            raise ValueError(
                f"PsyQAExample {self.example_id!r}: question must be non-empty"
            )
        if not self.counselor_response.strip():
            raise ValueError(
                f"PsyQAExample {self.example_id!r}: counselor_response must be non-empty"
            )

    @property
    def seeker_opening(self) -> str:
        """The utterance the simulated help-seeker opens with.

        Combines ``question`` + ``description`` into a single opening
        line — PsyQA records often carry the real emotional pressure in
        the longer ``description`` field, not the short ``question``
        header. The two are joined with a newline so the seeker's first
        turn reads as a complete help-seeking message.
        """
        q = self.question.strip()
        d = self.description.strip()
        if not d:
            return q
        return f"{q}\n{d}"


def iter_psyqa_examples(
    shim_path: Path,
    *,
    limit: Optional[int] = None,
) -> Iterator[PsyQAExample]:
    """Stream PsyQA examples from a JSONL shim file.

    ``limit`` caps the yielded count — useful for smoke runs where the
    full 2235-row test split is too expensive.
    """
    shim_path = Path(shim_path)
    yielded = 0
    with shim_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            yield _parse_example(obj, line_no)
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _parse_example(obj: dict, line_no: int) -> PsyQAExample:
    required = ("example_id", "split", "question", "description", "counselor_response")
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(
            f"PsyQA shim line {line_no}: missing required fields {missing}"
        )
    alt = obj.get("alt_responses") or []
    return PsyQAExample(
        example_id=str(obj["example_id"]),
        split=str(obj["split"]),
        question=str(obj["question"]),
        description=str(obj["description"]),
        keywords=str(obj.get("keywords", "")),
        counselor_response=str(obj["counselor_response"]),
        question_id=obj.get("question_id"),
        has_label=bool(obj.get("has_label", False)),
        n_alt_responses=int(obj.get("n_alt_responses", 0) or 0),
        alt_responses=tuple(str(a) for a in alt if a),
        metadata=dict(obj.get("metadata", {})),
    )


__all__ = [
    "PsyQAExample",
    "iter_psyqa_examples",
]
