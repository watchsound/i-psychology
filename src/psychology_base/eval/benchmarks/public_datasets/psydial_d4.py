"""PsyDial-D4 adapter — Chinese multi-turn counseling-training dialogues.

**Dataset provenance — honest attribution.** This corpus is NOT the
paper D4 (X-LANCE SJTU, EMNLP'22, 1339 psychiatrist-proofread
diagnostic dialogs, email-gated at ``binnieyao@gmail.com`` under DUA).
It is the public HuggingFace mirror ``qiuhuachuan/PsyDial-D4``
(Apache-2.0, observed 2026-10-10) — a **reconstructed-and-expanded**
Chinese counseling-training corpus (2382 dialogs, avg 37.8 turns) in
which the "counselor" side is a virtual AI counselor persona
("小天") trained on D4-shape conversations. Loading this corpus
under the "D4 slot" would mislabel the gate contribution; the Pillar F
sub-corpus ships it under its own name, documents the derivation, and
reports results as PsyDial-D4-specific — never as D4.

**Shape finding (2026-10-10 pin-shakeout).** Each record carries a
``messages`` array ``[{role: system|user|assistant, content: str}, ...]``.
``system`` is the counselor-persona instruction (meta, discarded).
``user`` turns are the help-seeker; ``assistant`` turns are the
AI-counselor target. Observed dialogs open with a greeting ritual
(``你好``, ``好的``, ``在我们正式开始之前麻烦你先填一下评估问卷``,
``好的，谢谢你。今天我们可以谈一谈``) spanning the first 2–4 user
turns before the first substantive client disclosure.

The adapter walks the ``messages`` array, skips the ``system`` turn,
and selects the first ``user`` turn with ``len(content) >= 20`` as
``seeker_opening`` (filters out ritual acknowledgements like
"你好"/"好的"/"嗯"/"是的"). The ``counselor_response`` is the first
``assistant`` turn **after** that user turn. Any later ``assistant``
turns in the dialog are preserved as ``alt_responses`` for future
multi-turn scorers; the first-pair-only scoring approach reuses the
PsyQA Chinese register scorer by design (same CN judge, same empathic
rate semantics).

**Shim schema** (one row per PsyDial-D4 record, JSONL)::

    {
      "example_id": "psydial-d4-00000",
      "split": "train",
      "n_messages": 42,
      "n_user_messages": 21,
      "n_assistant_messages": 21,
      "seeker_opening": "我想知道为什么自己现在会这么迷茫和空虚...",
      "counselor_response": "我明白了。你刚才提到了迷茫和空虚...",
      "seeker_opening_turn_index": 5,
      "counselor_response_turn_index": 6,
      "n_alt_responses": 20,
      "alt_responses": ["听起来你...", ...]
    }

Only ``example_id``, ``split``, ``seeker_opening``, and
``counselor_response`` are strictly required by the scorer; the rest
are preserved for diagnostic / multi-turn-scorer use.

**Not real human counseling.** The counselor side here is an idealized
AI-counselor training target. Cross-corpus reads should factor this
in: PsyQA refs are real online counselor voices; PsyDial-D4 refs are
curated training targets. Both can still be scored against the same
CN empathy judge for architectural-signature comparison, but the
reference distribution is NOT drawn from the same population.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class PsyDialD4Example:
    """One PsyDial-D4 Chinese counseling-training record.

    Duck-compatible with :class:`PsyQAExample` on the three attributes
    the shared Chinese scorer reads (``example_id``, ``seeker_opening``,
    ``counselor_response``) + ``keywords`` (consumed by
    :func:`make_psyqa_seeker`); ``keywords`` defaults to empty since
    PsyDial-D4 ships no topic tags.
    """

    example_id: str
    split: str
    seeker_opening: str
    counselor_response: str
    n_messages: int = 0
    n_user_messages: int = 0
    n_assistant_messages: int = 0
    seeker_opening_turn_index: int = -1
    counselor_response_turn_index: int = -1
    n_alt_responses: int = 0
    alt_responses: tuple[str, ...] = ()
    keywords: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("PsyDialD4Example.example_id must be non-empty")
        if not self.seeker_opening.strip():
            raise ValueError(
                f"PsyDialD4Example {self.example_id!r}: seeker_opening must be non-empty"
            )
        if not self.counselor_response.strip():
            raise ValueError(
                f"PsyDialD4Example {self.example_id!r}: counselor_response must be non-empty"
            )


def iter_psydial_d4_examples(
    shim_path: Path,
    *,
    limit: Optional[int] = None,
) -> Iterator[PsyDialD4Example]:
    """Stream PsyDial-D4 examples from a JSONL shim file.

    ``limit`` caps the yielded count — useful for smoke runs where the
    full 2382-row corpus is too expensive.
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


def _parse_example(obj: dict, line_no: int) -> PsyDialD4Example:
    required = ("example_id", "split", "seeker_opening", "counselor_response")
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(
            f"PsyDial-D4 shim line {line_no}: missing required fields {missing}"
        )
    alt = obj.get("alt_responses") or []
    return PsyDialD4Example(
        example_id=str(obj["example_id"]),
        split=str(obj["split"]),
        seeker_opening=str(obj["seeker_opening"]),
        counselor_response=str(obj["counselor_response"]),
        n_messages=int(obj.get("n_messages", 0) or 0),
        n_user_messages=int(obj.get("n_user_messages", 0) or 0),
        n_assistant_messages=int(obj.get("n_assistant_messages", 0) or 0),
        seeker_opening_turn_index=int(obj.get("seeker_opening_turn_index", -1)),
        counselor_response_turn_index=int(obj.get("counselor_response_turn_index", -1)),
        n_alt_responses=int(obj.get("n_alt_responses", 0) or 0),
        alt_responses=tuple(str(a) for a in alt if a),
        keywords=str(obj.get("keywords", "") or ""),
        metadata=dict(obj.get("metadata", {})),
    )


__all__ = [
    "PsyDialD4Example",
    "iter_psydial_d4_examples",
]
