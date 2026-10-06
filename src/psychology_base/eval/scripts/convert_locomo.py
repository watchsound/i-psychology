"""Convert the official LoCoMo JSON release to our JSONL shim.

The repo (``snap-research/locomo``, arXiv 2402.17753) ships
``locomo10.json`` as a top-level JSON array of ten dialogue instances.
Each instance has:

  - a ``conversation`` object: fields ``session_1``, ``session_2``, ...
    mapping to a list of turns; each turn has ``speaker`` + ``text``.
    Session timestamps live in sibling fields named
    ``session_1_date_time``, ``session_2_date_time``, ... alongside the
    conversation turns.
  - a ``qa`` list: each entry has ``question``, ``answer``,
    ``category`` (one of five), ``adversarial_answer`` (for category 5),
    and ``evidence`` (list of session names).

This converter reshapes each dialogue into our ordered JSONL shim schema
documented in :mod:`psychology_base.eval.benchmarks.locomo`. Sessions are
reindexed into a numerically-ordered list (``session_1`` before
``session_2``) so the adapter's observe order is deterministic.

Why JSONL and not direct JSON-array loading? Same reason as the
LongMemEval converter: streaming keeps memory flat and matches the
adapter's existing input convention.

Usage::

    python -m psychology_base.eval.scripts.convert_locomo \\
        --input ~/data/locomo10.json \\
        --output ~/.cache/psychology_base/benchmarks/locomo/locomo_shim.jsonl

Tolerant: a malformed instance is dropped with a stderr note, not fatal.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator


_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")

# Native LoCoMo stores ``category`` as the integer code 1..5 (string- or
# int-typed depending on serializer). The adapter's judge/F1 dispatcher
# keys on the semantic category name — ``single-hop`` goes through
# deterministic F1 (Mem0 92.5% is single-hop F1), the rest go through the
# LLM judge. Decode here so the shim files land already named and the
# adapter doesn't need to know about the native coding. Mapping verified
# by sampling questions per category on the real locomo10.json
# (2026-10-06):
#   1 → short-form single-fact ("What did X research?" / "Adoption
#       agencies")                                             → single-hop
#   2 → temporal ("When did X go to Y?" / "7 May 2023")        → temporal
#   3 → open-domain / commonsense inference                    → open-domain
#   4 → longer-form reasoning answers (multi-session grounded) → multi-hop
#   5 → adversarial yes/no with inverted trap in `answer`
#       field + `adversarial_answer` holding the plausible trap → adversarial
# Unknown codes pass through untouched so downstream code can log them.
_CATEGORY_MAP = {
    "1": "single-hop",
    "2": "temporal-reasoning",
    "3": "open-domain",
    "4": "multi-hop",
    "5": "adversarial",
}


def _decode_category(raw: object) -> str:
    key = str(raw).strip()
    return _CATEGORY_MAP.get(key, key)


def _iter_instances(payload: Any) -> Iterator[dict]:
    if isinstance(payload, list):
        yield from (p for p in payload if isinstance(p, dict))
    elif isinstance(payload, dict) and isinstance(payload.get("instances"), list):
        yield from (p for p in payload["instances"] if isinstance(p, dict))
    else:
        raise ValueError(
            "Unrecognized LoCoMo JSON structure. Expected a top-level array "
            "of dialogue objects, or an object with an 'instances' array."
        )


def _extract_sessions(conversation: dict) -> list[dict]:
    """Pull ``session_N`` keys out of the dialogue object, sorted
    numerically, with the matching ``session_N_date_time`` sibling pulled
    into each session's ``date`` field.

    Turns in the native format use ``speaker`` + ``text``; we normalize
    the text field name to ``content`` because our shim schema does.
    """
    sessions: list[tuple[int, str, list[dict], str]] = []
    for key, value in conversation.items():
        m = _SESSION_KEY_RE.match(key)
        if not m:
            continue
        if not isinstance(value, list):
            continue
        n = int(m.group(1))
        date = str(conversation.get(f"{key}_date_time", ""))
        sessions.append((n, key, value, date))
    sessions.sort(key=lambda t: t[0])

    out: list[dict] = []
    for _, sid, turns, date in sessions:
        shim_turns = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            speaker = turn.get("speaker")
            content = turn.get("text", turn.get("content"))
            if speaker is None or content is None:
                continue
            shim_turns.append({"speaker": str(speaker), "content": str(content)})
        out.append({"session_id": sid, "date": date, "turns": shim_turns})
    return out


def _project_questions(qa_list: list[dict], instance_id: str) -> list[dict]:
    """Convert LoCoMo's ``qa`` entries into our shim's ``questions``.

    Question IDs are not stable across the native format (no explicit
    field), so we synthesize ``<instance_id>-q<index>``. The adapter keys
    the judge cache on these, so stable synthesis matters for cache reuse
    across converter re-runs — don't reshuffle the input order.
    """
    out: list[dict] = []
    for i, qa in enumerate(qa_list):
        if not isinstance(qa, dict):
            continue
        if "question" not in qa or "answer" not in qa:
            continue
        out.append(
            {
                "id": f"{instance_id}-q{i}",
                "question": str(qa["question"]),
                "answer": str(qa["answer"]),
                "category": _decode_category(qa.get("category", "")),
                "adversarial_answer": str(qa.get("adversarial_answer", "")),
                "evidence": [str(e) for e in qa.get("evidence") or ()],
            }
        )
    return out


def _project_instance(raw: dict, idx: int) -> dict | None:
    conversation = raw.get("conversation")
    qa = raw.get("qa")
    if not isinstance(conversation, dict) or not isinstance(qa, list):
        return None
    instance_id = str(raw.get("sample_id") or raw.get("instance_id") or f"locomo-{idx}")
    sessions = _extract_sessions(conversation)
    if not sessions:
        return None
    questions = _project_questions(qa, instance_id)
    if not questions:
        return None
    return {
        "instance_id": instance_id,
        "sessions": sessions,
        "questions": questions,
    }


def convert(src: Path, dst: Path) -> tuple[int, int]:
    with src.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    dst.parent.mkdir(parents=True, exist_ok=True)
    input_count = 0
    output_count = 0
    dropped: list[str] = []
    with dst.open("w", encoding="utf-8") as out:
        for idx, raw in enumerate(_iter_instances(payload)):
            input_count += 1
            projected = _project_instance(raw, idx)
            if projected is None:
                dropped.append(
                    str(raw.get("sample_id") or raw.get("instance_id") or f"idx-{idx}")
                )
                continue
            out.write(json.dumps(projected, ensure_ascii=False))
            out.write("\n")
            output_count += 1

    if dropped:
        print(
            f"dropped {len(dropped)} instance(s) missing conversation/qa: "
            f"{dropped[:5]}{'...' if len(dropped) > 5 else ''}",
            file=sys.stderr,
        )
    return input_count, output_count


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convert_locomo",
        description=(
            "Convert official LoCoMo JSON (snap-research/locomo locomo10.json) "
            "to the adapter's JSONL shim."
        ),
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the official LoCoMo JSON file (locomo10.json).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the JSONL shim.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"input file not found: {args.input}", file=sys.stderr)
        return 2
    input_count, output_count = convert(args.input, args.output)
    print(
        f"converted {output_count}/{input_count} instances from "
        f"{args.input} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
