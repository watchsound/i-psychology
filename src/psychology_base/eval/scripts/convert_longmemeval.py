"""Convert the official LongMemEval JSON release to our JSONL shim.

The repo (``xiaowu0162/LongMemEval``, Hugging Face dataset ``xiaowu0162/longmemeval``)
ships three JSON array files:

  - ``longmemeval_s.json``       — short-haystack split (~30 sessions / instance)
  - ``longmemeval_m.json``       — medium-haystack split (~50 sessions / instance)
  - ``longmemeval_oracle.json``  — oracle split (answer sessions pre-labeled)

Each file is a single JSON array; each element is one evaluation instance
with the schema the adapter already consumes
(:mod:`psychology_base.eval.benchmarks.longmemeval`): ``question_id``,
``question_type``, ``question``, ``answer``, ``question_date``,
``haystack_sessions``, ``haystack_session_ids``, ``haystack_dates``,
``answer_session_ids``.

This converter is a one-shot pipe: load the array, drop any element
missing the required fields, emit one JSON object per line. The output is
what :class:`LongMemEvalLoader` streams from.

Why JSONL and not direct JSON-array loading? Memory. Each JSON file is
hundreds of megabytes; streaming line-by-line keeps the harness's memory
flat across the full 500-instance run.

Usage::

    python -m psychology_base.eval.scripts.convert_longmemeval \\
        --input ~/data/longmemeval_m.json \\
        --output ~/.cache/psychology_base/benchmarks/longmemeval/longmemeval_shim.jsonl

The script is intentionally tolerant: a missing or malformed instance is
dropped with a stderr note, not fatal. Count mismatches between input and
output are reported at the end so a silent-drop regression is visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator


# Fields the adapter requires. Instances missing any of these are dropped
# with a stderr note — the real data has stray partials from early paper
# revisions and we don't want them silently inflating the instance count.
_REQUIRED_FIELDS = ("question_id", "question", "answer", "haystack_sessions")

# Fields the adapter uses but tolerates missing (filled with sensible
# defaults by the loader). Listed for documentation; converter does not
# enforce.
_OPTIONAL_FIELDS = (
    "question_type",
    "question_date",
    "haystack_session_ids",
    "haystack_dates",
    "answer_session_ids",
)


def _iter_instances(payload: Any) -> Iterator[dict]:
    """LongMemEval ships a top-level JSON array. If a release ever flips to
    an object-with-'instances' envelope, handle that too so this converter
    doesn't need re-cutting."""
    if isinstance(payload, list):
        yield from (p for p in payload if isinstance(p, dict))
    elif isinstance(payload, dict) and isinstance(payload.get("instances"), list):
        yield from (p for p in payload["instances"] if isinstance(p, dict))
    else:
        raise ValueError(
            "Unrecognized LongMemEval JSON structure. Expected a top-level "
            "array of instance objects, or an object with an 'instances' array."
        )


def _is_valid(instance: dict) -> bool:
    for key in _REQUIRED_FIELDS:
        if key not in instance:
            return False
    sessions = instance["haystack_sessions"]
    if not isinstance(sessions, list):
        return False
    # Each session is a list of {role, content} turns.
    for session in sessions:
        if not isinstance(session, list):
            return False
        for turn in session:
            if not isinstance(turn, dict):
                return False
            if "role" not in turn or "content" not in turn:
                return False
    return True


def _project(instance: dict) -> dict:
    """Keep only the fields the adapter reads; drop anything else to keep
    the JSONL slim. Preserves field order for a stable diff when the real
    dataset is reconverted."""
    out: dict = {}
    for key in _REQUIRED_FIELDS + _OPTIONAL_FIELDS:
        if key in instance:
            out[key] = instance[key]
    return out


def convert(src: Path, dst: Path) -> tuple[int, int]:
    """Read ``src`` (a LongMemEval JSON array file), write ``dst`` as JSONL.

    Returns ``(input_count, output_count)``. The gap is the dropped-instance
    count, which the caller surfaces for operator inspection.
    """
    with src.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    dst.parent.mkdir(parents=True, exist_ok=True)
    input_count = 0
    output_count = 0
    dropped: list[str] = []
    with dst.open("w", encoding="utf-8") as out:
        for instance in _iter_instances(payload):
            input_count += 1
            if not _is_valid(instance):
                dropped.append(str(instance.get("question_id", "<no-id>")))
                continue
            projected = _project(instance)
            out.write(json.dumps(projected, ensure_ascii=False))
            out.write("\n")
            output_count += 1

    if dropped:
        print(
            f"dropped {len(dropped)} instance(s) missing required fields: "
            f"{dropped[:5]}{'...' if len(dropped) > 5 else ''}",
            file=sys.stderr,
        )
    return input_count, output_count


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convert_longmemeval",
        description=(
            "Convert official LongMemEval JSON (xiaowu0162/longmemeval) to "
            "the adapter's JSONL shim."
        ),
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the official LongMemEval JSON file (longmemeval_{s,m,oracle}.json).",
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
