"""Provision MathDial JSONL shim from the HuggingFace mirror.

Pulls `eth-nlped/mathdial` via HuggingFace datasets, parses the inline
move-tagged conversation into the shim format expected by
`psychology_base.eval.benchmarks.public_datasets.mathdial`, writes to
`.data/mathdial/train.jsonl` (gitignored).

Run::

    python benchmarks/scripts/provision_mathdial.py [--split train|validation|test]

Idempotent; re-running overwrites the shim.

**License.** MathDial is CC BY-SA 4.0. The shim is a derivative work and
inherits the share-alike obligation. `.data/` is gitignored precisely so
transitive distribution of our repo does not pull downstream users into
CC BY-SA via test fixtures. If you redistribute the shim, carry the
license forward.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".data" / "mathdial"


_MOVE_PATTERN = re.compile(r"^\((focus|probing|telling|generic)\)\s*", re.IGNORECASE)
_TURN_SEP = "|EOM|"


def _parse_conversation(raw: str) -> list[dict[str, Any]]:
    """MathDial native format: `Teacher: (move)text|EOM|Student: text|EOM|...`.

    Teacher turns carry an inline `(move)` prefix; student turns do not.
    """
    turns: list[dict[str, Any]] = []
    for chunk in raw.split(_TURN_SEP):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.lower().startswith("teacher:"):
            body = chunk[len("Teacher:"):].strip()
            m = _MOVE_PATTERN.match(body)
            if m is None:
                # Teacher turn without a move tag — happens for the opening
                # greeting. Skip since the alignment scorer only consumes
                # turns with gold moves.
                continue
            move = m.group(1).lower()
            utterance = _MOVE_PATTERN.sub("", body).strip()
            turns.append({"role": "teacher", "utterance": utterance, "move": move})
        elif chunk.lower().startswith("student:"):
            body = chunk[len("Student:"):].strip()
            turns.append({"role": "student", "utterance": body})
        else:
            # Malformed line — skip rather than fail the whole conversation.
            continue
    return turns


def _row_to_shim(row: dict[str, Any], idx: int) -> dict[str, Any]:
    turns = _parse_conversation(row.get("conversation", ""))
    self_correction = row.get("self-correctness")
    sc_bool: Any = None
    if isinstance(self_correction, str):
        sc_bool = self_correction.strip().lower() in {"yes", "true", "1"}
    return {
        "conversation_id": f"md_{row.get('qid', idx)}",
        "problem": row.get("question", ""),
        "ground_truth": row.get("ground_truth", ""),
        "student_incorrect_solution": row.get("student_incorrect_solution", ""),
        "self_correction": sc_bool,
        "turns": turns,
        "metadata": {
            "scenario": row.get("scenario"),
            "student_profile": row.get("student_profile"),
            "teacher_described_confusion": row.get("teacher_described_confusion"),
        },
    }


def _stream_rows(split: str) -> Iterator[dict[str, Any]]:
    ds = load_dataset("eth-nlped/mathdial", split=split)
    for idx, row in enumerate(ds):
        yield _row_to_shim(row, idx)


def provision(split: str = "train", limit: int | None = None) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{split}.jsonl"
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for shim in _stream_rows(split):
            if not shim["turns"]:
                # Drop empty conversations entirely.
                continue
            fh.write(json.dumps(shim, ensure_ascii=False) + "\n")
            written += 1
            if limit is not None and written >= limit:
                break
    print(f"Wrote {written} conversations to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    provision(split=args.split, limit=args.limit)
