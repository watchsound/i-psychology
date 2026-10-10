"""Provision ESConv JSONL shim from the HuggingFace mirror.

Pulls `thu-coai/esconv` via HuggingFace datasets, maps the native strategy
labels to our `ESConvStrategy` enum values, and writes shim rows to
`.data/esconv/<split>.jsonl` (gitignored).

Run::

    python benchmarks/scripts/provision_esconv.py [--split train|validation|test]

Idempotent; re-running overwrites the shim.

**License.** ESConv is CC BY-NC 4.0 (non-commercial). The shim is a
derivative work and inherits the non-commercial restriction. `.data/` is
gitignored to keep transitive repo redistribution from pulling downstream
users into CC BY-NC. If you redistribute the shim, carry the license
forward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".data" / "esconv"


# Native ESConv strategy label -> our ESConvStrategy enum value
_STRATEGY_MAP = {
    "Question": "question",
    "Others": "others",
    "Affirmation and Reassurance": "affirmation_and_reassurance",
    "Providing Suggestions": "providing_suggestions",
    "Self-disclosure": "self_disclosure",
    "Reflection of feelings": "reflection_of_feelings",
    "Information": "information",
    "Restatement or Paraphrasing": "restatement",
}


def _parse_dialog(dialog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map ESConv dialog turns to our shim schema.

    - `speaker="usr"` → `role="seeker"`
    - `speaker="sys"` → `role="supporter"` + optional `strategy` mapped to enum
    """
    turns: list[dict[str, Any]] = []
    for raw in dialog:
        speaker = raw.get("speaker")
        text = raw.get("text", "")
        if speaker == "usr":
            turns.append({"role": "seeker", "utterance": text})
        elif speaker == "sys":
            native_strategy = raw.get("strategy")
            shim_turn: dict[str, Any] = {"role": "supporter", "utterance": text}
            if native_strategy is not None:
                mapped = _STRATEGY_MAP.get(native_strategy)
                if mapped is None:
                    # Unknown strategy — skip the tag rather than fail the
                    # whole conversation; the scorer treats missing-strategy
                    # turns as unlabelled.
                    continue
                shim_turn["strategy"] = mapped
            turns.append(shim_turn)
        else:
            # Unknown speaker — skip.
            continue
    return turns


def _survey_score(raw_survey: Any) -> float | None:
    """ESConv survey is nested; collapse to a single scalar if possible.

    Use seeker.final_emotion_intensity as the proxy score — the dataset's own
    outcome measure. Return None if missing/malformed. (Keeps the shim
    `survey_score` field type-stable with `Optional[float]`.)
    """
    if not isinstance(raw_survey, dict):
        return None
    seeker = raw_survey.get("seeker")
    if not isinstance(seeker, dict):
        return None
    final = seeker.get("final_emotion_intensity")
    try:
        return float(final) if final is not None else None
    except (TypeError, ValueError):
        return None


def _row_to_shim(row: dict[str, Any], idx: int) -> dict[str, Any]:
    native = json.loads(row["text"])
    turns = _parse_dialog(native.get("dialog", []))
    return {
        "conversation_id": f"esc_{idx}",
        "emotion_type": native.get("emotion_type", ""),
        "problem_type": native.get("problem_type", ""),
        "situation": native.get("situation", ""),
        "turns": turns,
        "survey_score": _survey_score(native.get("survey_score")),
        "metadata": {
            "experience_type": native.get("experience_type"),
            "seeker_question1": native.get("seeker_question1"),
            "seeker_question2": native.get("seeker_question2"),
            "supporter_question1": native.get("supporter_question1"),
            "supporter_question2": native.get("supporter_question2"),
        },
    }


def _stream_rows(split: str) -> Iterator[dict[str, Any]]:
    ds = load_dataset("thu-coai/esconv", split=split)
    for idx, row in enumerate(ds):
        yield _row_to_shim(row, idx)


def provision(split: str = "train", limit: int | None = None) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{split}.jsonl"
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for shim in _stream_rows(split):
            if not shim["turns"]:
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
