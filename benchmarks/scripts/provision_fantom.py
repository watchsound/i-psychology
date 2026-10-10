"""Provision FANToM JSONL shim from the official AI2 Mosaic bucket.

**Why not HuggingFace.** The HF mirror ``clembench-playpen/fantom`` is
gated; attempting to pull it without an authenticated token raises
``DatasetNotFoundError``. The FANToM authors publish the raw tarball
unauthenticated at ``storage.googleapis.com/ai2-mosaic-public`` — the
exact URL the official `skywalker023/fantom` eval harness uses. We
mirror that behaviour here so provisioning works on a clean machine
without an HF account. Expected SHA256 is pinned to catch an upstream
rebuild silently changing semantics.

Normalises FANToM's native per-conversation record (one row per
`set_id` with `beliefQAs`, `factQA`, `answerabilityQAs_binary`,
`infoAccessibilityQAs_binary` groups) into one JSONL row per *question*
— the shape ``src/psychology_base/eval/benchmarks/public_datasets/
fantom.py`` consumes. Writes to ``.data/fantom/<split>.jsonl``
(gitignored).

Run::

    python benchmarks/scripts/provision_fantom.py              # full
    python benchmarks/scripts/provision_fantom.py --limit 50   # smoke
    python benchmarks/scripts/provision_fantom.py --dry-run    # print one shim row

**License.** FANToM is MIT per the official repo. The shim is a
derivative work and inherits MIT. ``.data/`` is gitignored to keep
transitive distribution clean.

**target_person extraction.** FANToM's raw belief QAs do not label a
``target_person`` field. The official eval script extracts it from the
question wording — ``"What does <Name> believe..."`` → ``<Name>`` via
``question.lower().split("does ")[1].split()[0]``. We replicate that
rule here. If a question text doesn't match the "What does Name" shape
(rare — most FANToM belief questions do), target_person falls back to
``joining_speaker`` for inaccessible belief questions (the character
who missed info) and ``None`` for accessible questions.

**question_type normalisation.** FANToM's raw ``question_type`` uses
colon-separated path strings like ``tom:belief:inaccessible``. We
collapse these to the shim's flat enum values
(``belief_inaccessible``, ``answerability``, ``info_accessibility``,
``fact``). ``:binary`` and ``:list`` sub-sub-types are flattened —
scorers that want to distinguish binary vs list should read the raw
field from ``metadata.raw_question_type``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".data" / "fantom"
RAW_DIR = DATA_DIR / "raw"

_TARBALL_URL = "https://storage.googleapis.com/ai2-mosaic-public/projects/fantom/fantom.tar.gz"
_TARBALL_SHA256 = "1d08dfa0ea474c7f83b9bc7e3a7b466eab25194043489dd618b4c5223e1253a4"
_RAW_JSON_FILENAME = "fantom_v1.json"


# ---------------------------------------------------------------------------
# Download + unpack
# ---------------------------------------------------------------------------


def _fetch_raw() -> Path:
    """Download the FANToM tarball and unpack to ``.data/fantom/raw/``.

    Idempotent: the second call sees the unpacked JSON and returns
    immediately. Hash is checked on fresh download; mismatch raises.
    """

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RAW_DIR / _RAW_JSON_FILENAME
    if json_path.exists():
        return json_path

    tarball = RAW_DIR / "fantom.tar.gz"
    print(f"Downloading {_TARBALL_URL}")
    urllib.request.urlretrieve(_TARBALL_URL, tarball)

    with tarball.open("rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    if got != _TARBALL_SHA256:
        raise RuntimeError(
            f"FANToM tarball SHA256 mismatch: expected {_TARBALL_SHA256}, got {got}"
        )

    with tarfile.open(tarball) as tar:
        tar.extractall(RAW_DIR)
    tarball.unlink()

    if not json_path.exists():
        raise RuntimeError(
            f"Expected {json_path} after unpacking; got {list(RAW_DIR.iterdir())}"
        )
    return json_path


# ---------------------------------------------------------------------------
# Parsing — raw row → one shim row per question
# ---------------------------------------------------------------------------


def _parse_context(full_context: str) -> list[dict[str, Any]]:
    """FANToM ``full_context`` is a newline-separated ``Speaker: utterance`` block."""
    out: list[dict[str, Any]] = []
    for line in full_context.splitlines():
        line = line.strip()
        if not line:
            continue
        speaker, _, body = line.partition(":")
        if body:
            out.append({"speaker": speaker.strip(), "utterance": body.strip()})
        else:
            out.append({"speaker": None, "utterance": line})
    return out


def _extract_participants(context: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for turn in context:
        speaker = turn.get("speaker")
        if speaker and speaker not in seen:
            seen.append(speaker)
    return seen


def _target_from_belief_question(text: str) -> str | None:
    """Replicate the official eval's target-character extraction.

    Official line (eval_fantom.py:352):
        belief_qas['target_character'] = belief_qas['question'].map(
            lambda x: x.lower().split("does ")[1].split()[0].lower()
        )

    We preserve the original case where possible by splitting the raw
    text, then returning the matching word; falls back to lowercase if
    the slice doesn't line up with the raw casing.
    """

    lowered = text.lower()
    if " does " not in lowered and not lowered.startswith("does "):
        return None
    try:
        after = text.split("oes ", 1)[1]  # case-insensitive-ish "does "
        candidate = after.split()[0]
        return candidate.strip(".,!?;:'\"")
    except (IndexError, ValueError):
        return None


def _target_from_binary_question(text: str) -> str | None:
    """"Does <Name> know..." → <Name>. Official rule, with light cleanup."""

    stripped = text.strip()
    if not stripped.lower().startswith("does "):
        return None
    rest = stripped[len("Does "):] if stripped[:5] == "Does " else stripped[5:]
    name = rest.split(" know")[0].strip()
    return name or None


def _normalize_qtype(raw_qtype: str) -> str:
    """Collapse FANToM's colon-path ``question_type`` to the shim's enum.

    Mapping:
        tom:belief:inaccessible                → belief_inaccessible
        tom:belief:inaccessible:multiple-choice → belief_inaccessible
        tom:belief:accessible                  → belief_accessible
        tom:answerability:*                    → answerability
        tom:info_accessibility:*               → info_accessibility
        fact                                   → fact
    Anything else returns the raw string so the adapter's
    ``FANToMQuestionType.from_shim`` falls back to None (defensive).
    """

    if raw_qtype == "fact":
        return "fact"
    parts = raw_qtype.split(":")
    if parts[:2] == ["tom", "belief"] and len(parts) >= 3:
        access = parts[2]
        if access == "inaccessible":
            return "belief_inaccessible"
        if access == "accessible":
            return "belief_accessible"
        return "belief"
    if parts[:2] == ["tom", "answerability"]:
        return "answerability"
    if parts[:2] == ["tom", "info_accessibility"]:
        return "info_accessibility"
    return raw_qtype


def _questions_from_row(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Expand one FANToM raw row into one normalised question dict per QA.

    Each yielded dict matches the ``_parse_question`` shape in
    ``public_datasets/fantom.py``. We expose both the normalised
    ``question_type`` and the raw ``raw_question_type`` under
    ``metadata`` (via the enclosing _row_to_shim) so scorers that need
    binary-vs-list granularity can reach for it.
    """

    joining_speaker = row.get("joining_speaker")

    # Belief QAs (per-target, inaccessible or accessible per entry)
    for entry in row.get("beliefQAs", []) or []:
        if not isinstance(entry, dict):
            continue
        raw_qtype = str(entry.get("question_type") or "")
        question_text = str(entry.get("question") or "")
        target = _target_from_belief_question(question_text)
        if target is None:
            # Inaccessible belief: the target is whoever "missed info" —
            # typically ``joining_speaker``. For accessible questions,
            # there is no canonical absent-target so we leave as None.
            access = entry.get("missed_info_accessibility")
            if access == "inaccessible":
                target = joining_speaker
        yield {
            "question_type": _normalize_qtype(raw_qtype),
            "question_text": question_text,
            "target_person": target,
            "correct_answer": entry.get("correct_answer"),
            "wrong_answer": entry.get("wrong_answer"),
            "tom_type": entry.get("tom_type"),
            "tom_order": None,
            "missed_info_accessibility": entry.get("missed_info_accessibility"),
            "choices": entry.get("choices") or entry.get("options"),
            "raw_question_type": raw_qtype,
        }

    # Factual question — one per row, exposed as dict (not list)
    fact_q = row.get("factQA")
    if isinstance(fact_q, dict) and fact_q.get("question"):
        yield {
            "question_type": "fact",
            "question_text": str(fact_q.get("question") or ""),
            "target_person": None,
            "correct_answer": fact_q.get("correct_answer"),
            "wrong_answer": fact_q.get("wrong_answer"),
            "tom_type": None,
            "tom_order": None,
            "missed_info_accessibility": None,
            "choices": None,
            "raw_question_type": "fact",
        }

    # Answerability binary + list
    for entry in row.get("answerabilityQAs_binary", []) or []:
        if not isinstance(entry, dict):
            continue
        qt = str(entry.get("question") or "")
        yield {
            "question_type": "answerability",
            "question_text": qt,
            "target_person": _target_from_binary_question(qt),
            "correct_answer": entry.get("correct_answer"),
            "wrong_answer": entry.get("wrong_answer"),
            "tom_type": None,
            "tom_order": None,
            "missed_info_accessibility": entry.get("missed_info_accessibility"),
            "choices": None,
            "raw_question_type": "tom:answerability:binary",
        }
    ans_list = row.get("answerabilityQA_list")
    if isinstance(ans_list, dict) and ans_list.get("question"):
        yield {
            "question_type": "answerability",
            "question_text": str(ans_list.get("question") or ""),
            "target_person": None,
            # list-type QAs store correct/wrong as list[str]; stringify so
            # the shim schema stays string|None.
            "correct_answer": _stringify_list(ans_list.get("correct_answer")),
            "wrong_answer": _stringify_list(ans_list.get("wrong_answer")),
            "tom_type": None,
            "tom_order": None,
            "missed_info_accessibility": ans_list.get("missed_info_accessibility"),
            "choices": None,
            "raw_question_type": "tom:answerability:list",
        }

    # Info accessibility binary + list
    for entry in row.get("infoAccessibilityQAs_binary", []) or []:
        if not isinstance(entry, dict):
            continue
        qt = str(entry.get("question") or "")
        yield {
            "question_type": "info_accessibility",
            "question_text": qt,
            "target_person": _target_from_binary_question(qt),
            "correct_answer": entry.get("correct_answer"),
            "wrong_answer": entry.get("wrong_answer"),
            "tom_type": None,
            "tom_order": None,
            "missed_info_accessibility": entry.get("missed_info_accessibility"),
            "choices": None,
            "raw_question_type": "tom:info_accessibility:binary",
        }
    info_list = row.get("infoAccessibilityQA_list")
    if isinstance(info_list, dict) and info_list.get("question"):
        yield {
            "question_type": "info_accessibility",
            "question_text": str(info_list.get("question") or ""),
            "target_person": None,
            "correct_answer": _stringify_list(info_list.get("correct_answer")),
            "wrong_answer": _stringify_list(info_list.get("wrong_answer")),
            "tom_type": None,
            "tom_order": None,
            "missed_info_accessibility": info_list.get("missed_info_accessibility"),
            "choices": None,
            "raw_question_type": "tom:info_accessibility:list",
        }


def _stringify_list(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def _row_to_shim(row: dict[str, Any], idx: int) -> dict[str, Any]:
    full_ctx = row.get("full_context") or ""
    context = _parse_context(str(full_ctx)) if isinstance(full_ctx, str) else []
    participants = _extract_participants(context)
    questions = list(_questions_from_row(row))
    conv_id = row.get("set_id") or row.get("conv_id") or f"fantom_{idx}"
    return {
        "conversation_id": str(conv_id),
        "participants": participants,
        "context": context,
        "questions": questions,
        "has_questions": bool(questions),
        "metadata": {
            "set_id": row.get("set_id"),
            "part_id": row.get("part_id"),
            "conv_id": row.get("conv_id"),
            "joining_speaker": row.get("joining_speaker"),
            "missed_info": row.get("missed_info"),
            "available_fields": sorted(k for k in row.keys()),
        },
    }


def _iter_shim_rows() -> Iterator[dict[str, Any]]:
    json_path = _fetch_raw()
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise RuntimeError(
            f"Expected list at top of {json_path}, got {type(data).__name__}"
        )
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        yield _row_to_shim(row, idx)


def provision(split: str = "test", limit: int | None = None) -> Path:
    """Write ``.data/fantom/<split>.jsonl``.

    FANToM is distributed as a single file (no split column in the raw
    data), so ``--split`` only controls the output filename. Downstream
    scorers can slice on `metadata.set_id` for reproducible subsamples.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{split}.jsonl"
    written = 0
    kept_empty = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for shim in _iter_shim_rows():
            if not shim["has_questions"]:
                kept_empty += 1
            fh.write(json.dumps(shim, ensure_ascii=False) + "\n")
            written += 1
            if limit is not None and written >= limit:
                break
    print(
        f"Wrote {written} rows to {out_path} "
        f"({written - kept_empty} with questions, {kept_empty} empty-but-retained)"
    )
    return out_path


def dry_run(split: str = "test") -> None:
    """Fetch, parse one row, print it. Useful when the schema shifts."""

    rows = _iter_shim_rows()
    first = next(rows, None)
    if first is None:
        print("No rows produced.")
        return
    print("=== First shim row ===")
    print(json.dumps(first, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(split=args.split)
    else:
        provision(split=args.split, limit=args.limit)
