"""Provision MSC JSONL shim from a HuggingFace mirror.

**Dataset provenance.** Xu et al. 2022 "Beyond Goldfish Memory: Long-Term
Open-Domain Conversation" (ACL 2022; arXiv 2107.07567). The canonical
distribution is through ParlAI at https://parl.ai/projects/msc/. The HF
mirrors are community-maintained reuploads of the same underlying tarball
with varying schema conventions; this script accepts a configurable
``--hf-dataset`` so a user can swap mirrors when one goes stale without
editing code.

**License.** CC BY-NC 4.0 per the MSC paper + ParlAI distribution. The
shim is a derivative work and inherits the non-commercial restriction.
``.data/`` is gitignored to keep transitive repo redistribution clean. If
you redistribute the shim, carry the license forward.

**Why no SHA256 pin.** Unlike the FANToM provisioning script which pulls
a fixed tarball from an immutable GCS URL, HuggingFace mirrors are
rebuildable and the ``datasets`` library's content hash is library-version
dependent. We pin instead by ``revision=<commit sha>`` when provided
(``--hf-revision``); when not provided the script fetches ``main`` and
records the resolved commit sha in the output shim's metadata so the pin
can be inferred post-hoc from the shipped artefact.

**Raw schema variance.** HF MSC mirrors differ in how they serialise the
5-session structure:
  - Shape A: one row per *session*, each row carries ``previous_dialogs``
    with prior sessions; ``conv_id`` / ``initial_data_id`` groups.
  - Shape B: one row per *conversation*, each row carries a ``sessions``
    list.
  - Shape C: one row per *session* with parallel ``dialogue``/``speaker``
    lists, grouped by ``dialoug_id`` (sic — the field is misspelt in the
    ``nayohan/multi_session_chat`` mirror) + ordered by ``session_id``.
    Personas live under ``persona1``/``persona2``. No per-session time
    interval; the provisioner falls back to the MSC-paper canonical gap
    schedule (see :data:`_SHAPE_C_SESSION_GAP_DAYS`).

This script detects shape at parse time and normalises all three to our
one-conversation-per-line shim schema (documented in
``src/psychology_base/eval/benchmarks/public_datasets/msc.py``).

**Smoke flow.** First invocation on a new machine::

    python benchmarks/scripts/provision_msc.py --dry-run --hf-dataset <name>

The dry run fetches one row, parses it, prints a redacted shim preview,
and exits — zero data on disk. Once the schema looks right::

    python benchmarks/scripts/provision_msc.py --hf-dataset <name> --limit 50

Writes ``.data/msc/train.jsonl`` (gitignored).

**Shim contract.** One MSC conversation per line, matching
``src/psychology_base/eval/benchmarks/public_datasets/msc.py``'s loader
schema. ``time_interval_days`` is parsed from MSC's natural-language gap
strings via :func:`_parse_time_interval` — ``"3 days"`` → ``3.0``,
``"about a week"`` → ``7.0``, missing/unparseable → ``0.0`` with a
warning counted in the summary.

**Mapping speaker ids.** MSC's native rows typically identify speakers by
``id`` fields or alternating position. We canonicalise to
``speaker_idx=0`` for the self-speaker (persona bullet set 0,
seeker-equivalent) and ``speaker_idx=1`` for the other-speaker (persona
bullet set 1, companion-equivalent). The mapping rule is "first speaker
in session 1 → speaker_idx=0"; adjust via ``--flip-speaker`` if the gate
needs the opposite polarity.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".data" / "msc"

_DEFAULT_HF_DATASET = "nayohan/multi_session_chat"
_DEFAULT_SPLIT = "train"

# Shape C mirrors drop MSC's natural-language gap labels. Fall back to a
# canonical gap schedule approximating the MSC paper's design (session 2
# ≈ hours after session 1; session 3 ≈ 1–3 days later; session 4 ≈ 1–7
# days later; session 5 ≈ 1–14 days later). The exact gap values don't
# gate Pillar A — session-5 Bond does — but non-trivial multi-week
# in-world time advance IS the signal that lets the memory substrate's
# aging policy run between sessions. These are the ``time_interval_days``
# values the shim writes for sessions 1..5 (session 1 is anchor = 0).
_SHAPE_C_SESSION_GAP_DAYS: tuple[float, ...] = (0.0, 0.25, 2.0, 5.0, 10.0)


# ---------------------------------------------------------------------------
# Time-interval parsing
# ---------------------------------------------------------------------------


_INT_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _parse_time_interval(raw: Any) -> float:
    """Convert MSC's natural-language gap to float days.

    Rules (order matters):
      - numeric → float
      - "N day(s)" / "N days" → N
      - "N hour(s)" → N / 24
      - "N week(s)" → N × 7
      - "N month(s)" → N × 30
      - any string with a leading integer → integer (coarse but safe)
      - anything else → 0.0

    Missing / unparseable returns 0.0 — the caller tracks the count so a
    high ratio surfaces as a provisioning warning, not a silent drop.
    """
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, str):
        return 0.0
    s = raw.strip().lower()
    if not s:
        return 0.0
    m = _INT_RE.search(s)
    if m is None:
        return 0.0
    try:
        n = float(m.group(1))
    except ValueError:
        return 0.0
    if "hour" in s:
        return n / 24.0
    if "week" in s:
        return n * 7.0
    if "month" in s:
        return n * 30.0
    # "day" or no explicit unit — default to days
    return n


# ---------------------------------------------------------------------------
# Shape detection + per-shape parsers
# ---------------------------------------------------------------------------


def _detect_shape(row: dict[str, Any]) -> str:
    """Return "A" (per-session w/ previous_dialogs) / "B" (per-conv) / "C"
    (per-session w/ dialoug_id + parallel dialogue+speaker) / "unknown".
    """
    if "previous_dialogs" in row and isinstance(row["previous_dialogs"], list):
        return "A"
    if "sessions" in row and isinstance(row["sessions"], list):
        return "B"
    # Some mirrors use "session_history" instead of "previous_dialogs"
    if "session_history" in row and isinstance(row["session_history"], list):
        return "A"
    # Shape C: nayohan/multi_session_chat and similar mirrors — parallel
    # dialogue + speaker lists grouped by dialoug_id.
    has_group_id = (
        "dialoug_id" in row or "dialog_id" in row or "conversation_id" in row
    )
    has_parallel = isinstance(row.get("dialogue"), list) and isinstance(
        row.get("speaker"), list
    )
    if has_group_id and has_parallel:
        return "C"
    return "unknown"


def _extract_personas(row: dict[str, Any]) -> list[list[str]]:
    """Pull the 2 persona bullet-lists in a shape-tolerant way."""
    candidates = (
        "personas",
        "init_personas",
        "personachat_personas",
        "your_persona",
    )
    for key in candidates:
        v = row.get(key)
        if isinstance(v, list) and len(v) >= 2:
            def _as_bullets(x: Any) -> list[str]:
                if isinstance(x, list):
                    return [str(b) for b in x if b is not None]
                if isinstance(x, str):
                    # some mirrors store a newline-joined bullet string
                    return [line.strip() for line in x.split("\n") if line.strip()]
                return []

            return [_as_bullets(v[0]), _as_bullets(v[1])]
    return [[], []]


def _extract_dialog_turns(dialog: Any) -> list[dict[str, Any]]:
    """Normalise one session's turns to list[{speaker_idx, utterance}].

    Accepts several raw shapes:
      - [{"id": "0", "text": "..."}, {"id": "1", "text": "..."}, ...]
      - [{"speaker": 0, "text": "..."}, ...]
      - [{"speaker_idx": 0, "utterance": "..."}, ...]
      - [{"role": "self", "text": "..."}, ...]
    """
    if not isinstance(dialog, list):
        return []
    turns: list[dict[str, Any]] = []
    first_speaker: str | int | None = None
    for raw in dialog:
        if not isinstance(raw, dict):
            continue
        text = (
            raw.get("utterance")
            or raw.get("text")
            or raw.get("content")
            or ""
        )
        if not isinstance(text, str) or not text.strip():
            continue
        speaker = (
            raw.get("speaker_idx")
            if "speaker_idx" in raw
            else raw.get("speaker")
            if "speaker" in raw
            else raw.get("id")
            if "id" in raw
            else raw.get("role")
        )
        if first_speaker is None and speaker is not None:
            first_speaker = speaker
        if speaker is None:
            # Alternating fallback — first turn is self, then flip.
            idx = len(turns) % 2
        else:
            try:
                idx = int(speaker)
            except (TypeError, ValueError):
                idx = 0 if speaker == first_speaker else 1
        turns.append({"speaker_idx": idx, "utterance": text.strip()})
    return turns


def _parse_shape_a_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Shape A: one row per session, grouped by conv_id.

    The last-session row typically carries ``previous_dialogs`` with every
    prior session's full dialog + ``time_interval`` string. We reconstruct
    the chain from that payload rather than merging many rows: it keeps
    one-row-per-conversation parsing + avoids a merge-key reliability
    hazard across HF mirrors.
    """
    # Prefer the row with the longest previous_dialogs — that is the
    # final session for the chain in all MSC mirrors I've seen.
    group_sorted = sorted(
        group,
        key=lambda r: len(r.get("previous_dialogs") or r.get("session_history") or []),
        reverse=True,
    )
    anchor = group_sorted[0]
    conv_id = str(
        anchor.get("conv_id")
        or anchor.get("initial_data_id")
        or anchor.get("set_id")
        or anchor.get("id")
        or "msc_unknown"
    )
    personas = _extract_personas(anchor)

    sessions: list[dict[str, Any]] = []
    prev = anchor.get("previous_dialogs") or anchor.get("session_history") or []
    for sidx, prev_session in enumerate(prev):
        if not isinstance(prev_session, dict):
            continue
        dialog = prev_session.get("dialog") or prev_session.get("turns") or []
        turns = _extract_dialog_turns(dialog)
        if not turns:
            continue
        sessions.append({
            "session_id": sidx + 1,
            "time_interval_days": _parse_time_interval(
                prev_session.get("time_interval") or prev_session.get("time_num")
            ),
            "turns": turns,
        })
    # The current session (the row's own ``dialog``) is the last entry.
    cur_dialog = anchor.get("dialog") or anchor.get("turns") or []
    cur_turns = _extract_dialog_turns(cur_dialog)
    if cur_turns:
        sessions.append({
            "session_id": len(sessions) + 1,
            "time_interval_days": _parse_time_interval(
                anchor.get("time_interval") or anchor.get("time_num")
            ),
            "turns": cur_turns,
        })
    return {
        "conversation_id": conv_id,
        "personas": personas,
        "sessions": sessions,
        "metadata": {
            "shape": "A",
            "source_rows_in_group": len(group),
        },
    }


def _parse_shape_b_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
    """Shape B: one row per conversation, carrying a ``sessions`` list."""
    conv_id = str(
        row.get("conv_id") or row.get("conversation_id") or f"msc_{idx}"
    )
    personas = _extract_personas(row)
    sessions_out: list[dict[str, Any]] = []
    for sidx, raw_session in enumerate(row.get("sessions") or []):
        if not isinstance(raw_session, dict):
            continue
        dialog = (
            raw_session.get("dialog")
            or raw_session.get("turns")
            or raw_session.get("utterances")
            or []
        )
        turns = _extract_dialog_turns(dialog)
        if not turns:
            continue
        sessions_out.append({
            "session_id": raw_session.get("session_id") or (sidx + 1),
            "time_interval_days": _parse_time_interval(
                raw_session.get("time_interval")
                or raw_session.get("time_num")
                or raw_session.get("time_interval_days")
            ),
            "turns": turns,
        })
    return {
        "conversation_id": conv_id,
        "personas": personas,
        "sessions": sessions_out,
        "metadata": {"shape": "B"},
    }


def _shape_c_personas(row: dict[str, Any]) -> list[list[str]]:
    """Shape C stores personas under flat ``persona1`` / ``persona2`` keys."""
    def _as_bullets(x: Any) -> list[str]:
        if isinstance(x, list):
            return [str(b).strip() for b in x if b is not None and str(b).strip()]
        if isinstance(x, str):
            return [line.strip() for line in x.split("\n") if line.strip()]
        return []

    p1 = _as_bullets(row.get("persona1"))
    p2 = _as_bullets(row.get("persona2"))
    return [p1, p2]


_SHAPE_C_SPEAKER_LABEL_RE = re.compile(r"(\d+)")


def _shape_c_speaker_idx(label: Any) -> int:
    """Map "Speaker 1"/"Speaker 2" (or similar) to speaker_idx 0/1.

    MSC paper's first-speaker-is-self convention: "Speaker 1" → 0,
    "Speaker 2" → 1. Falls back to alternating position via the caller.
    """
    if isinstance(label, (int, float)):
        n = int(label)
        return 0 if n <= 1 else 1
    if isinstance(label, str):
        m = _SHAPE_C_SPEAKER_LABEL_RE.search(label)
        if m is not None:
            n = int(m.group(1))
            return 0 if n <= 1 else 1
    return 0


def _parse_shape_c_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Shape C: per-session rows with parallel dialogue+speaker lists,
    grouped by ``dialoug_id`` (sic — misspelt field).

    Session ordering: ``session_id`` is 0-indexed in the mirror; we emit
    1..N in the shim. ``time_interval_days`` comes from
    :data:`_SHAPE_C_SESSION_GAP_DAYS` since this shape drops the paper's
    natural-language gap labels.
    """
    group_sorted = sorted(
        group,
        key=lambda r: int(r.get("session_id", 0) or 0),
    )
    anchor = group_sorted[0]
    raw_id = (
        anchor.get("dialoug_id")
        or anchor.get("dialog_id")
        or anchor.get("conversation_id")
        or "unknown"
    )
    conv_id = f"msc_{raw_id}"
    personas = _shape_c_personas(anchor)
    sessions: list[dict[str, Any]] = []
    for pos, row in enumerate(group_sorted, start=1):
        dialogue = row.get("dialogue") or []
        speakers = row.get("speaker") or []
        if not isinstance(dialogue, list) or not isinstance(speakers, list):
            continue
        turns: list[dict[str, Any]] = []
        for i, utterance in enumerate(dialogue):
            if not isinstance(utterance, str) or not utterance.strip():
                continue
            label = speakers[i] if i < len(speakers) else None
            if label is None:
                idx = i % 2
            else:
                idx = _shape_c_speaker_idx(label)
            turns.append({"speaker_idx": idx, "utterance": utterance.strip()})
        if not turns:
            continue
        gap_idx = pos - 1
        gap = (
            _SHAPE_C_SESSION_GAP_DAYS[gap_idx]
            if gap_idx < len(_SHAPE_C_SESSION_GAP_DAYS)
            else _SHAPE_C_SESSION_GAP_DAYS[-1]
        )
        sessions.append({
            "session_id": pos,
            "time_interval_days": float(gap),
            "turns": turns,
        })
    return {
        "conversation_id": conv_id,
        "personas": personas,
        "sessions": sessions,
        "metadata": {
            "shape": "C",
            "source_rows_in_group": len(group),
            "dialoug_id": str(raw_id),
            "gap_schedule": "canonical_msc_paper",
        },
    }


# ---------------------------------------------------------------------------
# Shape A grouping — rows arrive flat; we group by conv_id before parsing
# ---------------------------------------------------------------------------


def _group_rows_by_conv_id(
    rows: Iterator[dict[str, Any]], max_conversations: int | None
) -> Iterator[dict[str, Any]]:
    """Group shape-A rows by conv_id, yield one shim conversation per group.

    MSC HF mirrors emit per-session rows; we need per-conversation shims.
    We trust the row with the longest prior-dialog chain as the anchor
    (see _parse_shape_a_group) so collecting into a list per conv_id and
    emitting at EOF is correct — we don't need to see all rows before
    emitting any particular group.

    ``max_conversations`` caps emitted shim rows; grouping stops once the
    cap is hit so a smoke run doesn't scan the full stream.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    emitted = 0
    for row in rows:
        key = str(
            row.get("conv_id")
            or row.get("initial_data_id")
            or row.get("set_id")
            or ""
        )
        if not key:
            continue
        buckets.setdefault(key, []).append(row)
        # Opportunistic emit: once a bucket has a row with the full
        # previous_dialogs chain (len ≥ 4 for a 5-session MSC convo), emit
        # and drop. This keeps memory bounded.
        prev = row.get("previous_dialogs") or row.get("session_history") or []
        if len(prev) >= 4:
            shim = _parse_shape_a_group(buckets.pop(key))
            if shim["sessions"]:
                emitted += 1
                yield shim
            if max_conversations is not None and emitted >= max_conversations:
                return
    # EOF — emit any leftovers (short-chain conversations the opportunistic
    # emit missed).
    for key, group in buckets.items():
        if max_conversations is not None and emitted >= max_conversations:
            return
        shim = _parse_shape_a_group(group)
        if shim["sessions"]:
            emitted += 1
            yield shim


def _group_rows_by_dialoug_id(
    rows: Iterator[dict[str, Any]],
    max_conversations: int | None,
    min_sessions: int = 4,
) -> Iterator[dict[str, Any]]:
    """Group shape-C rows by dialoug_id, emit shims after full-dataset scan.

    We cannot opportunistically emit at a fixed row count because the
    ``nayohan/multi_session_chat`` mirror tops out at 4 sessions per
    conversation (``session_id`` 0..3), while other MSC mirrors have 5.
    Trusting a size trigger either loses the 5th session (fire-at-4) or
    never fires on 4-session mirrors (fire-at-5). Instead we buffer the
    whole stream — ~17.9k rows at ~1.3kB each is well under 50MB — then
    emit groups meeting ``min_sessions`` in dialoug_id order.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        key = str(
            row.get("dialoug_id")
            or row.get("dialog_id")
            or row.get("conversation_id")
            or ""
        )
        if not key:
            continue
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(row)
    emitted = 0
    for key in order:
        if max_conversations is not None and emitted >= max_conversations:
            return
        group = buckets[key]
        if len(group) < min_sessions:
            continue
        shim = _parse_shape_c_group(group)
        if shim["sessions"] and len(shim["sessions"]) >= min_sessions:
            emitted += 1
            yield shim


# ---------------------------------------------------------------------------
# HF loader + driver
# ---------------------------------------------------------------------------


def _iter_hf_rows(hf_dataset: str, split: str, hf_revision: str | None) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {}
    if hf_revision:
        kwargs["revision"] = hf_revision
    ds = load_dataset(hf_dataset, split=split, **kwargs)
    for row in ds:
        yield row


def _iter_shim_rows(
    hf_dataset: str,
    split: str,
    hf_revision: str | None,
    limit: int | None,
    flip_speaker: bool,
) -> Iterator[dict[str, Any]]:
    rows = _iter_hf_rows(hf_dataset, split, hf_revision)
    first = next(rows, None)
    if first is None:
        return
    shape = _detect_shape(first)
    # Rewind-via-chain so the first row isn't lost.
    def _prepended() -> Iterator[dict[str, Any]]:
        yield first
        yield from rows

    if shape == "A":
        shim_iter = _group_rows_by_conv_id(_prepended(), max_conversations=limit)
    elif shape == "B":
        def _b_iter() -> Iterator[dict[str, Any]]:
            emitted = 0
            for idx, row in enumerate(_prepended()):
                shim = _parse_shape_b_row(row, idx)
                if not shim["sessions"]:
                    continue
                emitted += 1
                yield shim
                if limit is not None and emitted >= limit:
                    return

        shim_iter = _b_iter()
    elif shape == "C":
        shim_iter = _group_rows_by_dialoug_id(_prepended(), max_conversations=limit)
    else:
        raise RuntimeError(
            f"Could not detect MSC schema shape from HF dataset {hf_dataset!r}. "
            f"First row keys: {sorted(first.keys())}. "
            f"Expected 'previous_dialogs' (shape A), 'sessions' (shape B), or "
            f"'dialoug_id' + parallel 'dialogue'/'speaker' lists (shape C)."
        )
    for shim in shim_iter:
        if flip_speaker:
            shim = _flip_speaker(shim)
        yield shim


def _flip_speaker(shim: dict[str, Any]) -> dict[str, Any]:
    """Swap speaker_idx 0 ↔ 1 across all turns + persona lists."""
    shim["personas"] = [shim["personas"][1], shim["personas"][0]]
    for session in shim["sessions"]:
        for turn in session["turns"]:
            turn["speaker_idx"] = 1 - int(turn["speaker_idx"])
    return shim


def provision(
    *,
    hf_dataset: str = _DEFAULT_HF_DATASET,
    split: str = _DEFAULT_SPLIT,
    hf_revision: str | None = None,
    limit: int | None = None,
    flip_speaker: bool = False,
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{split}.jsonl"
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for shim in _iter_shim_rows(
            hf_dataset, split, hf_revision, limit, flip_speaker
        ):
            fh.write(json.dumps(shim, ensure_ascii=False) + "\n")
            written += 1
            if limit is not None and written >= limit:
                break
    print(
        f"Wrote {written} MSC conversations to {out_path} "
        f"(hf_dataset={hf_dataset!r}, split={split!r}, "
        f"revision={hf_revision or 'main'})"
    )
    return out_path


def dry_run(
    *,
    hf_dataset: str = _DEFAULT_HF_DATASET,
    split: str = _DEFAULT_SPLIT,
    hf_revision: str | None = None,
    flip_speaker: bool = False,
) -> None:
    """Fetch, parse one conversation, print it. Verifies schema shape."""
    print(
        f"Dry-run — hf_dataset={hf_dataset!r} split={split!r} "
        f"revision={hf_revision or 'main'}"
    )
    rows = _iter_shim_rows(
        hf_dataset, split, hf_revision, limit=1, flip_speaker=flip_speaker
    )
    first = next(rows, None)
    if first is None:
        print("No shim rows produced — mirror may be empty or schema unrecognised.")
        return
    print("=== First shim conversation ===")
    print(json.dumps(first, ensure_ascii=False, indent=2)[:4000])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="provision_msc")
    p.add_argument("--hf-dataset", default=_DEFAULT_HF_DATASET)
    p.add_argument("--split", default=_DEFAULT_SPLIT)
    p.add_argument(
        "--hf-revision",
        default=None,
        help="HuggingFace commit sha for pinning; omit for 'main'.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--flip-speaker",
        action="store_true",
        help="Swap speaker_idx 0↔1 across all turns and persona lists.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.dry_run:
        dry_run(
            hf_dataset=args.hf_dataset,
            split=args.split,
            hf_revision=args.hf_revision,
            flip_speaker=args.flip_speaker,
        )
    else:
        provision(
            hf_dataset=args.hf_dataset,
            split=args.split,
            hf_revision=args.hf_revision,
            limit=args.limit,
            flip_speaker=args.flip_speaker,
        )
