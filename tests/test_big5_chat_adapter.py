"""Phase 8 Pillar C — BIG5-CHAT adapter loader + types tests.

In-memory JSONL only. No network, no HuggingFace, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.public_datasets.big5_chat import (
    Big5ChatDialog,
    iter_big5_chat_dialogs,
    partner_name,
    speaker_name,
)


# ---------------------------------------------------------------------------
# Dataclass validation
# ---------------------------------------------------------------------------


def test_dialog_accepts_valid_trait_and_level() -> None:
    d = Big5ChatDialog(
        dialog_id="big5_000001",
        env_idx=0,
        trait="openness",
        level="low",
        train_input="Hi.",
        train_output="Hello.",
    )
    assert d.trait == "openness"
    assert d.level == "low"


def test_dialog_rejects_invalid_trait() -> None:
    with pytest.raises(ValueError, match="trait"):
        Big5ChatDialog(
            dialog_id="d1",
            env_idx=0,
            trait="charisma",
            level="high",
            train_input="",
            train_output="hi",
        )


def test_dialog_rejects_invalid_level() -> None:
    with pytest.raises(ValueError, match="level"):
        Big5ChatDialog(
            dialog_id="d1",
            env_idx=0,
            trait="openness",
            level="medium",
            train_input="",
            train_output="hi",
        )


def test_dialog_rejects_empty_train_output() -> None:
    with pytest.raises(ValueError, match="train_output"):
        Big5ChatDialog(
            dialog_id="d1",
            env_idx=0,
            trait="openness",
            level="high",
            train_input="",
            train_output="",
        )


def test_dialog_normalises_optional_fields() -> None:
    d = Big5ChatDialog(
        dialog_id="d1",
        env_idx=0,
        trait="neuroticism",
        level="high",
        train_input="context",
        train_output="response",
    )
    assert d.narrative == ""
    assert d.literal == ""
    assert d.personx == ""
    assert d.persony == ""


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------


def _write_shim(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "shim.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def test_loader_parses_minimal_row(tmp_path: Path) -> None:
    shim = _write_shim(
        tmp_path,
        [
            {
                "dialog_id": "big5_000000",
                "env_idx": 0,
                "trait": "openness",
                "level": "low",
                "train_input": "Hey.",
                "train_output": "Hi back.",
            }
        ],
    )
    dialogs = list(iter_big5_chat_dialogs(shim))
    assert len(dialogs) == 1
    assert dialogs[0].dialog_id == "big5_000000"
    assert dialogs[0].trait == "openness"
    assert dialogs[0].level == "low"


def test_loader_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "shim.jsonl"
    row = {
        "dialog_id": "d1",
        "env_idx": 0,
        "trait": "neuroticism",
        "level": "high",
        "train_input": "x",
        "train_output": "y",
    }
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.write("\n")
        fh.write(json.dumps(row) + "\n")
    assert len(list(iter_big5_chat_dialogs(path))) == 2


def test_loader_normalises_trait_level_case(tmp_path: Path) -> None:
    shim = _write_shim(
        tmp_path,
        [
            {
                "dialog_id": "d1",
                "env_idx": 0,
                "trait": "OPENNESS",
                "level": "HIGH",
                "train_input": "",
                "train_output": "y",
            }
        ],
    )
    dialogs = list(iter_big5_chat_dialogs(shim))
    assert dialogs[0].trait == "openness"
    assert dialogs[0].level == "high"


def test_loader_rejects_missing_required_field(tmp_path: Path) -> None:
    shim = _write_shim(
        tmp_path,
        [
            {
                "dialog_id": "d1",
                "trait": "openness",
                "level": "high",
                # missing train_output
            }
        ],
    )
    with pytest.raises(ValueError, match="train_output"):
        list(iter_big5_chat_dialogs(shim))


def test_loader_preserves_metadata(tmp_path: Path) -> None:
    shim = _write_shim(
        tmp_path,
        [
            {
                "dialog_id": "d1",
                "env_idx": 42,
                "trait": "extraversion",
                "level": "high",
                "train_input": "x",
                "train_output": "y",
                "narrative": "scene",
                "literal": "lit",
                "personx": "Alice",
                "persony": "Bob",
                "metadata": {"head": "PersonX does Y", "relation": "xReact"},
            }
        ],
    )
    d = list(iter_big5_chat_dialogs(shim))[0]
    assert d.env_idx == 42
    assert d.personx == "Alice"
    assert d.persony == "Bob"
    assert d.metadata["relation"] == "xReact"


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


def test_speaker_name_prefers_persony() -> None:
    d = Big5ChatDialog(
        dialog_id="d1",
        env_idx=0,
        trait="openness",
        level="high",
        train_input="",
        train_output="y",
        personx="Alice",
        persony="Bob",
    )
    assert speaker_name(d) == "Bob"
    assert partner_name(d) == "Alice"


def test_speaker_name_falls_back_when_missing() -> None:
    d = Big5ChatDialog(
        dialog_id="d1",
        env_idx=0,
        trait="openness",
        level="high",
        train_input="",
        train_output="y",
    )
    assert speaker_name(d) == "speaker"
    assert partner_name(d) == "partner"
