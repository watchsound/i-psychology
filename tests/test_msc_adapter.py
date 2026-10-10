"""Phase 8 Pillar A — MSC adapter loader + types tests.

In-memory JSONL only. No network, no HuggingFace, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.public_datasets.msc import (
    MSCConversation,
    MSCSession,
    MSCTurn,
    OTHER_SPEAKER_IDX,
    SELF_SPEAKER_IDX,
    iter_msc_conversations,
    other_persona,
    self_opening_utterance,
    self_persona,
)


# ---------------------------------------------------------------------------
# Dataclass validation
# ---------------------------------------------------------------------------


def test_turn_accepts_valid_speaker_ids() -> None:
    assert MSCTurn(speaker_idx=0, utterance="hi").speaker_idx == 0
    assert MSCTurn(speaker_idx=1, utterance="hi").speaker_idx == 1


def test_turn_rejects_speaker_other_than_0_or_1() -> None:
    with pytest.raises(ValueError, match="speaker_idx"):
        MSCTurn(speaker_idx=2, utterance="hi")


def test_session_rejects_bad_session_id() -> None:
    with pytest.raises(ValueError, match="session_id"):
        MSCSession(session_id=0, time_interval_days=1.0, turns=())


def test_session_rejects_negative_time_interval() -> None:
    with pytest.raises(ValueError, match="time_interval_days"):
        MSCSession(session_id=1, time_interval_days=-1.0, turns=())


def test_conversation_cumulative_days_sums_intervals() -> None:
    convo = MSCConversation(
        conversation_id="msc_test",
        personas=(("a",), ("b",)),
        sessions=(
            MSCSession(session_id=1, time_interval_days=0.0, turns=()),
            MSCSession(session_id=2, time_interval_days=3.0, turns=()),
            MSCSession(session_id=3, time_interval_days=7.0, turns=()),
        ),
    )
    assert convo.cumulative_days_at(1) == 0.0
    assert convo.cumulative_days_at(2) == 3.0
    assert convo.cumulative_days_at(3) == 10.0
    assert convo.cumulative_days_at(99) is None


def test_conversation_session_lookup_by_id() -> None:
    convo = MSCConversation(
        conversation_id="msc_test",
        personas=(("a",), ("b",)),
        sessions=(
            MSCSession(session_id=1, time_interval_days=0.0, turns=()),
            MSCSession(session_id=5, time_interval_days=14.0, turns=()),
        ),
    )
    assert convo.session(1) is not None
    assert convo.session(5) is not None
    assert convo.session(3) is None


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------


def _shim_row() -> dict:
    return {
        "conversation_id": "msc_000001",
        "personas": [
            ["I love dogs.", "I work as a teacher."],
            ["I grew up in Kansas.", "I collect vintage cameras."],
        ],
        "sessions": [
            {
                "session_id": 1,
                "time_interval_days": 0.0,
                "turns": [
                    {"speaker_idx": 0, "utterance": "Hi, how are you today?"},
                    {"speaker_idx": 1, "utterance": "I'm doing great, you?"},
                ],
            },
            {
                "session_id": 2,
                "time_interval_days": 7.0,
                "turns": [
                    {"speaker_idx": 0, "utterance": "It's been a hard week."},
                    {"speaker_idx": 1, "utterance": "What happened?"},
                ],
            },
        ],
        "metadata": {"src": "test"},
    }


def test_loader_parses_a_single_row(tmp_path: Path) -> None:
    shim_path = tmp_path / "msc.jsonl"
    shim_path.write_text(json.dumps(_shim_row()) + "\n", encoding="utf-8")
    convs = list(iter_msc_conversations(shim_path))
    assert len(convs) == 1
    c = convs[0]
    assert c.conversation_id == "msc_000001"
    assert len(c.personas[0]) == 2
    assert c.num_sessions == 2
    assert c.sessions[0].session_id == 1
    assert c.sessions[1].time_interval_days == 7.0
    assert c.metadata["src"] == "test"


def test_loader_skips_blank_lines(tmp_path: Path) -> None:
    shim_path = tmp_path / "msc.jsonl"
    row = _shim_row()
    shim_path.write_text(
        "\n" + json.dumps(row) + "\n\n\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    assert len(list(iter_msc_conversations(shim_path))) == 2


def test_loader_rejects_missing_conversation_id(tmp_path: Path) -> None:
    row = _shim_row()
    del row["conversation_id"]
    shim_path = tmp_path / "msc.jsonl"
    shim_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conversation_id"):
        list(iter_msc_conversations(shim_path))


def test_loader_rejects_personas_not_a_2_element_list(tmp_path: Path) -> None:
    row = _shim_row()
    row["personas"] = [["only-one"]]
    shim_path = tmp_path / "msc.jsonl"
    shim_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="personas"):
        list(iter_msc_conversations(shim_path))


def test_loader_rejects_empty_sessions(tmp_path: Path) -> None:
    row = _shim_row()
    row["sessions"] = []
    shim_path = tmp_path / "msc.jsonl"
    shim_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sessions"):
        list(iter_msc_conversations(shim_path))


def test_loader_rejects_nonstring_utterance(tmp_path: Path) -> None:
    row = _shim_row()
    row["sessions"][0]["turns"][0]["utterance"] = 42  # type: ignore
    shim_path = tmp_path / "msc.jsonl"
    shim_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="utterance"):
        list(iter_msc_conversations(shim_path))


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


def _convo_fixture() -> MSCConversation:
    return MSCConversation(
        conversation_id="msc_test",
        personas=(("I love dogs.",), ("I live in Kansas.",)),
        sessions=(
            MSCSession(
                session_id=1,
                time_interval_days=0.0,
                turns=(
                    MSCTurn(speaker_idx=OTHER_SPEAKER_IDX, utterance="Hey!"),
                    MSCTurn(speaker_idx=SELF_SPEAKER_IDX, utterance="Hi there!"),
                ),
            ),
            MSCSession(
                session_id=2,
                time_interval_days=5.0,
                turns=(
                    MSCTurn(speaker_idx=SELF_SPEAKER_IDX, utterance="Been a while."),
                ),
            ),
        ),
    )


def test_self_persona_returns_speaker_0_bullets() -> None:
    assert self_persona(_convo_fixture()) == ("I love dogs.",)


def test_other_persona_returns_speaker_1_bullets() -> None:
    assert other_persona(_convo_fixture()) == ("I live in Kansas.",)


def test_self_opening_utterance_skips_other_speaker() -> None:
    # Session 1's first self-speaker turn is the second overall turn.
    assert self_opening_utterance(_convo_fixture(), session_id=1) == "Hi there!"


def test_self_opening_utterance_defaults_to_session_1() -> None:
    assert self_opening_utterance(_convo_fixture()) == "Hi there!"


def test_self_opening_utterance_returns_none_for_missing_session() -> None:
    assert self_opening_utterance(_convo_fixture(), session_id=99) is None
