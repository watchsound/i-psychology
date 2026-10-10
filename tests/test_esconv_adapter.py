"""ESConv adapter — shim loader + strategy-distribution scorer tests.

Phase 4b benchmark-pivot item D2 priority #2. In-memory fixture to keep
the repo license-neutral w.r.t. the ESConv payload.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.public_datasets.esconv import (
    ESConvConversation,
    ESConvStrategy,
    ESConvTurn,
    iter_esconv_conversations,
    seeker_opening_utterance,
    summarise_strategy_distribution,
)


def _write_shim(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "esconv_shim.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


# ---------------------------------------------------------------------------
# Loader — happy path + malformed inputs
# ---------------------------------------------------------------------------


def test_loader_round_trips_minimal_conversation(tmp_path):
    row = {
        "conversation_id": "esc_001",
        "emotion_type": "anxiety",
        "problem_type": "job crisis",
        "situation": "I lost my job last week.",
        "turns": [
            {"role": "seeker", "utterance": "I'm feeling overwhelmed."},
            {
                "role": "supporter",
                "utterance": "That sounds really hard.",
                "strategy": "affirmation_and_reassurance",
            },
        ],
        "survey_score": 4.2,
    }
    path = _write_shim(tmp_path, [row])
    convos = list(iter_esconv_conversations(path))
    assert len(convos) == 1
    convo = convos[0]
    assert convo.conversation_id == "esc_001"
    assert convo.emotion_type == "anxiety"
    assert convo.survey_score == pytest.approx(4.2)
    assert convo.turns[0].strategy is None
    assert convo.turns[1].strategy is ESConvStrategy.AFFIRMATION_AND_REASSURANCE


def test_loader_handles_missing_survey_score(tmp_path):
    row = {
        "conversation_id": "esc_002",
        "emotion_type": "sadness",
        "problem_type": "breakup",
        "situation": "x",
        "turns": [],
    }
    path = _write_shim(tmp_path, [row])
    convo = next(iter_esconv_conversations(path))
    assert convo.survey_score is None


def test_loader_skips_blank_lines(tmp_path):
    path = tmp_path / "shim.jsonl"
    path.write_text(
        json.dumps(
            {
                "conversation_id": "esc_x",
                "emotion_type": "anger",
                "problem_type": "x",
                "situation": "x",
                "turns": [],
            }
        )
        + "\n\n\n",
        encoding="utf-8",
    )
    convos = list(iter_esconv_conversations(path))
    assert len(convos) == 1


def test_loader_rejects_strategy_on_seeker_turn(tmp_path):
    row = {
        "conversation_id": "esc_bad",
        "emotion_type": "anxiety",
        "problem_type": "x",
        "situation": "x",
        "turns": [{"role": "seeker", "utterance": "x", "strategy": "question"}],
    }
    path = _write_shim(tmp_path, [row])
    with pytest.raises(ValueError, match="'strategy' only valid on supporter"):
        list(iter_esconv_conversations(path))


def test_loader_rejects_unknown_role(tmp_path):
    row = {
        "conversation_id": "esc_bad",
        "emotion_type": "x",
        "problem_type": "x",
        "situation": "x",
        "turns": [{"role": "narrator", "utterance": "x"}],
    }
    path = _write_shim(tmp_path, [row])
    with pytest.raises(ValueError, match="role must be"):
        list(iter_esconv_conversations(path))


def test_loader_rejects_unknown_strategy(tmp_path):
    row = {
        "conversation_id": "esc_bad",
        "emotion_type": "x",
        "problem_type": "x",
        "situation": "x",
        "turns": [{"role": "supporter", "utterance": "x", "strategy": "coaching"}],
    }
    path = _write_shim(tmp_path, [row])
    with pytest.raises(ValueError):
        list(iter_esconv_conversations(path))


# ---------------------------------------------------------------------------
# Strategy distribution + opening utterance
# ---------------------------------------------------------------------------


def _convo(turns: list[ESConvTurn], cid: str = "t") -> ESConvConversation:
    return ESConvConversation(
        conversation_id=cid,
        emotion_type="anxiety",
        problem_type="x",
        situation="x",
        turns=tuple(turns),
    )


def test_summarise_pools_supporter_strategy_counts():
    c1 = _convo(
        [
            ESConvTurn(role="seeker", utterance="hi"),
            ESConvTurn(
                role="supporter",
                utterance="x",
                strategy=ESConvStrategy.QUESTION,
            ),
            ESConvTurn(
                role="supporter",
                utterance="y",
                strategy=ESConvStrategy.AFFIRMATION_AND_REASSURANCE,
            ),
        ],
        cid="c1",
    )
    c2 = _convo(
        [
            ESConvTurn(
                role="supporter",
                utterance="z",
                strategy=ESConvStrategy.QUESTION,
            ),
        ],
        cid="c2",
    )
    report = summarise_strategy_distribution([c1, c2])
    assert report.sampled_supporter_turns == 3
    assert report.counts[ESConvStrategy.QUESTION] == 2
    assert report.counts[ESConvStrategy.AFFIRMATION_AND_REASSURANCE] == 1
    assert report.frequency(ESConvStrategy.QUESTION) == pytest.approx(2 / 3)


def test_summarise_skips_strategyless_supporter_turns():
    convo = _convo(
        [
            ESConvTurn(role="supporter", utterance="x"),  # no strategy — skipped
            ESConvTurn(
                role="supporter",
                utterance="y",
                strategy=ESConvStrategy.OTHERS,
            ),
        ]
    )
    report = summarise_strategy_distribution([convo])
    assert report.sampled_supporter_turns == 1
    assert report.counts[ESConvStrategy.OTHERS] == 1


def test_summarise_empty_list_returns_zero_counts():
    report = summarise_strategy_distribution([])
    assert report.sampled_supporter_turns == 0
    assert report.frequency(ESConvStrategy.QUESTION) == 0.0


def test_seeker_opening_utterance_returns_first_seeker_turn():
    convo = _convo(
        [
            ESConvTurn(role="supporter", utterance="How can I help?"),
            ESConvTurn(role="seeker", utterance="I need to vent."),
            ESConvTurn(role="seeker", utterance="(later)"),
        ]
    )
    assert seeker_opening_utterance(convo) == "I need to vent."


def test_seeker_opening_utterance_none_when_no_seeker():
    convo = _convo([ESConvTurn(role="supporter", utterance="hi")])
    assert seeker_opening_utterance(convo) is None


# ---------------------------------------------------------------------------
# Strategy enum coverage
# ---------------------------------------------------------------------------


def test_strategy_enum_covers_all_eight_canonical_moves():
    expected = {
        "reflection_of_feelings",
        "question",
        "restatement",
        "self_disclosure",
        "affirmation_and_reassurance",
        "providing_suggestions",
        "information",
        "others",
    }
    assert {s.value for s in ESConvStrategy} == expected
