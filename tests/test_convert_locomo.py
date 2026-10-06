"""Tests for the LoCoMo JSON → JSONL shim converter.

The real snap-research/locomo ``locomo10.json`` is ten long dialogues
and not fetched by CI; these tests exercise the converter on synthetic
JSON matching the native schema (nested ``conversation`` object with
``session_N`` + ``session_N_date_time`` siblings; ``qa`` list) at tiny
scale. Round-trip through ``LoCoMoLoader`` is the end-to-end integration
check — if the converter emits something the loader can't read, the real
locomo10 gate check would also fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.locomo import (
    ADVERSARIAL,
    SINGLE_HOP,
    LoCoMoLoader,
)
from psychology_base.eval.scripts.convert_locomo import (
    _extract_sessions,
    _iter_instances,
    _project_instance,
    _project_questions,
    convert,
    main,
)


def _sample_instance(sample_id: str = "s0", with_adversarial: bool = False) -> dict:
    """Shape-match the real locomo10.json: ``conversation.session_N`` turn
    lists with sibling ``session_N_date_time``; a ``qa`` list of entries."""
    inst = {
        "sample_id": sample_id,
        "conversation": {
            "session_1_date_time": "2026-03-10T19:00:00",
            "session_1": [
                {"speaker": "Jennifer", "text": "I bought a Westerly 28."},
                {"speaker": "David", "text": "Nice."},
            ],
            "session_2_date_time": "2026-04-02T20:00:00",
            "session_2": [
                {"speaker": "Jennifer", "text": "Sailed her to Hull."},
            ],
        },
        "qa": [
            {
                "question": "What kind of boat did Jennifer buy?",
                "answer": "Westerly 28",
                "category": "single-hop",
                "evidence": ["session_1"],
            },
            {
                "question": "Where did Jennifer sail her boat to?",
                "answer": "Hull",
                "category": "single-hop",
                "evidence": ["session_2"],
            },
        ],
    }
    if with_adversarial:
        inst["qa"].append(
            {
                "question": "Did David ever sail with Jennifer?",
                "answer": "cannot be determined from the dialogue",
                "adversarial_answer": "cannot be determined from the dialogue",
                "category": "adversarial",
                "evidence": [],
            }
        )
    return inst


# ---------- _iter_instances ----------


def test_iter_instances_accepts_top_level_array():
    payload = [_sample_instance("a"), _sample_instance("b")]
    assert [i["sample_id"] for i in _iter_instances(payload)] == ["a", "b"]


def test_iter_instances_accepts_envelope_object():
    payload = {"instances": [_sample_instance("a")]}
    assert [i["sample_id"] for i in _iter_instances(payload)] == ["a"]


def test_iter_instances_rejects_unrecognized_structure():
    with pytest.raises(ValueError, match="Unrecognized LoCoMo"):
        list(_iter_instances({"not-the-field": []}))


def test_iter_instances_skips_non_dict_elements():
    payload = [_sample_instance("a"), "garbage", _sample_instance("b"), None]
    assert [i["sample_id"] for i in _iter_instances(payload)] == ["a", "b"]


# ---------- _extract_sessions ----------


def test_extract_sessions_numerically_orders_keys():
    """``session_10`` must sort AFTER ``session_2`` (numeric, not lexical)."""
    conversation = {
        "session_10_date_time": "2026-05-01T10:00:00",
        "session_10": [{"speaker": "A", "text": "late"}],
        "session_2_date_time": "2026-04-02T20:00:00",
        "session_2": [{"speaker": "A", "text": "middle"}],
        "session_1_date_time": "2026-03-10T19:00:00",
        "session_1": [{"speaker": "A", "text": "early"}],
    }
    out = _extract_sessions(conversation)
    assert [s["session_id"] for s in out] == ["session_1", "session_2", "session_10"]


def test_extract_sessions_normalizes_text_to_content_field():
    conversation = {
        "session_1_date_time": "2026-03-10T19:00:00",
        "session_1": [{"speaker": "Jennifer", "text": "the real native field name"}],
    }
    out = _extract_sessions(conversation)
    assert out[0]["turns"][0]["content"] == "the real native field name"
    assert "text" not in out[0]["turns"][0]


def test_extract_sessions_tolerates_content_field_already_present():
    conversation = {
        "session_1_date_time": "2026-03-10T19:00:00",
        "session_1": [{"speaker": "A", "content": "already-shim-shaped"}],
    }
    out = _extract_sessions(conversation)
    assert out[0]["turns"][0]["content"] == "already-shim-shaped"


def test_extract_sessions_drops_malformed_turns():
    conversation = {
        "session_1_date_time": "2026-03-10T19:00:00",
        "session_1": [
            {"speaker": "A", "text": "ok"},
            "not-a-dict",
            {"speaker": "B"},  # missing text/content
            {"text": "no speaker"},
        ],
    }
    out = _extract_sessions(conversation)
    assert len(out[0]["turns"]) == 1


def test_extract_sessions_fills_missing_date_as_empty_string():
    conversation = {
        "session_1": [{"speaker": "A", "text": "no date sibling"}],
    }
    out = _extract_sessions(conversation)
    assert out[0]["date"] == ""


def test_extract_sessions_skips_non_list_session_values():
    conversation = {
        "session_1": "not a list — bogus",
        "session_2_date_time": "2026-04-02T20:00:00",
        "session_2": [{"speaker": "A", "text": "ok"}],
    }
    out = _extract_sessions(conversation)
    assert [s["session_id"] for s in out] == ["session_2"]


# ---------- _project_questions ----------


def test_project_questions_synthesizes_stable_ids():
    """Question IDs are synthesized from (instance_id, index) and MUST be
    stable across converter re-runs — the judge cache keys on them."""
    qa = [
        {"question": "q0", "answer": "a0", "category": "single-hop"},
        {"question": "q1", "answer": "a1", "category": "single-hop"},
    ]
    out = _project_questions(qa, instance_id="inst-7")
    assert [q["id"] for q in out] == ["inst-7-q0", "inst-7-q1"]


def test_project_questions_drops_entries_missing_required_fields():
    qa = [
        {"question": "ok", "answer": "yes", "category": "single-hop"},
        {"answer": "missing question"},
        {"question": "missing answer"},
        "not a dict",
    ]
    out = _project_questions(qa, instance_id="x")
    assert len(out) == 1


def test_project_questions_decodes_native_numeric_categories():
    """Native locomo10.json uses the integer codes 1..5; the adapter's
    judge/F1 dispatcher keys on the semantic name. Decode here so the
    shim lands already named. Mapping verified by sampling the real
    data (2026-10-06)."""
    qa = [
        {"question": "q1", "answer": "a1", "category": 1},
        {"question": "q2", "answer": "a2", "category": "2"},
        {"question": "q3", "answer": "a3", "category": 3},
        {"question": "q4", "answer": "a4", "category": "4"},
        {"question": "q5", "answer": "a5", "category": 5, "adversarial_answer": "trap"},
    ]
    out = _project_questions(qa, instance_id="inst")
    assert [q["category"] for q in out] == [
        "single-hop",
        "temporal-reasoning",
        "open-domain",
        "multi-hop",
        "adversarial",
    ]


def test_project_questions_preserves_adversarial_answer():
    qa = [
        {
            "question": "Did X happen?",
            "answer": "cannot be determined",
            "adversarial_answer": "cannot be determined",
            "category": "adversarial",
            "evidence": [],
        }
    ]
    out = _project_questions(qa, instance_id="x")
    assert out[0]["adversarial_answer"] == "cannot be determined"
    assert out[0]["category"] == "adversarial"


# ---------- _project_instance ----------


def test_project_instance_returns_none_on_missing_conversation():
    inst = {"sample_id": "x", "qa": [{"question": "q", "answer": "a"}]}
    assert _project_instance(inst, 0) is None


def test_project_instance_returns_none_on_missing_qa():
    inst = {"sample_id": "x", "conversation": {"session_1": [{"speaker": "A", "text": "hi"}]}}
    assert _project_instance(inst, 0) is None


def test_project_instance_synthesizes_id_when_sample_id_absent():
    inst = {
        "conversation": {
            "session_1_date_time": "2026-01-01T00:00:00",
            "session_1": [{"speaker": "A", "text": "hi"}],
        },
        "qa": [{"question": "q", "answer": "a", "category": "single-hop"}],
    }
    projected = _project_instance(inst, idx=5)
    assert projected is not None
    assert projected["instance_id"] == "locomo-5"


# ---------- convert (end-to-end + loader round-trip) ----------


def test_convert_round_trip_through_loader(tmp_path):
    src = tmp_path / "locomo.json"
    dst = tmp_path / "locomo_shim.jsonl"
    payload = [_sample_instance("rt-0", with_adversarial=True)]
    src.write_text(json.dumps(payload), encoding="utf-8")

    input_count, output_count = convert(src, dst)
    assert (input_count, output_count) == (1, 1)

    loader = LoCoMoLoader(shim_path=dst)
    instances = list(loader.iter_instances())
    assert len(instances) == 1
    inst = instances[0]
    assert inst.instance_id == "rt-0"
    # 2 sessions: 2 + 1 turns.
    assert len(inst.history) == 3
    # First turn: Jennifer in session_1.
    first = inst.history[0]
    assert first.speaker == "Jennifer"
    assert first.session_id == "session_1"
    # 3 questions: 2 single-hop + 1 adversarial.
    assert len(inst.questions) == 3
    categories = [q.question_type for q in inst.questions]
    assert categories == [SINGLE_HOP, SINGLE_HOP, ADVERSARIAL]
    # Adversarial gold flows from the adversarial_answer field.
    assert inst.questions[2].gold == "cannot be determined from the dialogue"


def test_convert_drops_malformed_instances(tmp_path, capsys):
    src = tmp_path / "locomo.json"
    dst = tmp_path / "locomo_shim.jsonl"
    good = _sample_instance("good")
    bad_no_conversation = {"sample_id": "bad-1", "qa": [{"question": "q", "answer": "a"}]}
    bad_no_qa = {
        "sample_id": "bad-2",
        "conversation": {"session_1": [{"speaker": "A", "text": "hi"}]},
    }
    payload = [good, bad_no_conversation, bad_no_qa]
    src.write_text(json.dumps(payload), encoding="utf-8")

    input_count, output_count = convert(src, dst)
    assert (input_count, output_count) == (3, 1)
    err = capsys.readouterr().err
    assert "dropped 2 instance" in err
    assert "bad-1" in err and "bad-2" in err


def test_convert_session_order_is_numerical_in_output(tmp_path):
    """End-to-end: ``session_10`` lands AFTER ``session_2`` in the loader's
    flattened turn history. Guards against lexical sort regressing the
    adapter's observe order."""
    src = tmp_path / "locomo.json"
    dst = tmp_path / "locomo_shim.jsonl"
    inst = {
        "sample_id": "ord-0",
        "conversation": {
            "session_10_date_time": "2026-05-01T10:00:00",
            "session_10": [{"speaker": "A", "text": "late"}],
            "session_2_date_time": "2026-04-02T20:00:00",
            "session_2": [{"speaker": "A", "text": "middle"}],
            "session_1_date_time": "2026-03-10T19:00:00",
            "session_1": [{"speaker": "A", "text": "early"}],
        },
        "qa": [{"question": "q", "answer": "a", "category": "single-hop"}],
    }
    src.write_text(json.dumps([inst]), encoding="utf-8")

    convert(src, dst)
    loader = LoCoMoLoader(shim_path=dst)
    history = list(loader.iter_instances())[0].history
    assert [t.content for t in history] == ["early", "middle", "late"]
    assert [t.session_id for t in history] == ["session_1", "session_2", "session_10"]


def test_convert_handles_empty_evidence_list(tmp_path):
    src = tmp_path / "locomo.json"
    dst = tmp_path / "locomo_shim.jsonl"
    inst = _sample_instance("ev-0", with_adversarial=True)
    src.write_text(json.dumps([inst]), encoding="utf-8")

    convert(src, dst)
    loader = LoCoMoLoader(shim_path=dst)
    questions = list(loader.iter_instances())[0].questions
    assert questions[0].metadata["evidence"] == ("session_1",)
    assert questions[2].metadata["evidence"] == ()


# ---------- CLI ----------


def test_cli_missing_input_returns_nonzero(tmp_path, capsys):
    rc = main(["--input", str(tmp_path / "nope.json"), "--output", str(tmp_path / "out.jsonl")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "input file not found" in err


def test_cli_happy_path(tmp_path, capsys):
    src = tmp_path / "locomo.json"
    dst = tmp_path / "locomo_shim.jsonl"
    src.write_text(json.dumps([_sample_instance("cli-0")]), encoding="utf-8")

    rc = main(["--input", str(src), "--output", str(dst)])
    assert rc == 0
    assert dst.exists()
    assert "converted 1/1" in capsys.readouterr().out
