"""Tests for the LongMemEval HF JSON → JSONL shim converter.

The real HF dataset is hundreds of megabytes and not fetched by CI; these
tests exercise the converter on synthetic JSON that matches the real
schema (field names + nesting) at tiny scale. Round-trip through the real
``LongMemEvalLoader`` is the end-to-end integration check — if the
converter's output doesn't load, the gate check on real data would also
fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.longmemeval import LongMemEvalLoader
from psychology_base.eval.scripts.convert_longmemeval import (
    convert,
    main,
    _is_valid,
    _iter_instances,
)


def _sample_instance(qid: str = "x", with_optional: bool = True) -> dict:
    inst = {
        "question_id": qid,
        "question_type": "single-session-user",
        "question": "What color is the user's bike?",
        "answer": "red",
        "haystack_sessions": [
            [
                {"role": "user", "content": "I rode my red bike today."},
                {"role": "assistant", "content": "Cool, where?"},
            ],
        ],
    }
    if with_optional:
        inst["question_date"] = "2026-05-01T10:00:00"
        inst["haystack_session_ids"] = ["s0"]
        inst["haystack_dates"] = ["2026-04-20T09:00:00"]
        inst["answer_session_ids"] = ["s0"]
    return inst


# ---------- _is_valid ----------


def test_is_valid_accepts_minimal_required():
    inst = {
        "question_id": "x",
        "question": "q",
        "answer": "a",
        "haystack_sessions": [[{"role": "user", "content": "hi"}]],
    }
    assert _is_valid(inst)


@pytest.mark.parametrize("missing", ["question_id", "question", "answer", "haystack_sessions"])
def test_is_valid_rejects_missing_required(missing):
    inst = _sample_instance(with_optional=False)
    del inst[missing]
    assert not _is_valid(inst)


def test_is_valid_rejects_sessions_not_a_list():
    inst = _sample_instance(with_optional=False)
    inst["haystack_sessions"] = "not-a-list"
    assert not _is_valid(inst)


def test_is_valid_rejects_turn_without_role_or_content():
    inst = _sample_instance(with_optional=False)
    inst["haystack_sessions"] = [[{"role": "user"}]]
    assert not _is_valid(inst)
    inst["haystack_sessions"] = [[{"content": "hi"}]]
    assert not _is_valid(inst)


# ---------- _iter_instances ----------


def test_iter_instances_accepts_top_level_array():
    payload = [_sample_instance("a"), _sample_instance("b")]
    assert [i["question_id"] for i in _iter_instances(payload)] == ["a", "b"]


def test_iter_instances_accepts_envelope_object():
    payload = {"instances": [_sample_instance("a")]}
    assert [i["question_id"] for i in _iter_instances(payload)] == ["a"]


def test_iter_instances_rejects_unrecognized_structure():
    with pytest.raises(ValueError, match="Unrecognized LongMemEval"):
        list(_iter_instances({"not-the-field": []}))


def test_iter_instances_skips_non_dict_elements():
    payload = [_sample_instance("a"), "garbage", _sample_instance("b"), None]
    assert [i["question_id"] for i in _iter_instances(payload)] == ["a", "b"]


# ---------- convert ----------


def test_convert_round_trip_through_loader(tmp_path):
    """End-to-end: convert → load → read back the fields the loader uses."""
    src = tmp_path / "lme.json"
    dst = tmp_path / "lme_shim.jsonl"
    payload = [_sample_instance("roundtrip-0"), _sample_instance("roundtrip-1")]
    src.write_text(json.dumps(payload), encoding="utf-8")

    input_count, output_count = convert(src, dst)
    assert input_count == 2
    assert output_count == 2

    loader = LongMemEvalLoader(shim_path=dst)
    loaded = list(loader.iter_instances())
    assert [i.instance_id for i in loaded] == ["roundtrip-0", "roundtrip-1"]
    assert loaded[0].questions[0].gold == "red"
    assert loaded[0].questions[0].question_type == "single-session-user"
    assert loaded[0].questions[0].metadata["answer_session_ids"] == ("s0",)


def test_convert_drops_malformed_instances(tmp_path, capsys):
    src = tmp_path / "lme.json"
    dst = tmp_path / "lme_shim.jsonl"
    good = _sample_instance("good")
    bad_missing_answer = _sample_instance("bad-1")
    del bad_missing_answer["answer"]
    bad_sessions_shape = _sample_instance("bad-2")
    bad_sessions_shape["haystack_sessions"] = [[{"role": "user"}]]  # no content
    payload = [good, bad_missing_answer, bad_sessions_shape]
    src.write_text(json.dumps(payload), encoding="utf-8")

    input_count, output_count = convert(src, dst)
    assert input_count == 3
    assert output_count == 1
    err = capsys.readouterr().err
    assert "dropped 2 instance" in err
    assert "bad-1" in err and "bad-2" in err


def test_convert_projects_only_adapter_fields(tmp_path):
    src = tmp_path / "lme.json"
    dst = tmp_path / "lme_shim.jsonl"
    inst = _sample_instance("proj-0")
    inst["extraneous_field_the_adapter_ignores"] = {"keep_jsonl_slim": True}
    src.write_text(json.dumps([inst]), encoding="utf-8")

    convert(src, dst)
    line = dst.read_text(encoding="utf-8").strip()
    parsed = json.loads(line)
    assert "extraneous_field_the_adapter_ignores" not in parsed
    assert parsed["question_id"] == "proj-0"


def test_convert_handles_optional_fields_absent(tmp_path):
    src = tmp_path / "lme.json"
    dst = tmp_path / "lme_shim.jsonl"
    inst = _sample_instance("minimal", with_optional=False)
    src.write_text(json.dumps([inst]), encoding="utf-8")

    input_count, output_count = convert(src, dst)
    assert (input_count, output_count) == (1, 1)
    loader = LongMemEvalLoader(shim_path=dst)
    loaded = list(loader.iter_instances())
    assert len(loaded) == 1
    assert loaded[0].questions[0].metadata["answer_session_ids"] == ()


# ---------- CLI ----------


def test_cli_missing_input_returns_nonzero(tmp_path, capsys):
    rc = main(["--input", str(tmp_path / "nope.json"), "--output", str(tmp_path / "out.jsonl")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "input file not found" in err


def test_cli_happy_path(tmp_path, capsys):
    src = tmp_path / "lme.json"
    dst = tmp_path / "lme_shim.jsonl"
    src.write_text(json.dumps([_sample_instance("cli-0")]), encoding="utf-8")

    rc = main(["--input", str(src), "--output", str(dst)])
    assert rc == 0
    assert dst.exists()
    assert "converted 1/1" in capsys.readouterr().out
