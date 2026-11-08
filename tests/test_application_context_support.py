"""Frozen software contracts; injected replies are not application-quality evidence."""
import copy
import json
from datetime import datetime, timezone

import pytest

from psychology_base.eval.benchmarks.evaluation_support import AuditedSystem, complete_keys, measure_recall
from psychology_base.eval.adapters import Observation
from psychology_base.applications.learning.tutor_agent import _render_context_block


def test_audit_preserves_actual_recall_and_detaches_metadata():
    system = AuditedSystem(byte_cap=None)
    system.observe(Observation(datetime(2026, 10, 19, tzinfo=timezone.utc), "user", "I enjoy chess."))
    result = system.recall("chess")
    assert system.last_context_fragments == result.fragments_used[:6]
    assert system.last_recall == result and system.audit_recall_count == 1
    result.metadata["review_mutation"] = True
    assert "review_mutation" not in system.last_recall.metadata
    assert system.activation.recall_count == system.witness_state.activation.recall_count == 1
    assert copy.deepcopy(system).last_recall == system.last_recall


def test_tutor_renderer_marks_legacy_fragments_as_unknown_speakers():
    from psychology_base.eval.adapters import make_fragment
    stamp = datetime(2026, 10, 19, tzinfo=timezone.utc)
    fragments = [make_fragment("我先算了总量。", stamp), make_fragment("Then I divided by two.", stamp)]
    assert list(map(json.loads, _render_context_block(fragments).splitlines())) == [
        {'speaker': None, 'session': None, 'text': f.content, 'source_order': None} for f in fragments]
    assert _render_context_block(()) == ""


@pytest.mark.parametrize("actual,expected", [([], set()), ([(1,), (1,)], {(1,)}), ([(2,)], {(1,)})])
def test_matrix_rejects_empty_duplicate_or_wrong_keys(actual, expected):
    with pytest.raises(ValueError):
        complete_keys(actual, expected)


def test_matrix_reports_complete_keys():
    assert complete_keys([(1,), (2,)], {(1,), (2,)}) == {"expected": 2, "actual": 2, "unique": 2}


def test_measurement_owns_tracing_and_does_one_real_recall():
    system = AuditedSystem(byte_cap=None)
    system.observe(Observation(datetime(2026, 10, 19, tzinfo=timezone.utc), "user", "actual source text"))
    recalled, metrics = measure_recall(system, "actual source")
    assert recalled.fragments_used and system.audit_recall_count == 1
    assert metrics["recall_ms"] >= 0 and metrics["recall_peak"] > 0
    import tracemalloc
    tracemalloc.start()
    try:
        with pytest.raises(ValueError):
            measure_recall(system, "actual source")
    finally:
        tracemalloc.stop()
    assert system.audit_recall_count == 1
