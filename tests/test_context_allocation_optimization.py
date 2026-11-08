"""Exact archived-function parity, including numerical and serializer boundaries.

Authored inputs exercise software invariants; they are not semantic quality data.
"""
import importlib.util
from pathlib import Path
import random
import sys

import pytest

from psychology_base.companion import reconstruction as memory
from tests.test_context_allocation import MODES, render
from tests.test_cue_context import fragment


@pytest.fixture(scope="module")
def original_allocator():
    path = Path(".data/application_context/context-allocation-run-v1/executed-source/"
                "src/psychology_base/companion/reconstruction.py")
    if not path.exists():
        pytest.skip("Original local allocation experiment archive unavailable")
    spec = importlib.util.spec_from_file_location("allocation_optimization_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compare(original, monkeypatch, fs, scores, cue, mode, limit, renderer, byte_limit):
    snapshot = memory.recall_tokens(fs, cue, "unicode")
    option = memory.ResearchSelection(mode, renderer, byte_limit)
    actual = memory.ordered_selection(fs, scores, limit, option, snapshot=snapshot)
    with monkeypatch.context() as patch:
        patch.setattr(memory, "_allocate_occurrences", original._allocate_occurrences)
        expected = memory.ordered_selection(fs, scores, limit, option, snapshot=snapshot)
    assert actual == expected  # Includes original scores, every trace field and search counts.
    return actual


def order_sensitive(fs):
    return render(fs) + ("x" * 73 if fs and fs[0].content.endswith("blue") else "")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("limit", [0, 1, 3, 6])
@pytest.mark.parametrize("renderer", [render, order_sensitive])
def test_exact_search_parity_across_ties_input_orders_and_byte_boundaries(
        original_allocator, monkeypatch, mode, limit, renderer):
    fs = [fragment(i, text, group=group) for group in ("first", "second", "third")
          for i, text in enumerate(["red blue", "question", "green outcome"])]
    rng = random.Random(27)
    for iteration in range(9):
        rng.shuffle(fs)
        values = ([1.0] * 9, [1e16, 1., -1e16, .5, .5, 0., -1., 1., .01],
                  [rng.choice([-.5, 0., .1, .2, .3]) for _ in fs])[iteration % 3]
        scores = {f.id: score for f, score in zip(fs, values)}
        byte_limit = [0, len(renderer(fs[:1]).encode("utf-8")), 6000][iteration % 3]
        compare(original_allocator, monkeypatch, fs, scores, "red green absent", mode,
                limit, renderer, byte_limit)


@pytest.mark.parametrize("mode", MODES)
def test_large_sparse_bit_positions_and_changed_scores_are_recall_local(original_allocator, monkeypatch, mode):
    fs = [fragment(0, "query blue" if i in (0, 34, 70) else f"detail {i}", group=str(i))
          for i in range(72)]
    for reverse in (False, True):
        ordered = list(reversed(fs)) if reverse else fs
        scores = {f.id: (1. if i in (0, 34, 70) else 0.) for i, f in enumerate(ordered)}
        compare(original_allocator, monkeypatch, ordered, scores, "query", mode, 6, render, 6000)


@pytest.mark.parametrize("mode", MODES)
def test_empty_no_support_and_expansion_fallback_match_original(original_allocator, monkeypatch, mode):
    fs = [fragment(i, text) for i, text in enumerate(["query", "question", "outcome"])]
    for inventory, scores, cue in (([], {}, "query"), (fs, {}, "query"),
                                    (fs, {fs[0].id: 1.}, "absent")):
        compare(original_allocator, monkeypatch, inventory, scores, cue, mode, 6, render, 6000)
    with monkeypatch.context() as patch:
        patch.setattr(memory, "_ALLOCATION_MAX_EXPANSIONS", 0)
        patch.setattr(original_allocator, "_ALLOCATION_MAX_EXPANSIONS", 0)
        _, trace = compare(original_allocator, patch, fs, {fs[0].id: 1.}, "query", mode, 6, render, 6000)
    assert trace["fallback"] == "expansion_limit"
