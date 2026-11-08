"""Trace integrity and real state transitions for the focused-policy review."""
from __future__ import annotations

import copy
from itertools import combinations

import pytest

from benchmarks.scripts import phase9_carryover_run as cli
from psychology_base.companion import reconstruction as memory
from psychology_base.eval.benchmarks import cue_context as observed
from psychology_base.eval.benchmarks import focused_context as study
from psychology_base.eval.benchmarks import recall_carryover as carry
from psychology_base.eval.benchmarks import turn_association as association
from tests.test_cue_context import fragment
from tests.test_focused_context_study import frozen, validate_archived


@pytest.mark.parametrize("report_kind", ["broad", "focused"])
@pytest.mark.parametrize("corruption", ["missing_fragment", "foreign_id", "duplicate_id", "source_pointer",
    "rank_gap", "rank_swap", "rank_duplicate", "rank_bool"])
def test_trace_inventory_and_lower_ranks_cannot_be_forged(report_kind, corruption):
    if report_kind == "broad":
        from tests.test_cue_context_study import frozen as earlier, validate_archived as validate_earlier
        _, cases, controls, report = earlier()
        validate = validate_earlier
    else:
        _, cases, controls, report = frozen()
        validate = validate_archived
    trace = report["traces"][0]
    tail = next(f for f in trace["fragments"] if f["rank"] == 7)
    if corruption == "missing_fragment":
        trace["fragments"].remove(tail)
    elif corruption == "foreign_id":
        tail["id"] = "foreign"
    elif corruption == "duplicate_id":
        tail["id"] = trace["selected_ids"][0]
    elif corruption == "source_pointer":
        tail["source"] = [-1, -1]
    elif corruption == "rank_gap":
        tail["rank"] = 999
    elif corruption == "rank_swap":
        other = next(f for f in trace["fragments"] if f["rank"] == 8)
        tail["rank"], other["rank"] = other["rank"], tail["rank"]
    elif corruption == "rank_duplicate":
        tail["rank"] = 8
    else:
        tail["rank"] = True
    with pytest.raises(ValueError, match="trace"):
        validate(report, cases, controls)


@pytest.mark.parametrize("corruption", ["focus", "foreign_ids", "duplicate_ids", "budget",
    "foreign_activation", "nan_activation", "boolean_activation", "negative_activation"])
def test_matching_baseline_and_primary_corruptions_still_fail_ownership_validation(corruption):
    _, cases, controls, report = frozen()
    primary = next(r for r in report["trials"] if r["candidate"] == "focus_companion")
    baseline = next(r for r in report["trials"] if (r["case"], r["state"], r["absent"], r["candidate"])
                    == (primary["case"], primary["state"], primary["absent"], "metadata_only"))
    inventory = next(i["ids"] for i in report["source_inventories"] if
                     (i["case"], i["candidate"], i["absent"])
                     == (primary["case"], "metadata_only", primary["absent"]))
    owned_id = next(iter(inventory))
    if corruption == "focus":
        field, value = "focus_ref", [-1, -1]
    elif corruption == "foreign_ids":
        field, value = "witness_ids", ["foreign"]
    elif corruption == "duplicate_ids":
        field, value = "witness_ids", [owned_id, owned_id]
    elif corruption == "budget":
        field, value = "witness_ids", list(inventory)[:7]
    elif corruption == "foreign_activation":
        field, value = "witness_activation", {"foreign": 1.0}
    else:
        amount = {"nan_activation": float("nan"), "boolean_activation": True, "negative_activation": -1.0}[corruption]
        field, value = "witness_activation", {owned_id: amount}
    baseline[field] = primary[field] = value
    # Parity alone cannot authenticate either view. Keep it satisfied so this
    # exercises the independent input/ownership/value guards.
    assert all(study.extra_checks(report["trials"]).values())
    with pytest.raises(ValueError, match="input or Witness view"):
        validate_archived(report, cases, controls)


def test_new_ablation_arm_cannot_change_annotated_observation_ids():
    _, cases, controls, report = frozen()
    item = next(i for i in report["source_inventories"] if i["candidate"] == "focus_shared")
    item["ids"]["changed-id"] = item["ids"].pop(next(iter(item["ids"])))
    with pytest.raises(ValueError, match="observation IDs"):
        validate_archived(report, cases, controls)


def test_control_arm_names_are_checked_independently_of_control_results():
    _, cases, controls, report = frozen()
    report["software_controls"]["unregistered"] = report["software_controls"].pop("focus_shared")
    with pytest.raises(ValueError, match="control arms"):
        validate_archived(report, cases, controls)


def test_unregistered_focused_arm_has_no_fallback_policy():
    with pytest.raises(ValueError, match="Unknown focused-context arm"):
        study.system_for_arm("unregistered", cjk_tokenizer="unicode")


def selection_pool():
    return [fragment(0, "current question", group="focus"), fragment(1, "answer detail", group="focus"),
        *(fragment(0, f"history old irrelevant broad topic clutter {i}", group=f"d{i}") for i in range(6))]


@pytest.mark.parametrize("arm", ["focus_companion", "focus_shared"])
def test_focus_api_changes_the_real_pool_and_learns_once_from_that_pool(arm):
    system = study.system_for_arm(arm, cjk_tokenizer="unicode", top_k=2)
    baseline = study.system_for_arm("metadata_only", cjk_tokenizer="unicode", top_k=2)
    fs = selection_pool()
    for item in (system, baseline):
        for f in fs:
            item._ingest_fragment(f)
    before, witness_before = copy.deepcopy(system.edges), copy.deepcopy(system.witness_edges)
    cue, focus = "history old irrelevant broad topic clutter current question", "current question"
    ordinary = baseline.recall(cue, scope=("alice", "memory"), focus_cue=focus)
    result = system.recall(cue, scope=("alice", "memory"), focus_cue=focus)
    assert fs[1] in result.fragments_used and fs[1] not in ordinary.fragments_used
    selected = {f.id for f in result.fragments_used}
    assert len(selected) == 2 and result.cue == cue
    assert set(memory.tokens(result.content)) <= set().union(*(set(memory.tokens(f.content)) for f in result.fragments_used))
    for a, b in combinations(fs, 2):
        delta = 0.05 if {a.id, b.id} <= selected else 0.0
        assert system.edges.get(a.id, b.id) == pytest.approx(before.get(a.id, b.id) + delta)
    assert system.witness_edges == witness_before
    assert system.activation.recall_count == system.witness_state.activation.recall_count == 1
    assert system.companion.store.all() == baseline.companion.store.all()


@pytest.mark.parametrize("focus", [None, ""])
def test_focus_omission_and_empty_string_reach_actual_reconstruction(focus):
    system = study.system_for_arm("focus_companion", cjk_tokenizer="unicode")
    fs = selection_pool()
    for f in fs:
        system._ingest_fragment(f)
    cue = "current question"
    result = system.recall(cue, scope=("alice", "memory"), focus_cue=focus)
    if focus is None:
        assert fs[1] in result.fragments_used
    else:
        baseline = study.system_for_arm("metadata_only", cjk_tokenizer="unicode")
        for f in fs:
            baseline._ingest_fragment(f)
        expected = baseline.recall(cue, scope=("alice", "memory"))
        expected.metadata["fragment_context_selection"] = {
            "name": "research:focused_context_scores", "version": None}
        assert result == expected


@pytest.fixture(scope="module")
def public_sources():
    pytest.importorskip("jieba", reason="Optional Chinese backend not installed")
    from psychology_base.i18n import tokenizer_identity
    assert tokenizer_identity("jieba")["module_version"] == "0.42.1"
    data_root = cli.ROOT / ".data/kdconv/data"
    if not all((data_root / path).exists() for path in ("film/test.json", "music/test.json", "film/train.json", "music/train.json")):
        pytest.skip("Optional pinned KdConv sources not provisioned")
    datasets, _, _ = observed.regression_inputs(data_root)
    return {case["id"]: (sources[case["source"]], case, protocol)
            for sources, cases, protocol in datasets for case in cases}


@pytest.mark.parametrize("case_id", ["film/test/82/13", "film/train-development/499/19",
    "music/train-development/849/12", "music/train-development/431/11"])
@pytest.mark.parametrize("state", carry.STATES)
@pytest.mark.parametrize("absent", [False, True])
def test_actual_public_recall_preserves_both_gains_and_all_three_recorded_losses(public_sources, case_id, state, absent):
    source, case, protocol = public_sources[case_id]
    _, _, _, report = frozen()
    expected = {(r["case"], r["candidate"], r["state"], r["absent"]): r for r in report["trials"]}
    for arm in study.ARMS:
        base, ids, _ = association.setup(source, case, arm, protocol,
                                        absent=absent, arms=study.ARMS, system_factory=study.system_for_arm)
        actual = carry.evaluate(source, case, base, ids, "production", state, absent, focus_current=True)
        archived = expected[case_id, arm, state, absent]
        for field in ("selected_ids", "selected_sources", "gold_ids", "recall_at_6", "mrr_at_6",
                      "support_precision", "nonempty", "focus_ref", "witness_ids", "witness_activation"):
            assert actual[field] == archived[field], (case_id, arm, state, absent, field)
