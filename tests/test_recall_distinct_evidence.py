"""Recall must spend its finite context budget on distinct, auditable evidence."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from benchmarks.scripts import phase9_recall_diagnosis as diagnosis
from psychology_base.companion.edges import EdgeStore
from psychology_base.companion.pam_export import pam_export, pam_import
from psychology_base.companion.reconstruction import (
    ActivationState, _distinct_ranked_fragments, reconstruct, spread_activation,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation, make_fragment
from psychology_base.eval.benchmarks import attachment_longitudinal as harness


BASE = datetime(2026, 10, 18, tzinfo=timezone.utc)


def fragment(text, index=0, **kwargs):
    return make_fragment(text, BASE + timedelta(minutes=index), **kwargs)


def test_duplicate_replies_cannot_displace_distinct_user_evidence():
    system = CompanionSystem(top_k=6, byte_cap=None)
    fact = "My brother's guide dog is named Miso."
    for index, (speaker, text) in enumerate(
        [("user", fact)] + [("companion", "I hear you. I am here with you.")] * 18
    ):
        system.observe(Observation(
            BASE + timedelta(minutes=index), speaker, text,
            tags=("user:alice", f"session:{index}", f"speaker:{speaker}"),
        ))
    cue = "Do you remember my brother's guide dog?"
    raw = diagnosis.ranked_pool(system, cue, legacy=True)
    assert all(f.content != fact for f, _ in raw[:6])  # Reproduces the old failure.
    recalled = system.recall(cue)
    assert any(f.content == fact for f in recalled.fragments_used)
    assert len(recalled.fragments_used) == 2
    assert len(system.companion.store) == 19  # Storage and provenance were not collapsed.
    assert {f.id for f in recalled.fragments_used} <= {f.id for f in system.companion.store.all()}


@pytest.mark.parametrize("difference", ["owner", "speaker", "subject", "entity", "copresence", "tag", "affect", "parents"])
def test_identical_words_with_different_evidence_provenance_remain_distinct(difference):
    original = fragment("I am ready.", tags=("user:alice", "session:first", "speaker:user"))
    kwargs = {"tags": ("user:alice", "session:second", "speaker:user")}
    if difference == "owner":
        kwargs["tags"] = ("user:bob", "session:second", "speaker:user")
    elif difference == "speaker":
        kwargs["tags"] = ("user:alice", "session:second", "speaker:companion")
    elif difference in {"subject", "entity", "copresence", "tag"}:
        kwargs["tags"] += (f"{difference}:different",)
    elif difference == "affect":
        kwargs["affect"] = {"valence": -0.8, "arousal": 0.9}
    elif difference == "parents":
        kwargs["parent_ids"] = (original.id,)
    other = fragment(original.content, 1, **kwargs)
    selected = _distinct_ranked_fragments([(original, 2.0), (other, 1.0)])
    assert [f.id for f, _ in selected] == [original.id, other.id]


@pytest.mark.parametrize("session_tags", [("session:first", "session:second"), ("session_1", "session_2")])
def test_only_repeated_occurrence_metadata_is_collapsed(session_tags):
    a = fragment("A shared sentence.", tags=("speaker:user", session_tags[0]))
    b = fragment(a.content, 1, tags=(session_tags[1], "speaker:user"))
    assert _distinct_ranked_fragments([(b, 3.0), (a, 2.0)]) == [(b, 3.0)]
    assert pam_import(pam_export((a, b))) == [a, b]


def test_exact_grouping_does_not_merge_changed_facts_or_negation():
    texts = ("My dog is Miso.", "My dog is Milo.", "My dog is not Miso.")
    ranked = [(fragment(text, i), 3.0 - i) for i, text in enumerate(texts)]
    assert _distinct_ranked_fragments(ranked) == ranked


def test_distinct_budget_does_not_pin_an_old_fact_or_expand_top_k():
    unrelated = fragment("Pottery on weekends.")
    matches = [fragment(f"Astronomy observation {chr(97 + i)}.", i + 1) for i in range(9)]
    result = reconstruct([unrelated, *matches], "astronomy", None, ActivationState(), 0,
                         edges=EdgeStore(), top_k=6)
    assert len(result.fragments_used) == 6
    assert unrelated.id not in {f.id for f in result.fragments_used}
    assert all("Astronomy" in f.content for f in result.fragments_used)


def test_hebbian_learning_uses_the_same_distinct_pool_as_context():
    repeated = [fragment("Remember this topic.", i, tags=(f"session:{i}",)) for i in range(8)]
    fact = fragment("Remember this useful fact.", 9)
    edges = EdgeStore()
    result = reconstruct([*repeated, fact], "remember", None, ActivationState(), 0, edges=edges)
    assert len(result.fragments_used) == 2
    assert len(edges.weights) == 1
    assert edges.get(result.fragments_used[0].id, result.fragments_used[1].id) > 0
    assert all(edges.get(a.id, b.id) == 0 for a in repeated for b in repeated)


@pytest.mark.parametrize("name", [
    "supportive-en", "supportive-zh", "neutral-en", "neutral-zh", "rupture-en", "rupture-zh",
    "sycophancy-en", "repair-en", "repair-zh", "short-duration-en", "fifteen-sessions-en",
    "extreme-emotion-en",
])
def test_real_longitudinal_controls_recall_original_fact_at_every_boundary(name):
    result = harness.run_trajectory(harness.build_trajectory(name))
    report = result.report()
    assert report["trajectory_complete"]
    assert report["continuity_probe_pass"]
    assert all(t["original_user_fact_in_context"] for t in report["block_transitions"])
    assert not report["model_response_quality_measured"]
    assert not report["attachment_validity_established"]


def test_checkpoint_rejects_a_different_recall_selection_policy(tmp_path):
    value = harness.build_trajectory("supportive-en").model_dump(mode="json")
    value["sessions"] = value["sessions"][:4]
    manifest = harness.TrajectoryManifest.model_validate(value)
    path = tmp_path / "checkpoint.json"
    harness.run_trajectory(manifest, checkpoint_path=path, stop_after=2)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    del envelope["payload"]["configuration"]["fragment_recall_policy"]
    envelope["sha256"] = harness.digest(envelope["payload"])
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration mismatch"):
        harness.load_checkpoint(path, manifest)


def test_diagnostic_counterfactuals_do_not_mutate_live_state():
    system = CompanionSystem(byte_cap=None)
    system.observe(Observation(BASE, "user", "My guide dog is Miso.", tags=("speaker:user",)))
    system.recall("guide dog")
    before = copy.deepcopy(system.inspect())
    weights = copy.deepcopy(system.edges.weights)
    for kwargs in ({}, {"reset_residual": True}, {"reset_edges": True},
                   {"sleep_ticks": 1}, {"reset_residual": True, "direct_only": True}):
        assert diagnosis.ranked_pool(system, "Miso", **kwargs)
    assert system.inspect() == before
    assert system.edges.weights == weights


def test_diagnostic_edge_reset_preserves_configured_chinese_priors(chinese_cases):
    pytest.importorskip("jieba")
    system = CompanionSystem(cjk_tokenizer="jieba")
    for text in chinese_cases["edge_pair"]:
        system.observe(Observation(BASE, "user", text))
    # Before Hebbian updates, rebuilding priors must reproduce the real graph.
    actual = diagnosis.ranked_pool(system, "焦虑")
    rebuilt = diagnosis.ranked_pool(system, "焦虑", reset_edges=True)
    assert [f.id for f, _ in actual] == [f.id for f, _ in rebuilt]
    assert [score for _, score in actual] == pytest.approx([score for _, score in rebuilt])


@pytest.mark.parametrize("count", [2, 8, 32])
def test_dense_graph_cannot_manufacture_activation_mass(count):
    fragments = [fragment(f"Topic {chr(97 + i)}.", i) for i in range(count)]
    edges = EdgeStore()
    for a in fragments:
        for b in fragments:
            edges.set(a.id, b.id, 3.0)
    state = ActivationState(residual={f.id: 10_000.0 for f in fragments})
    scores = spread_activation(fragments, "topic", state, edges=edges)
    # At most 1 cue + 1 residual, followed by two mass-conserving hops.
    assert sum(scores.values()) <= 2.0 * (1.0 + 0.5 + 0.25) + 1e-12


@pytest.mark.parametrize("mode,active_sources", [("unmatched", 0), ("isolated", 1), ("connected_pair", 2)])
def test_spreading_only_scans_neighbors_of_active_sources(mode, active_sources):
    class CountingEdges(EdgeStore):
        lookups = 0

        def get(self, a, b):
            self.lookups += 1
            return super().get(a, b)

    fragments = [fragment("uniqueanchor")] + [fragment("unrelated stored note", i) for i in range(1, 32)]
    edges = CountingEdges()
    if mode == "connected_pair":
        edges.set(fragments[0].id, fragments[1].id, 0.6)
    cue = "missingquery" if mode == "unmatched" else "uniqueanchor"
    scores = spread_activation(fragments, cue, ActivationState(), edges=edges)
    assert edges.lookups == active_sources * (len(fragments) - 1)
    if mode == "unmatched":
        assert scores == {}
    elif mode == "isolated":
        assert scores == {fragments[0].id: 1.0}
    else:
        assert scores == pytest.approx({fragments[0].id: 1.25, fragments[1].id: 0.5})


def test_specific_cue_overlap_beats_a_shared_function_word_without_graph_edges():
    fact = fragment("I paint tiny ceramic boats on Saturdays.")
    reply = fragment("I hear you.", 1)
    scores = spread_activation([reply, fact], "What do I paint on Saturdays?", ActivationState(),
                               edges=EdgeStore())
    assert scores[fact.id] > scores[reply.id]
    assert sum(scores.values()) == pytest.approx(1.0)


def test_off_scope_or_evicted_residual_cannot_consume_the_live_activation_budget():
    live = fragment("Guide dog Miso.")
    state = ActivationState(residual={"outside-scope": 1e9})
    scores = spread_activation([live], "guide dog", state, edges=EdgeStore())
    assert scores == {live.id: 1.0}


@pytest.mark.parametrize("fact,cue", [
    ("My brother has a guide dog called Miso.", "What is my brother's guide dog called?"),
    ("I paint tiny ceramic boats on Saturdays.", "What do I paint on Saturdays?"),
])
@pytest.mark.parametrize("varied_replies", [False, True])
@pytest.mark.parametrize("fact_session", [0, 2])
def test_non_marker_facts_survive_repetitive_and_varied_dialogue(fact, cue, varied_replies, fact_session):
    system = CompanionSystem(top_k=6, byte_cap=None)
    anchor = None
    for index in range(16):
        utterance = fact if index == fact_session else cue
        recall = system.recall(utterance)
        if index in (4, 8, 12):
            assert anchor in {f.id for f in recall.fragments_used}
        reply = "I hear you. We can take this slowly."
        if varied_replies:
            reply += f" We could discuss {chr(97 + index)} next."
        for speaker, content in (("user", utterance), ("companion", reply)):
            system.observe(Observation(
                BASE + timedelta(days=index), speaker, content,
                tags=("user:alice", f"session:{index}", f"speaker:{speaker}"),
            ))
            if index == fact_session and speaker == "user":
                anchor = system.companion.store.all()[-1].id
        system.mark_session_boundary()


def diagnostic_evidence(tmp_path):
    value = harness.build_trajectory("supportive-en").model_dump(mode="json")
    value["sessions"] = value["sessions"][:5]
    manifest = harness.TrajectoryManifest.model_validate(value)
    report = harness.run_trajectory(manifest).report()
    directory = tmp_path / "evidence" / manifest.trajectory_id
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return directory.parent


def test_diagnostic_cli_replays_real_contexts_and_reports_its_scope(tmp_path, capsys):
    directory = diagnostic_evidence(tmp_path)
    output = tmp_path / "diagnosis.json"
    assert diagnosis.main(["--evidence-dir", str(directory), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["all_replayed_contexts_match_reference"]
    assert report["actual_all_boundary_pass_count"] == 1
    assert report["selected_trajectories"] == 1
    assert "not chain interventions" in report["evidence_scope"]
    assert "actual_all_boundary_pass_count" in capsys.readouterr().out


def test_diagnostic_cli_does_not_overwrite_prior_evidence(tmp_path):
    directory = diagnostic_evidence(tmp_path)
    output = tmp_path / "existing.json"
    output.write_bytes(b"previous evidence\n")
    with pytest.raises(FileExistsError):
        diagnosis.main(["--evidence-dir", str(directory), "--output", str(output)])
    assert output.read_bytes() == b"previous evidence\n"


def test_diagnostic_refuses_to_invent_missing_trajectories(tmp_path):
    with pytest.raises(ValueError, match="no trajectory manifests"):
        diagnosis.run(tmp_path)


def test_diagnostic_reports_reference_drift_instead_of_silent_agreement(tmp_path):
    directory = diagnostic_evidence(tmp_path)
    report_path = next(directory.glob("*/report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["per_session"][1]["context_fragments"][0] = []
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = diagnosis.run(directory)
    assert not result["all_replayed_contexts_match_reference"]
