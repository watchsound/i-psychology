"""Study integrity and actual-source integration, distinct from performance claims."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tracemalloc

import pytest

from benchmarks.scripts import phase9_chinese_retrieval_run as cli
from psychology_base.companion import reconstruction as memory
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks import chinese_retrieval as study
from psychology_base.i18n import tokenizer_identity


@pytest.fixture
def source_study():
    protocol = json.loads((cli.FIXTURES / "protocol.json").read_text(encoding="utf-8"))
    catalog_path = cli.FIXTURES / "catalog.json"
    assert study.digest(catalog_path) == protocol["catalog_sha256"]
    if not cli.SOURCE.exists():
        pytest.skip("Optional public KdConv source not provisioned; study CLI fails if absent")
    source = study.load_source(cli.SOURCE, protocol)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    return source, catalog, protocol


@pytest.fixture
def real_jieba():
    pytest.importorskip("jieba", reason="Optional Chinese backend not installed")
    assert tokenizer_identity("jieba")["module_version"] == "0.42.1"


def test_catalog_labels_splits_and_spans_match_actual_public_source(source_study):
    source, catalog, protocol = source_study
    assert catalog == study.build_catalog(source, protocol)
    splits = catalog["split_dialogues"]
    assert not set(splits["development"]) & set(splits["evaluation"])
    assert len({c["dialogue"] for c in catalog["cases"]}) == len(catalog["cases"])
    assert {s: sum(c["split"] == s for c in catalog["cases"]) for s in splits} == {
        "development": 12, "evaluation": 21}
    for case in catalog["cases"]:
        assert len(case["history"]) == len(case["absent_history"]) == 24
        assert len(set(map(tuple, case["history"]))) == 24
        assert set(map(tuple, case["gold"])) <= set(map(tuple, case["history"]))
        assert not set(map(tuple, case["gold"])) & set(map(tuple, case["absent_history"]))
        assert {ref[0] for ref in case["history"] + case["absent_history"] + case["cue_refs"]} <= set(splits[case["split"]])
        answer = study.message(source, case["answer"])
        assert case["fact"] in answer["attrs"]
        start, end = case["span"]
        assert answer["message"][start:end] == case["fact"]["attrvalue"]
        cue = " ".join(study.message(source, ref)["message"] for ref in case["cue_refs"])
        assert case["fact"]["attrvalue"] not in cue


def test_tampered_source_is_rejected_before_evaluation(source_study, tmp_path):
    _, _, protocol = source_study
    path = tmp_path / "changed.json"
    path.write_bytes(cli.SOURCE.read_bytes() + b" ")
    with pytest.raises(ValueError, match="frozen KdConv"):
        study.load_source(path, protocol)


def test_missing_or_tampered_catalog_fails_cli_without_report(source_study, tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text("{}", encoding="utf-8")
    output = tmp_path / "report.json"
    assert cli.main(["run", "--catalog", str(path), "--output", str(output)]) == 1
    assert not output.exists()
    assert cli.main(["run", "--catalog", str(tmp_path / "absent"), "--output", str(output)]) == 1
    assert not output.exists()


def test_reports_cannot_overwrite_existing_evidence(tmp_path):
    path = tmp_path / "evidence.json"
    cli.write_new(path, {"original": True})
    with pytest.raises(FileExistsError):
        cli.write_new(path, {"replacement": True})
    assert json.loads(path.read_text()) == {"original": True}


@pytest.mark.parametrize("setting,value", [("top_k", 12), ("witness_active", False), ("feature_cap_per_channel", 128)])
def test_protocol_settings_cannot_be_silently_ignored(setting, value, tmp_path):
    protocol = json.loads((cli.FIXTURES / "protocol.json").read_text(encoding="utf-8"))
    protocol["fixed"][setting] = value
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="outside this fixed study"):
        cli.load_inputs(cli.SOURCE, path, cli.FIXTURES / "catalog.json")


def test_existing_allocation_tracing_is_not_stopped_or_misreported():
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="owns recall allocation tracing"):
            study.evaluate_case([], {}, "unicode")
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("policy", study.CANDIDATES)
def test_english_feature_adapter_preserves_existing_memory_tokens(policy, real_jieba):
    text = "It's Python ABC-123, not somebody else's code."
    assert study.features(text, policy) == memory.tokens(text)


def test_real_search_expands_compounds_and_width_ids_preserve_original_offsets(real_jieba):
    original = "北京大学的Ｐｙｔｈｏｎ项目ＡＢＣ－１２３"
    assert "大学" not in study.features("北京大学", "precise")
    assert "大学" in study.features("北京大学", "search")
    assert "京大" in study.features("北京大学", "bigram")
    folded = study.fold_width(original)
    assert len(folded) == len(original)
    assert folded == "北京大学的Python项目ABC-123"
    assert "w:abc-123" in study.features(original, "width_ids")
    assert "w:abc-456" not in study.features(original, "width_ids")
    start = original.index("Ｐ")
    assert study.fold_width(original[start:start + 6]) == folded[start:start + 6] == "Python"
    assert original == "北京大学的Ｐｙｔｈｏｎ项目ＡＢＣ－１２３"


def test_bounded_channels_have_a_label_independent_feature_budget(real_jieba):
    text = "北京大学研究自然语言处理和人工智能以及心理学与数学物理化学"
    result = study.features(text, "bounded", cap=3)
    assert len([f for f in result if f.startswith("w:")]) == 3
    assert len([f for f in result if f.startswith("h:")]) == 3
    assert len(set(result)) == len(result)
    with pytest.raises(ValueError, match="Unknown"):
        study.features(text, "unregistered")
    with pytest.raises(ValueError, match="Unknown"):
        study.study_system("unregistered")


def test_experiment_binds_real_code_only_to_companion_seed_extraction(real_jieba):
    original_globals = memory.reconstruct.__globals__
    original_spread = memory.spread_activation
    a = study.study_system("search")
    b = study.study_system("bigram")
    assert a.recall.__func__.__code__ is CompanionSystem.recall.__code__
    reconstruct = a.recall.__func__.__globals__["reconstruct"]
    assert reconstruct.__code__ is memory.reconstruct.__code__
    spread = reconstruct.__globals__["spread_activation"]
    assert spread.__code__ is original_spread.__code__
    assert reconstruct.__globals__["coalesce"] is memory.coalesce
    assert spread.__globals__["jaccard_prior"] is memory.jaccard_prior
    assert a.recall.__func__.__globals__["witness_couple"] is CompanionSystem.recall.__globals__["witness_couple"]
    assert spread.__globals__["tokens"]("北京大学") != b.recall.__func__.__globals__["reconstruct"].__globals__["spread_activation"].__globals__["tokens"]("北京大学")
    assert memory.reconstruct.__globals__ is original_globals
    assert memory.reconstruct.__globals__["spread_activation"] is original_spread


@pytest.mark.parametrize("policy", ["search", "bigram", "bounded", "width_ids"])
def test_bound_representation_reaches_actual_spreading_snapshot(policy, real_jieba):
    from psychology_base.eval.adapters import make_fragment

    system = study.study_system(policy)
    reconstruct = system.recall.__func__.__globals__["reconstruct"]
    fragments = [make_fragment("北京大学", study.BASE), make_fragment("大学生活", study.BASE)]
    cue = "大学"
    snapshot = reconstruct.__globals__["recall_tokens"](fragments, cue, "jieba")
    assert snapshot.cue == set(study.features(cue, policy))
    assert snapshot.fragments == [(f.id, set(study.features(f.content, policy))) for f in fragments]
    direct = reconstruct.__globals__["spread_activation"](fragments, cue, memory.ActivationState(), hops=0,
                                                           cjk_tokenizer="jieba")
    overlap = {fid: len(words & snapshot.cue) for fid, words in snapshot.fragments}
    total = sum(overlap.values())
    assert direct == {fid: count / total for fid, count in overlap.items() if count}
    def inspect_focus(pool, scores, snapshot):
        assert snapshot.focus == set(study.features("北京大学", policy))
        return scores
    inspected = study._bind(reconstruct, _selection_scores=inspect_focus)
    inspected(fragments, cue, None, memory.ActivationState(), 0,
              cjk_tokenizer="jieba", focus_cue="北京大学")


@pytest.mark.parametrize("policy", study.CANDIDATES)
@pytest.mark.parametrize("absent,warm", [(False, False), (True, False), (False, True), (True, True)])
def test_actual_public_source_uses_real_recall_budget_scope_and_provenance(source_study, real_jieba, policy, absent, warm):
    source, catalog, _ = source_study
    result = study.evaluate_case(source, catalog["cases"][0], policy, absent=absent, warm=warm)
    assert len(result["selected_ids"]) <= 6
    assert result["stored_occurrences"] == 25
    assert len(result["selected_ids"]) == len(result["selected_sources"])
    if absent:
        assert result["gold_ids"] == []
        assert result["recall_at_6"] is None
    else:
        assert len(result["gold_ids"]) == 1
        assert result["recall_at_6"] in (0.0, 1.0)  # A miss stays a miss, not a test failure.


@pytest.mark.parametrize("policy,mode", [("unicode", "unicode"), ("precise", "jieba")])
def test_canonical_study_system_is_exact_existing_engine_over_stateful_trace(source_study, real_jieba, policy, mode):
    source, catalog, _ = source_study
    candidate = study.study_system(policy)
    baseline = CompanionSystem(top_k=6, byte_cap=None, seed=0, cjk_tokenizer=mode)
    case = catalog["cases"][0]
    for i, ref in enumerate(case["history"]):
        event = study.Observation(study.BASE + study.timedelta(minutes=i), "user",
                                  study.message(source, ref)["message"], tags=("user:study",))
        candidate.observe(event)
        baseline.observe(event)
    for ref in case["cue_refs"]:
        cue = study.message(source, ref)["message"]
        assert candidate.recall(cue) == baseline.recall(cue)
        assert candidate.inspect() == baseline.inspect()
        assert candidate.activation == baseline.activation
        assert candidate.witness_state == baseline.witness_state
        assert candidate.edges.weights == baseline.edges.weights


def test_exploratory_diagnostic_preserves_source_and_uses_separate_state_copies(source_study, real_jieba):
    source, catalog, _ = source_study
    original = copy.deepcopy(source)
    result = study.diagnose_case(source, catalog["cases"][0])
    assert source == original
    assert set(result["variants"]) == {"cold", "warm", "warm_reset_residual"}
    assert all(len(v["selected_ids"]) <= 6 for v in result["variants"].values())
    assert all(isinstance(v["gold_hit"], bool) for v in result["variants"].values())


def test_metrics_do_not_credit_distractors_missing_evidence_or_low_rank():
    scores = study.score_context(["noise", "gold", "other"], {"gold"})
    assert scores["recall_at_6"] == 1
    assert scores["mrr_at_6"] == 0.5
    assert scores["support_precision"] == 1 / 3
    assert study.score_context([], {"gold"})["recall_at_6"] == 0
    assert study.score_context(["noise"], set())["recall_at_6"] is None
    assert study.score_context([], set())["nonempty"] is False


def test_selection_uses_development_only_and_cannot_promote_production():
    protocol = json.loads((cli.FIXTURES / "protocol.json").read_text(encoding="utf-8"))
    metric = {"recall_at_6": 0.5, "mrr_at_6": 0.4, "support_precision": 0.1,
              "absent_nonempty_rate": 0.5, "p95_latency_ms": 10, "peak_python_bytes": 100}
    summary = {split: {p: {s: copy.deepcopy(metric) for s in ("cold", "warm")}
                       for p in study.CANDIDATES} for split in ("development", "evaluation")}
    summary["development"]["search"]["cold"]["recall_at_6"] = 0.6
    # A better evaluation candidate cannot replace the development choice.
    summary["evaluation"]["bigram"]["cold"]["recall_at_6"] = 1.0
    result = study.decide(summary, protocol)
    assert result["development_selected"] == "search"
    assert not result["advance_to_independent_replication"]
    assert not result["production_adoption"]
    summary["evaluation"]["search"]["cold"]["recall_at_6"] = 0.6
    assert study.decide(summary, protocol)["advance_to_independent_replication"]
    summary["development"]["search"]["cold"]["absent_nonempty_rate"] = 0.6
    assert study.decide(summary, protocol)["development_selected"] is None


def test_incomplete_trial_matrix_is_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        study.summarize([])
