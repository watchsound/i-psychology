"""Tests for the Argilla + κ label-collection pipeline (Phase 4b §11 item 7).

Tested here (local-only — no Docker, no HTTP):

- ``run_probe_cohort`` invokes a canned responder on each probe.
- ``to-argilla`` round-trip: unlabeled rows → ArgillaRecord JSONL → back
  through ``import_argilla_labels`` → :class:`LabeledResponse`.
- ``_extract_label`` handles the four export-shape variants Argilla has
  shipped across versions.
- ``compute_pairwise_kappa`` pairs on ``external_id`` across auditors,
  including unmatched labels on either side.
- CLI driver end-to-end: run-probes → to-argilla → from-argilla × 2 →
  kappa, with stub mode and simulated Argilla export JSONLs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychology_base.applications.safety.argilla_pipeline import (
    PairwiseKappaReport,
    UnlabeledProbeResponse,
    _extract_label,
    compute_pairwise_kappa,
    export_responses_to_argilla,
    import_argilla_labels,
    main,
    read_labeled_responses,
    read_probe_responses,
    run_probe_cohort,
    write_labeled_responses,
    write_probe_responses,
)
from psychology_base.applications.safety.hypocrisy_gap import (
    HypocrisyProbe,
    LabeledResponse,
    Language,
    LoadingCategory,
    PILOT_PROBE_SET,
    ResponseLabel,
    Severity,
)


# ---------------------------------------------------------------------------
# run_probe_cohort
# ---------------------------------------------------------------------------


def _canned_probe_responder(probe: HypocrisyProbe) -> str:
    return f"ack:{probe.probe_id}:{probe.user_claim[:16]}"


def test_run_probe_cohort_shape_and_turn_indices():
    probes = PILOT_PROBE_SET[:3]
    rows = run_probe_cohort(
        probes,
        _canned_probe_responder,
        user_id="t01",
        session_id="s0001",
        start_turn_index=5,
    )
    assert len(rows) == 3
    assert rows[0].turn_index == 5
    assert rows[1].turn_index == 6
    assert rows[2].turn_index == 7
    for row, probe in zip(rows, probes):
        assert row.probe_id == probe.probe_id
        assert row.probe_user_claim == probe.user_claim
        assert row.loading_category is probe.loading_category
        assert row.severity is probe.severity
        assert row.locale is probe.locale
        assert row.companion_response.startswith("ack:")
        assert row.external_id == f"t01:s0001:{probe.probe_id}"


def test_run_probe_cohort_empty_probes_returns_empty():
    rows = run_probe_cohort(
        (), _canned_probe_responder, user_id="t01", session_id="s0001"
    )
    assert rows == []


# ---------------------------------------------------------------------------
# write_probe_responses / read_probe_responses round-trip
# ---------------------------------------------------------------------------


def test_probe_responses_jsonl_roundtrip(tmp_path):
    probes = PILOT_PROBE_SET[:2]
    rows = run_probe_cohort(
        probes, _canned_probe_responder, user_id="u", session_id="s"
    )
    path = tmp_path / "resp.jsonl"
    write_probe_responses(rows, path)
    recovered = read_probe_responses(path)
    assert recovered == rows


# ---------------------------------------------------------------------------
# export_responses_to_argilla + import_argilla_labels round-trip
# ---------------------------------------------------------------------------


def test_export_to_argilla_writes_native_schema(tmp_path):
    probes = PILOT_PROBE_SET[:2]
    rows = run_probe_cohort(
        probes, _canned_probe_responder, user_id="u", session_id="s"
    )
    path = tmp_path / "upload.jsonl"
    n = export_responses_to_argilla(rows, path)
    assert n == 2

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert set(first.keys()) == {"fields", "metadata", "external_id"}
    assert first["fields"]["probe_user_claim"] == probes[0].user_claim
    assert first["fields"]["companion_response"].startswith("ack:")
    assert first["fields"]["loading_category"] == probes[0].loading_category.value
    assert first["external_id"] == f"u:s:{probes[0].probe_id}"
    assert first["metadata"]["locale"] == probes[0].locale.value


def _simulate_argilla_export_modern(
    uploaded_lines: list[dict], labels: list[str]
) -> list[dict]:
    """Attach modern-schema responses: list-of-dicts with question_name=label."""
    out = []
    for line, label in zip(uploaded_lines, labels):
        enriched = dict(line)
        enriched["responses"] = [{"question_name": "label", "value": label}]
        out.append(enriched)
    return out


def test_argilla_import_roundtrip_modern_schema(tmp_path):
    probes = PILOT_PROBE_SET[:3]
    rows = run_probe_cohort(
        probes, _canned_probe_responder, user_id="u", session_id="s"
    )
    upload_path = tmp_path / "upload.jsonl"
    export_responses_to_argilla(rows, upload_path)
    uploaded = [json.loads(l) for l in upload_path.read_text("utf-8").splitlines()]

    labels = ["gently_corrects", "redirects", "hedges"]
    exported = _simulate_argilla_export_modern(uploaded, labels)
    export_path = tmp_path / "argilla_export.jsonl"
    with export_path.open("w", encoding="utf-8") as fh:
        for line in exported:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    labelled = import_argilla_labels(
        export_path, auditor_id="auditor-a", label_field="label"
    )
    assert len(labelled) == 3
    assert [r.auditor_label for r in labelled] == [
        ResponseLabel.GENTLY_CORRECTS,
        ResponseLabel.REDIRECTS,
        ResponseLabel.HEDGES,
    ]
    for row, probe in zip(labelled, probes):
        assert row.probe_id == probe.probe_id
        assert row.user_id == "u"
        assert row.session_id == "s"
        assert row.auditor_id == "auditor-a"
        assert row.loading_category is probe.loading_category
        assert row.severity is probe.severity
        assert row.locale is probe.locale


# ---------------------------------------------------------------------------
# _extract_label — robustness across export shapes.
# ---------------------------------------------------------------------------


def test_extract_label_modern_list_schema():
    line = {"responses": [{"question_name": "label", "value": "gently_corrects"}]}
    assert _extract_label(line, "label") is ResponseLabel.GENTLY_CORRECTS


def test_extract_label_dict_schema():
    line = {"responses": {"label": {"value": "redirects"}}}
    assert _extract_label(line, "label") is ResponseLabel.REDIRECTS


def test_extract_label_flat_top_level():
    line = {"label": "hedges"}
    assert _extract_label(line, "label") is ResponseLabel.HEDGES


def test_extract_label_metadata_fallback():
    line = {"metadata": {"label": "validates_falsehood"}}
    assert _extract_label(line, "label") is ResponseLabel.VALIDATES_FALSEHOOD


def test_extract_label_falls_back_to_first_response_when_no_question_names():
    """Older-schema export: responses list carries no `question_name` key at all.

    Fallback to the first entry's `value` is safe here because there is
    no "wrong question" to confuse it with.
    """
    line = {"responses": [{"value": "gently_corrects"}]}
    assert _extract_label(line, "sycophancy_label") is ResponseLabel.GENTLY_CORRECTS


def test_extract_label_refuses_fallback_when_other_questions_exist():
    """Multi-question workspace, `label_field` matches none → return None.

    Guards against silently reading a different question's value (e.g.,
    a confidence score or notes field) as the label. The caller raises
    with the external_id, prompting the operator to override
    ``--label-field``.
    """
    line = {
        "responses": [
            {"question_name": "confidence", "value": "0.8"},
            {"question_name": "notes", "value": "unsure between hedges and redirects"},
        ],
    }
    assert _extract_label(line, "label") is None


def test_extract_label_matches_named_question_in_multi_question_export():
    line = {
        "responses": [
            {"question_name": "confidence", "value": "0.8"},
            {"question_name": "label", "value": "gently_corrects"},
            {"question_name": "notes", "value": "clear truth-preservation"},
        ],
    }
    assert _extract_label(line, "label") is ResponseLabel.GENTLY_CORRECTS


def test_extract_label_mixed_entries_some_named_some_not_still_refuses_fallback():
    """Edge case: at least one entry declares question_name, others don't.

    Treat as ambiguous — the labelled dataset is non-canonical and the
    operator should override ``--label-field``.
    """
    line = {
        "responses": [
            {"value": "gently_corrects"},
            {"question_name": "confidence", "value": "0.8"},
        ],
    }
    assert _extract_label(line, "label") is None


def test_extract_label_unknown_returns_none():
    assert _extract_label({"responses": []}, "label") is None
    assert _extract_label({"label": "not_a_real_label"}, "label") is None


def test_extract_label_is_case_insensitive_and_strips_whitespace():
    assert (
        _extract_label({"label": "GENTLY_CORRECTS"}, "label")
        is ResponseLabel.GENTLY_CORRECTS
    )
    assert (
        _extract_label({"label": "  redirects  "}, "label")
        is ResponseLabel.REDIRECTS
    )
    assert _extract_label({"label": "GentlyCorrect"}, "label") is None


def test_extract_label_custom_label_field_name():
    line = {"responses": [{"question_name": "sycophancy_label", "value": "hedges"}]}
    assert _extract_label(line, "sycophancy_label") is ResponseLabel.HEDGES


# ---------------------------------------------------------------------------
# import_argilla_labels — error path when label unparseable.
# ---------------------------------------------------------------------------


def test_import_argilla_labels_raises_on_missing_label(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "fields": {
                    "probe_user_claim": "x",
                    "companion_response": "y",
                    "loading_category": "self_worth",
                },
                "metadata": {},
                "external_id": "u:s:p1",
                "responses": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="u:s:p1"):
        import_argilla_labels(path, auditor_id="x", label_field="label")


def test_import_argilla_labels_recovers_ids_from_external_id(tmp_path):
    """When metadata is missing probe_id/user_id/session_id, parse them from external_id."""
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        json.dumps(
            {
                "fields": {
                    "probe_user_claim": "x",
                    "companion_response": "y",
                    "loading_category": "self_worth",
                },
                "metadata": {},
                "external_id": "alice:sess01:probe-xyz",
                "label": "redirects",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = import_argilla_labels(path, auditor_id="A", label_field="label")
    assert len(rows) == 1
    r = rows[0]
    assert r.user_id == "alice"
    assert r.session_id == "sess01"
    assert r.probe_id == "probe-xyz"
    assert r.auditor_label is ResponseLabel.REDIRECTS


# ---------------------------------------------------------------------------
# compute_pairwise_kappa — pairing semantics.
# ---------------------------------------------------------------------------


def _mk_labeled(
    probe_id: str,
    *,
    user_id: str = "u",
    session_id: str = "s",
    label: ResponseLabel = ResponseLabel.GENTLY_CORRECTS,
    auditor_id: str = "A",
) -> LabeledResponse:
    return LabeledResponse(
        probe_id=probe_id,
        user_id=user_id,
        session_id=session_id,
        turn_index=0,
        probe_user_claim="",
        companion_response="",
        auditor_label=label,
        auditor_id=auditor_id,
        loading_category=LoadingCategory.SELF_WORTH,
        severity=Severity.MEDIUM,
        locale=Language.EN,
    )


def test_compute_pairwise_kappa_perfect_agreement():
    a = [_mk_labeled("p1", auditor_id="A"), _mk_labeled("p2", auditor_id="A",
         label=ResponseLabel.REDIRECTS)]
    b = [_mk_labeled("p1", auditor_id="B"), _mk_labeled("p2", auditor_id="B",
         label=ResponseLabel.REDIRECTS)]
    report = compute_pairwise_kappa(a, b)
    assert report.n_paired == 2
    assert report.n_only_a == 0
    assert report.n_only_b == 0
    assert report.kappa == pytest.approx(1.0)
    assert report.meets_pilot_gate is True
    assert report.auditor_a == "A"
    assert report.auditor_b == "B"


def test_compute_pairwise_kappa_total_disagreement_fails_gate():
    a = [
        _mk_labeled("p1", auditor_id="A", label=ResponseLabel.GENTLY_CORRECTS),
        _mk_labeled("p2", auditor_id="A", label=ResponseLabel.REDIRECTS),
    ]
    b = [
        _mk_labeled("p1", auditor_id="B", label=ResponseLabel.VALIDATES_FALSEHOOD),
        _mk_labeled("p2", auditor_id="B", label=ResponseLabel.HEDGES),
    ]
    report = compute_pairwise_kappa(a, b)
    assert report.kappa <= 0.0
    assert report.meets_pilot_gate is False


def test_compute_pairwise_kappa_reports_unmatched_rows():
    a = [_mk_labeled("p1", auditor_id="A"), _mk_labeled("p2", auditor_id="A")]
    b = [_mk_labeled("p1", auditor_id="B"), _mk_labeled("p3", auditor_id="B")]
    report = compute_pairwise_kappa(a, b)
    assert report.n_paired == 1
    assert report.unmatched_a == ("u:s:p2",)
    assert report.unmatched_b == ("u:s:p3",)


def test_compute_pairwise_kappa_zero_overlap_reports_no_gate():
    a = [_mk_labeled("p1", auditor_id="A")]
    b = [_mk_labeled("p2", auditor_id="B")]
    report = compute_pairwise_kappa(a, b)
    assert report.n_paired == 0
    assert report.meets_pilot_gate is False
    assert report.kappa == 0.0


def test_compute_pairwise_kappa_floor_override():
    a = [
        _mk_labeled("p1", auditor_id="A", label=ResponseLabel.GENTLY_CORRECTS),
        _mk_labeled("p2", auditor_id="A", label=ResponseLabel.REDIRECTS),
        _mk_labeled("p3", auditor_id="A", label=ResponseLabel.HEDGES),
        _mk_labeled("p4", auditor_id="A", label=ResponseLabel.VALIDATES_FALSEHOOD),
    ]
    b = [
        _mk_labeled("p1", auditor_id="B", label=ResponseLabel.GENTLY_CORRECTS),
        _mk_labeled("p2", auditor_id="B", label=ResponseLabel.REDIRECTS),
        _mk_labeled("p3", auditor_id="B", label=ResponseLabel.HEDGES),
        _mk_labeled("p4", auditor_id="B", label=ResponseLabel.GENTLY_CORRECTS),
    ]
    strict = compute_pairwise_kappa(a, b, pilot_kappa_floor=0.95)
    assert strict.meets_pilot_gate is False
    lenient = compute_pairwise_kappa(a, b, pilot_kappa_floor=0.40)
    assert lenient.meets_pilot_gate is True


# ---------------------------------------------------------------------------
# LabeledResponse JSONL round-trip.
# ---------------------------------------------------------------------------


def test_labeled_response_jsonl_roundtrip(tmp_path):
    rows = [
        _mk_labeled("p1", auditor_id="A", label=ResponseLabel.GENTLY_CORRECTS),
        _mk_labeled("p2", auditor_id="A", label=ResponseLabel.HEDGES),
    ]
    path = tmp_path / "labels.jsonl"
    write_labeled_responses(rows, path)
    recovered = read_labeled_responses(path)
    assert recovered == rows


# ---------------------------------------------------------------------------
# CLI driver — end-to-end stub flow.
# ---------------------------------------------------------------------------


def test_cli_end_to_end_stub_flow(tmp_path, capsys):
    out_responses_a = tmp_path / "resp-a.jsonl"
    rc = main(
        [
            "run-probes",
            "--app",
            "emotional_companion",
            "--mode",
            "stub",
            "--probe-source",
            "model",
            "--user-id",
            "u",
            "--session-id",
            "s0001",
            "--out",
            str(out_responses_a),
        ]
    )
    assert rc == 0
    assert out_responses_a.exists()
    lines = out_responses_a.read_text("utf-8").splitlines()
    assert len(lines) == len(PILOT_PROBE_SET)

    out_upload = tmp_path / "upload.jsonl"
    rc = main(
        [
            "to-argilla",
            "--responses",
            str(out_responses_a),
            "--out",
            str(out_upload),
        ]
    )
    assert rc == 0
    uploaded = [
        json.loads(l) for l in out_upload.read_text("utf-8").splitlines()
    ]
    assert len(uploaded) == len(PILOT_PROBE_SET)

    # Simulate Argilla export for two auditors. Auditor-A perfectly
    # agrees; Auditor-B flips one row to generate <1.0 but still-passing κ.
    def _write_export(path: Path, labels: list[str]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for line, label in zip(uploaded, labels):
                enriched = dict(line)
                enriched["responses"] = [
                    {"question_name": "label", "value": label}
                ]
                fh.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    # Mix labels so κ has variance to compute against. Auditor A assigns a
    # 3-category pattern; auditor B agrees on all but one row.
    cycle = ["gently_corrects", "redirects", "hedges"]
    labels_a = [cycle[i % 3] for i in range(len(PILOT_PROBE_SET))]
    labels_b = list(labels_a)
    labels_b[0] = "redirects"  # one disagreement
    export_a = tmp_path / "export-a.jsonl"
    export_b = tmp_path / "export-b.jsonl"
    _write_export(export_a, labels_a)
    _write_export(export_b, labels_b)

    out_labels_a = tmp_path / "labels-a.jsonl"
    out_labels_b = tmp_path / "labels-b.jsonl"
    rc = main(
        [
            "from-argilla",
            "--labelled",
            str(export_a),
            "--auditor-id",
            "auditor-a",
            "--out",
            str(out_labels_a),
        ]
    )
    assert rc == 0
    rc = main(
        [
            "from-argilla",
            "--labelled",
            str(export_b),
            "--auditor-id",
            "auditor-b",
            "--out",
            str(out_labels_b),
        ]
    )
    assert rc == 0

    out_report = tmp_path / "kappa.json"
    rc = main(
        [
            "kappa",
            "--auditor-a",
            str(out_labels_a),
            "--auditor-b",
            str(out_labels_b),
            "--out",
            str(out_report),
        ]
    )
    report = json.loads(out_report.read_text("utf-8"))
    assert report["n_paired"] == len(PILOT_PROBE_SET)
    assert report["n_only_a"] == 0
    assert report["n_only_b"] == 0
    # One disagreement out of 20 → high κ (pe > 0 because B introduces REDIRECTS).
    assert report["kappa"] > 0.60
    assert report["meets_pilot_gate"] is True
    assert rc == 0


def test_cli_kappa_exit_nonzero_when_gate_fails(tmp_path):
    rows_a = [_mk_labeled("p1", auditor_id="A", label=ResponseLabel.GENTLY_CORRECTS)]
    rows_b = [_mk_labeled("p1", auditor_id="B", label=ResponseLabel.VALIDATES_FALSEHOOD)]
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    write_labeled_responses(rows_a, path_a)
    write_labeled_responses(rows_b, path_b)
    rc = main(
        [
            "kappa",
            "--auditor-a",
            str(path_a),
            "--auditor-b",
            str(path_b),
            "--out",
            str(tmp_path / "r.json"),
        ]
    )
    assert rc == 1


def test_cli_run_probes_rejects_unknown_app(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "run-probes",
                "--app",
                "not_an_app",
                "--mode",
                "stub",
                "--user-id",
                "u",
                "--session-id",
                "s",
                "--out",
                str(tmp_path / "out.jsonl"),
            ]
        )


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])


# ---------------------------------------------------------------------------
# --probe-source agent — the §9.4b κ-gating target.
# ---------------------------------------------------------------------------


def _enrol_test_user(
    pilot_root: Path,
    app_value: str,
    user_id: str,
    *,
    policy_id: str = "S1",
    age_bracket: str = "26-45",
    region: str = "CN",
) -> None:
    """Create an enrolled UserManifest in-tree so the agent-source path has
    something to load. Mirrors the companion/tutor enrol flow directly to
    avoid depending on the CLI."""
    from psychology_base.applications.pilot.layout import (
        PilotApp,
        PilotLayout,
        UserManifest,
        utc_now_iso,
        write_manifest,
    )

    layout = PilotLayout(pilot_root=pilot_root, app=PilotApp(app_value))
    manifest = UserManifest(
        user_id=user_id,
        app=PilotApp(app_value),
        policy_id=policy_id,
        enrolled_at=utc_now_iso(),
        age_bracket=age_bracket,
        region=region,
        language="en",
        guardian_consent_artefact_id=None,
        pilot_root=str(pilot_root.resolve()),
    )
    write_manifest(layout, manifest)


def test_cli_run_probes_agent_source_requires_pilot_root(tmp_path):
    """`--probe-source agent` without `--pilot-root` must error, not fall
    through to a bare-LLM path. This guards the §9.4b κ-gating semantics."""
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "run-probes",
                "--app",
                "emotional_companion",
                "--mode",
                "stub",
                "--user-id",
                "u",
                "--session-id",
                "s0001",
                "--out",
                str(tmp_path / "out.jsonl"),
            ]
        )
    assert "pilot-root" in str(exc_info.value)


def test_cli_run_probes_agent_source_companion_builds_through_agent(tmp_path):
    """Agent-source responses must come from CompanionAgent, not the bare
    stub. We assert the response_text carries the LLM-stub signature
    (`"(stub responder)"`) that `tutor_cli._stub_responder` emits —
    the pipeline's own `_stub_probe_responder` emits a different
    signature (`"I hear how heavy this feels"`), so the two paths are
    distinguishable."""
    _enrol_test_user(tmp_path, "emotional_companion", "c01")

    out_path = tmp_path / "resp.jsonl"
    rc = main(
        [
            "run-probes",
            "--app",
            "emotional_companion",
            "--mode",
            "stub",
            "--probe-source",
            "agent",
            "--pilot-root",
            str(tmp_path),
            "--user-id",
            "c01",
            "--session-id",
            "c01-s0001",
            "--probes",
            "pilot",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    rows = read_probe_responses(out_path)
    assert len(rows) == len(PILOT_PROBE_SET)
    # Any non-crisis probe response must have gone through the agent's
    # LLM-responder path, which carries the stub-responder signature.
    agent_signature_count = sum(
        1 for r in rows if "(stub responder)" in r.companion_response
    )
    assert agent_signature_count >= 1, (
        "At least one probe response should carry the agent's LLM-stub "
        f"signature; got responses: {[r.companion_response[:60] for r in rows]!r}"
    )
    # And no response should carry the pipeline's model-source stub.
    model_signature_count = sum(
        1 for r in rows if "I hear how heavy this feels" in r.companion_response
    )
    assert model_signature_count == 0


def test_cli_run_probes_agent_source_tutor_builds_through_agent(tmp_path):
    _enrol_test_user(tmp_path, "learning_tutor", "t01", age_bracket="18-25")

    out_path = tmp_path / "resp.jsonl"
    rc = main(
        [
            "run-probes",
            "--app",
            "learning_tutor",
            "--mode",
            "stub",
            "--probe-source",
            "agent",
            "--pilot-root",
            str(tmp_path),
            "--user-id",
            "t01",
            "--session-id",
            "t01-s0001",
            "--probes",
            "pilot",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    rows = read_probe_responses(out_path)
    assert len(rows) == len(PILOT_PROBE_SET)
    agent_signature_count = sum(
        1 for r in rows if "(stub responder)" in r.companion_response
    )
    assert agent_signature_count >= 1


def test_cli_run_probes_agent_source_missing_user_errors(tmp_path):
    """Agent-source with a pilot_root that has no enrolled user must fail,
    not silently fall through to the bare-LLM path."""
    (tmp_path / "emotional_companion" / "users").mkdir(parents=True)
    with pytest.raises((FileNotFoundError, SystemExit)):
        main(
            [
                "run-probes",
                "--app",
                "emotional_companion",
                "--mode",
                "stub",
                "--probe-source",
                "agent",
                "--pilot-root",
                str(tmp_path),
                "--user-id",
                "ghost",
                "--session-id",
                "ghost-s0001",
                "--out",
                str(tmp_path / "out.jsonl"),
            ]
        )


def test_cli_run_probes_agent_source_default_is_agent(tmp_path):
    """`--probe-source` is omitted → default must be `agent` (κ-gating target).

    Without `--pilot-root`, this must raise rather than fall through to
    the bare-LLM path.
    """
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "run-probes",
                "--app",
                "emotional_companion",
                "--mode",
                "stub",
                "--user-id",
                "u",
                "--session-id",
                "s",
                "--out",
                str(tmp_path / "o.jsonl"),
            ]
        )
    assert "pilot-root" in str(exc_info.value)
