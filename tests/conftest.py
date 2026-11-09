"""Shared text controls; lifecycle and assertions stay in their existing suites."""

import json
from pathlib import Path

import pytest


# Tracked tests whose module-top-level code loads research artifacts under
# `docs/experiments/phase-9-*` paths that are gitignored (research-sprawl bucket
# under `.gitignore:46 docs/experiments/`). On a fresh clone those paths do not
# exist and the tests raise `FileNotFoundError` at collection time. Each entry
# below names a sentinel path the test's top-level code reads; when the sentinel
# is absent, the test is added to `collect_ignore` so a fresh clone's suite
# stays green. On a dev checkout with the sprawl present, nothing is ignored
# and these tests collect and run as before.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PHASE_9_SPRAWL_DEPENDENT_TESTS = {
    "test_isolated_interpretation_review.py": "docs/experiments/phase-9-memory-navigation/compare-isolated-interpretation-reviews-v1.py",
    "test_bilingual_authored_review.py": "docs/experiments/phase-9-memory-navigation/review-bilingual-authored-contrasts-v1.py",
    "test_bilingual_public_review.py": "docs/experiments/phase-9-memory-navigation/review-bilingual-public-windows-v1.py",
    "test_claim_attribution.py": "docs/experiments/phase-9-application-context/claim-attribution-evidence-v1.json",
    "test_development_evidence_audit.py": "docs/experiments/phase-9-memory-navigation/audit-development-evidence-v1.py",
    "test_evidence_boundary_amendment.py": "docs/experiments/phase-9-memory-navigation/apply-evidence-boundary-amendment-v1.py",
    "test_evidence_boundary_independent_review.py": "docs/experiments/phase-9-memory-navigation/compare-evidence-boundary-independent-review-v1.py",
    "test_independent_membership.py": "docs/experiments/phase-9-memory-navigation/run-independent-membership-v1.py",
    "test_interpretation_contract.py": "docs/experiments/phase-9-memory-navigation/interpretation-contract-cases-v1.json",
    "test_interpretation_local_integration.py": "docs/experiments/phase-9-memory-navigation/interpretation-contract-cases-v1.json",
    "test_interpretation_supervision.py": "docs/experiments/phase-9-memory-navigation/prepare-interpretation-supervision-v1.py",
    "test_interpretation_validation_preparation.py": "docs/experiments/phase-9-memory-navigation/prepare-interpretation-validation-v1.py",
    "test_local_memory_admission.py": "docs/experiments/phase-9-memory-navigation/run-membership-local-sentinel-v2.py",
    "test_membership_independent_review.py": "docs/experiments/phase-9-memory-navigation/compare-paired-context-reviews-v1.py",
    "test_native_qualification_comparison.py": "docs/experiments/phase-9-memory-navigation/compare-native-qualification-reviews-v1.py",
    "test_nonliteral_order_controls.py": "docs/experiments/phase-9-memory-navigation/prepare-nonliteral-review-v1.py",
    "test_nonliteral_review.py": "docs/experiments/phase-9-memory-navigation/prepare-nonliteral-review-v1.py",
    "test_paired_context_membership.py": "docs/experiments/phase-9-memory-navigation/paired-context-membership-v1.py",
    "test_pointwise_membership_packet.py": "docs/experiments/phase-9-memory-navigation/run-pointwise-membership-v1.py",
    "test_query_binding.py": "docs/experiments/phase-9-memory-navigation/query-binding-development-freeze-v1.json",
    "test_reference_extraction.py": "docs/experiments/phase-9-memory-navigation/run-reference-possibility-v1.py",
    "test_reference_interface.py": "docs/experiments/phase-9-memory-navigation/run-reference-possibility-v1.py",
    "test_reference_possibility.py": "docs/experiments/phase-9-memory-navigation/run-reference-possibility-v1.py",
    "test_reference_selections.py": "docs/experiments/phase-9-memory-navigation/run-reference-possibility-v1.py",
    "test_reference_selections_integration.py": "docs/experiments/phase-9-memory-navigation/run-reference-possibility-v1.py",
    "test_reflection_context.py": "docs/experiments/phase-9-memory-navigation/run-reflection-context-v1.py",
    "test_semantic_membership_experiment.py": "docs/experiments/phase-9-memory-navigation/run-semantic-membership-v1.py",
    "test_semantic_reuse.py": "docs/experiments/phase-9-memory-navigation/interpretation-contract-cases-v1.json",
    # Tracked tests that import gitignored `benchmarks/scripts/phase9_*` modules
    # (see `.gitignore:101-103`). Same sentinel pattern: ignore when absent.
    "test_chinese_retrieval_study.py": "benchmarks/scripts/phase9_chinese_retrieval_run.py",
    "test_recall_distinct_evidence.py": "benchmarks/scripts/phase9_recall_diagnosis.py",
    # Tracked tests whose test-function bodies read gitignored research
    # artifacts or import gitignored sibling tests. They collect fine but fail
    # at runtime on a fresh clone. Same sentinel pattern.
    "test_context_allocation_integration.py": "docs/experiments/phase-9-memory-navigation/run-episode-context-v1.py",
    "test_interpretation_packet.py": "docs/experiments/phase-9-memory-navigation/run-interpretation-source-v2.py",
    "test_native_local_backend.py": "docs/experiments/phase-9-memory-navigation/run-membership-native-sentinel-v1.py",
    "test_prompt_interference.py": "docs/experiments/phase-9-memory-navigation/run-prompt-interference-v1.py",
    "test_prompt_interference_audit.py": "docs/experiments/phase-9-memory-navigation/run-prompt-interference-v1.py",
    "test_recall_carryover.py": "docs/experiments/phase-9-recall-carryover/evaluation-v2.json",
    "test_request_headroom.py": "docs/experiments/phase-9-memory-navigation/request-headroom-development-freeze-v1.json",
    "test_runtime_context_selection.py": "docs/experiments/phase-9-runtime-context-selection/before.json",
    "test_tongyi_interpretation_screen.py": "docs/experiments/phase-9-memory-navigation/run-tongyi-interpretation-screen-v1.py",
    # Runtime body reads `.data/` payloads provisioned by `benchmarks/scripts/
    # provision_*.py` (gitignored per `.gitignore:38 .data/`). Fresh clones
    # don't have `.data/` until a provision step runs.
    "test_runtime_decision_separation.py": ".data/application_context/runtime-report-residuals-v1/hosted-run/report.json",
    # Tracked tests that import gitignored sibling test modules. The sentinel
    # is the sibling file itself; its absence means the import will fail.
    "test_compact_membership_format4.py": "tests/test_reporting_membership_session.py",
    "test_focused_context_review.py": "tests/test_cue_context_study.py",
    "test_focused_context_study.py": "tests/test_cue_context_study.py",
    "test_native_membership_batch.py": "tests/test_native_semantics_packet.py",
    "test_offline_local_deadlines.py": "tests/test_membership_local_sentinel.py",
}
collect_ignore = [
    name
    for name, sentinel in _PHASE_9_SPRAWL_DEPENDENT_TESTS.items()
    if not (_REPO_ROOT / sentinel).exists()
]

# Unconditional collect_ignore: ported from the gitignored root `conftest.py`
# because pytest's `collect_ignore = ["tests/foo.py"]` is a cross-directory
# no-op — only `collect_ignore_glob` works across directories (verified via
# minimal repro 2026-11-09). Same-dir `collect_ignore` DOES work, so moving
# these filename-only entries into tests/conftest.py makes the ignore take
# effect regardless of root conftest.py presence. All entries are gitignored
# under .gitignore (buckets at :66+ one-off names); on fresh clone they're
# absent so this list is harmless, on dev checkout they're ignored so
# optional-dependency import failures and known-drift tests don't fail the
# suite. Mirrors the semantics the root conftest.py always intended.
collect_ignore += [
    # Known-drift tracked test (listed in root conftest.py:66 as unconditional).
    # Also appears in `_PHASE_9_SPRAWL_DEPENDENT_TESTS` above as a fresh-clone
    # sentinel; the unconditional entry here covers dev-checkout too.
    "test_isolated_interpretation_review.py",
    # One-off gitignored test files (ported from root conftest.py:69-103).
    "test_annotated_dataset_reuse.py",
    "test_chinese_zero_seed_recall.py",
    "test_cue_context_study.py",
    "test_evidence_utility.py",
    "test_evidence_utility_integration.py",
    "test_focused_context.py",
    "test_independent_membership_continuation.py",
    "test_interpretation_policy_v3.py",
    "test_interpretation_policy_v4.py",
    "test_local_sentinel_resumption.py",
    "test_membership_checkpoint_finalization.py",
    "test_membership_contract_boundaries.py",
    "test_membership_evidence_boundary_adjudication.py",
    "test_membership_ordinal_data.py",
    "test_membership_resume.py",
    "test_membership_teacher_capability.py",
    "test_memory_navigation.py",
    "test_metadata_publication.py",
    "test_native_english_review.py",
    "test_native_membership_preparation.py",
    "test_native_membership_sources.py",
    "test_native_qualification_coverage_comparison.py",
    "test_native_qualification_coverage_sources.py",
    "test_native_qualification_extension.py",
    "test_native_semantics_packet.py",
    "test_natural_interpretation_validation.py",
    "test_original_32_ledger.py",
    "test_pointwise_membership.py",
    "test_pointwise_membership_audit.py",
    "test_runtime_ambiguity.py",
    "test_source_blocks.py",
    "test_source_support.py",
    "test_turn_association_study.py",
]


@pytest.fixture(scope="session")
def chinese_cases():
    path = Path(__file__).resolve().parents[1] / "benchmarks/fixtures/chinese_tokenization/cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def unavailable_jieba(monkeypatch):
    """Negative-path injection; positive cases always use the installed backend."""
    from dataclasses import replace
    import psychology_base.i18n as i18n
    import psychology_base.llm.config as config

    cfg = replace(config.load_config(), i18n=config.I18nConfig(tokenizer="jieba"))
    monkeypatch.setattr(config, "load_config", lambda: cfg)
    monkeypatch.setattr(i18n, "_load_jieba", lambda: (None, {}, "not installed"))
