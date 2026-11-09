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
}
collect_ignore = [
    name
    for name, sentinel in _PHASE_9_SPRAWL_DEPENDENT_TESTS.items()
    if not (_REPO_ROOT / sentinel).exists()
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
