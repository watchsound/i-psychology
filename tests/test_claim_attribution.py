"""Evidence/fixture controls only: no generated semantic judgments or API calls."""
import copy
import hashlib
import json

from psychology_base.applications.context import render_context_block
from psychology_base.eval.benchmarks import application_context as study
from tests.test_speaker_grounding import fragments
from tests.test_speaker_perspective import (
    test_every_frozen_boundary_reaches_the_existing_judge_without_label_leak as assert_request_transport,
)


DOCS = study.ROOT / 'docs/experiments/phase-9-application-context'
BANK = study.read(study.FIXTURES / 'claim-attribution-boundaries-v1.json')
TRACE = study.read(DOCS / 'claim-attribution-evidence-v1.json')


def test_contract_contrasts_preserve_source_ids_and_both_existing_display_versions():
    assert len({c['id'] for c in BANK['cases']}) == len(BANK['cases']) == 12
    assert BANK['semantic_policy_status'] == 'proposed-not-applied-to-production'
    assert BANK['qualification_eligible'] is BANK['empirical_model_results'] is False
    assert BANK['provider_calls'] == 0
    basics = BANK['reused_basic_controls']
    path = study.ROOT / basics['path']
    assert study.shared.digest(path) == basics['sha256']
    assert set(basics['ids']) <= {c['id'] for c in study.read(path)['cases']}
    for case in BANK['cases']:
        assert case['origin'] == 'authored-contract-contrast'
        assert 'observed_judgment' not in case
        assert set(case['expected_criteria']) == set(study.MEMORY_FIELDS)
        assert not study.memory_consistency_errors(case['expected_criteria'])
        source = fragments(case)  # Reconstructs and verifies content-addressed IDs.
        before = copy.deepcopy(source)
        assert render_context_block(source, assistant_speaker='tutor', format_version=2) == case['context']
        assert [json.loads(line)['text'] for line in render_context_block(source, assistant_speaker='tutor').splitlines()] == [f.content for f in source]
        old = render_context_block(source, assistant_speaker='tutor', format_version=1)
        assert hashlib.sha256(old.encode()).hexdigest() == case['legacy_context_sha256']
        assert source == before


def test_new_contrasts_reuse_existing_request_transport_assertions():
    for case in BANK['cases']:
        # Existing capture raises before a completion; it supplies no model output.
        assert_request_transport(case)


def test_minimal_contrasts_isolate_content_error_and_known_or_unknown_ownership():
    cases = {c['id']: c for c in BANK['cases']}
    supported = cases['supported-user-composite']
    changed = cases['user-composite-wrong-first-stage']
    assert supported['fragments'] == changed['fragments']
    assert supported['query'] == changed['query']
    assert supported['reply'].replace('first stage took 12 days', 'first stage took 13 days') == changed['reply']
    for name in ('other-actor-direct-first-stage', 'tutor-hypothetical-first-stage',
                 'tutor-question-first-stage', 'tutor-paraphrase-first-stage',
                 'unknown-actor-first-stage', 'unrelated-number-same-actor'):
        variant = cases[name]
        assert variant['fragments'][:1] == changed['fragments']
        assert variant['reply'] == changed['reply'] and variant['query'] == changed['query']
    known = cases['other-actor-direct-first-stage']['fragments'][1]
    unknown = cases['unknown-actor-first-stage']['fragments'][1]
    assert {k: v for k, v in known.items() if k not in ('id', 'tags')} == {
        k: v for k, v in unknown.items() if k not in ('id', 'tags')}
    assert known['tags'] == ['speaker:tutor', 'session:source']
    assert unknown['tags'] == ['speaker:source-actor-0', 'session:source']
    assert cases['other-actor-whole-composite']['reply'] == cases['user-endorses-same-composite']['reply']
    assert {c['id'] for c in BANK['cases'] if c['proposed_boundary_interpretation']} == {
        'tutor-hypothetical-first-stage', 'tutor-question-first-stage', 'tutor-paraphrase-first-stage'}


def test_claim_trace_binds_real_source_spans_and_preserves_the_failed_result():
    bank_path = study.ROOT / TRACE['historical_bank_path']
    assert study.shared.digest(bank_path) == TRACE['historical_bank_sha256'] == BANK['historical_bank_sha256']
    assert study.shared.digest(study.ROOT / TRACE['new_bank_path']) == TRACE['new_bank_sha256']
    validation_path = study.ROOT / TRACE['historical_validation_path']
    assert study.shared.digest(validation_path) == TRACE['historical_validation_sha256']
    assert study.shared.digest(DOCS / 'speaker-perspective-diagnostic-analysis-v1.json') == TRACE['historical_analysis_sha256']
    cases = {c['id']: c for c in study.read(bank_path)['cases']}
    old, new = cases[TRACE['legacy_case']], cases[TRACE['named_case']]
    assert all(old[k] == new[k] for k in ('fragments', 'reply', 'query', 'reference', 'source_absent', 'application'))
    source = fragments(old)
    assert render_context_block(source, assistant_speaker='tutor', format_version=1) == old['context']
    assert render_context_block(source, assistant_speaker='tutor', format_version=2) == new['context']
    source_text = {f['id']: f['content'] for f in old['fragments']}
    for anchor in TRACE['anchors']:
        text = source_text[anchor['fragment_id']] if anchor['fragment_id'] else old[anchor['field']]
        assert text[anchor['start']:anchor['end']] == anchor['quote']
        assert hashlib.sha256(anchor['quote'].encode()).hexdigest() == anchor['quote_sha256']
    validation = study.read(validation_path)
    assert validation['diagnostic_gates_passed'] is False
    assert validation['metrics']['first_pass_exact_matches'] == 19
    judgments = {r['case']: r for r in validation['judgments'] if r['repeat'] == 1}
    observations = TRACE['observations']
    for prefix, case in [('legacy', old), ('named', new)]:
        row = judgments[case['id']]
        assert row['dispatch'] == observations[prefix + '_dispatch']
        assert row['observed_criteria'] == observations[prefix + '_criteria']
        assert row['expected_criteria'] == case['expected_criteria']
    assert [k for k in study.MEMORY_FIELDS if observations['legacy_criteria'][k] != observations['named_criteria'][k]] == ['attribution_correct']
