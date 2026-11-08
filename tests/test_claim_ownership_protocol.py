"""Protocol selection/transport regressions; no model-quality judgments are supplied."""
import copy
import json

import pytest

from psychology_base.eval.benchmarks import application_context as study
from tests.test_application_context_study import real_sources
from tests.test_speaker_grounding import CapturedRequest, RequestCapture


LEGACY_PROMPT_HASHES = {
    'base': '07550e6129eabe967dbee3c033aa05d216885eab06690772b8fc11475d846f53',
    'query': 'd25c946ea0839646acf30cb2969dbd5444037d3540a09a2c450b8d24b7ebc99d',
    'speaker': '6f72785871b92ba55e8338e5d003b3c078d702a266ab63c6de238967e619113f',
}


def prior_protocol(mode):
    if mode == 'fresh':
        return {'budget': {'max_calls': 745, 'max_usd': '1'}}
    return study.revised_grading_protocol(prior_protocol('fresh'),
        current_query_evidence=mode == 'query', explicit_speaker_evidence=mode == 'speaker')


@pytest.mark.parametrize('mode', LEGACY_PROMPT_HASHES)
def test_existing_prompt_versions_keep_their_prechange_hashes(mode):
    assert study.shared.stable_key(prior_protocol(mode)['memory_prompt']) == LEGACY_PROMPT_HASHES[mode]


def test_version_one_selector_is_identical_to_the_existing_speaker_option():
    assert study.revised_grading_protocol({}, speaker_evidence_version=1) == study.revised_grading_protocol(
        {}, explicit_speaker_evidence=True)


@pytest.mark.parametrize('mode', ['fresh', 'base', 'query', 'speaker'])
def test_claim_ownership_upgrades_each_existing_stage_without_mutation_or_budget_change(mode):
    previous = prior_protocol(mode)
    before = copy.deepcopy(previous)
    new = study.revised_grading_protocol(previous, speaker_evidence_version=2)
    expected_prefix = prior_protocol('query')['memory_prompt'] + '\n'
    assert new['memory_prompt'] == expected_prefix + study.MEMORY_SPEAKER_PERSPECTIVE_CLARIFICATION_V2
    assert previous == before and new['budget'] == before['budget']
    assert new['grading_contract_version'] == 2  # Boolean schema version, not speaker-prompt version.
    new['budget']['max_calls'] = 1
    assert previous['budget'] == before['budget']


@pytest.mark.parametrize('selection', [{}, {'explicit_speaker_evidence': True},
    {'current_query_evidence': True}, {'speaker_evidence_version': 2}])
def test_new_speaker_mode_is_idempotent_and_legacy_options_do_not_downgrade_it(selection):
    new = study.revised_grading_protocol({}, speaker_evidence_version=2)
    assert study.revised_grading_protocol(new, **selection) == new


@pytest.mark.parametrize('value', [True, False, 0, -1, 5, 1.0, 2.0, 3.0, '2', {}, []])
def test_unknown_or_ambiguous_speaker_versions_fail_closed(value):
    with pytest.raises(ValueError, match='Speaker evidence version'):
        study.revised_grading_protocol({}, speaker_evidence_version=value)


def test_explicit_downgrade_requires_starting_from_the_original_protocol():
    new = study.revised_grading_protocol({}, speaker_evidence_version=2)
    before = copy.deepcopy(new)
    with pytest.raises(ValueError, match='downgrade'):
        study.revised_grading_protocol(new, speaker_evidence_version=1)
    assert new == before


@pytest.mark.parametrize('change', ['append', 'truncate'])
def test_new_mode_rejects_changed_prompt_suffixes(change):
    new = study.revised_grading_protocol({}, speaker_evidence_version=2)
    new['memory_prompt'] = new['memory_prompt'] + 'changed' if change == 'append' else new['memory_prompt'][:-1]
    with pytest.raises(ValueError, match='prompt changed'):
        study.revised_grading_protocol(new)


def test_new_mode_replaces_only_the_ambiguous_rule_and_reuses_roles_relevance_and_request_shape():
    old = study.MEMORY_SPEAKER_PERSPECTIVE_CLARIFICATION_V1
    new = study.MEMORY_SPEAKER_PERSPECTIVE_CLARIFICATION_V2
    old_start = old.index('For an earlier-memory assertion,')
    old_end = old.index('The rationale should identify')
    assert new.startswith(old[:old_start]) and new.endswith(old[old_end:])
    assert old[old_start:old_end] not in new
    assert new.count('Claim ownership (speaker evidence v2):') == 1
    assert study.MEMORY_FIELDS == ('addresses_request', 'grounded', 'attribution_correct',
        'abstains_appropriately', 'critical_error')


def test_all_frozen_contexts_reach_the_same_judge_without_labels_and_fit_existing_input_limit(real_sources):
    profile = study.local_profile(study.FIXTURES / 'hybrid-screen-v4.json', real_sources[0])
    old = study.revised_grading_protocol(profile, explicit_speaker_evidence=True)
    new = study.revised_grading_protocol(old, speaker_evidence_version=2)
    cases = study.read(study.FIXTURES / 'memory-query-boundaries-v2.json')['cases']
    cases += study.read(study.FIXTURES / 'claim-attribution-boundaries-v1.json')['cases']
    assert len(cases) == 32
    for case in cases:
        requests = []
        for protocol in (old, new):
            capture = RequestCapture()
            with pytest.raises(CapturedRequest):
                study.judge_memory(capture, {'app': case['application']}, case['source_absent'],
                    {'query': case['query'], 'context': case['context']}, case['reply'], case['reference'], protocol=protocol)
            requests.append(capture)
            assert capture.messages[0].content == protocol['memory_prompt']
            assert json.loads(capture.messages[1].content) == dict(APPLICATION=case['application'], QUERY=case['query'],
                VISIBLE_CONTEXT=case['context'], REPLY=case['reply'], SOURCE_ABSENT=case['source_absent'], WITHHELD_REFERENCE=case['reference'])
            size = sum(len(m.content.encode('utf-8')) + 64 for m in capture.messages) + 512
            assert size <= 12000, (case['id'], size)
        assert requests[0].messages[1:] == requests[1].messages[1:]
        assert requests[0].kwargs == requests[1].kwargs
        assert requests[1].kwargs['max_tokens'] == 300 and requests[1].kwargs['temperature'] == 0


def test_new_identity_cannot_inherit_frozen_qualification(real_sources):
    old = study.local_profile(study.FIXTURES / 'hybrid-screen-v4.json', real_sources[0])
    new = study.revised_grading_protocol(old, speaker_evidence_version=2)
    assert study.protocol_sha(new) != study.protocol_sha(old)
    before, after = study.qualification_identity(old), study.qualification_identity(new)
    assert before['tutor_contract_prompt_sha256'] == after['tutor_contract_prompt_sha256']
    assert before['memory_contract_prompt_sha256'] != after['memory_contract_prompt_sha256']
    assert new['budget'] == old['budget']
    with pytest.raises(ValueError, match='pre-model freeze'):
        study.calibration_items(new, real_sources[2])


def test_archived_real_boundary_judgments_reparse_without_repair():
    bank = study.read(study.FIXTURES / 'memory-query-boundaries-v1.json')
    observed = [c['observed_judgment'] for c in bank['cases'] if 'observed_judgment' in c]
    assert len(observed) == 3
    for old in observed:
        assert study.parse_memory(old['raw'], old['finish_reason'], old['proxy'], contract_version=2) == old
