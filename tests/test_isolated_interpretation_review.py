"""Review isolation/commitment and strict comparison checks, not semantic gold."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from psychology_base.applications.measurement.reference_readings import (
    decode_interpretation_response, interpretation_candidate_view,
)

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'docs/experiments/phase-9-memory-navigation/compare-isolated-interpretation-reviews-v1.py'
spec=importlib.util.spec_from_file_location('isolated_review_comparison_test_owner',PATH)
owner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


def example():
    focus=dict(context_scope='Only this supplied turn.',prior_utterances=[],
               focus_utterance=dict(speaker='user',text='I waved.'))
    raw=dict(coverage='enumerated',readings=[dict(bindings=[dict(span=[0,0],kind='deictic',role='agent',
        event=[0,1],target=dict(speaker='user'))],events=[dict(span=[0,1],predicate=[1,1],
        polarity='affirmed',modality='asserted',attribution=dict(speaker='user'))])])
    return focus,raw,decode_interpretation_response(json.dumps(raw),focus)


def test_exact_comparison_keeps_semantic_changes_visible():
    focus,raw,analysis=example()
    same=owner.compare_pair(analysis,deepcopy(analysis),focus)
    assert same['correct'] and same['exact_supported_agreement']
    assert same['category']=='exact_supported_agreement'
    changed=deepcopy(raw);changed['readings'][0]['events'][0]['polarity']='negated'
    changed=decode_interpretation_response(json.dumps(changed),focus)
    result=owner.compare_pair(analysis,changed,focus)
    assert not result['correct'] and result['category']=='disagreement'


def test_shared_abstention_is_not_a_concrete_graph_agreement():
    focus,_,_=example()
    raw=dict(coverage='unsupported',readings=[dict(bindings=[],events=[])])
    analysis=decode_interpretation_response(json.dumps(raw),focus)
    result=owner.compare_pair(analysis,analysis,focus)
    assert result['correct'] and result['both_unsupported']
    assert not result['exact_supported_agreement'] and result['category']=='shared_unsupported'


def test_multi_reading_person_or_unknown_targets_do_not_claim_finite_object_coverage():
    def reading(target):
        return dict(bindings=[dict(span=[4,4],event=[3,4],kind='anaphoric',role='patient',target=target)])
    analysis=dict(coverage='enumerated',readings=[reading(dict(span=[0,0],type='person')),
                                                reading(dict(span=[1,1],type='person'))])
    assert not owner.finite_object_alternatives(analysis)
    for row in analysis['readings']:
        row['bindings'][0]['target']['type']='object'
    assert owner.finite_object_alternatives(analysis)
    analysis['coverage']='open'
    assert not owner.finite_object_alternatives(analysis)


def test_both_seals_are_required_before_either_submission_is_read(tmp_path,monkeypatch):
    monkeypatch.setattr(owner,'CAPSULE',tmp_path)
    directory=tmp_path/'reviewer-B';directory.mkdir()
    (directory/'submission-verification.json').write_text('not yet safe to read',encoding='utf-8')
    with patch.object(owner,'read',side_effect=AssertionError('Comparison opened a review too soon')):
        with pytest.raises(FileNotFoundError,match='Both'):
            owner.sealed_submissions()


def test_changed_sealed_bytes_are_rejected_before_answers_are_validated(tmp_path,monkeypatch):
    monkeypatch.setattr(owner,'CAPSULE',tmp_path)
    for reviewer in ['reviewer-B','reviewer-C']:
        directory=tmp_path/reviewer;directory.mkdir()
        (directory/'submission.json').write_text('{}',encoding='utf-8')
        (directory/'submission-verification.json').write_text(json.dumps(dict(passed=True,submission_sha256='0'*64)),encoding='utf-8')
    with patch.object(owner.validator,'validate',side_effect=AssertionError('Tampered review parsed')):
        with pytest.raises(ValueError,match='bytes changed'):
            owner.sealed_submissions()


def test_access_attestation_rejects_prior_project_memory():
    submission=dict(access_attestation=dict(prior_answers_seen=False,peer_submission_seen=False,
        files_read=['docs/project-memory.md']))
    with pytest.raises(ValueError,match='unapproved'):
        owner.attestation(submission,'reviewer-B')


def test_source_only_attestation_preserves_shared_family_limitation():
    source=owner.CAPSULE/'reviewer-B/input.json'
    submission=dict(access_attestation=dict(prior_answers_seen=False,peer_submission_seen=False,
        files_read=[str(source)]))
    result=owner.attestation(submission,'reviewer-B')
    assert result['source_only_access_self_attested'] and result['shared_model_family']
    assert not result['human_review']


def test_own_working_drafts_are_allowed_but_peer_data_remains_forbidden():
    own=owner.CAPSULE/'reviewer-B/build_submission.py'
    submission=dict(access_attestation=dict(prior_answers_seen=False,peer_submission_seen=False,
        files_read=[str(own)]))
    assert owner.attestation(submission,'reviewer-B')['source_only_access_self_attested']
    submission['access_attestation']['files_read']=[str(owner.CAPSULE/'reviewer-C/input.json')]
    with pytest.raises(ValueError,match='unapproved'):
        owner.attestation(submission,'reviewer-B')


def review_fixture(tmp_path):
    focus,raw,_=example()
    source=interpretation_candidate_view(focus)
    task=dict(case='software-only',record=0,source=dict(role='user',content=json.dumps(source)),
        input_sha256='a'*64,source_occurrence_sha256='b'*64,protocol_identity='c'*64,annotation_policy_sha256='d'*64)
    row={k:task[k] for k in ['case','record','input_sha256','source_occurrence_sha256','protocol_identity','annotation_policy_sha256']}
    row.update(meaning='The user reports waving.',references='I is the user.',uncertainty='No recipient supplied.',
        coverage_basis='One explicit act and actor.',observed_condition='no_object_reference',
        observed_realization='explicit_subject',confidence='high',flags=[],evidence=[dict(turn=0,quote='I waved.')],answer=raw)
    packet=dict(reviewer_id='reviewer-B',tasks=[task])
    submission=dict(schema='interpretation-isolated-ai-review-submission-v1',reviewer_id='reviewer-B',reviewer_type='AI',
        prior_answers_seen=False,peer_submission_seen=False,human_approved=False,reviews=[row])
    return packet,submission,tmp_path/'input.json',tmp_path/'submission.json'


def test_source_validator_accepts_a_pinned_complete_software_submission(tmp_path):
    packet,submission,input_path,submission_path=review_fixture(tmp_path)
    input_path.write_text(json.dumps(packet),encoding='utf-8')
    submission_path.write_text(json.dumps(submission),encoding='utf-8')
    result=owner.validator.validate(input_path,submission_path)
    assert result['passed'] and result['reviewed_cases']==result['decoder_accepted']==1
    assert not result['independently_verified_semantics']


@pytest.mark.parametrize('damage',['stale_pin','invented_quote','missing_case','duplicate_case','human_approval','prior_answers_seen','invalid_coverage'])
def test_review_validation_rejects_stale_incomplete_or_misattributed_submissions(tmp_path,damage):
    packet,submission,input_path,submission_path=review_fixture(tmp_path)
    row=submission['reviews'][0]
    if damage=='stale_pin':row['annotation_policy_sha256']='f'*64
    elif damage=='invented_quote':row['evidence'][0]['quote']='I sent a postcard.'
    elif damage=='missing_case':submission['reviews']=[]
    elif damage=='duplicate_case':submission['reviews'].append(deepcopy(row))
    elif damage=='human_approval':submission['human_approved']=True
    elif damage=='prior_answers_seen':submission['prior_answers_seen']=True
    elif damage=='invalid_coverage':row['answer']['coverage']='complete'
    input_path.write_text(json.dumps(packet),encoding='utf-8')
    submission_path.write_text(json.dumps(submission),encoding='utf-8')
    with pytest.raises(ValueError):
        owner.validator.validate(input_path,submission_path)
