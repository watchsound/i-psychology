"""Authored software oracles exercise real decoders; no model accuracy claims."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from psychology_base.applications.measurement.reference_readings import (
    INTERPRETATION_SCHEMA, interpretation_candidate_view, decode_interpretation_selections,
    interpretation_membership_payload, reduce_interpretation_labels, decode_reference_selections,
)
from tests.test_reference_selections import focus_input, phrase


CASES_PATH = Path(__file__).resolve().parents[1] / 'docs/experiments/phase-9-memory-navigation/interpretation-contract-cases-v1.json'
BANK = json.loads(CASES_PATH.read_text(encoding='utf-8'))
CASES = BANK['cases']


def software_example(case=CASES[0]):
    """Explicit reviewed-answer construction for tests only, never inference."""
    focus = focus_input(case['focus'], prior=() if case['prior'] is None else (case['prior'],))
    from psychology_base.eval.benchmarks.reference_extraction import authored_interpretation_example
    raw,_=authored_interpretation_example(case,focus)
    return focus, raw


def decode(focus, raw):
    return decode_interpretation_selections(json.dumps(raw,ensure_ascii=False),focus)


def labels(focus, raw, rows, *, open_labels=None, descriptions=('first description','second description')):
    payload=interpretation_membership_payload(focus,raw,descriptions)
    response=dict(labels=rows,open_labels=open_labels)
    return reduce_interpretation_labels(json.dumps(response),focus,raw,descriptions,request_identity=payload['request_identity'])


@pytest.mark.parametrize('case',CASES,ids=lambda c:c['id'])
def test_frozen_bilingual_software_examples_preserve_all_source_choices(case):
    focus,raw=software_example(case);before=deepcopy((focus,raw))
    assert decode(focus,raw)==raw
    assert (focus,raw)==before
    assert len(raw['readings'])==len(case['targets'])
    assert all(r['events'][0]['polarity']==case['polarity'] and r['events'][0]['modality']==case['modality'] for r in raw['readings'])
    assert 'descriptions' not in interpretation_candidate_view(focus)
    if case.get('quoted'):
        assert all(r['events'][0]['attribution'].get('type')=='person' for r in raw['readings'])
        assert all(r['bindings'][0]['target'].get('type')=='person' for r in raw['readings'])
        assert all(len(r['events'])==2 for r in raw['readings'])
    ref=BANK['membership_references'][case['id']]
    assert labels(focus,raw,ref['labels'],open_labels=ref['open_labels'],descriptions=ref['descriptions'])==ref['reduced']


def test_same_implicit_clause_can_bind_distinct_subject_and_object_slots():
    focus,raw=software_example(CASES[2]);row=raw['readings'][0]
    row['bindings'][0].update(span=row['events'][0]['span'],kind='implicit')
    assert row['bindings'][0]['span']==row['bindings'][1]['span']
    assert decode(focus,raw)==raw
    row['bindings'].append(deepcopy(row['bindings'][0]))
    with pytest.raises(ValueError,match='Repeated interpretation slot'):decode(focus,raw)


@pytest.mark.parametrize('damage',[
    'schema','identity','extra','empty','too_many_readings','duplicate_reading','reordered_duplicate',
    'boolean_id','missing_id','reversed','cross_turn','prior_binding','unknown_kind','unknown_role',
    'target_type','unknown_speaker','dangling_event','outside_predicate','outside_binding',
    'duplicate_event','extra_event','polarity','modality','object_attribution','frame_drift',
    'slot_drift','kind_drift','enumerated_unknown','open_without_unknown',
])
def test_invalid_v2_outputs_reject_whole_response_without_repair(damage):
    focus,raw=software_example();row=raw['readings'][0];binding=row['bindings'][0];event=row['events'][0]
    if damage=='schema':raw['schema']='reference-source-selections-v1'
    if damage=='identity':raw['source_identity']='0'*64
    if damage=='extra':raw['labels']=['match','nonmatch']
    if damage=='empty':raw['readings']=[]
    if damage=='too_many_readings':raw['readings']*=5
    if damage=='duplicate_reading':raw['readings'].append(deepcopy(row))
    if damage=='reordered_duplicate':
        other=deepcopy(row);other['bindings'].reverse();raw['readings'].append(other)
    if damage=='boolean_id':binding['span']=[True,True]
    if damage=='missing_id':binding['span']=[999,999]
    if damage=='reversed':binding['span']=list(reversed(event['span']))
    if damage=='cross_turn':binding['span']=[0,event['span'][1]]
    if damage=='prior_binding':binding['span']=[0,0]
    if damage=='unknown_kind':binding['kind']='resolved'
    if damage=='unknown_role':binding['role']='owner'
    if damage=='target_type':row['bindings'][1]['target']['type']='fact'
    if damage=='unknown_speaker':binding['target']={'speaker':'tutor'}
    if damage=='dangling_event':binding['event']=event['predicate']
    if damage=='outside_predicate':event['predicate']=[0,0]
    if damage=='outside_binding':event['span']=event['predicate']
    if damage=='duplicate_event':row['events'].append(deepcopy(event))
    if damage=='extra_event':event['confidence']=1
    if damage=='polarity':event['polarity']='false'
    if damage=='modality':event['modality']='true'
    if damage=='object_attribution':event['attribution']=row['bindings'][1]['target']
    if damage=='frame_drift':raw['readings'][1]['events'][0]['predicate']=raw['readings'][1]['events'][0]['span']
    if damage=='slot_drift':raw['readings'][1]['bindings'].pop()
    if damage=='kind_drift':raw['readings'][1]['bindings'][0]['kind']='implicit'
    if damage=='enumerated_unknown':binding['target']=None
    if damage=='open_without_unknown':raw['coverage']='open'
    with pytest.raises(ValueError):decode(focus,raw)


def test_legacy_decoder_does_not_silently_accept_typed_v2_and_does_not_share_mutable_views():
    from psychology_base.applications.measurement.reference_readings import reference_candidate_view
    focus,raw=software_example();legacy=reference_candidate_view(focus)
    assert legacy['schema']=='reference-source-selections-v1'
    view=interpretation_candidate_view(focus)
    assert view['source_identity']==legacy['source_identity']
    view['turns'][0]['text']='changed'
    assert reference_candidate_view(focus)==legacy
    with pytest.raises(ValueError):decode_reference_selections(json.dumps(raw),focus)


def test_open_coverage_cannot_be_promoted_by_agreement_only_within_listed_readings():
    focus,raw=software_example(CASES[4])
    assert labels(focus,raw,[['nonmatch','nonmatch']],open_labels=['uncertain','uncertain'])==['uncertain','uncertain']
    assert labels(focus,raw,[['nonmatch','nonmatch']],open_labels=['nonmatch','uncertain'])==['nonmatch','uncertain']
    with pytest.raises(ValueError):labels(focus,raw,[['nonmatch','nonmatch']])


def test_enumerated_disagreement_is_not_confidence_voting_and_keeps_positive_agreement():
    focus,raw=software_example()
    assert labels(focus,raw,[['match','nonmatch'],['nonmatch','match']])==['uncertain','uncertain']
    assert labels(focus,raw,[['match','nonmatch'],['match','nonmatch']])==['match','nonmatch']
    with pytest.raises(ValueError):labels(focus,raw,[['match','nonmatch']])
    with pytest.raises(ValueError):labels(focus,raw,[['match','nonmatch']]*2,open_labels=['match','match'])


@pytest.mark.parametrize('change',['text','scope','description','target','coverage','missing_labels','extra_key'])
def test_conditional_table_binds_exact_inputs_and_rejects_stale_or_incomplete_outputs(change):
    focus,raw=software_example();descriptions=['first description','second description']
    payload=interpretation_membership_payload(focus,raw,descriptions)
    response=dict(labels=[['match','nonmatch']]*2,open_labels=None)
    if change=='text':focus['focus_utterance']['text']+=' Extra.'
    if change=='scope':focus['context_scope']+=' Changed.'
    if change=='description':descriptions[0]='different description'
    if change=='target':raw['readings'][0]['bindings'][0]['target']={'speaker':'companion'}
    if change=='coverage':raw['coverage']='unsupported'
    if change=='missing_labels':response['labels'].pop()
    if change=='extra_key':response['confidence']=1
    with pytest.raises(ValueError):reduce_interpretation_labels(json.dumps(response),focus,raw,descriptions,
        request_identity=payload['request_identity'])


def test_source_valid_empty_proposal_is_not_a_semantic_success_certificate():
    focus,raw=software_example()
    raw['readings']=[dict(bindings=[],events=[])]
    assert decode(focus,raw)==raw  # Deliberately only a structural statement.
    assert raw['readings'][0]!=software_example()[1]['readings'][0]
    raw['coverage']='unsupported'
    with pytest.raises(ValueError,match='Unsupported'):interpretation_membership_payload(focus,raw,['a','b'])


def test_implicit_argument_cannot_select_only_the_predicate_inside_its_clause():
    focus,raw=software_example()
    for reading in raw['readings']:
        for binding in reading['bindings']:
            if binding['kind']=='implicit':binding['span']=reading['events'][0]['predicate'][:]
    with pytest.raises(ValueError,match='full event clause'):decode(focus,raw)


def test_full_implicit_clause_can_exclude_only_boundary_punctuation():
    focus,raw=software_example()
    full=phrase(focus,1,focus['focus_utterance']['text'])
    for reading in raw['readings']:
        reading['events'][0]['span']=full[:]
        for binding in reading['bindings']:binding['event']=full[:]
    assert decode(focus,raw)==raw


@pytest.mark.parametrize('which',['bindings','events'])
def test_reading_bounds_reject_before_accepting_a_truncated_prefix(which):
    focus,raw=software_example(CASES[2])
    raw['readings'][0][which]*=9
    with pytest.raises(ValueError,match='unbounded'):decode(focus,raw)


def test_total_target_bound_covers_the_whole_joint_reading_set():
    focus=focus_input('I handed it.',prior=('Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel.',))
    clause=phrase(focus,1,'I handed it');predicate=phrase(focus,1,'handed')
    worlds=[]
    for name in ('Alpha','Bravo','Charlie','Delta','Echo','Foxtrot','Golf','Hotel'):
        worlds.append(dict(bindings=[
            dict(span=phrase(focus,1,'I'),kind='deictic',role='agent',event=clause,target={'speaker':'user'}),
            dict(span=phrase(focus,1,'it'),kind='anaphoric',role='patient',event=clause,
                 target=dict(span=phrase(focus,0,name),type='object'))],
            events=[dict(span=clause,predicate=predicate,polarity='affirmed',modality='asserted',attribution={'speaker':'user'})]))
    raw=dict(schema=INTERPRETATION_SCHEMA,source_identity=interpretation_candidate_view(focus)['source_identity'],coverage='enumerated',readings=worlds)
    with pytest.raises(ValueError,match='target bounds'):decode(focus,raw)


def test_event_unknowns_require_open_coverage_even_without_an_unbound_person():
    focus,raw=software_example(CASES[2])
    raw['readings'][0]['events'][0]['modality']='unknown'
    with pytest.raises(ValueError,match='coverage'):decode(focus,raw)
    raw['coverage']='open'
    assert decode(focus,raw)==raw


def test_standalone_nominal_and_unknown_attribution_use_existing_source_rules():
    focus=focus_input('That one.',prior=('The chemistry report is here.',),speaker='unknown')
    raw=dict(schema=INTERPRETATION_SCHEMA,source_identity=interpretation_candidate_view(focus)['source_identity'],coverage='enumerated',
        readings=[dict(events=[],bindings=[dict(span=phrase(focus,1,'That one'),kind='nominal',role='referent',event=None,
            target=dict(span=phrase(focus,0,'chemistry report'),type='object'))])])
    assert decode(focus,raw)==raw
    raw['readings'][0]['bindings'][0]['target']={'speaker':'user'}
    with pytest.raises(ValueError,match='known speaker'):decode(focus,raw)


def test_duplicate_keys_and_mutable_conditional_inputs_cannot_hide_changes():
    focus,raw=software_example()
    content=json.dumps(raw)
    with pytest.raises(ValueError):decode_interpretation_selections(content[:-1]+',"coverage":"open"}',focus)
    descriptions=['a','b'];before=deepcopy((focus,raw,descriptions))
    payload=interpretation_membership_payload(focus,raw,descriptions)
    payload['source']['turns'][0]['text']='changed'
    payload['interpretation']['coverage']='unsupported';payload['descriptions'][0]='changed'
    assert (focus,raw,descriptions)==before


def test_code_attaches_source_identity_without_asking_the_model_to_copy_it():
    from psychology_base.applications.measurement.reference_readings import decode_interpretation_response
    focus,raw=software_example()
    wire=dict(coverage=raw['coverage'],readings=raw['readings'])
    assert decode_interpretation_response(json.dumps(wire),focus)==raw
    with pytest.raises(ValueError,match='semantic choices'):
        decode_interpretation_response(json.dumps(raw),focus)


def test_conditional_identity_is_owned_by_the_caller_and_not_a_model_echo():
    focus,raw=software_example();descriptions=['a','b']
    payload=interpretation_membership_payload(focus,raw,descriptions)
    wire=dict(labels=[['match','nonmatch']]*2,open_labels=None)
    assert reduce_interpretation_labels(json.dumps(wire),focus,raw,descriptions,
        request_identity=payload['request_identity'])==['match','nonmatch']
    with pytest.raises(ValueError):reduce_interpretation_labels(json.dumps(wire),focus,raw,descriptions,request_identity='0'*64)
    wire['request_identity']=payload['request_identity']
    with pytest.raises(ValueError):reduce_interpretation_labels(json.dumps(wire),focus,raw,descriptions,
        request_identity=payload['request_identity'])
