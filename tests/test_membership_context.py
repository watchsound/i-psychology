"""Portable input/reference contracts; no model-quality assertions."""
from copy import deepcopy
import json

import pytest

from psychology_base.applications.measurement.semantic_membership import membership_inputs, reviewed_membership_rows


def sample():
    return dict(id='context-contract', language='en', descriptions=["my statement of Mira's occupation", "my question about Mira"],
        records=[dict(id=i, speaker='user', text=text, source_order=dict(group='a'*32, position=i))
                 for i, text in enumerate(['We are discussing Mira.', 'She is a biologist.',
                                          'What does Mira do?', 'She is a physicist.'])])


def reference(case):
    return dict(id=case['id'], contract_version=3, reviewed=True, reviewer='authored software-contract test',
        judgments=[dict(slot=slot, record=r['id'], label='nonmatch', reason='different_act',
            rationale='Inert annotation for structural validation only.',
            evidence=[dict(start=0, end=len(r['text']), quote=r['text'])], context_evidence=[])
            for slot in range(2) for r in case['records']])


def row_for(case, record, slot=0):
    return next(r for r in membership_inputs(case, format_version=3) if r['record']==record and r['slot']==slot)


def test_context_uses_at_most_two_preceding_source_turns_and_marks_focus():
    row=row_for(sample(),3)
    payload=json.loads(row['pair'][0])
    assert row['context_record_ids']==(1,2)
    assert payload['focus_utterance']['text']=='She is a physicist.'
    assert [r['text'] for r in payload['prior_utterances']]==['She is a biologist.','What does Mira do?']
    assert 'earlier context may be missing' in payload['context_scope']
    assert row_for(sample(),0)['context_record_ids']==()
    assert 'focus utterance-act membership' in row['pair'][1]


def test_recall_reversal_cannot_change_prior_context_or_any_pair():
    case=sample()
    expected={(r['slot'],r['record']):r for r in membership_inputs(case,format_version=3)}
    case['records'].reverse()
    assert {(r['slot'],r['record']):r for r in membership_inputs(case,format_version=3)}==expected


def test_cross_group_future_and_unpositioned_turns_cannot_supply_antecedents():
    case=sample()
    case['records'][0]['source_order']['group']='b'*32
    assert row_for(case,1)['context_record_ids']==()
    case['records'][0]['source_order']=None
    assert row_for(case,1)['context_record_ids']==()
    assert json.loads(row_for(case,0)['pair'][0])['prior_utterances']==[]


def test_context_text_cannot_change_the_authoritative_outer_role():
    case=sample();case['records'][1]['speaker']=None
    row=row_for(case,1)
    assert row['constraint']=='uncertain'
    assert json.loads(row['pair'][0])['focus_utterance']['speaker']=='unknown'
    case['records'][1]['speaker']='tutor'
    assert row_for(case,1)['constraint']=='nonmatch'


@pytest.mark.parametrize('coordinate',[{'group':'bad','position':0}, {'group':'a'*32,'position':True},
    {'group':'a'*32,'position':-1}, {'group':'a'*32,'position':2**63}, {'group':'a'*32,'position':0,'extra':1}])
def test_invalid_source_metadata_is_not_a_context_hint(coordinate):
    case=sample();case['records'][0]['source_order']=coordinate
    with pytest.raises(ValueError,match='source-order'):
        membership_inputs(case,format_version=3)
    assert membership_inputs(case,format_version=2)


def test_conflicting_positions_fail_and_legal_repeated_text_is_preserved():
    case=sample();case['records'][2]['text']=case['records'][0]['text']
    assert row_for(case,3)['context_record_ids']==(1,2)
    case['records'][2]['source_order']['position']=0
    with pytest.raises(ValueError,match='unambiguous source'):
        membership_inputs(case,format_version=3)


def test_large_gapped_positions_use_only_supplied_rows_without_fabricating_prefix():
    case=sample()
    for r in case['records']:r['source_order']['position']+=10**12
    assert row_for(case,3)['context_record_ids']==(1,2)


def test_missing_intervening_turns_cannot_silently_link_a_pronoun_to_a_stale_topic():
    case=sample();case['records'][3]['source_order']['position']=10
    assert row_for(case,3)['context_record_ids']==()
    case=sample();case['records'][0]['source_order']['position']=100
    assert row_for(case,1)['context_record_ids']==()
    assert row_for(case,3)['context_record_ids']==(1,2)


def test_payload_contains_no_ids_positions_source_groups_annotations_or_future_turns():
    case=sample();case['expected_label']='match';case['reviewer']='secret-reference'
    row=row_for(case,1)
    payload=json.loads(row['pair'][0])
    assert set(payload)=={'context_scope','prior_utterances','focus_utterance'}
    assert all(set(r)=={'speaker','text'} for r in payload['prior_utterances']+[payload['focus_utterance']])
    assert 'secret-reference' not in str(row['pair']) and 'She is a physicist.' not in row['pair'][0]
    assert case['descriptions'][0] in row['pair'][1]


def test_reference_requires_exact_focus_evidence_and_visible_context_citations():
    case=sample();ref=reference(case)
    judgment=next(j for j in ref['judgments'] if j['slot']==0 and j['record']==1)
    text=case['records'][0]['text']
    judgment['context_evidence']=[dict(record=0,start=0,end=len(text),quote=text)]
    assert len(reviewed_membership_rows(case,ref))==8


@pytest.mark.parametrize('mutation',[
    lambda r:r['judgments'][1].pop('context_evidence'),
    lambda r:r['judgments'][1].update(context_evidence=None),
    lambda r:r['judgments'][1].update(context_evidence=[None]),
    lambda r:r['judgments'][1].update(context_evidence=[dict(record=2,start=0,end=3,quote='Wha')]),
    lambda r:r['judgments'][1].update(context_evidence=[dict(record=1,start=0,end=3,quote='She')]),
    lambda r:r['judgments'][1].update(context_evidence=[dict(record=True,start=0,end=1,quote='W')]),
    lambda r:r['judgments'][1].update(context_evidence=[dict(record=0,start=True,end=2,quote='We')]),
    lambda r:r['judgments'][1].update(context_evidence=[dict(record=0,start=0,end=2,quote='XX')]),
    lambda r:r.update(contract_version=3.0),
])
def test_invalid_or_undelivered_context_reference_is_refused(mutation):
    case=sample();ref=reference(case);mutation(ref)
    with pytest.raises(ValueError):reviewed_membership_rows(case,ref)


def test_same_group_but_out_of_window_evidence_is_refused():
    case=sample();ref=reference(case);text=case['records'][0]['text']
    ref['judgments'][3]['context_evidence']=[dict(record=0,start=0,end=len(text),quote=text)]
    with pytest.raises(ValueError,match='absent from'):
        reviewed_membership_rows(case,ref)


def test_reference_changes_cannot_choose_context_or_mutate_inputs():
    case=sample();original=deepcopy(case);ref=reference(case)
    before=membership_inputs(case,format_version=3)
    reviewed_membership_rows(case,ref)
    for j in ref['judgments']:j.update(label='uncertain',reason='missing_reference',rationale='Different review.')
    reviewed_membership_rows(case,ref)
    assert before==membership_inputs(case,format_version=3) and case==original
