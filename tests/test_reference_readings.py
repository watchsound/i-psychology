"""Pure source/reading contracts; no model predictions or provider calls."""
from copy import deepcopy
from itertools import product
import json

import pytest

from psychology_base.applications.measurement.reference_readings import decode_reference_readings, reduce_reading_labels


def example():
    focus=dict(context_scope='Only supplied contiguous prior turns.',
        prior_utterances=[dict(speaker='companion',text='Mira and Jo are women.')],
        focus_utterance=dict(speaker='user',text='She is a nurse.'))
    raw=dict(mentions=[dict(turn=0,quote='Mira',occurrence=0,kind='named'),
        dict(turn=0,quote='Jo',occurrence=0,kind='named'),dict(turn=1,quote='She',occurrence=0,kind='anaphoric')],
        entities=[dict(mentions=[0]),dict(mentions=[1])],focus_references=[2],readings=[[0],[1]],coverage='enumerated')
    return focus,raw


def test_exact_source_offsets_and_all_joint_readings_are_preserved():
    focus,raw=example();before=deepcopy((focus,raw))
    result=decode_reference_readings(json.dumps(raw),focus)
    assert result['readings']==[[0],[1]] and (focus,raw)==before
    assert result['mentions'][1]['start']==9 and result['mentions'][1]['end']==11
    for mention in result['mentions']:
        turns=focus['prior_utterances']+[focus['focus_utterance']]
        assert turns[mention['turn']]['text'][mention['start']:mention['end']]==mention['quote']


def test_chinese_unicode_and_repeated_quote_use_code_derived_offsets():
    focus=dict(context_scope='Supplied.',prior_utterances=[],focus_utterance=dict(speaker='user',text='我说：“我没有取消。”'))
    raw=dict(mentions=[dict(turn=0,quote='我',occurrence=i,kind='deictic') for i in (0,1)],
        entities=[dict(speaker='user',turn=0)],focus_references=[0,1],readings=[[0,None]],coverage='open')
    result=decode_reference_readings(json.dumps(raw,ensure_ascii=False),focus)
    assert [(m['start'],m['end']) for m in result['mentions']]==[(0,1),(4,5)]
    assert result['readings']==[[0,None]]


@pytest.mark.parametrize('damage',[
    'extra_key','missing_key','empty_readings','too_many_readings','too_many_mentions','too_many_entities',
    'boolean_turn','boolean_occurrence','absent_quote','future_turn','negative_turn','bad_kind','duplicate_mention',
    'negative_occurrence','duplicate_key','boolean_entity_member','duplicate_entity_member','unknown_entity_anchor',
    'pronoun_entity','duplicate_entity','bad_speaker','unknown_speaker','wrong_speaker_turn','boolean_speaker_turn',
    'boolean_reference','prior_reference','omitted_focus','duplicate_reference','duplicate_reading','short_reading',
    'boolean_target','dangling_target','open_without_null','closed_with_null','bad_coverage','changed_context',
])
def test_invalid_proposals_and_context_fail_without_repairs(damage):
    focus,raw=example()
    if damage=='extra_key':raw['complete']=True
    elif damage=='missing_key':raw.pop('coverage')
    elif damage=='empty_readings':raw['readings']=[]
    elif damage=='too_many_readings':raw['readings']=[[0]]*9
    elif damage=='too_many_mentions':raw['mentions']*=6
    elif damage=='too_many_entities':raw['entities']*=5
    elif damage=='boolean_turn':raw['mentions'][0]['turn']=False
    elif damage=='boolean_occurrence':raw['mentions'][0]['occurrence']=False
    elif damage=='absent_quote':raw['mentions'][0]['quote']='Invented person'
    elif damage=='future_turn':raw['mentions'][0]['turn']=2
    elif damage=='negative_turn':raw['mentions'][0]['turn']=-1
    elif damage=='bad_kind':raw['mentions'][0]['kind']='certain'
    elif damage=='duplicate_mention':raw['mentions'].append(raw['mentions'][0])
    elif damage=='negative_occurrence':raw['mentions'][0]['occurrence']=-1
    elif damage=='boolean_entity_member':raw['entities'][0]['mentions']=[False]
    elif damage=='duplicate_entity_member':raw['entities'][0]['mentions']=[0,0]
    elif damage=='unknown_entity_anchor':raw['entities'][0]['mentions']=[10]
    elif damage=='pronoun_entity':raw['entities'][0]['mentions']=[2]
    elif damage=='duplicate_entity':raw['entities'].append(dict(mentions=[0]))
    elif damage=='bad_speaker':raw['entities'][0]=dict(speaker='someone',turn=0)
    elif damage=='unknown_speaker':focus['prior_utterances'][0]['speaker']='unknown';raw['entities'][0]=dict(speaker='unknown',turn=0)
    elif damage=='wrong_speaker_turn':raw['entities'][0]=dict(speaker='user',turn=0)
    elif damage=='boolean_speaker_turn':raw['entities'][0]=dict(speaker='companion',turn=False)
    elif damage=='boolean_reference':raw['focus_references']=[True]
    elif damage=='prior_reference':raw['focus_references']=[0]
    elif damage=='omitted_focus':raw['focus_references']=[];raw['readings']=[[]]
    elif damage=='duplicate_reference':raw['focus_references']=[2,2];raw['readings']=[[0,0]]
    elif damage=='duplicate_reading':raw['readings']=[[0],[0]]
    elif damage=='short_reading':raw['readings']=[[]]
    elif damage=='boolean_target':raw['readings']=[[False]]
    elif damage=='dangling_target':raw['readings']=[[2]]
    elif damage=='open_without_null':raw['coverage']='open'
    elif damage=='closed_with_null':raw['readings']=[[None]]
    elif damage=='bad_coverage':raw['coverage']=True
    elif damage=='changed_context':focus['focus_utterance']['role']='user'
    content=json.dumps(raw)
    if damage=='duplicate_key':content=content[:-1]+',"coverage":"enumerated"}'
    with pytest.raises(ValueError):decode_reference_readings(content,focus)


def test_named_focus_cannot_be_bound_to_another_anchored_entity():
    focus,raw=example();focus['focus_utterance']['text']='Jo is a nurse.'
    raw['mentions'][2]=dict(turn=1,quote='Jo',occurrence=0,kind='named')
    raw['entities'][1]['mentions'].append(2)
    with pytest.raises(ValueError):decode_reference_readings(json.dumps(raw),focus)


def test_empty_reference_assignment_is_not_an_empty_set_of_readings():
    focus,raw=example();focus['focus_utterance']['text']='The cover is blue.'
    raw.update(mentions=[],entities=[],focus_references=[],readings=[[]])
    assert decode_reference_readings(json.dumps(raw),focus)['readings']==[[]]


def test_implicit_reference_is_anchored_to_real_clause_and_stays_unresolved():
    focus,raw=example();focus['prior_utterances']=[];focus['focus_utterance']['text']='在医院当护士。'
    raw.update(mentions=[dict(turn=0,quote='在医院当护士。',occurrence=0,kind='implicit')],
        entities=[],focus_references=[0],readings=[[None]],coverage='open')
    assert decode_reference_readings(json.dumps(raw),focus)['mentions'][0]['end']==7


@pytest.mark.parametrize('count',[1,2,3,4])
def test_reduction_is_order_and_duplicate_invariant_and_never_invents_certainty(count):
    vectors=list(product(('match','nonmatch','uncertain'),repeat=2))
    for rows in product(vectors,repeat=count):
        matrix=[list(row) for row in rows]
        result=reduce_reading_labels(matrix,expected_readings=count)
        assert result==reduce_reading_labels(matrix[::-1],expected_readings=count)
        assert result==reduce_reading_labels(matrix+[matrix[0]],expected_readings=count+1)
        for slot,label in enumerate(result):
            if label!='uncertain':assert all(row[slot]==label for row in matrix)


def test_omitted_world_and_reference_independent_act_counterexamples():
    assert reduce_reading_labels([['nonmatch','match']],expected_readings=1)==['nonmatch','match']
    assert reduce_reading_labels([['nonmatch','match'],['match','nonmatch']],expected_readings=2)==['uncertain','uncertain']
    assert reduce_reading_labels([['nonmatch','nonmatch']],expected_readings=1)==['nonmatch','nonmatch']
    assert reduce_reading_labels([['match','match']],expected_readings=1)==['match','match']
    with pytest.raises(ValueError):reduce_reading_labels([],expected_readings=0)
    with pytest.raises(ValueError):reduce_reading_labels([['match','nonmatch']],expected_readings=2)
