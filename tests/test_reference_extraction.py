"""Authored software inputs/oracles, distinct from measured model quality."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from psychology_base.applications.measurement.local_query_binding import LocalChatMembershipAssessor, REFERENCE_PROMPT
from psychology_base.applications.measurement.reference_readings import decode_reference_readings
from psychology_base.eval.benchmarks.local_membership_audit import membership_wire_payload
from psychology_base.eval.benchmarks.reference_extraction import build_packet, rows_for, grade, authored_schema_example, compare_analysis
from psychology_base.llm.types import Completion, TaskTier
from tests.test_local_chat_membership import router
from tests.test_reference_possibility import packet as parent_packet

ROOT=Path(__file__).parents[1]


def extraction_router(raw, response=None):
    """Explicit software completion; the wire body never acquires descriptions."""
    value,seen=router()
    def chat(messages,**kwargs):
        seen.append((messages,kwargs))
        completion=Completion(content=json.dumps(raw,ensure_ascii=False),provider='ollama',model_id='authored-test',
            tier=TaskTier.ADVANCED,prompt_tokens=100,completion_tokens=100,finish_reason='stop',
            metadata=dict(local_only=True,done=True,response_model='authored-test',
                model_attestation=dict(digest='a'*64),context_length=8192,seed=0,json_mode=True))
        return response(completion) if response else completion
    value.chat=chat
    return value,seen


def packet():
    parent,refs=parent_packet()
    bank=json.loads((ROOT/'benchmarks/fixtures/application_context/reference-extraction-controls-v1.json').read_text(encoding='utf-8'))
    descriptions=json.loads((ROOT/'benchmarks/fixtures/application_context/reference-possibility-controls-v1.json').read_text(encoding='utf-8'))
    # The inherited portable corpus has authored arithmetic/reflection text,
    # not the ignored public MathDial excerpt. Declare its actual references.
    # The frozen experiment uses the unchanged real-source references in BANK.
    bank['preserved']['original-calculation/with-prior'].update(focus=[],readings=[[]])
    bank['preserved']['original-reflection/with-prior'].update(focus=[['I',0,'deictic']],readings=[['speaker:user']])
    return build_packet(parent,refs,bank,descriptions)


def oracle(inputs,refs):
    observed=[]
    for entry in inputs['schedule']:
        rows=rows_for(inputs['cases'][entry['case']],entry)
        _,analysis=authored_schema_example(refs[entry['case']],json.loads(rows[0]['pair'][0]))
        observed.append(dict(id=entry['id'],analysis=analysis))
    return observed


def test_packet_preserves_all_previous_cases_and_membership_references():
    previous,inherited=parent_packet();inputs,refs=packet()
    assert len(inputs['cases'])==24 and len(inputs['schedule'])==28 and len(refs)==24
    for key,case in previous['cases'].items():
        assert inputs['cases'][key]==case and refs[key]['inherited_membership_reference']==inherited[key]
    assert len([e for e in inputs['schedule'] if e['repeat']])==4
    assert all(r['reviewed'] and r['rationale'] and r['context_sha256'] for r in refs.values())
    for entry in inputs['schedule']:
        rows=rows_for(inputs['cases'][entry['case']],entry)
        payload=membership_wire_payload(rows,'authored-test',assessment_kind='reference-extraction-v1',max_tokens=1024)
        body=json.loads(payload['messages'][1]['content'])
        assert set(body)=={'focus_input'} and len(body['focus_input']['prior_utterances'])<=2
        assert payload['messages'][0]['content']==REFERENCE_PROMPT
    before=deepcopy((inputs,refs));assert grade(inputs['cases'],refs,inputs['schedule'],oracle(inputs,refs))['candidate_screen_passed']
    assert (inputs,refs)==before


def test_authored_chinese_control_encoding_and_exact_implicit_question_are_preserved():
    parent,inherited=parent_packet()
    bank=json.loads((ROOT/'benchmarks/fixtures/application_context/reference-extraction-controls-v1.json').read_text(encoding='utf-8'))
    descriptions=json.loads((ROOT/'benchmarks/fixtures/application_context/reference-possibility-controls-v1.json').read_text(encoding='utf-8'))
    implicit=next(c for c in bank['new_cases'] if c['id']=='reference-implicit-zh')
    assert implicit['prior'][1]=='她们中的某一位从事什么工作？'
    implicit['prior'][1]='??????????????'
    with pytest.raises(ValueError,match='source script'):
        build_packet(parent,inherited,bank,descriptions)


@pytest.mark.parametrize('lang',['en','zh'])
def test_correlation_and_implicit_subjects_are_real_required_readings(lang):
    inputs,refs=packet();observed=oracle(inputs,refs);entries={e['id']:e for e in inputs['schedule']}
    key=f'reference-correlated-{lang}'
    obs=next(o for o in observed if entries[o['id']]['case']==key)
    result=compare_analysis(obs['analysis'],refs[key])
    assert result['correct'] and result['normalized_readings']==[['jo','jo'],['mira','mira']]
    bad=deepcopy(obs['analysis']);a,b=bad['readings'];bad['readings'] += [[a[0],b[1]],[b[0],a[1]]]
    result=compare_analysis(bad,refs[key]);assert not result['correct'] and result['extra_readings']==2
    bad=deepcopy(obs['analysis']);bad['readings']=bad['readings'][:1]
    assert compare_analysis(bad,refs[key])['omitted_readings']==1
    implicit=refs[f'reference-implicit-{lang}'];assert implicit['focus'][0]['kind']=='implicit'
    assert {tuple(row) for row in implicit['readings']}=={('mira',),('jo',)}


def test_equivalent_named_source_anchors_do_not_force_one_exact_alias_span():
    inputs,refs=packet();key='reference-alias-en'
    entry=next(e for e in inputs['schedule'] if e['case']==key)
    focus=json.loads(rows_for(inputs['cases'][key],entry)[0]['pair'][0])
    raw,_=authored_schema_example(refs[key],focus)
    candidate=next(i for i,m in enumerate(raw['mentions']) if m['quote']=='Dr Lane')
    raw['entities'][0]['mentions']=[candidate]
    analysis=decode_reference_readings(json.dumps(raw),focus)
    assert compare_analysis(analysis,refs[key])['correct']


def test_missing_context_and_unrelated_act_share_open_reference_without_fake_labels():
    inputs,refs=packet()
    for key in ('reference-missing-antecedent-en','reference-incompatible-act-en'):
        assert refs[key]['readings']==[[None]] and refs[key]['coverage']=='open'
    assert refs['both-en']['readings']==[['speaker:user',None,None]]
    assert refs['both-en']['coverage']=='open'


def test_unaligned_mentions_do_not_acquire_invented_semantic_coverage_measurements():
    inputs,refs=packet();entry=inputs['schedule'][0];case=inputs['cases'][entry['case']]
    focus=json.loads(rows_for(case,entry)[0]['pair'][0])
    raw,_=authored_schema_example(refs[case['id']],focus)
    index=raw['focus_references'][0]
    raw['mentions'][index]['quote']=focus['focus_utterance']['text']
    analysis=decode_reference_readings(json.dumps(raw),focus)
    result=compare_analysis(analysis,refs[case['id']])
    assert not result['correct'] and not result['readings_comparable']
    assert result['missing_focus_mentions']==result['extra_focus_mentions']==1
    assert result['omitted_readings'] is result['extra_readings'] is result['normalized_readings'] is None
    observed=oracle(inputs,refs);observed[0]['analysis']=analysis
    report=grade(inputs['cases'],refs,inputs['schedule'],observed)
    assert report['summary']['all']['unaligned_reading_conditions']==1
    assert report['summary']['all']['omitted_readings']==0
    assert not report['candidate_screen_passed']


@pytest.mark.parametrize('damage',['missing','failed','wrong_person','wrong_coverage','all_empty'])
def test_gate_cannot_pass_partial_or_semantically_wrong_extractions(damage):
    inputs,refs=packet();observed=oracle(inputs,refs)
    if damage=='missing':observed.pop()
    elif damage=='failed':observed[0]=dict(id=observed[0]['id'],error='Declared software format failure')
    elif damage=='wrong_person':observed[0]['analysis']['readings']=observed[0]['analysis']['readings'][:1]
    elif damage=='wrong_coverage':observed[0]['analysis']['coverage']='unsupported'
    else:
        for o in observed:o['analysis'].update(mentions=[],entities=[],focus_references=[],readings=[[]],coverage='enumerated')
    result=grade(inputs['cases'],refs,inputs['schedule'],observed)
    assert not result['candidate_screen_passed'] and not result['qualified']
    assert result['summary']['all']['planned']==28
    if damage in ('missing','failed'):assert result['summary']['all']['completed']==27


def test_query_independent_adapter_uses_one_pinned_local_call_and_canonical_payload():
    inputs,refs=packet();entry=inputs['schedule'][0];rows=rows_for(inputs['cases'][entry['case']],entry)
    raw,expected=authored_schema_example(refs[entry['case']],json.loads(rows[0]['pair'][0]))
    value,seen=extraction_router(raw)
    assessor=LocalChatMembershipAssessor(value,assessment_kind='reference-extraction-v1',format_version=4,
        max_input=7168,max_tokens=1024)
    assert assessor(rows)==expected and len(seen)==1
    payload=membership_wire_payload(rows,'authored-test',assessment_kind='reference-extraction-v1',max_tokens=1024)
    assert [dict(role=m.role,content=m.content) for m in seen[0][0]]==payload['messages']
    changed=deepcopy(rows)
    for row in changed:row['pair']=(row['pair'][0],row['pair'][1]+' arbitrary changed description')
    assert assessor(changed)==expected
    assert seen[0][0]==seen[1][0]


@pytest.mark.parametrize('damage',['mode','profile','format','grouping','output','rows','extra_focus','mutation','attestation'])
def test_adapter_rejects_unfrozen_configuration_or_incomplete_input(damage):
    inputs,refs=packet();entry=inputs['schedule'][0];rows=rows_for(inputs['cases'][entry['case']],entry)
    value,seen=router();kwargs=dict(assessment_kind='reference-extraction-v1',format_version=4,max_input=7168,max_tokens=1024)
    if damage=='mode':value.execution=replace(value.execution,mode='research')
    elif damage=='profile':kwargs['prompt_profile']='independent-membership-v1'
    elif damage=='format':kwargs['format_version']=3
    elif damage=='grouping':kwargs['descriptions_per_call']=1
    elif damage=='output':kwargs['max_tokens']=1025
    if damage in ('mode','profile','format','grouping','output'):
        with pytest.raises(ValueError):LocalChatMembershipAssessor(value,**kwargs)
        assert not seen;return
    assessor=LocalChatMembershipAssessor(value,**kwargs)
    if damage=='rows':rows.pop()
    elif damage=='extra_focus':
        more=deepcopy(rows)
        for r in more:r['record']+=1
        rows+=more
    elif damage=='mutation':assessor.assessment_kind='membership'
    else:
        raw,_=authored_schema_example(refs[entry['case']],json.loads(rows[0]['pair'][0]))
        def response(c):
            m=deepcopy(c.metadata);m['model_attestation']['digest']='b'*64
            return replace(c,content=json.dumps(raw),metadata=m)
        value.chat=extraction_router(raw,response)[0].chat
    with pytest.raises(ValueError):assessor(rows)
    if damage!='attestation':assert not seen
