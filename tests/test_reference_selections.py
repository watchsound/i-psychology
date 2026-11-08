"""Pure-code oracles: no provider calls or claims of model interpretation quality."""
from copy import deepcopy
import json

import pytest

from psychology_base.applications.measurement.reference_readings import (
    SCHEMA, reference_source_units, reference_candidate_view, reference_selection_schema,
    decode_reference_selections, reference_turns,
)
from psychology_base.applications.reporting_order import _phrase_targets
from psychology_base.eval.benchmarks.reference_extraction import rows_for, compare_analysis, authored_selection_example
from tests.test_reference_extraction import packet


def focus_input(text, prior=(), speaker='user'):
    return dict(context_scope='Authored software source window',
        prior_utterances=[dict(text=t,speaker='companion') for t in prior],
        focus_utterance=dict(text=text,speaker=speaker))


def select(focus, turn, start, end):
    units = reference_source_units(focus)
    return [next(i for i,u in enumerate(units) if u.record_id==turn and u.start==start),
            next(i for i,u in enumerate(units) if u.record_id==turn and u.end==end)]


def phrase(focus, turn, quote, occurrence=0):
    span = _phrase_targets([dict(reference_turns(focus)[turn],id=turn)],quote)[occurrence]
    return select(focus,turn,span.start,span.end)


def wire_reference(span, target=None, kind='anaphoric'):
    return dict(span=span,kind=kind,target=target)


def decode(focus, readings, coverage='enumerated'):
    return decode_reference_selections(json.dumps(dict(coverage=coverage,readings=readings),ensure_ascii=False),focus)


def test_existing_28_authored_graph_oracles_round_trip_through_one_canonical_decoder():
    inputs, refs = packet()
    count = 0
    for entry in inputs['schedule']:
        focus = json.loads(rows_for(inputs['cases'][entry['case']],entry)[0]['pair'][0])
        before = deepcopy(focus)
        view = reference_candidate_view(focus)
        raw,oracle = authored_selection_example(refs[entry['case']],focus)
        result = decode_reference_selections(json.dumps(raw,ensure_ascii=False),focus)
        assert result==oracle and result['schema']==SCHEMA and compare_analysis(result,refs[entry['case']])['correct']
        assert reference_candidate_view(focus)==view and focus==before
        count += 1
    assert count==28


@pytest.mark.parametrize('extra',[False,True])
def test_reviewed_software_oracle_never_truncates_misaligned_expected_readings(extra):
    inputs,refs=packet();key='reference-correlated-en'
    entry=next(e for e in inputs['schedule'] if e['case']==key)
    focus=json.loads(rows_for(inputs['cases'][key],entry)[0]['pair'][0])
    reference=deepcopy(refs[key])
    if extra:reference['readings'][0].append(reference['readings'][0][0])
    else:reference['readings'][0].pop()
    with pytest.raises(ValueError):authored_selection_example(reference,focus)


def test_view_keeps_exact_unicode_roles_text_and_every_nonwhitespace_source_position():
    focus = focus_input('“我”说：Jose\u0301，李\U00020000！\n',prior=('Dr Lane — Mira.',))
    units = reference_source_units(focus); view = reference_candidate_view(focus)
    for turn,record in enumerate(reference_turns(focus)):
        actual = [u for u in units if u.record_id==turn]
        assert all(u.text==record['text'][u.start:u.end] for u in actual)
        covered = {i for u in actual for i in range(u.start,u.end)}
        expected = {i for i,c in enumerate(record['text']) if not c.isspace()}
        assert covered==expected
        assert view['turns'][turn]['text']==record['text']
    assert [u.text for u in units if u.record_id==1]==['“','我','”','说','：','Jose\u0301','，','李','\U00020000','！']
    view['turns'][0]['units'][0][1]='changed'
    assert reference_candidate_view(focus)['turns'][0]['units'][0][1]=='Dr'
    changed = deepcopy(focus);changed['focus_utterance']['text'] += '.'
    assert reference_candidate_view(changed)['source_identity']!=reference_candidate_view(focus)['source_identity']


def test_repeated_quotes_get_code_derived_occurrences_not_distinct_name_ordinals():
    focus = focus_input('I, I and I.',prior=('Mira and Jo are neighbors.',))
    row = [wire_reference(phrase(focus,1,'I',i),dict(speaker='user'),'deictic') for i in range(3)]
    result = decode(focus,[list(reversed(row))])
    mentions = result['mentions']
    assert [(m['quote'],m['occurrence']) for m in mentions]==[('I',0),('I',1),('I',2)]
    assert result['focus_references']==[0,1,2] and result['readings']==[[0,0,0]]
    assert result['entities']==[dict(speaker='user',turn=1)]


def test_alias_target_is_an_explicit_source_choice_and_not_string_similarity():
    focus = focus_input('She is a nurse.',prior=('Mira is also called Dr Lane.',))
    ref = phrase(focus,1,'She')
    result = decode(focus,[[wire_reference(ref,dict(name=phrase(focus,0,'Dr Lane')))]])
    anchor = result['mentions'][result['entities'][0]['mentions'][0]]
    assert anchor['quote']=='Dr Lane' and anchor['occurrence']==0
    assert [m['quote'] for m in result['mentions']]==['Dr Lane','She']


def test_quoted_first_person_is_not_automatically_the_outer_speaker():
    focus = focus_input('Jo said, “I am a nurse.”',prior=('Jo arrived.',))
    row = [wire_reference(phrase(focus,1,'Jo'),dict(name=phrase(focus,0,'Jo')),'named'),
           wire_reference(phrase(focus,1,'I'),dict(name=phrase(focus,0,'Jo')),'deictic')]
    result = decode(focus,[row])
    assert result['readings']==[[0,0]] and all('speaker' not in e for e in result['entities'])


def test_correlated_readings_are_preserved_without_cartesian_expansion():
    focus = focus_input('She said, “I am a nurse.”',prior=('Mira and Jo arrived.',))
    worlds = []
    for person in ('Mira','Jo'):
        target = dict(name=phrase(focus,0,person))
        worlds.append([wire_reference(phrase(focus,1,'She'),target),
            wire_reference(phrase(focus,1,'I'),target,'deictic')])
    result = decode(focus,worlds)
    assert len(result['readings'])==2 and all(a==b for a,b in result['readings'])
    assert result['coverage']=='enumerated'


def test_chinese_omitted_subject_uses_its_clause_and_retains_unknown():
    focus = focus_input('从事护理工作。',speaker='unknown')
    span = phrase(focus,0,'从事护理工作')
    result = decode(focus,[[wire_reference(span,None,'implicit')]],coverage='open')
    assert result['mentions'][0]['quote']=='从事护理工作'
    assert result['mentions'][0]['kind']=='implicit' and result['readings']==[[None]]
    assert result['coverage']=='open' and reference_candidate_view(focus)['speakers']==[]


def test_missing_semantic_completeness_is_not_inferred_from_nonnull_assignments():
    focus = focus_input('I agree.')
    row = [wire_reference(phrase(focus,0,'I'),dict(speaker='user'),'deictic')]
    result = decode(focus,[row],coverage='unsupported')
    assert result['coverage']=='unsupported' and result['readings']==[[0]]
    with pytest.raises(ValueError,match='Coverage claim'):
        decode(focus,[[wire_reference(phrase(focus,0,'I'))]])
    with pytest.raises(ValueError,match='Coverage claim'):
        decode(focus,[row],coverage='open')
    result = decode(focus,[[wire_reference(phrase(focus,0,'I'))]],coverage='unsupported')
    assert result['coverage']=='unsupported' and result['readings']==[[None]]


def test_multiline_source_slice_keeps_exact_whitespace():
    focus=focus_input('Keep\n  trying!')
    result=decode(focus,[[wire_reference([0,1],None,'implicit')]],coverage='open')
    assert result['mentions'][0]['quote']=='Keep\n  trying'


def test_alias_anchors_are_not_merged_without_a_semantic_decision():
    focus=focus_input('She is a nurse.',prior=('Mira is also called Dr Lane.',))
    span=phrase(focus,1,'She')
    result=decode(focus,[[wire_reference(span,dict(name=phrase(focus,0,name)))] for name in ('Mira','Dr Lane')])
    assert len(result['entities'])==2 and len(result['readings'])==2


def test_existing_nonoverlapping_literal_occurrence_policy_is_not_silently_changed():
    focus=focus_input('哈哈哈')
    with pytest.raises(ValueError,match='literal boundary'):
        decode(focus,[[wire_reference([1,2],None,'implicit')]],coverage='open')


def test_entity_limit_applies_after_explicit_targets_are_collected():
    names='Al Be Ce De Ef Ge Hi Ix Jay'.split()
    focus=focus_input('She and he arrived.',prior=(' '.join(names)+'.',))
    spans=[phrase(focus,1,'She'),phrase(focus,1,'he')]
    worlds=[]
    for i in range(0,10,2):
        worlds.append([wire_reference(span,dict(name=phrase(focus,0,names[min(i+j,8)])))
            for j,span in enumerate(spans)])
    with pytest.raises(ValueError,match='mention/entity bounds'):decode(focus,worlds)


def test_empty_or_incomplete_mentions_are_not_evidence_of_semantic_success():
    inputs,refs = packet();key='reference-correlated-en'
    entry = next(e for e in inputs['schedule'] if e['case']==key)
    focus = json.loads(rows_for(inputs['cases'][key],entry)[0]['pair'][0])
    result = decode(focus,[[]])
    assert result['coverage']=='enumerated'
    assert not compare_analysis(result,refs[key])['correct']
    assert compare_analysis(result,refs[key])['missing_focus_mentions']==2


@pytest.mark.parametrize('damage',['root','coverage','boolean_id','missing_id','reverse','cross_turn',
    'prior_focus','kind','unknown_role','future_role','extra','duplicate','missing_world','kind_drift','duplicate_world'])
def test_invalid_selections_are_rejected_without_source_or_semantic_repair(damage):
    focus = focus_input('She said I agree.',prior=('Mira arrived.',))
    row = [wire_reference(phrase(focus,1,'She'),dict(name=phrase(focus,0,'Mira')))]
    value = dict(coverage='enumerated',readings=[row])
    if damage=='root': value['focus_references']=[0]
    elif damage=='coverage':value['coverage']=True
    elif damage=='boolean_id':row[0]['span'][0]=True
    elif damage=='missing_id':row[0]['span'][0]=999
    elif damage=='reverse':row[0]['span']=[len(reference_source_units(focus))-1,row[0]['span'][0]]
    elif damage=='cross_turn':row[0]['target']['name']=[0,len(reference_source_units(focus))-1]
    elif damage=='prior_focus':row[0]['span']=phrase(focus,0,'Mira')
    elif damage=='kind':row[0]['kind']='speaker'
    elif damage=='unknown_role':row[0]['target']=dict(speaker='unknown')
    elif damage=='future_role':row[0]['target']=dict(speaker='tutor')
    elif damage=='extra':row[0]['target']['repair']=True
    elif damage=='duplicate':row.append(deepcopy(row[0]))
    elif damage=='missing_world':value['readings'].append([])
    elif damage=='kind_drift':
        second=deepcopy(row);second[0]['kind']='deictic';value['readings'].append(second)
    else:value['readings'].append(deepcopy(row))
    before=deepcopy(value)
    with pytest.raises(ValueError):decode_reference_selections(json.dumps(value),focus)
    assert value==before


def test_name_target_cannot_reclassify_the_same_anaphoric_focus_span():
    focus = focus_input('She arrived.')
    span = phrase(focus,0,'She')
    with pytest.raises(ValueError,match='inconsistent mention kinds'):
        decode(focus,[[wire_reference(span,dict(name=span))]])


def test_mixed_script_interval_keeps_the_existing_literal_boundary_rule():
    focus = focus_input('I我')
    with pytest.raises(ValueError,match='literal boundary'):
        decode(focus,[[wire_reference([0,0],dict(speaker='user'),'deictic')]])
    result = decode(focus,[[wire_reference([1,1],None,'deictic')]],coverage='open')
    assert result['mentions'][0]['quote']=='我'


@pytest.mark.parametrize('text,error',[('a '*257,'unit bound'),('a'*4097,'byte bound'),('米'*1400,'byte bound')])
def test_overflow_rejects_whole_source_view_instead_of_truncating(text,error):
    focus=focus_input(text)
    for build in (reference_source_units,reference_candidate_view,reference_selection_schema):
        with pytest.raises(ValueError,match=error):build(focus)


def test_schema_limits_focus_ids_to_actual_focus_and_target_ids_to_actual_window():
    focus = focus_input('I agree.',prior=('Mira arrived.',));units=reference_source_units(focus)
    schema=reference_selection_schema(focus);reference=schema['properties']['readings']['items']['items']
    ids=reference['properties']['span']['items']
    assert ids['minimum']==next(i for i,u in enumerate(units) if u.record_id==1)
    assert ids['maximum']==len(units)-1
    names=reference['properties']['target']['anyOf'][1]['properties']['name']['items']
    assert names['minimum']==0 and names['maximum']==len(units)-1
    assert reference['properties']['target']['anyOf'][2]['properties']['speaker']['enum']==['companion','user']
    schema['properties']['coverage']['enum'].append('invented')
    assert reference_selection_schema(focus)['properties']['coverage']['enum']==['enumerated','open','unsupported']


@pytest.mark.parametrize('content',['{"coverage":"enumerated","coverage":"open","readings":[[]]}',
    '{"coverage":"enumerated","readings":[]}', '{"mentions":[],"entities":[],"focus_references":[],"readings":[[]],"coverage":"enumerated"}'])
def test_duplicate_keys_empty_world_set_and_old_protocols_cannot_be_reinterpreted(content):
    with pytest.raises(ValueError):decode_reference_selections(content,focus_input('No people.'))
