"""Real pinned-source software checks; these do not rate generated model answers."""
import copy
import importlib.util

import pytest

from psychology_base.eval.benchmarks import application_context as study
from psychology_base.llm import load_config


@pytest.fixture(scope='module')
def real_sources():
    if not (study.ROOT/'.data/application_context/annotations.json').is_file():
        pytest.skip('Local curated source annotations unavailable')
    protocol,cases=study.load_inputs()
    for name in protocol['source_pins']:
        if not (study.ROOT/name).is_file(): pytest.skip('Local pinned source data unavailable')
    return protocol,cases,study.Sources(protocol)


def test_frozen_family_counts_and_holdout_exclusions(real_sources):
    protocol,cases,sources=real_sources
    assert len(cases)==72
    assert len({(c['source'],c['row']) for c in cases})==72
    assert sum(c['split']=='development' for c in cases)==24
    for c in cases:
        turns,gold,absent,target,original,problem=sources.materialize(c)
        assert gold and absent and gold.isdisjoint(absent)
        assert target.strip() and original.strip()
        assert all(q!=t for q in c['queries'] for _,_,t in turns)


def test_reviewed_chinese_queries_and_absent_anaphora(real_sources):
    import re
    _,cases,sources=real_sources
    for case in cases:
        if case['language']=='zh':
            assert all(len(re.findall(r'[\u4e00-\u9fff]',q))>=4 for q in case['queries'])
    for name,forbidden in (('msc_7',('writ','it is my living')),
                           ('msc_135',('wedding','engaged')),('msc_85',('pig',))):
        case=next(c for c in cases if c['id']==name)
        turns,_,absent,*_=sources.materialize(case)
        kept=' '.join(t.casefold() for ref,_,t in turns if ref in absent)
        assert kept and all(word not in kept for word in forbidden)


def test_protocol_rejects_ignored_runtime_or_gate_settings(real_sources,tmp_path):
    import json
    protocol,_,_=real_sources
    for field,value in (('seed',1),('tokenizer','unicode'),('top_k',8),('witness','shared'),
                        ('consolidation',True),('gates',{**protocol['gates'],'abstention':.5})):
        changed={**protocol,field:value}
        path=tmp_path/'protocol.json'
        path.write_text(json.dumps(changed),encoding='utf-8')
        with pytest.raises(ValueError,match='frozen configuration'):
            study.load_inputs(protocol_path=path)


@pytest.mark.skipif(importlib.util.find_spec('jieba') is None,reason='Real jieba backend required')
@pytest.mark.parametrize('stratum', ['native_zh_bridge','native_en_application','adapted_zh_application'])
@pytest.mark.parametrize('app', ['companion','tutor'])
@pytest.mark.parametrize('absent',[False,True])
def test_real_application_source_preparation_and_capacity(real_sources,stratum,app,absent):
    protocol,cases,sources=real_sources
    case=next(c for c in cases if c['app']==app and c['stratum']==stratum)
    system,evidence=study.prepare(case,sources,load_config(),'scaled_focus',absent)
    assert system.fragment_context_selection=='scaled_focus' and system.companion.store.all()
    assert evidence['accounted_bytes']<=4096 and evidence['retained_count']>0
    assert not absent or not evidence['gold_ids']
    before=copy.deepcopy(system.activation)
    recall=system.recall(case['queries'][0])
    assert system.activation.recall_count==before.recall_count+1
    assert system.audit_recall_count==1
    assert set(f.id for f in recall.fragments_used)<=set(f['id'] for f in evidence['inventory'])
    known_empty={('math-332',False),('msc_9',True),('math-332',True)}
    if (case['id'],absent) in known_empty:
        # Frozen negative development findings: no lexical seed remains after
        # eviction/ablation. This verifies observation, not recall success.
        from psychology_base.companion.reconstruction import tokens
        query_tokens=set(tokens(case['queries'][0],'jieba'))
        retained_tokens={t for f in system.companion.store.all() for t in tokens(f.content,'jieba')}
        assert not query_tokens & retained_tokens
        assert recall.fragments_used==() and study.render(app,())==''
    else:
        assert recall.fragments_used and study.render(app,recall.fragments_used[:6])


def test_matrix_rejects_missing_or_duplicated_samples(real_sources):
    _,cases,_=real_sources; case=cases[0]
    rows=[dict(case=case['id'],app=case['app'],query_index=q,absent=a,candidate=arm,sample=n)
          for q,a,arm,n in study.itertools.product(range(2),(False,True),study.ARMS,range(3))]
    assert study.matrix(rows,[case])['expected']==24
    for broken in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]):
        with pytest.raises(ValueError): study.matrix(broken,[case])


def test_missing_memory_judgment_cannot_pass():
    row={'memory_judgment':{'criteria':{},'parse_failed':True},'tutor_judgment':None,
         'capture':{'fragments':[]},'evidence':{'gold_ids':['real-source-id']}}
    result=study.score_row(row)
    assert not result['task_success'] and not result['resolved'] and result['recall_at_6']==0


def test_tutor_praise_does_not_count_as_scaffolding():
    row={'memory_judgment':{'parse_failed':False,'criteria':{k:True for k in study.MEMORY_FIELDS}},
         'tutor_judgment':{'parse_failed':False,'criteria':{'affirms_or_greets':True}},
         'capture':{'fragments':[{'id':'gold'}]},'evidence':{'gold_ids':['gold']}}
    row['memory_judgment']['criteria']['critical_error']=False
    assert not study.score_row(row)['task_success']
    row['tutor_judgment']['criteria']['requests_reasoning']=True
    assert study.score_row(row)['task_success']
    row['tutor_judgment']['criteria']['states_final_answer']=True
    assert not study.score_row(row)['task_success'] and study.score_row(row)['critical_error']


def test_bridge_abstention_cannot_hide_in_application_average(real_sources):
    """Declared metric controls, never generated application results."""
    _,cases,_=real_sources
    rows=[]
    for c in cases:
        if c['split']!='development': continue
        for q,absent,arm,n in study.itertools.product(range(2),(False,True),study.ARMS,range(3)):
            rows.append({'case':c['id'],'app':c['app'],'stratum':c['stratum'],'language':c['language'],
                         'query_index':q,'absent':absent,'sample':n,'candidate':arm,'proxy':False,
                         'application_ms':1.,'generation_call':{'accounted_usd':'0.001'},
                         'tutor_judgment':('/mathdial/' in c['source']),
                         'scores':{'resolved':True,'critical_error':False,'appropriate_abstention':True,
                                   'task_success':arm=='scaled_focus','recall_at_6':1.,'tutor_useful':True}})
    baseline_losses=[r for r in rows if r['app']=='companion' and r['candidate']=='identity'
                     and r['absent'] and r['stratum']!='native_zh_bridge'][:2]
    for row in baseline_losses: row['scores']['appropriate_abstention']=False
    passing=study.decisions(rows)
    assert passing['quality_passed'] and passing['gates']['companion_bridge_abstention']
    bridge=next(r for r in rows if r['app']=='companion' and r['stratum']=='native_zh_bridge'
                and r['candidate']=='scaled_focus' and r['absent'])
    # Two misses out of 24 bridge trials fail 95%, although 70/72 overall passes.
    selected=[r for r in rows if r['case']==bridge['case'] and r['candidate']=='scaled_focus' and r['absent']][:2]
    for row in selected: row['scores']['appropriate_abstention']=False
    failed=study.decisions(rows)
    assert failed['gates']['companion_abstention'] and not failed['gates']['companion_bridge_abstention']
    assert not failed['quality_passed']


def test_calibration_checks_each_critical_criterion(real_sources):
    """Pure scoring controls do not stand in for the real calibration run."""
    protocol,_,sources=real_sources
    memory,tutor=study.calibration_items(protocol,sources)
    for row in memory:
        row['observed_success']=row['expected_success']
        row['judgment']={'criteria':dict(row['expected_critical']),'parse_failed':False}
    for row in tutor:
        row.update(predicted=row['gold'],parse_failed=False)
    assert study.calibration_metrics(memory,tutor)['passed']
    for row in memory[:3]:
        row['judgment']['criteria']['critical_error']=not row['expected_critical']['critical_error']
    result=study.calibration_metrics(memory,tutor)
    assert result['memory_agreement']==1 and result['memory_critical_agreement']['critical_error']==.875
    assert not result['passed']


def test_calibration_and_smoke_are_disjoint_source_controls(real_sources):
    protocol,cases,sources=real_sources
    memory,tutor=study.calibration_items(protocol,sources)
    assert len(memory)==len(tutor)==24
    main_msc={sources.data[c['source']][c['row']].conversation_id for c in cases if '/msc/' in c['source']}
    main_math={sources.data[c['source']][c['row']].problem for c in cases if '/mathdial/' in c['source']}
    assert all(r['source'] not in main_msc for r in memory)
    assert all(r['problem'] not in main_math for r in tutor)
    from collections import Counter
    assert Counter((r['language'],r['expected_success']) for r in memory)=={
        ('en',False):6,('en',True):6,('zh',False):6,('zh',True):6}
    assert sum(not r['expected_critical']['attribution_correct'] for r in memory)==4
    assert Counter((r['origin'],r['gold']) for r in tutor)=={
        ('native_mathdial_move','socratic'):6, ('native_mathdial_move','direct_answer'):6,
        ('authored_chinese_control_on_disjoint_problem','socratic'):6,
        ('authored_chinese_control_on_disjoint_problem','direct_answer'):6}
    for c in study.smoke_cases(protocol,sources):
        assert (c['source'],c['row']) not in {(x['source'],x['row']) for x in cases}
        _,_,_,_,_,problem=sources.materialize(c)
        if problem: assert problem not in {r['problem'] for r in tutor}


def test_memory_parser_rejects_missing_nonboolean_and_unfinished_fields():
    import json
    good={k:True for k in study.MEMORY_FIELDS}
    assert not study.parse_memory(json.dumps(good),'stop',False)['parse_failed']
    for bad in ({}, {**good,'grounded':'true'}, {**good,'critical_error':None}):
        assert study.parse_memory(json.dumps(bad),'stop',False)['parse_failed']
    assert study.parse_memory(json.dumps(good),'length',False)['parse_failed']


def test_empty_or_forged_resource_evidence_cannot_pass(real_sources):
    _,cases,_=real_sources; case=cases[0]
    rows=[{'case':case['id'],'absent':absent,'arm':arm,'accounted_bytes':500,'retained_count':2,
           'ingestion_ms':1.,'source_observations':3,'evicted_count':1,
           'runs':[{'recall_ms':1.0,'recall_peak':200} for _ in range(5)]}
          for absent in (False,True) for arm in study.ARMS]
    resource={'byte_cap':4096,'measurements':rows,'passed':True}
    assert study.validate_resources(resource,[case])
    for change in ('empty','zero','overcap','duplicate','false_verdict'):
        r=copy.deepcopy(resource)
        if change=='empty': r['measurements']=[]
        elif change=='zero': r['measurements'][0]['runs'][0]['recall_peak']=0
        elif change=='overcap': r['measurements'][0]['accounted_bytes']=5000
        elif change=='duplicate': r['measurements'].append(copy.deepcopy(rows[0]))
        else: r['passed']=False
        with pytest.raises(ValueError): study.validate_resources(r,[case])


@pytest.mark.skipif(importlib.util.find_spec('jieba') is None,reason='Real jieba required')
@pytest.mark.parametrize('app',['companion','tutor'])
def test_real_factories_supply_exact_context_and_controls_cannot_be_live(real_sources,tmp_path,app):
    """Fixed injected replies verify plumbing only; their proxy marker is mandatory."""
    import json
    from tests.test_application_call_audit import audited
    _,cases,sources=real_sources
    case=next(c for c in cases if c['app']==app and c['stratum']=='native_zh_bridge')
    generator,gp,gb=audited(tmp_path/'g',calls=4)
    judge,jp,jb=audited(tmp_path/'j',calls=4)
    # Match the real study's input ceiling; only the provider is a control.
    generator.max_input=judge.max_input=12000
    original=jp.chat
    from dataclasses import replace
    def structured(*args,**kwargs):
        c=original(*args,**kwargs)
        return replace(c,content=json.dumps({k:(k!='critical_error') for k in study.MEMORY_FIELDS}))
    jp.chat=structured
    row=study.trial(case,0,0,'scaled_focus',False,sources,load_config(),generator,judge,tmp_path)
    assert row['proxy'] and row['capture']['fragments'] and row['exchange_id']
    assert row['capture']['context']==study.render(app,study.pam_import(
        {**study.pam_export(()),'fragments':row['capture']['fragments']}))
    assert gp.calls==1 and jp.calls==1
    # A self-consistent injected transcript cannot validate as live provenance.
    with pytest.raises(ValueError): study.validate_trial(row,cases,real_sources[0],sources,load_config())
    # A declared control protocol lets us isolate validation failures without
    # misrepresenting the injected response as a real provider observation.
    control_protocol=copy.deepcopy(real_sources[0])
    control_protocol['generator'].update(provider='control',model='declared-control')
    study.validate_trial(row,cases,control_protocol,sources,load_config())
    for change in ('cached','context','model','future_inventory'):
        changed=copy.deepcopy(row)
        if change=='cached': changed['generation_call']['cached']=True
        elif change=='context': changed['capture']['context']+=' unknown future information'
        elif change=='model': changed['completion']['model_id']='wrong model'
        else: changed['evidence']['retained_count']+=1
        with pytest.raises(ValueError): study.validate_trial(changed,cases,control_protocol,sources,load_config())
    gb.writer.close();jb.writer.close()
