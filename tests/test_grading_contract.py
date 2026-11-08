"""Prospective contract controls; authored booleans are not judge accuracy evidence."""
import copy
import itertools
import importlib.util
import json
from types import SimpleNamespace

import pytest

from psychology_base.applications.measurement import empath_judge as empath
from psychology_base.applications.measurement.socratic_compliance import ComplianceLabel
from psychology_base.eval.benchmarks import application_context as study
from tests.test_application_context_study import real_sources


def memory(**changes):
    return dict(addresses_request=True, grounded=True, attribution_correct=True,
                abstains_appropriately=True, critical_error=False) | changes


def tutor(**changes):
    return {k.value:False for k in empath.EMPATHSubCriterion} | changes


class ControlRouter:
    """A parser/transport control that cannot provide model-quality evidence."""
    controls=True

    def __init__(self, criteria, finish='stop'):
        self.raw=json.dumps(criteria)
        self.finish=finish
        self.requests=[]

    def chat(self, messages, **kwargs):
        self.requests.append((messages,kwargs))
        self.last_completion=SimpleNamespace(content=self.raw,finish_reason=self.finish)
        return self.last_completion


@pytest.mark.parametrize('criteria',[
    memory(grounded=True, critical_error=True),
    memory(grounded=False, critical_error=False),
    memory(attribution_correct=False),
    memory(grounded=False, critical_error=True, abstains_appropriately=True),
])
def test_memory_contradictions_preserve_raw_values_and_fail_resolution(criteria):
    raw=json.dumps(criteria)
    result=study.parse_memory(raw,'stop',False,contract_version=2)
    assert result['criteria']==criteria and result['raw']==raw
    assert not result['parse_failed'] and result['consistency_errors']
    row={'memory_judgment':result,'tutor_judgment':None,
         'capture':{'fragments':[]},'evidence':{'gold_ids':[]}}
    assert not study.score_row(row)['resolved']
    assert not study.memory_control_success({'source_absent':False},result)
    # Clearing the reported error list cannot bypass recomputation at scoring.
    result['consistency_errors']=[]
    assert not study.score_row(row)['resolved']
    assert study.parse_memory(raw,'stop',False)['criteria']==criteria
    assert 'consistency_errors' not in study.parse_memory(raw,'stop',False)


def test_memory_boolean_contract_exhaustively_accepts_only_the_eight_coherent_patterns():
    allowed={(address,True,True,abstain,False) for address,abstain in itertools.product((False,True),repeat=2)}
    allowed|={(address,False,attribution,False,True) for address,attribution in itertools.product((False,True),repeat=2)}
    for values in itertools.product((False,True),repeat=5):
        criteria=dict(zip(study.MEMORY_FIELDS,values))
        result=study.parse_memory(json.dumps(criteria),'stop',False,contract_version=2)
        assert (not result['consistency_errors'])==(values in allowed)


@pytest.mark.parametrize('criteria',[
    tutor(states_final_answer=True, requests_reasoning=True),
    tutor(corrects_with_fact=True, provides_scaffold_hint=True),
    tutor(affirms_or_greets=True, requests_reasoning=True),
])
def test_empath_contradictions_keep_the_original_aggregation_and_raw_text(criteria):
    router=ControlRouter(criteria)
    verdict=empath.LLMEMPATHJudge(router,contract_version=2).judge('declared parser control')
    assert {k.value:v for k,v in verdict.sub_criteria.items()}==criteria
    assert verdict.rationale==router.raw
    assert verdict.label==empath.aggregate_sub_criteria(verdict.sub_criteria)
    assert not verdict.parse_failed and verdict.consistency_errors and not verdict.resolved
    old=empath.LLMEMPATHJudge(ControlRouter(criteria),strict=True).judge('declared parser control')
    assert old.resolved and not old.consistency_errors


def test_empath_boolean_contract_exhaustively_accepts_seven_coherent_patterns():
    allowed={tuple(tutor(**{key:True})[k.value] for k in empath.EMPATHSubCriterion)
             for key in ('states_final_answer','corrects_with_fact','affirms_or_greets')}
    allowed|={(reason,False,hint,False,False) for reason,hint in itertools.product((False,True),repeat=2)}
    for values in itertools.product((False,True),repeat=5):
        criteria=dict(zip((k.value for k in empath.EMPATHSubCriterion),values))
        verdict=empath.LLMEMPATHJudge(ControlRouter(criteria),contract_version=2).judge('parser control')
        assert verdict.resolved==(values in allowed)


@pytest.mark.parametrize('raw',[
    '{"grounded":true,"grounded":false,"addresses_request":true,"attribution_correct":true,"abstains_appropriately":false,"critical_error":true}',
    '{"requests_reasoning":true,"requests_reasoning":false,"states_final_answer":false,"provides_scaffold_hint":false,"corrects_with_fact":false,"affirms_or_greets":false}',
])
def test_new_contract_rejects_duplicate_boolean_keys(raw):
    if 'grounded' in raw:
        result=study.parse_memory(raw,'stop',False,contract_version=2)
        assert result['raw']==raw and result['parse_failed']
    else:
        router=ControlRouter({}); router.raw=raw
        assert empath.LLMEMPATHJudge(router,contract_version=2).judge('parser control').parse_failed


@pytest.mark.parametrize('version',[True,False,0,3,'2',None])
def test_unknown_contract_rejected_before_dispatch(version):
    router=ControlRouter(tutor())
    with pytest.raises(ValueError): empath.LLMEMPATHJudge(router,contract_version=version)
    with pytest.raises(ValueError): study.parse_memory('{}','stop',False,contract_version=version)
    assert not router.requests


def test_revised_protocol_is_explicit_nonmutating_and_changes_qualification_identity():
    base={'memory_prompt':study.MEMORY_PROMPT,'profile_sha256':'a'*64}
    before=copy.deepcopy(base)
    revised=study.revised_grading_protocol(base)
    assert base==before and revised['grading_contract_version']==2
    assert revised['memory_prompt'].startswith(base['memory_prompt'])
    assert study.protocol_sha(revised)!=study.protocol_sha(base)
    assert study.qualification_identity(revised)['grading_contract_version']==2
    assert study.revised_grading_protocol(revised)==revised
    router=ControlRouter(memory())
    result=study.judge_memory(router,{'app':'companion'},False,{'query':'q','context':'user said: fact'},
                              'reply','fact',protocol=revised)
    assert result['contract_version']==2 and result['proxy']
    assert router.requests[0][0][0].content==revised['memory_prompt']
    tutoring=ControlRouter(tutor(provides_scaffold_hint=True))
    verdict=study.judge_tutor(tutoring,'review your work',problem_context='problem',student_utterance='help',protocol=revised)
    assert verdict.contract_version==2 and verdict.resolved
    assert 'quoted earlier work' in tutoring.requests[0][0][0].content


@pytest.mark.parametrize('case',study.read(study.FIXTURES/'grading-boundaries-v2.json')['cases'],ids=lambda c:c['id'])
def test_authored_boundary_examples_are_consistent_contracts_not_model_evidence(case):
    # These are reviewed expected labels, not an automated semantic classifier.
    if case['kind']=='memory':
        result=study.parse_memory(json.dumps(case['expected_criteria']),'stop',True,contract_version=2)
        assert not result['consistency_errors'] and not result['parse_failed']
        assert result['proxy']
    else:
        result=empath.parse_empath_verdict(json.dumps(case['expected_criteria']),'stop',contract_version=2)
        assert result.resolved and not result.consistency_errors


def test_fresh_profile_keeps_models_budget_thresholds_and_excludes_all_observed_families(real_sources):
    base,cases,sources=real_sources
    new=study.local_profile(study.FIXTURES/'hybrid-screen-v4.json',base)
    old=study.local_profile(study.FIXTURES/'hybrid-screen-v3.json',base)
    assert new['budget']=={'max_calls':615,'max_usd':'1','local_only':False}
    assert all(new[k]==old[k] for k in ('judge','generator','source_pins','gates','samples'))
    assert new['grading_contract_version']==2 and new['calibration_revision']==4
    assert 'memory_evidence' not in new
    m,t=study.calibration_items(new,sources)
    previous=[study.calibration_items(base if r==1 else {**base,'calibration_revision':r},sources) for r in (1,2,3)]
    prior_memory={x['source'] for memory_items,_ in previous for x in memory_items}
    prior_tutor={x['source'] for _,tutor_items in previous for x in tutor_items}
    normalize=lambda x:' '.join(x.casefold().split())
    prior_problems={normalize(x['problem']) for _,ts in previous for x in ts}
    reserved=cases+study.smoke_cases(base,sources)
    for c in reserved:
        if '/msc/' in c['source']: prior_memory.add(sources.data[c['source']][c['row']].conversation_id)
        if '/mathdial/' in c['source']:
            convo=sources.data[c['source']][c['row']]
            prior_tutor.add(convo.conversation_id); prior_problems.add(normalize(convo.problem))
    assert len(m)==len(t)==24
    assert len({r['source'] for r in m})==12
    assert {r['source'] for r in m}.isdisjoint(prior_memory)
    assert len({r['source'] for r in t})==24
    assert {r['source'] for r in t}.isdisjoint(prior_tutor)
    assert {normalize(r['problem']) for r in t}.isdisjoint(prior_problems)
    for row in m:
        turn=sources.data['.data/msc/train.jsonl'][row['source_row']].sessions[0].turns[row['source_turn']]
        assert turn.speaker_idx==0 and turn.utterance==row['reference']
    assert study.calibration_catalog(m,t,protocol=new)==study.read(study.FIXTURES/'qualification-v4.json')


@pytest.mark.parametrize('mutation',['budget','contract_hash','catalog_hash'])
def test_revised_profile_rejects_scope_or_contract_drift(real_sources,tmp_path,mutation):
    profile=study.read(study.FIXTURES/'hybrid-screen-v4.json')
    if mutation=='budget': profile['budget']['max_calls']=616
    else: profile['grading_contract' if mutation=='contract_hash' else 'qualification']['sha256']='0'*64
    path=tmp_path/'profile.json'; path.write_text(json.dumps(profile),encoding='utf-8')
    with pytest.raises(ValueError): study.local_profile(path,real_sources[0])


def test_old_qualification_cannot_be_reused_for_revised_contract(real_sources):
    old=study.local_profile(study.FIXTURES/'hybrid-screen-v3.json',real_sources[0])
    revised=study.revised_grading_protocol(old)
    with pytest.raises(ValueError,match='fresh version-4'):
        study.calibration_items(revised,real_sources[2])


def test_contradiction_blocks_calibration_without_improving_any_label_metric(real_sources):
    base,_,sources=real_sources
    m,t=study.calibration_items(base,sources)
    for row in m:
        row['judgment']={'criteria':dict(row['expected_critical']),'parse_failed':False}
        row['observed_success']=row['expected_success']
    for row in t: row.update(predicted=row['gold'],parse_failed=False)
    before=study.calibration_metrics(m,t)
    assert before['passed']
    t[0].update(contract_version=2,criteria=tutor(states_final_answer=True,requests_reasoning=True),consistency_errors=[])
    after=study.calibration_metrics(m,t)
    assert not after['passed'] and study.calibration_failure_is_final(m,t)
    assert after['memory_agreement']==before['memory_agreement']==1
    assert after['tutor_kappa']==before['tutor_kappa']==1
    assert len(m)==len(t)==24


def test_native_mathdial_gate_counts_contradictions_without_dropping_their_pairs():
    from psychology_base.eval.benchmarks.public_datasets.mathdial_empath_scorer import EMPATHMathDialReport
    bad=empath.parse_empath_verdict(json.dumps(tutor(requests_reasoning=True,affirms_or_greets=True)),
                                   'stop',contract_version=2)
    good=empath.parse_empath_verdict(json.dumps(tutor(states_final_answer=True)),'stop',contract_version=2)
    verdicts=(bad,good)*50
    report=EMPATHMathDialReport(sampled_teacher_turns=100,taxonomy_matches=100,
        auto_socratic_count=100,auto_direct_answer_count=0,empath_verdicts=verdicts,
        empath_gold_pairs=tuple((v.label,v.label) for v in verdicts),all_proxy_verdicts=False)
    assert report.empath_kappa().kappa==1 and len(report.empath_gold_pairs)==100
    assert report.judge_consistency_failures()==50
    assert not report.meets_pivot_gate()


@pytest.mark.parametrize('kind',['memory','tutor'])
@pytest.mark.parametrize('mutation',['missing','string','integer','null','unfinished'])
def test_prospective_contract_requires_complete_boolean_finished_outputs(kind,mutation):
    criteria=memory() if kind=='memory' else tutor()
    field=next(iter(criteria))
    if mutation=='missing': criteria.pop(field)
    elif mutation!='unfinished': criteria[field]={'string':'false','integer':0,'null':None}[mutation]
    raw=json.dumps(criteria); finish='length' if mutation=='unfinished' else 'stop'
    result=(study.parse_memory(raw,finish,True,contract_version=2) if kind=='memory' else
            empath.parse_empath_verdict(raw,finish,contract_version=2))
    assert result['parse_failed'] if kind=='memory' else result.parse_failed


def test_actual_recorded_judgments_replay_unchanged_and_new_checks_do_not_relabel_them():
    path=study.ROOT/'.data/application_context/judge-repeatability-v1.json'
    if not path.exists(): pytest.skip('Recorded hosted diagnostic unavailable')
    before=study.shared.digest(path)
    report=study.read(path)
    flagged=[]
    for row in report['results']:
        completion=row['call']['completion']
        raw=completion['content']; finish=completion['finish_reason']
        if row['kind']=='memory':
            assert study.parse_memory(raw,finish,False)==row['judgment']
            revised=study.parse_memory(raw,finish,False,contract_version=2)
            assert revised['raw']==raw and revised['criteria']==row['judgment']['criteria']
            if revised['consistency_errors']: flagged.append((row['control'],row['repeat']))
        else:
            legacy=empath.parse_empath_verdict(raw,finish,strict=True)
            assert legacy.label.value==row['judgment']['label']
            assert {k.value:v for k,v in legacy.sub_criteria.items()}==row['judgment']['criteria']
    assert ('memory-en-present',2) in flagged
    assert study.shared.digest(path)==before


@pytest.mark.skipif(importlib.util.find_spec('jieba') is None,reason='Real jieba backend required')
def test_normal_trial_and_replay_keep_contradictions_unresolved_and_reject_forged_metadata(real_sources,tmp_path):
    from dataclasses import replace
    from psychology_base.llm import load_config
    from tests.test_application_call_audit import audited
    base,cases,sources=real_sources
    protocol=study.local_profile(study.FIXTURES/'hybrid-screen-v4.json',base)
    # Real source/factory/router flow with an explicitly declared provider
    # control. This transcript cannot qualify as real model-quality evidence.
    protocol['screening']=False
    protocol['generator'].update(provider='control',model='declared-control')
    case=next(c for c in cases if c['split']=='development' and '/mathdial/' in c['source'])
    generation,_,gb=audited(tmp_path/'g',calls=1)
    judge,provider,jb=audited(tmp_path/'j',calls=2)
    generation.max_input=judge.max_input=12000
    original=provider.chat
    values=[memory(critical_error=True),tutor(corrects_with_fact=True,requests_reasoning=True)]
    def scripted(*args,**kwargs):
        completion=original(*args,**kwargs)
        return replace(completion,content=json.dumps(values[provider.calls-1]))
    provider.chat=scripted
    try:
        row=study.trial(case,0,0,'scaled_focus',False,sources,load_config(),generation,judge,tmp_path,protocol=protocol)
        assert row['proxy'] and not row['scores']['resolved']
        assert row['memory_judgment']['contract_version']==row['tutor_judgment']['contract_version']==2
        assert row['memory_judgment']['consistency_errors'] and row['tutor_judgment']['consistency_errors']
        study.validate_trial(row,cases,protocol,sources,load_config())
        for kind in ('memory','tutor'):
            forged=copy.deepcopy(row); forged[kind+'_judgment']['consistency_errors']=[]
            with pytest.raises(ValueError): study.validate_trial(forged,cases,protocol,sources,load_config())
        requests=[json.loads(line) for line in jb.writer.path.read_text(encoding='utf-8').splitlines()]
        requests=[r for r in requests if r['event']=='reservation']
        assert requests[0]['messages'][0]['content']==protocol['memory_prompt']
        assert 'quoted earlier work' in requests[1]['messages'][0]['content']
        assert all(r['temperature']==0 for r in requests)
    finally:
        gb.writer.close(); jb.writer.close()


def test_calibration_forwards_revised_contract_and_stops_on_a_real_parser_contradiction(real_sources,tmp_path):
    from dataclasses import replace
    from tests.test_application_call_audit import audited
    protocol=study.local_profile(study.FIXTURES/'hybrid-screen-v4.json',real_sources[0])
    judge,provider,budget=audited(tmp_path,calls=1)
    judge.max_input=12000
    original=provider.chat
    def scripted(*args,**kwargs):
        return replace(original(*args,**kwargs),content=json.dumps(memory(critical_error=True)))
    provider.chat=scripted
    try:
        report=study.calibration(protocol,real_sources[2],judge)
        assert len(report['memory_controls'])==1 and not report['tutor_controls']
        assert report['stopped_early'] and not report['passed'] and report['proxy']
        judgment=report['memory_controls'][0]['judgment']
        assert not judgment['parse_failed'] and judgment['consistency_errors']
        assert judgment['criteria']['grounded'] and judgment['criteria']['critical_error']
    finally: budget.writer.close()


@pytest.mark.parametrize('mode',['stable','drift','accuracy_failure'])
def test_existing_calibration_runs_eight_interleaved_sentinels_and_cannot_bypass_drift(real_sources,tmp_path,mode):
    """Scripted transport test only; proxy=True forbids model-quality evidence."""
    from dataclasses import replace
    from tests.test_application_call_audit import audited
    protocol=study.local_profile(study.FIXTURES/'hybrid-screen-v4.json',real_sources[0])
    m,t=study.calibration_items(protocol,real_sources[2])
    values={}
    for i,item in enumerate(m):
        quoted=item['reference'] in item['response']
        values['memory',i]=memory(addresses_request=quoted,
            abstains_appropriately=not item['expected_critical']['critical_error'] and (item['source_absent'] or quoted),
            **item['expected_critical'])
    for i,item in enumerate(t):
        socratic=(item['gold']=='socratic')!=(mode=='accuracy_failure')
        values['tutor',i]=tutor(**({'requests_reasoning':True} if socratic else {'states_final_answer':True}))
    selected=study.calibration_sentinels(m,t)
    assert len(selected)==len(set(selected))==8
    sequence=[values['memory',i] for i in range(24)]+[values['tutor',i] for i in range(24)]
    sequence += [copy.deepcopy(values[key]) for key in selected+list(reversed(selected))]
    if mode=='drift': sequence[48]['addresses_request']=not sequence[48]['addresses_request']
    judge,provider,budget=audited(tmp_path,calls=64)
    judge.max_input=12000; original=provider.chat
    def scripted(*args,**kwargs):
        completion=original(*args,**kwargs)
        return replace(completion,content=json.dumps(sequence[provider.calls-1]))
    provider.chat=scripted
    try:
        report=study.calibration(protocol,real_sources[2],judge)
        assert report['proxy']
        assert len(report['memory_controls'])==len(report['tutor_controls'])==24
        assert report['memory_agreement']==1
        assert report['tutor_kappa']==(-1 if mode=='accuracy_failure' else 1)
        assert report['passed'] is (mode=='stable')
        assert report['complete'] is (mode=='stable')
        assert len(report['repeated_controls'])=={'stable':16,'drift':1,'accuracy_failure':0}[mode]
        assert provider.calls=={'stable':64,'drift':49,'accuracy_failure':48}[mode]
        if mode=='stable':
            bad=copy.deepcopy(report['repeated_controls'])
            bad[1]=copy.deepcopy(bad[0])
            with pytest.raises(ValueError,match='schedule'):
                study.repeatability_status(report['memory_controls'],report['tutor_controls'],bad)
            assert not study.calibration_metrics(report['memory_controls'],report['tutor_controls'])['passed']
            # Even a relabelled proxy transcript cannot reuse one dispatch as
            # several independent repetitions. This is a negative integrity test.
            forged=copy.deepcopy(report)
            forged.update(proxy=False,code_sha256=study.fingerprints(),protocol_sha256=study.protocol_sha(protocol))
            forged['repeated_controls'][0]['assessment']['call']=copy.deepcopy(forged['memory_controls'][selected[0][1]]['call'])
            with pytest.raises(ValueError,match='distinct uncached'):
                study.validate_calibration(forged,protocol,real_sources[2])
        assert report['repeatability']==study.repeatability_status(report['memory_controls'],report['tutor_controls'],report['repeated_controls'])
    finally: budget.writer.close()


def test_new_verdict_cannot_qualify_with_an_aggregate_label_that_disagrees_with_its_criteria():
    from dataclasses import replace
    verdict=empath.parse_empath_verdict(json.dumps(tutor(states_final_answer=True)),'stop',contract_version=2)
    assert verdict.resolved
    assert not replace(verdict,label=ComplianceLabel.SOCRATIC).resolved
