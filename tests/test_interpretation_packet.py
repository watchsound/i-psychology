"""Offline packet and grading controls; no provider/model invocation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from psychology_base.applications.measurement.local_query_binding import reference_messages
from psychology_base.eval.benchmarks.reference_extraction import compare_interpretation_analysis,grade_interpretation_packet

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def packet():
    file=ROOT/'docs/experiments/phase-9-memory-navigation/run-interpretation-source-v2.py'
    spec=importlib.util.spec_from_file_location('interpretation_packet_test',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    bank=json.loads(module.BANK.read_text(encoding='utf-8'))
    inputs,refs=module.build_packet(bank)
    return module,inputs,refs


def test_four_source_inputs_are_label_free_and_preserve_bilingual_context(packet):
    module,inputs,refs=packet
    assert len(inputs['cases'])==len(inputs['schedule'])==len(refs)==4
    for entry in inputs['schedule']:
        case=inputs['cases'][entry['case']]
        rows=module.rows_for(case,entry)
        focus=json.loads(rows[0]['pair'][0])
        payload=json.loads(reference_messages(focus,interpretation=True)[1].content)
        assert set(payload)=={'schema','source_identity','context_scope','focus_turn','speakers','turns'}
        assert payload['focus_turn']==1 and payload['turns'][0]['speaker']=='tutor'
        assert payload['turns'][1]['speaker']=='user'
        assert 'expected' not in case and 'membership_references' not in case
        assert 'descriptions' not in payload and 'readings' not in payload
        assert compare_interpretation_analysis(refs[case['id']]['expected'],refs[case['id']],focus)['correct']
    assert module.LIMITS['max_native_attempts']==4 and module.LIMITS['paid_api_usd']=='0'


def test_missing_wrong_and_empty_outputs_keep_the_complete_denominator(packet):
    module,inputs,refs=packet
    schedule=inputs['schedule']
    empty=grade_interpretation_packet(inputs['cases'],refs,schedule,[])
    assert empty['summary']['all']==dict(planned=4,attempted=0,completed=0,correct=0,not_attempted=4)
    observed=[dict(id=e['id'],analysis=deepcopy(refs[e['case']]['expected'])) for e in schedule]
    assert grade_interpretation_packet(inputs['cases'],refs,schedule,observed)['candidate_screen_passed']
    observed[0]['analysis']['readings']=observed[0]['analysis']['readings'][:1]
    wrong=grade_interpretation_packet(inputs['cases'],refs,schedule,observed)
    assert wrong['summary']['all']['correct']==3 and not wrong['candidate_screen_passed']
    observed[0]['analysis']['readings']=[dict(bindings=[],events=[])]
    assert not grade_interpretation_packet(inputs['cases'],refs,schedule,observed)['records'][0]['correct']
    observed[0]=dict(id=schedule[0]['id'],error='Declared transport failure')
    failed=grade_interpretation_packet(inputs['cases'],refs,schedule,observed[:1])
    assert failed['summary']['all']==dict(planned=4,attempted=1,completed=0,correct=0,not_attempted=3)
    assert failed['qualified'] is False


def test_predeclared_source_equivalences_allow_heads_punctuation_and_order_not_wrong_targets(packet):
    module,inputs,refs=packet
    entry=inputs['schedule'][0];reference=refs[entry['case']]
    focus=json.loads(module.rows_for(inputs['cases'][entry['case']],entry)[0]['pair'][0])
    actual=deepcopy(reference['expected'])
    for reading in actual['readings']:
        for binding in reading['bindings']:
            binding['event']=reference['span_equivalences'][0][1]
            if binding['role']=='patient':
                binding['span']=reference['span_equivalences'][0][1]
                original=binding['target']['span']
                group=next(g for g in reference['span_equivalences'][1:] if g[0]==original)
                binding['target']['span']=group[-1]
        reading['events'][0]['span']=reference['span_equivalences'][0][1]
        reading['bindings'].reverse()
    actual['readings'].reverse()
    assert compare_interpretation_analysis(actual,reference,focus)['correct']
    actual['readings'][0]['events'][0]['polarity']='affirmed'
    assert not compare_interpretation_analysis(actual,reference,focus)['correct']


def test_bad_reference_aliases_and_unscheduled_observations_are_not_accepted(packet):
    module,inputs,refs=packet
    entry=inputs['schedule'][0];focus=json.loads(module.rows_for(inputs['cases'][entry['case']],entry)[0]['pair'][0])
    reference=deepcopy(refs[entry['case']]);reference['span_equivalences'].append(reference['span_equivalences'][0])
    with pytest.raises(ValueError,match='overlap'):compare_interpretation_analysis(reference['expected'],reference,focus)
    with pytest.raises(ValueError):grade_interpretation_packet(inputs['cases'],refs,inputs['schedule'],[dict(id='not scheduled')])


def test_expanded_budget_reuses_exact_source_packet_and_keeps_a_single_bounded_scope(packet):
    module,inputs,refs=packet
    before=deepcopy((inputs,refs))
    original=module.OUT
    module.expanded_budget()
    assert module.build_packet(json.loads(module.BANK.read_text(encoding='utf-8')))==before
    assert module.OUT!=original and module.BASELINE[0]==original
    assert module.THINKING_TOKENS==2048
    assert module.LIMITS['max_tokens']==3072 and module.LIMITS['max_input']==5120
    assert module.LIMITS['context_length']==8192
    assert module.LIMITS['max_native_attempts']==1 and module.LIMITS['max_metadata_attempts']==12
    assert module.LIMITS['request_timeout_s']==module.LIMITS['operation_timeout_s']==3600
    assert module.LIMITS['max_wall_s']==4200 and module.LIMITS['paid_api_usd']=='0'
    with pytest.raises(ValueError):module.expanded_budget()


def test_convention_probe_keeps_references_and_has_a_separate_bounded_scope(packet):
    module,inputs,refs=packet
    before=deepcopy((inputs,refs));original=module.OUT
    module.annotation_conventions()
    assert module.build_packet(json.loads(module.BANK.read_text(encoding='utf-8')))==before
    assert module.OUT!=original and module.BASELINE[0]==original
    assert not module.EXPANDED and module.CONVENTIONS
    assert module.LIMITS['max_native_attempts']==1 and module.LIMITS['max_metadata_attempts']==12
    assert module.LIMITS['max_tokens']==1536 and module.THINKING_TOKENS==512
    assert module.LIMITS['max_input']+module.LIMITS['max_tokens']==8192
    assert module.LIMITS['request_timeout_s']==module.LIMITS['operation_timeout_s']==1200
    assert module.LIMITS['max_wall_s']==1500 and module.LIMITS['paid_api_usd']=='0'
    with pytest.raises(ValueError):module.expanded_budget()
    with pytest.raises(ValueError):module.annotation_conventions()


@pytest.mark.parametrize('samples,admitted',[
    ([0,30,60],True),([60,60,60],True),([60.1,20,20],False),
    ([20,95,20],False),([-1,20,20],False),([float('nan'),20,20],False),
    ([float('inf'),20,20],False),([True,20,20],False),
])
def test_convention_cpu_admission_precedes_dispatch_and_retains_refusal(packet,tmp_path,monkeypatch,samples,admitted):
    import psutil
    module,_,_=packet;module.annotation_conventions()
    module.OUT=tmp_path;module.RESULT=tmp_path/'result.json'
    calls=[];pending=iter(samples);intervals=[]
    def sample(*,interval):
        intervals.append(interval)
        return next(pending)
    monkeypatch.setattr(psutil,'cpu_percent',sample)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=32*1024**3))
    base=SimpleNamespace(save=lambda p,v:p.write_text(json.dumps(v),encoding='utf-8'))
    monkeypatch.setattr(module,'load_plan',lambda:(SimpleNamespace(base=base),{}))
    monkeypatch.setattr(module,'run_local_membership_diagnostic',lambda **kwargs:calls.append(kwargs))
    module.run()
    assert len(calls)==int(admitted) and intervals==[1,1,1]
    receipt=json.loads((tmp_path/'resource-preflight-001.json').read_text(encoding='utf-8'))
    assert receipt['admitted'] is admitted and receipt['native_attempts']==receipt['metadata_attempts']==0
    assert receipt['policy']==module.CPU_ADMISSION and receipt['paid_api_usd']=='0'
    assert not (tmp_path/'started.json').exists() and not module.RESULT.exists()
    if not admitted:
        original=(tmp_path/'resource-preflight-001.json').read_bytes()
        monkeypatch.setattr(psutil,'cpu_percent',lambda **kwargs:99)
        module.run()
        assert not calls and (tmp_path/'resource-preflight-001.json').read_bytes()==original
        assert (tmp_path/'resource-preflight-002.json').exists()
    module.RESULT.write_text('{}',encoding='utf-8')
    with pytest.raises(FileExistsError):module.run()
    assert not (tmp_path/'resource-preflight-003.json').exists()
