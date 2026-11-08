"""Software-only supervision fixtures; simulated admission is not fresh gold."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from psychology_base.applications.measurement.reference_readings import decode_interpretation_response
from psychology_base.applications.measurement.semantic_membership import membership_inputs
from psychology_base.eval.benchmarks.membership_preparation import (
    _digest, build_source_ledger, interpretation_review_projection,
    prepare_interpretation_bank, interpretation_sft_records,
)
from psychology_base.eval.benchmarks.reference_extraction import authored_interpretation_example
from tests.test_interpretation_contract import CASES


def bank(declared=CASES[0], *, key='software-case', split='train', application='tutor', loss=True):
    """Synthetic ledger/review claims test admission mechanics only."""
    texts=([declared['prior']] if declared['prior'] is not None else [])+[declared['focus']]
    group=hashlib.sha256(key.encode()).hexdigest()[:32]
    case=dict(id=key,source_entry=key,origin='authored',split=split,application=application,
        language=declared['language'],descriptions=['FIRST HIDDEN DESCRIPTION','SECOND HIDDEN DESCRIPTION'],
        focus_id=len(texts)-1,records=[dict(id=i,text=t,speaker=application if i<len(texts)-1 else 'user',
            source_order=dict(group=group,position=i)) for i,t in enumerate(texts)])
    task=interpretation_review_projection(case)[0]
    focus=json.loads(next(r for r in membership_inputs(case,application=application,format_version=4)
        if r['record']==case['focus_id'])['pair'][0])
    analysis,_=authored_interpretation_example(declared,focus)
    reference=dict(case=key,record=task['record'],analysis=analysis,reviewer='software-fixture-reviewer')
    review=dict(case=key,record=task['record'],state='implementer_reviewed',reference_loss=loss,
        reviewers=['software-fixture-reviewer'],independent_review=False,
        rationale='Explicit authored software target, not a model observation.',
        coverage_basis='Use all declared joint readings and preserve explicit unknowns.',
        analysis_sha256=_digest(analysis),source_admission=dict(admitted=True,
            source_basis='Simulated test-only source admission.',attribution_status='Authored software fixture.'),
        **{k:task[k] for k in ('input_sha256','source_occurrence_sha256','protocol_identity')})
    ledger=build_source_ledger([dict(id=key,case_id=key,parents=[],source_keys=['test/'+key],exposure=[])])
    return [case],[reference],[review],ledger


@pytest.mark.parametrize('declared',CASES,ids=lambda c:c['id'])
@pytest.mark.parametrize('application',['tutor','companion'])
def test_full_interpretations_roundtrip_through_trainer_messages_without_losing_readings(declared,application):
    args=bank(declared,application=application);before=deepcopy(args)
    eligible,complete=prepare_interpretation_bank(*args)
    exported=interpretation_sft_records(*args,split='train')
    assert args==before and len(eligible)==len(complete)==len(exported)==1
    messages=exported[0]['messages']
    assert set(exported[0])=={'messages'} and [m['role'] for m in messages]==['system','user','assistant']
    assert all(set(m)=={'role','content'} for m in messages)
    source=json.loads(next(r for r in membership_inputs(args[0][0],application=application,format_version=4)
        if r['record']==args[0][0]['focus_id'])['pair'][0])
    assert decode_interpretation_response(messages[-1]['content'],source)==args[1][0]['analysis']
    assert len(json.loads(messages[-1]['content'])['readings'])==len(declared['targets'])
    assert set(json.loads(messages[-1]['content']))=={'coverage','readings'}
    assert 'HIDDEN DESCRIPTION' not in json.dumps(messages)
    assert 'software-fixture-reviewer' not in json.dumps(messages)
    messages[0]['content']='Changed detached export'
    assert args==before


def test_description_and_query_changes_cannot_change_source_training_inputs():
    args=bank();case=args[0][0]
    before=interpretation_sft_records(*args,split='train')
    case.update(descriptions=['OTHER A','OTHER B'],query='SECRET QUERY',annotation='SECRET GOLD')
    assert interpretation_sft_records(*args,split='train')==before
    assert 'SECRET' not in json.dumps(before)


@pytest.mark.parametrize('damage',[
    'input','occurrence','protocol','target_pin','missing_reference','missing_review','duplicate_reference',
    'duplicate_review','bool_record','state','loss_type','source_admission','source_basis','reviewer',
    'duplicate_reviewer','rationale','coverage_basis','no_target','draft_loss','qualification_review',
])
def test_unpinned_unreviewed_or_incomplete_interpretation_bank_is_rejected(damage):
    cases,refs,reviews,ledger=bank();ref,review=refs[0],reviews[0]
    if damage=='input':review['input_sha256']='0'*64
    elif damage=='occurrence':cases[0]['records'][0]['source_order']['group']='f'*32
    elif damage=='protocol':review['protocol_identity']='0'*64
    elif damage=='target_pin':ref['analysis']['readings']=ref['analysis']['readings'][:1]
    elif damage=='missing_reference':refs.clear()
    elif damage=='missing_review':reviews.clear()
    elif damage=='duplicate_reference':refs.append(deepcopy(ref))
    elif damage=='duplicate_review':reviews.append(deepcopy(review))
    elif damage=='bool_record':ref['record']=True
    elif damage=='state':review['state']='approved'
    elif damage=='loss_type':review['reference_loss']=1
    elif damage=='source_admission':review['source_admission']['admitted']=False
    elif damage=='source_basis':review['source_admission']['source_basis']=''
    elif damage=='reviewer':ref['reviewer']='unrecorded-reviewer'
    elif damage=='duplicate_reviewer':review['reviewers']*=2
    elif damage=='rationale':review['rationale']=''
    elif damage=='coverage_basis':review['coverage_basis']=''
    elif damage=='no_target':
        ref['analysis']=None;review['analysis_sha256']=_digest(None)
    elif damage=='draft_loss':review['state']='draft'
    elif damage=='qualification_review':cases[0]['split']='qualification'
    with pytest.raises(ValueError):prepare_interpretation_bank(cases,refs,reviews,ledger)


@pytest.mark.parametrize('damage',['coverage','partial_implicit','source_id','unknown_target','duplicate_reading','orphan'])
def test_even_re_pinned_targets_must_pass_the_existing_strict_decoder(damage):
    args=bank();raw=args[1][0]['analysis'];row=raw['readings'][0]
    if damage=='coverage':raw['coverage']='open'
    elif damage=='partial_implicit':
        for world in raw['readings']:world['bindings'][1]['span']=world['events'][0]['predicate'][:]
    elif damage=='source_id':raw['source_identity']='0'*64
    elif damage=='unknown_target':row['bindings'][1]['target']['span']=[999,999]
    elif damage=='duplicate_reading':raw['readings'].append(deepcopy(row))
    elif damage=='orphan':row['bindings'][0]['event']=row['events'][0]['predicate'][:]
    args[2][0]['analysis_sha256']=_digest(raw)
    with pytest.raises(ValueError):prepare_interpretation_bank(*args)


def test_prompt_and_decoder_revision_changes_require_new_reviews(monkeypatch):
    from psychology_base.applications.measurement import local_query_binding,reference_readings
    args=bank()
    monkeypatch.setattr(local_query_binding,'INTERPRETATION_PROMPT',local_query_binding.INTERPRETATION_PROMPT+'\nAmendment.')
    with pytest.raises(ValueError,match='stale'):prepare_interpretation_bank(*args)
    monkeypatch.undo()
    monkeypatch.setattr(reference_readings,'INTERPRETATION_VALIDATION_REVISION','changed-review-contract')
    with pytest.raises(ValueError,match='stale'):prepare_interpretation_bank(*args)


@pytest.mark.parametrize('state',['draft','projected_only','unresolved','invalid'])
def test_masked_pending_tasks_stay_in_complete_denominator_without_becoming_sft(state):
    args=bank(loss=False);ref,review=args[1][0],args[2][0]
    ref['analysis']=None;review.update(state=state,analysis_sha256=_digest(None),reviewers=[])
    eligible,complete=prepare_interpretation_bank(*args)
    assert not eligible and len(complete)==1 and complete[0]['analysis'] is None
    assert interpretation_sft_records(*args,split='train')==[]


def test_valid_unsupported_target_can_teach_abstention_with_explicit_review():
    args=bank();args[1][0]['analysis']['coverage']='unsupported'
    args[2][0]['analysis_sha256']=_digest(args[1][0]['analysis'])
    exported=interpretation_sft_records(*args,split='train')
    assert json.loads(exported[0]['messages'][-1]['content'])['coverage']=='unsupported'


def merged(left,right,*,related=False):
    entries=left[3]['entries']+right[3]['entries']
    if related:entries[1]['parents']=[entries[0]['id']]
    return tuple(left[i]+right[i] for i in range(3))+(build_source_ledger(entries),)


@pytest.mark.parametrize('masked',[False,True])
def test_source_only_leakage_is_rejected_despite_different_descriptions_and_review_masks(masked):
    left,right=bank(key='first'),bank(key='second',split='tuning',loss=not masked)
    right[0][0]['descriptions']=['DIFFERENT A','DIFFERENT B']
    args=merged(left,right)
    with pytest.raises(ValueError,match='source-only input crosses'):interpretation_sft_records(*args,split='train')


def test_source_translation_component_is_checked_before_export_filtering():
    left,right=bank(CASES[0],key='first'),bank(CASES[1],key='translated',split='tuning')
    with pytest.raises(ValueError,match='component crosses'):
        interpretation_sft_records(*merged(left,right,related=True),split='train')


def test_order_equivalent_targets_deduplicate_but_conflicting_targets_do_not():
    left,right=bank(key='first'),bank(key='second')
    raw=right[1][0]['analysis'];raw['readings'].reverse()
    for row in raw['readings']:row['bindings'].reverse()
    right[2][0]['analysis_sha256']=_digest(raw)
    args=merged(left,right)
    eligible,complete=prepare_interpretation_bank(*args)
    assert len(eligible)==1 and len(complete)==2 and complete[1]['duplicate_of']==complete[0]['id']
    for row in raw['readings']:row['events'][0]['polarity']='affirmed'
    right[2][0]['analysis_sha256']=_digest(raw)
    with pytest.raises(ValueError,match='conflicting interpretation'):prepare_interpretation_bank(*merged(left,right))


def test_a_masked_duplicate_cannot_suppress_a_later_eligible_target():
    left,right=bank(key='masked',loss=False),bank(key='reviewed')
    eligible,complete=prepare_interpretation_bank(*merged(left,right))
    assert len(eligible)==1 and eligible[0]['case']=='reviewed' and len(complete)==2


@pytest.mark.parametrize('exposure',['regression','diagnostic','reserved','unclassified_prior_use','tuning'])
def test_old_exposure_is_never_relabelled_as_training(exposure):
    args=bank();args[3]['entries'][0]['exposure']=[exposure]
    ledger=build_source_ledger(args[3]['entries'])
    with pytest.raises(ValueError,match='exposure'):prepare_interpretation_bank(*args[:3],ledger)


def test_independent_qualification_remains_evaluation_only():
    args=bank(split='qualification');review=args[2][0]
    review.update(state='adjudicated',independent_review=True,
        reviewers=['software-fixture-reviewer','simulated-independent-reviewer'])
    eligible,complete=prepare_interpretation_bank(*args)
    assert len(eligible)==len(complete)==1
    assert interpretation_sft_records(*args,split='train')==[]
    for split in ('qualification','reserved','regression'):
        with pytest.raises(ValueError,match='train or tuning'):interpretation_sft_records(*args,split=split)


@pytest.mark.parametrize('focus',[False,999])
def test_focus_selection_cannot_invent_or_alias_a_source_occurrence(focus):
    args=bank();args[0][0]['focus_id']=focus
    with pytest.raises(ValueError,match='actual source'):interpretation_review_projection(args[0][0])


def test_omitting_an_explicit_focus_means_reviewing_all_occurrences():
    args=bank();args[0][0].pop('focus_id')
    assert len(interpretation_review_projection(args[0][0]))==2
    with pytest.raises(ValueError,match='Every interpretation occurrence'):prepare_interpretation_bank(*args)


def test_future_source_and_array_order_cannot_change_the_reviewed_focus():
    args=bank();case=args[0][0]
    original=interpretation_sft_records(*args,split='train')
    case['records'].append(dict(id=2,text='SECRET FUTURE CLARIFICATION',speaker='user',
        source_order=dict(group=case['records'][0]['source_order']['group'],position=2)))
    case['records'].reverse()
    assert interpretation_sft_records(*args,split='train')==original


def test_correlated_joint_assignments_are_not_expanded_into_cross_pairs():
    args=bank();raw=args[1][0]['analysis']
    for reading in raw['readings']:
        patient=deepcopy(reading['bindings'][1])
        patient['role']='recipient'
        reading['bindings'].append(patient)
    args[2][0]['analysis_sha256']=_digest(raw)
    output=json.loads(interpretation_sft_records(*args,split='train')[0]['messages'][-1]['content'])
    assert len(output['readings'])==2
    for reading in output['readings']:
        assert reading['bindings'][1]['target']==reading['bindings'][2]['target']


@pytest.mark.parametrize('oversize',['units','bytes'])
def test_source_registry_limits_are_preserved_without_clipping(oversize):
    args=bank();case=args[0][0]
    case['records'][0]['text']='word '*300 if oversize=='units' else 'x'*5000
    with pytest.raises(ValueError):interpretation_review_projection(case)


def producer():
    path=Path(__file__).resolve().parents[1]/'docs/experiments/phase-9-memory-navigation/prepare-interpretation-supervision-v1.py'
    spec=importlib.util.spec_from_file_location('interpretation_supervision_export_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_existing_control_packet_is_complete_and_cannot_become_training():
    module=producer();packet=module.authored_control_packet()
    args=tuple(packet[k] for k in ('cases','references','annotations','ledger'))
    eligible,complete=prepare_interpretation_bank(*args)
    assert len(eligible)==len(complete)==16
    assert {r['split'] for r in complete}=={'regression'}
    assert {r['language'] for r in complete}=={'en','zh'}
    assert interpretation_sft_records(*args,split='train')==interpretation_sft_records(*args,split='tuning')==[]
    assert all(not r['review']['independent_review'] for r in complete)


def test_real_export_command_writes_only_train_targets_and_refuses_overwrite_or_wrong_pin():
    module=producer();left=bank(key='training-software-control')
    right=bank(CASES[1],key='evaluation-software-control',split='qualification')
    right[2][0].update(state='adjudicated',independent_review=True,
        reviewers=['software-fixture-reviewer','simulated-independent-reviewer'])
    args=merged(left,right)
    packet=dict(schema=module.PACKET_SCHEMA,**dict(zip(('cases','references','annotations','ledger'),args)))
    # Keep the real archive's relative paths within its intended workspace.
    with tempfile.TemporaryDirectory(dir=module.ROOT/'.cache',prefix='interpretation-export-test-') as directory:
        workspace=Path(directory).resolve();workspace.relative_to(module.ROOT)
        source=workspace/'packet.json';source.write_text(json.dumps(packet,ensure_ascii=False),encoding='utf-8')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        out,plan=workspace/'out',workspace/'freeze.json'
        command=[sys.executable,str(Path(module.__file__)),'--packet',str(source),'--sha256',digest,
            '--out',str(out),'--plan',str(plan)]
        failed=subprocess.run(command[:-5]+['0'*64,'--out',str(out),'--plan',str(plan)],
            capture_output=True,text=True,timeout=30)
        assert failed.returncode and not out.exists() and not plan.exists()
        result=subprocess.run(command,capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr
        freeze=json.loads(plan.read_text(encoding='utf-8'))
        assert freeze['sft_rows']=={'train':1,'tuning':0} and freeze['model_calls']==freeze['training_steps']==0
        assert freeze['training_ready'] is False
        records=[json.loads(line) for line in (out/'train.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(records)==1 and records==interpretation_sft_records(*args,split='train')
        assert (out/'tuning.jsonl').read_bytes()==b''
        for name,pin in freeze['output_sha256'].items():
            assert hashlib.sha256((out/name).read_bytes()).hexdigest()==pin
        before=plan.read_bytes(),(out/'train.jsonl').read_bytes()
        rerun=subprocess.run(command,capture_output=True,text=True,timeout=30)
        assert rerun.returncode and before==(plan.read_bytes(),(out/'train.jsonl').read_bytes())
