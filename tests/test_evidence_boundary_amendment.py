"""Explicit software fixtures for scoped reviewer amendments; no accuracy claim."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.membership_preparation import compile_membership_review
from tests.test_evidence_boundary_independent_review import arguments

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('boundary_amendment_test_owner',
    ROOT/'docs/experiments/phase-9-memory-navigation/apply-evidence-boundary-amendment-v1.py')
stage=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def fixture():
    args=arguments()
    raw,packet=args[6],args[8]
    key=args[0][0]['id']; cell=(key,0,1)
    pair=next(r for c in packet['cases'] if c['id']==key for r in c['rows'] if (r['slot'],r['record'])==(0,1))
    judgment=deepcopy(next(j for j in args[1][0]['judgments'] if (j['slot'],j['record'])==(0,1)))
    judgment.update(label='uncertain',reason='missing_reference',rationale='Explicit authored software amendment control.')
    pins=dict(packet_sha256='packet',first_pass_sha256='first',interpretation_note_sha256='note')
    amendment=dict(reviewer=raw['reviewer'],**pins,all_other_first_pass_cells_retained=True,
        amendments=[dict(case=key,slot=0,record=1,input_sha256=pair['input_sha256'],
            before='nonmatch',after='uncertain',judgment=judgment)])
    return args,raw,amendment,packet,pins,{cell}


def test_amendment_preserves_raw_review_and_changes_only_the_named_cell():
    args,raw,amendment,packet,pins,scope=fixture()
    before=deepcopy((raw,amendment,packet))
    revised=stage.apply_amendment(raw,amendment,packet,pins,scope)
    assert (raw,amendment,packet)==before
    key=args[0][0]['id']
    assert raw['cases'][key]['vectors'][0]=='NN' and revised['cases'][key]['vectors'][0]=='NU'
    assert revised['cases']['authored-control']==raw['cases']['authored-control']
    assert revised['post_first_pass_amendment'] and not revised['fresh_blind_sample']
    assert revised['cases'][key]['pairs']==raw['cases'][key]['pairs']


@pytest.mark.parametrize('change',['reviewer','packet','first','note','retain','scope','duplicate',
    'before','same_after','input','missing_judgment','label_mismatch','float_record','bool_slot'])
def test_unpinned_or_out_of_scope_amendment_cannot_be_applied(change):
    _,raw,amendment,packet,pins,scope=fixture()
    item=amendment['amendments'][0]
    if change=='reviewer':amendment['reviewer']='Different reviewer'
    elif change in ('packet','first','note'):
        field={'packet':'packet_sha256','first':'first_pass_sha256','note':'interpretation_note_sha256'}[change]
        amendment[field]='stale'
    elif change=='retain':amendment['all_other_first_pass_cells_retained']=False
    elif change=='scope':scope=set()
    elif change=='duplicate':amendment['amendments']*=2
    elif change=='before':item['before']='match'
    elif change=='same_after':item['after']='nonmatch'
    elif change=='input':item['input_sha256']='stale'
    elif change=='missing_judgment':item.pop('judgment')
    elif change=='float_record':item['record']=1.0
    elif change=='bool_slot':item['slot']=False
    else:item['judgment']['label']='match'
    with pytest.raises(ValueError):stage.apply_amendment(raw,amendment,packet,pins,scope)


def test_existing_compiler_owns_exact_evidence_and_rejects_corrupted_amendment_span():
    args,raw,amendment,packet,pins,scope=fixture()
    revised=stage.apply_amendment(raw,amendment,packet,pins,scope)
    case=args[0][0]
    selected=deepcopy(revised);selected['cases']={case['id']:selected['cases'][case['id']]}
    metadata={case['id']:dict(admitted=True,attribution_status='authored_software_control',source_basis='Unit guard only.')}
    refs,_=compile_membership_review([case],selected,metadata)
    stage.verify_amended_judgments(refs,amendment)
    amendment['amendments'][0]['judgment']['evidence'][0]['quote']='Corrupted quote'
    with pytest.raises(ValueError,match='canonical source/span'):
        stage.verify_amended_judgments(refs,amendment)


def process():
    return dict(process_disclosure=dict(files_read=[str(stage.first.HANDOFF)],own_prior_judgments_seen=True,
        new_blind_sample=False,first_pass_modified=False,interpretation_note_modified=False,
        preparer_proposed_labels_seen=False,other_reviews_or_comparison_outcomes_seen=False,
        future_rows_used_as_evidence_for_earlier_focus=False,other_agents_invoked=False,
        web_or_external_model_providers_or_downloads_or_paid_apis_used=False))


def test_post_first_pass_scope_is_disclosed_without_pretending_a_new_blind_sample():
    stage.verify_amendment_process(process())
    for change in ('read_status','new_blind_sample','preparer_proposed_labels_seen','first_pass_modified'):
        amendment=process();d=amendment['process_disclosure']
        if change=='read_status':d['files_read'].append(str(ROOT/'docs/project-memory.md'))
        else:d[change]=True
        with pytest.raises(ValueError):stage.verify_amendment_process(amendment)
