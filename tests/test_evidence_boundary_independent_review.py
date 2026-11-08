"""Authored software guards for independent rule review, not model accuracy."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.membership_preparation import build_source_ledger, review_projection
from tests.test_native_qualification_comparison import qualification_inputs, packet_for, process_fixture
from tests.test_bilingual_membership_preparation import entry

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('independent_boundary_review_test_owner',
    ROOT/'docs/experiments/phase-9-memory-navigation/compare-evidence-boundary-independent-review-v1.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def arguments():
    native, refs, annotations, second, ledger, pin = qualification_inputs()
    control = deepcopy(native[0])
    control.update(id='authored-control', source_entry='authored-control', split='regression')
    control['records'][1]['text'] = 'An explicitly authored software control succeeded.'
    for r in control['records']:
        r['source_order']['group'] = 'f'*32
    ref, annotation = deepcopy(refs[0]), deepcopy(annotations[0])
    ref['id'] = annotation['id'] = control['id']
    for j in ref['judgments']:
        if j['record']==1:
            text = control['records'][1]['text']
            j['evidence'] = [dict(start=0,end=len(text),quote=text)]
    pairs = [{k:r[k] for k in ('slot','record','input_sha256')} for r in review_projection(control)]
    annotation.update(membership_loss=False,pairs=pairs)
    second['cases'][control['id']] = deepcopy(second['cases'][native[0]['id']])
    second['cases'][control['id']]['pairs'] = pairs
    second['rule_review'] = dict(rules=[dict(id=k,verdict='accept',severity='none',
        rationale='Explicit authored software-control rule review.') for k in ('a','b')],
        overall_conclusion='The two authored software-control rules are coherent.')
    ledger = build_source_ledger(ledger['entries']+[entry(control['id'], exposure=['diagnostic'])])
    return native, refs, annotations, [control], [ref], [annotation], second, ledger, packet_for(native+[control]), pin, {'a','b'}


def test_agreeing_rules_controls_and_native_reference_use_existing_export_without_mutating_input():
    args = arguments(); before = deepcopy(args)
    independent, annotations, semantic, complete, report, gate, controls = stage.evaluate(*args)
    assert args == before and gate['clarification_check_passed']
    assert len(semantic)==2 and len(complete)==4
    assert annotations[0]['state']=='independently_reviewed'
    assert controls['comparison']['semantic_label_matches']==2
    assert all(not r['membership_loss'] for r in controls['complete_pairs'])


def test_blocking_rule_concern_holds_native_even_when_raw_dispute_list_is_empty():
    args = arguments()
    args[6]['rule_review']['rules'][0].update(verdict='concern',severity='blocking')
    _, annotations, semantic, complete, report, gate, _ = stage.evaluate(*args)
    assert not gate['clarification_check_passed'] and not semantic and len(complete)==4
    assert annotations[0]['state']=='unresolved' and not annotations[0]['membership_loss']
    assert report['reviewer_disputes'][-1]['blocking']


def test_control_label_disagreement_cannot_pass_rule_gate_or_admit_native_reference():
    args = arguments()
    record = args[6]['cases']['authored-control']
    record['vectors'][0]='NM'
    record['notes'][0][1]='direct_act|Authored review disagreement for guard verification.'
    _, _, semantic, _, native_report, gate, controls = stage.evaluate(*args)
    assert not gate['clarification_check_passed'] and not gate['control_labels_agree']
    assert len(controls['comparison']['label_disagreements'])==1 and not semantic
    assert native_report['semantic_label_matches']==native_report['semantic_label_total']==2


def test_nonblocking_rule_concern_is_preserved_without_inventing_label_disagreement():
    args = arguments()
    args[6]['rule_review']['rules'][0].update(verdict='concern',severity='nonblocking')
    *_, gate, controls = stage.evaluate(*args)
    assert gate['clarification_check_passed'] and gate['rule_review']['concerns'][0]['severity']=='nonblocking'


@pytest.mark.parametrize('change', ['missing_rule','duplicate_rule','unknown_rule','missing_conclusion',
    'blank_reason','accept_blocking','concern_none','missing_review','missing_case',
    'changed_pair','source_metadata','unknown_dispute','bad_dispute_flag','bad_dispute_type'])
def test_malformed_or_leaking_review_cannot_establish_clarification_admission(change):
    args = arguments(); raw=args[6]
    if change=='missing_rule': raw['rule_review']['rules'].pop()
    elif change=='duplicate_rule': raw['rule_review']['rules'].append(deepcopy(raw['rule_review']['rules'][0]))
    elif change=='unknown_rule': raw['rule_review']['rules'][0]['id']='invented'
    elif change=='missing_conclusion': raw['rule_review'].pop('overall_conclusion')
    elif change=='blank_reason': raw['rule_review']['rules'][0]['rationale']=' '
    elif change=='accept_blocking': raw['rule_review']['rules'][0]['severity']='blocking'
    elif change=='concern_none': raw['rule_review']['rules'][0]['verdict']='concern'
    elif change=='missing_review': raw.pop('rule_review')
    elif change=='missing_case': raw['cases'].pop('authored-control')
    elif change=='changed_pair': args[8]['cases'][0]['rows'][0]['input']=['changed','changed']
    elif change=='source_metadata': args[8]['cases'][0]['source']='Labels must not be inferred from source'
    elif change=='unknown_dispute': raw['disputes']=[dict(case='unknown',blocking=True,rationale='Unknown scope')]
    elif change=='bad_dispute_flag': raw['disputes']=[dict(case=None,blocking='yes',rationale='Invalid flag')]
    else: raw['disputes']=['Invalid dispute']
    with pytest.raises(ValueError): stage.evaluate(*args)


def test_native_label_difference_still_holds_component_after_rules_and_controls_pass():
    args = arguments()
    key=args[0][0]['id']
    args[6]['cases'][key]['vectors'][0]='NM'
    args[6]['cases'][key]['notes'][0][1]='direct_act|Authored native-review disagreement.'
    _, annotations, semantic, _, report, gate, _ = stage.evaluate(*args)
    assert gate['clarification_check_passed'] and not semantic
    assert report['directly_blocked_cases']==[key] and annotations[0]['state']=='unresolved'


def test_new_read_scope_uses_exact_handoff_and_rejects_status_or_other_packets():
    raw, delegation, completion = process_fixture()
    disclosure=raw['process_disclosure']
    disclosure['files_read']=[str(stage.HANDOFF),str(ROOT/'CLAUDE.md')]
    stage.process_owner.verify_review_process(raw,delegation,completion,'packet-pin','raw-pin')
    for bad in (stage.prep.OUT/'inputs.json',ROOT/'docs/project-memory.md',stage.adjudication.OUT/'references.json'):
        disclosure['files_read']=[str(stage.HANDOFF),str(bad)]
        with pytest.raises(ValueError,match='blinded packet scope'):
            stage.process_owner.verify_review_process(raw,delegation,completion,'packet-pin','raw-pin')


def test_selection_is_label_free_and_covers_every_bilingual_condition_once():
    # This handoff is consumed. Verify its real archived inputs and source pins;
    # rebuilding it invokes the fresh-execution guard against a changing HEAD.
    # The production guard remains unchanged and must continue rejecting drift.
    fields=stage.prep.read(stage.prep.PLAN)
    stage.prep.study.old.verify_parent(stage.prep.OUT,fields,stage.prep._digest)
    for name,pin in fields['source'].items():
        assert stage.prep.sha(stage.prep.OUT/'executed-source'/name)==pin
    cases=stage.prep.read(stage.prep.OUT/'inputs.json')
    packet=stage.prep.read(stage.prep.OUT/'blinded-review.json')
    controls=[c for c in cases if c['split']=='regression']
    assert fields['native_cases']==13 and len(controls)==18 and fields['complete_cells']==186
    assert len({(c['family'],c['variant'],c['language']) for c in controls})==18
    assert sum(c['application']=='tutor' for c in controls)==9
    assert all(set(c)=={'id','rows'} for c in packet['cases'])
    assert all(set(r)=={'slot','record','input_sha256','context_record_ids','input'}
        for c in packet['cases'] for r in c['rows'])
