"""Qualification quarantine controls with explicitly authored software fixtures."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.membership_preparation import build_source_ledger, prepare_membership_bank
from tests.test_membership_independent_review import inputs, study as comparison

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('native_qualification_comparison_test_owner',
    ROOT/'docs/experiments/phase-9-memory-navigation/compare-native-qualification-reviews-v1.py')
native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)


def qualification_inputs(two=False):
    cases, refs, annotations, second, ledger, pin = inputs(two=two)
    for c in cases:
        c['split'] = 'qualification'
    entries = deepcopy(ledger['entries'])
    for e in entries:
        e['exposure'] = []
        e['qualification_only'] = True
    return cases, refs, annotations, second, build_source_ledger(entries), pin


def test_qualification_disagreement_retains_complete_masked_source_component():
    args = qualification_inputs(two=True)
    key = args[0][0]['id']
    args[3]['cases'][key]['vectors'][0] = 'NM'
    args[3]['cases'][key]['notes'][0][1] = 'direct_act|Authored software-control disagreement.'
    before = deepcopy(args)
    _, annotations, semantic, complete, report = comparison.compare_reviews(*args)
    assert args == before
    assert semantic == [] and len(complete) == 8
    assert report['directly_blocked_cases'] == [key]
    assert len(report['quarantined_components']) == 1
    assert all(a['state']=='unresolved' and a['independent_review'] and not a['membership_loss'] for a in annotations)
    assert all(len(set(a['reviewers']))==2 for a in annotations)
    assert all(not r['membership_loss'] for r in complete)


def test_qualification_blocking_review_dispute_retains_labels_without_admission():
    args = qualification_inputs()
    args[3]['disputes'] = [dict(case=None, blocking=True, rationale='Authored blocking-contract control.')]
    _, annotations, semantic, complete, report = comparison.compare_reviews(*args)
    assert semantic == [] and len(complete) == 4
    assert report['semantic_label_matches']==report['semantic_label_total']==2
    assert annotations[0]['state']=='unresolved' and not annotations[0]['membership_loss']


def test_qualification_agreement_keeps_qualification_partition_and_independent_gate():
    args = qualification_inputs()
    _, annotations, semantic, complete, report = comparison.compare_reviews(*args)
    assert len(semantic)==2 and len(complete)==4
    # The authored fixture has two structural actor cells in complete references.
    assert all(r['split']=='qualification' for r in semantic)
    assert annotations[0]['state']=='independently_reviewed' and report['admitted_semantic_rows']==2


@pytest.mark.parametrize('change', ['single_reviewer', 'duplicate_reviewer', 'not_independent',
    'loss_enabled', 'missing_dispute', 'blank_reason', 'wrong_dispute_type', 'missing_primary',
    'unreviewed_state', 'changed_pair', 'missing_cell'])
def test_reviewed_qualification_hold_cannot_weaken_review_or_export_guards(change):
    args = qualification_inputs()
    args[3]['disputes'] = [dict(case=None, blocking=True, rationale='Authored hold validation.')]
    _, annotations, _, _, _ = comparison.compare_reviews(*args)
    case, ref, annotation = args[0][0], deepcopy(args[1][0]), annotations[0]
    if change=='single_reviewer': annotation['reviewers']=annotation['reviewers'][:1]
    elif change=='duplicate_reviewer': annotation['reviewers']=[ref['reviewer']]*2
    elif change=='not_independent': annotation['independent_review']=False
    elif change=='loss_enabled': annotation['membership_loss']=True
    elif change=='missing_dispute': annotation.pop('dispute')
    elif change=='blank_reason': annotation['dispute']={'reason':'   '}
    elif change=='wrong_dispute_type': annotation['dispute']='A string cannot establish a review hold.'
    elif change=='missing_primary': annotation['reviewers']=['Invented A','Invented B']
    elif change=='unreviewed_state': annotation['state']='draft'
    elif change=='changed_pair': annotation['pairs'][0]['input_sha256']='stale'
    else: ref['judgments'].pop()
    with pytest.raises(ValueError):
        prepare_membership_bank([case], [ref], [annotation], args[4])


def test_unresolved_first_pass_still_cannot_enter_any_qualification_export():
    cases, refs, annotations, _, ledger, _ = qualification_inputs()
    annotations[0].update(state='unresolved', membership_loss=False,
                          dispute={'reason':'Only one reviewing agent has seen this case.'})
    with pytest.raises(ValueError, match='Independent review'):
        prepare_membership_bank(cases, refs, annotations, ledger)


def packet_for(cases):
    from psychology_base.eval.benchmarks.membership_preparation import review_projection
    return native.study.review_owner.blinded_packet([dict(id=c['id'],rows=review_projection(c)) for c in cases])


def test_native_stage_reuses_comparison_keeps_raw_review_and_corrects_scope_prose():
    cases, refs, annotations, second, ledger, pin = qualification_inputs()
    packet = packet_for(cases)
    before = deepcopy((cases, refs, annotations, second, ledger, packet))
    expected = comparison.compare_reviews(cases, refs, annotations, second, ledger, pin)
    actual = native.compare_packet(cases, refs, annotations, second, ledger, packet, pin)
    assert (cases, refs, annotations, second, ledger, packet) == before
    assert actual[:4] == expected[:4]
    assert {k:v for k,v in actual[4].items() if k!='limits'} == {k:v for k,v in expected[4].items() if k!='limits'}
    assert 'native English companion' in actual[4]['limits']
    assert 'four authored development' not in actual[4]['limits']


@pytest.mark.parametrize('change', ['split', 'version', 'input', 'context', 'case_order',
    'missing_cell', 'source_metadata', 'label_metadata'])
def test_native_stage_rejects_changed_or_unblinded_packet_before_comparison(change):
    cases, refs, annotations, second, ledger, pin = qualification_inputs(two=True)
    packet = packet_for(cases)
    if change=='split': cases[0]['split']='train'
    elif change=='version': packet['input_format_version']=3
    elif change=='input': packet['cases'][0]['rows'][0]['input']=['Changed premise','Changed target']
    elif change=='context': packet['cases'][0]['rows'][0]['context_record_ids']=[999]
    elif change=='case_order': packet['cases'].reverse()
    elif change=='missing_cell': packet['cases'][0]['rows'].pop()
    elif change=='source_metadata': packet['cases'][0]['source']='Not part of the blinded input'
    else: packet['cases'][0]['rows'][0]['expected']='match'
    with pytest.raises(ValueError):
        native.compare_packet(cases, refs, annotations, second, ledger, packet, pin)


def test_native_stage_reports_disagreement_without_erasing_it_or_exporting_its_family():
    args = qualification_inputs(two=True)
    case = args[0][0]
    args[3]['cases'][case['id']]['vectors'][0]='NM'
    args[3]['cases'][case['id']]['notes'][0][1]='direct_act|Authored independent disagreement control.'
    _, annotations, semantic, complete, report = native.compare_packet(*args[:5], packet_for(args[0]), args[5])
    assert semantic == [] and len(complete)==8
    assert len(report['label_disagreements'])==1
    assert all(a['state']=='unresolved' and not a['membership_loss'] for a in annotations)


def process_fixture():
    second = dict(process_disclosure=dict(files_read=[str(native.PARENT/'qualification-blinded-review.json'),
        str(ROOT/'CLAUDE.md')], prior_proposed_labels_seen=False, other_reviews_or_comparison_outcomes_seen=False,
        future_rows_used_as_evidence_for_earlier_focus=False, other_agents_invoked=False,
        web_or_external_model_providers_or_downloads_or_paid_apis_used=False))
    delegation = dict(packet_sha256='packet-pin', fork_turns='none', user_authorized=True,
                      project_provider_calls=0, agent='Separate authored test reviewer')
    completion = dict(raw_review_sha256='raw-pin', agent=delegation['agent'])
    return second, delegation, completion


def test_review_process_requires_exact_handoff_and_exposes_its_attestation_limit():
    args = process_fixture();before=deepcopy(args)
    native.verify_review_process(*args, 'packet-pin', 'raw-pin')
    assert args==before


@pytest.mark.parametrize('change', ['packet', 'raw', 'history_fork', 'not_authorized', 'provider_calls',
    'wrong_agent', 'no_disclosure', 'no_files', 'empty_path', 'missing_packet', 'other_reference',
    'prior_labels', 'comparison_outcomes', 'future_context', 'other_agent', 'remote_calls'])
def test_unestablished_or_contaminated_review_process_cannot_be_accepted(change):
    second, delegation, completion = process_fixture()
    disclosure = second['process_disclosure']
    if change=='packet': delegation['packet_sha256']='wrong'
    elif change=='raw': completion['raw_review_sha256']='changed'
    elif change=='history_fork': delegation['fork_turns']='all'
    elif change=='not_authorized': delegation['user_authorized']=False
    elif change=='provider_calls': delegation['project_provider_calls']=1
    elif change=='wrong_agent': completion['agent']='Different reviewer'
    elif change=='no_disclosure': second.pop('process_disclosure')
    elif change=='no_files': disclosure['files_read']=None
    elif change=='empty_path': disclosure['files_read'].append('')
    elif change=='missing_packet': disclosure['files_read']=[str(ROOT/'CLAUDE.md')]
    elif change=='other_reference': disclosure['files_read'].append(str(ROOT/'docs/project-memory.md'))
    else:
        key={'prior_labels':'prior_proposed_labels_seen','comparison_outcomes':'other_reviews_or_comparison_outcomes_seen',
             'future_context':'future_rows_used_as_evidence_for_earlier_focus','other_agent':'other_agents_invoked',
             'remote_calls':'web_or_external_model_providers_or_downloads_or_paid_apis_used'}[change]
        disclosure[key]=True
    with pytest.raises(ValueError):
        native.verify_review_process(second, delegation, completion, 'packet-pin', 'raw-pin')
