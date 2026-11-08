"""Software audit guards with authored controls, not semantic accuracy tests."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.membership_preparation import prepare_membership_bank
from tests.test_membership_independent_review import inputs

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('development_evidence_audit_test_owner',
    ROOT/'docs/experiments/phase-9-memory-navigation/audit-development-evidence-v1.py')
stage=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def fixture():
    cases,refs,annotations,_,ledger,_=inputs(two=True)
    annotations[1].update(state='unresolved',membership_loss=False,
        dispute=dict(reason='Explicit authored software-control hold.'))
    rows,complete=prepare_membership_bank(cases,refs,annotations,ledger)
    exported={r['id'] for r in rows};pins={'baseline':'frozen-software-control'};rules={'a','b'}
    manual=dict(pins=pins,independent_of_preparer=False,blinded_to_baseline_labels=False,
        audit_type='unblinded_primary_consistency_review',reviewer='Disclosed software-fixture auditor',
        case_reviews={c['id']:dict(disposition='retain',rationale='An authored audit-guard test.') for c in cases},
        semantic_reviews=[dict(id=r['id'],input_sha256=r['input_sha256'],before_label=r['label'],
            after_label=r['label'],baseline_exported=r['id'] in exported,
            disposition='retain' if r['membership_loss'] else 'retain_existing_hold',
            assessed_rules=sorted(rules),rationale='An explicit authored metadata retention control.')
            for r in complete if r['constraint'] is None])
    return complete,manual,pins,{c['id'] for c in cases},rules,exported


def test_audit_covers_exported_and_held_semantic_cells_without_mutating_them():
    args=fixture();before=deepcopy(args)
    result=stage.validate_audit(*args)
    assert args==before and len(result)==4
    assert sum(r['baseline_exported'] for r in result)==2
    assert sum(r['disposition']=='retain_existing_hold' for r in result)==2


@pytest.mark.parametrize('change',['pins','independence','blind','reviewer','missing_case','case_default',
    'missing_row','duplicate_row','extra_structural','stale_input','before','after','release_hold',
    'export_flag','bad_flag_type','missing_rule','blank_reason','blank_case_reason'])
def test_incomplete_or_misrepresented_audit_cannot_certify_a_bank(change):
    complete,manual,pins,cases,rules,exported=fixture()
    record=manual['semantic_reviews'][0]
    held=next(r for r in manual['semantic_reviews'] if r['disposition']=='retain_existing_hold')
    if change=='pins':manual['pins']={'baseline':'changed'}
    elif change=='independence':manual['independent_of_preparer']=True
    elif change=='blind':manual['blinded_to_baseline_labels']=True
    elif change=='reviewer':manual['reviewer']=' '
    elif change=='missing_case':manual['case_reviews'].pop(next(iter(cases)))
    elif change=='case_default':manual['case_reviews'][next(iter(cases))]['disposition']='unreviewed'
    elif change=='missing_row':manual['semantic_reviews'].pop()
    elif change=='duplicate_row':manual['semantic_reviews'][-1]=deepcopy(record)
    elif change=='extra_structural':manual['semantic_reviews'].append(deepcopy(next(r for r in complete if r['constraint'])))
    elif change=='stale_input':record['input_sha256']='stale'
    elif change=='before':record['before_label']='uncertain'
    elif change=='after':record['after_label']='uncertain'
    elif change=='release_hold':held['disposition']='retain'
    elif change=='export_flag':held['baseline_exported']=True
    elif change=='bad_flag_type':record['baseline_exported']=1
    elif change=='missing_rule':record['assessed_rules']=[]
    elif change=='blank_reason':record['rationale']=' '
    else:manual['case_reviews'][next(iter(cases))]['rationale']=' '
    with pytest.raises(ValueError):stage.validate_audit(complete,manual,pins,cases,rules,exported)


def test_export_identity_is_not_inferred_from_the_membership_loss_flag():
    complete,manual,pins,cases,rules,exported=fixture()
    # An exact-input duplicate can retain loss eligibility without exporting
    # another semantic row. The canonical export ID set remains authoritative.
    key=next(iter(exported));exported=exported-{key}
    record=next(r for r in manual['semantic_reviews'] if r['id']==key)
    record['baseline_exported']=False
    assert stage.validate_audit(complete,manual,pins,cases,rules,exported)
