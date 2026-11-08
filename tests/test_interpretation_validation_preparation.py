"""Offline packet integrity; these checks do not validate semantic references."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import socket
from unittest.mock import patch

import pytest

from psychology_base.eval.benchmarks.membership_preparation import (
    _digest, interpretation_review_projection, interpretation_sft_records,
    prepare_interpretation_bank, validate_membership_partitions,
)

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'docs/experiments/phase-9-memory-navigation/prepare-interpretation-validation-v1.py'
spec = importlib.util.spec_from_file_location('interpretation_validation_preparation_test_owner', PATH)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


@pytest.fixture(scope='module')
def prepared():
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')), \
         patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('Network forbidden')), \
         patch.object(socket, 'create_connection', side_effect=AssertionError('Network forbidden')):
        return owner.build()


def bank_args(artifacts):
    b = artifacts['pending-bank.json']
    return [b[k] for k in ['cases', 'references', 'annotations', 'ledger']]


def test_selection_uses_metadata_and_keeps_each_source_stratum(prepared):
    pool, _, ledger, _ = owner.source_pool()
    selected = owner.select_native(pool, ledger)
    changed = deepcopy(pool)
    changed.reverse()
    for case in changed:
        case['descriptions'] = ['HIDDEN NEW DESCRIPTION A', 'HIDDEN NEW DESCRIPTION B']
        case['query'] = 'HIDDEN QUERY'
        for record in case['records']:
            record['text'] = 'Source text changed without changing source identity.'
    assert owner.select_native(changed, ledger) == selected == prepared[0]['source-selection.json']
    assert [len(s['selected']) for s in selected['strata']] == [2, 2, 2, 2]
    assert sum(s['candidates'] for s in selected['strata']) == 25
    assert selected['redraw_permitted'] is False


def test_source_shortage_fails_instead_of_redrawing():
    pool, _, ledger, _ = owner.source_pool()
    with pytest.raises(ValueError, match='quota'):
        owner.select_native(pool, ledger, quota=3)


def test_pending_targets_cannot_enter_evaluation_loss_or_supervision(prepared):
    args = bank_args(prepared[0])
    eligible, complete = prepare_interpretation_bank(*args)
    assert not eligible and len(complete) == 32
    assert all(r['analysis'] is None and not r['reference_loss'] for r in complete)
    assert all(r['review']['state'] == 'draft' and r['review']['reviewers'] == []
               and not r['review']['independent_review'] for r in complete)
    assert all(not interpretation_sft_records(*args, split=s) for s in ['train', 'tuning'])
    with pytest.raises(ValueError, match='train or tuning'):
        interpretation_sft_records(*args, split='qualification')
    damaged = deepcopy(args)
    damaged[2][0]['reference_loss'] = True
    with pytest.raises(ValueError, match='Independent interpretation review'):
        prepare_interpretation_bank(*damaged)
    damaged = deepcopy(args)
    next(a for a in damaged[2] if a['case'].startswith('validation-authored-'))['reference_loss'] = True
    with pytest.raises(ValueError, match='Unreviewed'):
        prepare_interpretation_bank(*damaged)


@pytest.mark.parametrize('origin', ['public-source', 'authored'])
@pytest.mark.parametrize('partition', ['train', 'tuning'])
def test_source_ancestry_blocks_training_and_tuning_repartition(prepared, origin, partition):
    cases, _, _, ledger = deepcopy(bank_args(prepared[0]))
    case = next(c for c in cases if (c['origin'] == 'authored') == (origin == 'authored'))
    case['split'] = partition
    with pytest.raises(ValueError, match='exposure'):
        validate_membership_partitions(cases, ledger)


def test_native_views_preserve_occurrences_and_exclude_future_turns(prepared):
    pool, _, _, _ = owner.source_pool()
    originals = {c['id']: c for c in pool}
    native = [c for c in prepared[0]['pending-bank.json']['cases'] if c['split'] == 'qualification']
    saw_trailing_turn = False
    for case in native:
        assert case['records'] == originals[case['id']]['records']
        assert 'phenomena' not in case and case['descriptions'] == owner.NEUTRAL_DESCRIPTIONS
        task, = interpretation_review_projection(case)
        positions = {r['id']: r['source_order']['position'] for r in case['records']}
        assert all(positions[rid] < positions[task['record']] for rid in task['context_record_ids'])
        future = [rid for rid, pos in positions.items() if pos > positions[task['record']]]
        assert not set(future) & set(task['context_record_ids'])
        saw_trailing_turn |= bool(future)
    assert saw_trailing_turn
    assert prepared[1]['native_source_records_replayed'] == 24


def test_blind_review_excludes_intentions_old_labels_and_model_answers(prepared):
    artifacts = prepared[0]
    blind = artifacts['blind-review-packet.json']
    assert not blind['interpretation_targets_supplied'] and not blind['independent_review_completed']
    forbidden = ['intended_antecedent_condition', 'intended_realization', 'expected', 'analysis',
                 'descriptions', 'phenomena', 'prediction', 'completion', 'historical_grade']
    for task in blind['tasks']:
        assert not set(task) & set(forbidden)
        serialized = json.dumps(task)
        assert all('"' + key + '"' not in serialized for key in forbidden)
    for slot in ['A', 'B']:
        template = artifacts[f'reviewer-{slot}-template.json']
        assert template['reviewer'] is None and template['required_independence']
        assert len(template['tasks']) == len(blind['tasks'])
        assert all(t['analysis'] is None and t['rationale'] is None and t['state'] == 'pending'
                   for t in template['tasks'])
    conditions = artifacts['authored-review-intentions.json']
    assert len({(c['language'], c['application'], c['intended_antecedent_condition'], c['intended_realization'])
                for c in conditions}) == 24
    assert all(c['interpretation_reference'] is None for c in conditions)


def test_prospective_payload_retains_fixed_v2_prefix_without_dispatch(prepared):
    candidate = owner.read(owner.CANDIDATE)
    prefix = candidate['requests'][0]['messages'][:-1]
    requests = prepared[0]['prospective-v2-requests.json']['requests']
    assert len(requests) == 32
    assert len({r['full_messages_sha256'] for r in requests}) == 32
    for row in requests:
        assert len(row['messages']) == 8 and row['messages'][:-1] == prefix
        assert row['prefix_sha256'] == _digest(prefix)
        assert row['full_messages_sha256'] == _digest(row['messages'])
        assert row['fits_input_limit'] and not row['dispatch_permitted'] and not row['allowance_created']
    changed = deepcopy(candidate)
    changed['requests'][1]['messages'][0]['content'] += 'changed'
    task = dict(case='test', record=0, messages=[{}, {'role': 'user', 'content': 'source'}])
    with pytest.raises(ValueError, match='prefix'):
        owner.prospective_request(task, changed)


def test_overbound_source_is_retained_as_hold_without_truncation():
    source = {'role': 'user', 'content': 'x' * 20000}
    task = dict(case='overbound', record=7, messages=[{}, source])
    row = owner.prospective_request(task, owner.read(owner.CANDIDATE))
    assert row['messages'][-1] == source
    assert row['status'] == 'held_input_bound' and not row['fits_input_limit']
    assert not row['dispatch_permitted'] and not row['allowance_created']


def test_pinned_input_rejects_changed_bytes(tmp_path):
    path = tmp_path / 'input.json'
    path.write_text('{"source": 1}', encoding='utf-8')
    pin = owner.sha(path)
    assert owner.pinned(path, pin) == {'source': 1}
    path.write_text('{"source": 2}', encoding='utf-8')
    with pytest.raises(ValueError, match='Pinned input changed'):
        owner.pinned(path, pin)
