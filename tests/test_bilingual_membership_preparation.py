"""Disclosed authored software fixtures; these do not measure model accuracy."""
from copy import deepcopy
import hashlib
import json

import pytest

from psychology_base.applications.measurement.semantic_membership import membership_inputs
from psychology_base.eval.benchmarks.membership_preparation import (
    build_source_ledger, pinned_json, prepare_membership_bank, recover_authored_coordinates,
    recover_public_coordinates, review_projection,
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_fixture(tmp_path, kind='kdconv'):
    texts = ['讨论小林。', '她在学校工作。', '她教什么？', '她教数学。', '她喜欢教学。', '为什么？']
    if kind == 'mathdial':
        texts = ['What is two plus two?', 'Four.', 'How did you calculate it?',
                 'I added two to two.', 'How confident are you?', 'I am confident.']
    records = [dict(id=i, speaker=None if kind == 'kdconv' else ('tutor' if i % 2 == 0 else 'user'), text=t)
               for i, t in enumerate(texts)]
    raw = ([dict(name='test-topic', messages=[dict(message=t) for t in texts])] if kind == 'kdconv' else
           [dict(conversation_id='test-conversation', turns=[dict(role='teacher' if i % 2 == 0 else 'student',
                                                                utterance=t) for i, t in enumerate(texts)])])
    path = tmp_path / 'corpus.json'
    digest = write_json(path, raw)
    case = dict(id='authored-test-source', source='test/corpus.json', row=0,
                identity='test-topic' if kind == 'kdconv' else 'test-conversation', origin='public-source',
                language='zh' if kind == 'kdconv' else 'en', application='tutor',
                descriptions=['询问教学科目', '表达信心'] if kind == 'kdconv' else
                             ["the learner's first calculation", "the learner's expression of confidence"],
                records=records)
    return case, dict(corpus_path=path, corpus_sha256=digest, corpus_kind=kind, imported=deepcopy(case))


@pytest.mark.parametrize('kind', ['kdconv', 'mathdial'])
def test_public_recovery_uses_verified_occurrences_and_canonical_projection(tmp_path, kind):
    case, kwargs = public_fixture(tmp_path, kind)
    before = deepcopy(case)
    recovered = recover_public_coordinates(case, **kwargs)
    assert case == before and 'source_order' not in case['records'][0]
    case['records'].reverse()
    reversed_case = recover_public_coordinates(case, **kwargs)
    assert {r['id']: r['source_order'] for r in recovered['records']} == {
        r['id']: r['source_order'] for r in reversed_case['records']}
    rows = membership_inputs(recovered, application='tutor', format_version=4)
    last = next(r for r in rows if r['slot'] == 0 and r['record'] == 5)
    assert last['context_record_ids'] == (3, 4)
    assert len({r['source_order']['group'] for r in recovered['records']}) == 1
    if kind == 'kdconv':
        assert all(r['speaker'] is None for r in recovered['records'])
        assert json.loads(last['pair'][0])['focus_utterance']['speaker'] == 'unknown'
    else:
        assert [r['speaker'] for r in recovered['records']] == ['tutor', 'user'] * 3


@pytest.mark.parametrize('change', ['text', 'speaker', 'identity', 'row', 'missing', 'duplicate',
                                   'bool_id', 'import_order', 'import_text', 'coordinate', 'unknown_role'])
def test_public_recovery_refuses_unverified_or_incomplete_source(tmp_path, change):
    case, kwargs = public_fixture(tmp_path, 'mathdial')
    if change == 'text': case['records'][2]['text'] = 'Invented text'
    elif change == 'speaker': case['records'][0]['speaker'] = None
    elif change == 'identity': case['identity'] = 'wrong'
    elif change == 'row': case['row'] = True
    elif change == 'missing': case['records'].pop()
    elif change == 'duplicate': case['records'][1]['id'] = 0
    elif change == 'bool_id': case['records'][1]['id'] = True
    elif change == 'import_order': kwargs['imported']['records'].reverse()
    elif change == 'import_text': kwargs['imported']['records'][0]['text'] = 'Invented import'
    elif change == 'coordinate': case['records'][0]['source_order'] = dict(group='a' * 32, position=0)
    elif change == 'unknown_role':
        raw = json.loads(kwargs['corpus_path'].read_text())
        raw[0]['turns'][0]['role'] = 'moderator'
        kwargs['corpus_sha256'] = write_json(kwargs['corpus_path'], raw)
    with pytest.raises(ValueError): recover_public_coordinates(case, **kwargs)


def test_raw_snapshot_digest_and_identity_are_checked(tmp_path):
    case, kwargs = public_fixture(tmp_path)
    kwargs['corpus_path'].write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='Pinned source changed'): recover_public_coordinates(case, **kwargs)
    kwargs['corpus_sha256'] = write_json(kwargs['corpus_path'], [dict(name='changed', messages=[])])
    with pytest.raises(ValueError, match='identity'): recover_public_coordinates(case, **kwargs)


def test_jsonl_snapshot_and_repeated_occurrences_keep_distinct_positions(tmp_path):
    case, kwargs = public_fixture(tmp_path)
    case['records'][1]['text'] = case['records'][0]['text']
    kwargs['imported'] = deepcopy(case)
    raw = [dict(name=case['identity'], messages=[dict(message=r['text']) for r in case['records']])]
    path = tmp_path / 'corpus.jsonl'
    kwargs.update(corpus_path=path, corpus_sha256=write_json(path, raw[0]))
    result = recover_public_coordinates(case, **kwargs)
    assert result['records'][0]['text'] == result['records'][1]['text']
    assert result['records'][0]['source_order']['position'] == 0
    assert result['records'][1]['source_order']['position'] == 1


def test_authored_contrasts_require_a_separate_frozen_chronology_declaration(tmp_path):
    case, _ = public_fixture(tmp_path)
    case.update(origin='authored', id='declared-zh')
    path = tmp_path / 'declaration.json'
    old_array = dict(schema='membership-training-authored-v1', families=[case['records']])
    digest = write_json(path, old_array)
    with pytest.raises(ValueError, match='explicit authored chronology'):
        recover_authored_coordinates(case, declaration_path=path, declaration_sha256=digest)
    declaration = dict(schema='authored-membership-sequences-v1', sequences=[dict(
        id=case['id'], language='zh', author='Authored structural-test fixture', chronology='authored',
        records=deepcopy(case['records']))])
    digest = write_json(path, declaration)
    result = recover_authored_coordinates(case, declaration_path=path, declaration_sha256=digest)
    case['records'].reverse()
    assert sorted(result['records'], key=lambda r: r['id']) == sorted(
        recover_authored_coordinates(case, declaration_path=path, declaration_sha256=digest)['records'],
        key=lambda r: r['id'])
    declaration['sequences'][0]['author'] = ''
    digest = write_json(path, declaration)
    with pytest.raises(ValueError, match='author and language'):
        recover_authored_coordinates(case, declaration_path=path, declaration_sha256=digest)


def entry(key, *, keys=None, parents=None, exposure=None):
    return dict(id=key, case_id=key, source_keys=keys or ['source/' + key], parents=parents or [], exposure=exposure or [])


def test_fresh_qualification_lock_propagates_to_every_source_derivative():
    source = entry('locked', keys=['dialogue/held'])
    source['qualification_only'] = True
    derivative = entry('child', parents=['locked'])
    alternate = entry('other-window', keys=['dialogue/held'])
    ledger = build_source_ledger([source, derivative, alternate, entry('fresh')])
    groups = {key: c for c in ledger['components'] for key in c['members']}
    assert groups['locked']['allowed_partitions'] == ['qualification', 'regression']
    assert groups['child'] == groups['other-window'] == groups['locked']
    assert 'train' in groups['fresh']['allowed_partitions']
    assert source['qualification_only'] is True and 'qualification_only' not in derivative


@pytest.mark.parametrize('exposure', ['train', 'tuning', 'reserved', 'regression', 'diagnostic', 'unclassified_prior_use'])
def test_qualification_lock_never_overrides_prior_exposure(exposure):
    source = entry('locked', keys=['dialogue/held'])
    source['qualification_only'] = True
    old = entry('old', keys=['dialogue/held'], exposure=[exposure])
    ledger = build_source_ledger([source, old])
    assert ledger['components'][0]['allowed_partitions'] == ['regression']


@pytest.mark.parametrize('flag', ['true', 1, 0, None, [], {}])
def test_qualification_lock_requires_an_explicit_boolean(flag):
    source = entry('locked')
    source['qualification_only'] = flag
    with pytest.raises(ValueError, match='exposure history'):
        build_source_ledger([source])


def test_components_close_transitively_over_sources_translations_and_derivatives():
    entries = [entry('original', keys=['episode/one']),
               entry('translation', keys=['episode/two'], parents=['original']),
               entry('new-window', keys=['episode/two']),
               entry('unrelated', keys=['episode/three'])]
    before = deepcopy(entries)
    ledger = build_source_ledger(entries)
    assert entries == before
    assert ledger == build_source_ledger(entries[::-1])
    assert sorted(len(c['members']) for c in ledger['components']) == [1, 3]
    assert next(c for c in ledger['components'] if 'original' in c['members'])['members'] == [
        'new-window', 'original', 'translation']


@pytest.mark.parametrize('exposure,allowed', [
    ([], {'train', 'tuning', 'qualification', 'regression', 'reserved'}),
    (['train'], {'train', 'regression'}), (['tuning'], {'tuning', 'regression'}),
    (['train', 'tuning'], {'tuning', 'regression'}), (['reserved'], {'reserved', 'regression'}),
    (['reserved', 'train'], {'regression'}), (['diagnostic'], {'regression'}),
    (['regression'], {'regression'}), (['unclassified_prior_use'], {'regression'}),
])
def test_exposure_propagates_to_the_entire_component(exposure, allowed):
    ledger = build_source_ledger([entry('one', exposure=exposure), entry('child', parents=['one'])])
    assert set(ledger['components'][0]['allowed_partitions']) == allowed


@pytest.mark.parametrize('entries', [
    [entry('one'), entry('one')], [entry('one', parents=['absent'])],
    [entry('one', parents=['one'])], [entry('one', parents=['two']), entry('two', parents=['one'])],
    [entry('one', exposure=['fresh'])], [dict(id='one', parents=[], exposure=[], source_keys=[])],
])
def test_invalid_or_cyclic_source_history_is_rejected(entries):
    with pytest.raises(ValueError): build_source_ledger(entries)


def annotated_case(key='example', application='tutor', split='train'):
    case = dict(id=key, language='en', origin='authored', application=application, split=split,
                source_entry=key, descriptions=['my question', 'my statement'], records=[
                    dict(id=0, text='What happened?', speaker='user', source_order=dict(group='a' * 32, position=0)),
                    dict(id=1, text='It worked.', speaker='user', source_order=dict(group='a' * 32, position=1))])
    # Explicit structural-test annotations, not independent gold labels.
    reference = dict(id=key, contract_version=4, reviewed=True, reviewer='structural-fixture-author', judgments=[
        dict(slot=s, record=r['id'], label='nonmatch', reason='different_act',
             evidence=[dict(start=0, end=len(r['text']), quote=r['text'])], context_evidence=[],
             rationale='Inert reference for software-boundary tests only.')
        for s in range(2) for r in case['records']])
    review = dict(id=key, state='implementer_reviewed', reviewers=['structural-fixture-author'],
                  membership_loss=True,
                  pairs=[{k: row[k] for k in ('slot', 'record', 'input_sha256')} for row in review_projection(case)])
    return case, reference, review


def test_mixed_applications_share_canonical_rows_and_labels_never_enter_inputs():
    a, ra, ma = annotated_case()
    b, rb, mb = annotated_case('other', application='companion')
    b['records'][0]['text'] = 'What happened next?'
    rb['judgments'][0]['evidence'] = rb['judgments'][2]['evidence'] = [dict(start=0, end=19, quote='What happened next?')]
    mb['pairs'] = [{k: r[k] for k in ('slot', 'record', 'input_sha256')} for r in review_projection(b)]
    ledger = build_source_ledger([entry('example'), entry('other')])
    semantic, complete = prepare_membership_bank([a, b], [ra, rb], [ma, mb], ledger)
    assert {r['application'] for r in complete} == {'tutor', 'companion'}
    assert len(semantic) == len({r['input_sha256'] for r in complete})
    before = review_projection(a)
    a.update(expected_label='match', reviewer='not-in-input', reference=ra)
    assert review_projection(a) == before and 'not-in-input' not in str(before)


def test_qualification_locked_source_cannot_enter_training_by_case_partition_change():
    case, reference, review = annotated_case('held')
    source = entry('held')
    source['qualification_only'] = True
    ledger = build_source_ledger([source])
    with pytest.raises(ValueError, match='Source exposure forbids'):
        prepare_membership_bank([case], [reference], [review], ledger)
    case['split'] = 'qualification'
    with pytest.raises(ValueError, match='Independent review'):
        prepare_membership_bank([case], [reference], [review], ledger)


@pytest.mark.parametrize('change', ['old_contract', 'application', 'unresolved', 'projected', 'missing_pin',
                                   'stale_pin', 'no_reviewer', 'adjudication_not_independent', 'changed_context'])
def test_incomplete_stale_or_unreviewed_exports_fail_closed(change):
    case, ref, review = annotated_case()
    ledger = build_source_ledger([entry(case['id'])])
    if change == 'old_contract': ref['contract_version'] = 2
    elif change == 'application': case.pop('application')
    elif change == 'unresolved': review['state'] = 'unresolved'
    elif change == 'projected': review['state'] = 'projected_only'
    elif change == 'missing_pin': review.pop('pairs')
    elif change == 'stale_pin': review['pairs'][0]['input_sha256'] = '0' * 64
    elif change == 'no_reviewer': review['reviewers'] = []
    elif change == 'adjudication_not_independent':
        case['split'] = 'qualification'; review['state'] = 'adjudicated'
    elif change == 'changed_context':
        case['records'][0]['source_order']['group'] = 'b' * 32
    with pytest.raises(ValueError): prepare_membership_bank([case], [ref], [review], ledger)


def test_review_masks_do_not_turn_disagreement_into_a_semantic_label():
    case, ref, review = annotated_case()
    review.update(state='unresolved', membership_loss=False)
    semantic, complete = prepare_membership_bank([case], [ref], [review], build_source_ledger([entry(case['id'])]))
    assert semantic == [] and len(complete) == 4
    assert all(r['label'] == 'nonmatch' and not r['membership_loss'] for r in complete)


@pytest.mark.parametrize('exposure', ['tuning', 'reserved', 'diagnostic', 'regression', 'unclassified_prior_use'])
def test_historical_nontraining_use_cannot_enter_training(exposure):
    case, ref, review = annotated_case()
    ledger = build_source_ledger([entry(case['id'], exposure=[exposure])])
    with pytest.raises(ValueError, match='exposure'):
        prepare_membership_bank([case], [ref], [review], ledger)


def test_duplicate_format4_inputs_are_not_weighted_twice_and_cannot_cross_splits():
    a, ra, ma = annotated_case()
    b, rb, mb = annotated_case('variant')
    ledger = build_source_ledger([entry('example'), entry('variant')])
    semantic, complete = prepare_membership_bank([a, b], [ra, rb], [ma, mb], ledger)
    assert len(semantic) == 4 and len(complete) == 8
    assert all(r['duplicate_of'] for r in complete[4:])
    b['split'] = 'tuning'
    with pytest.raises(ValueError, match='model input crosses'):
        prepare_membership_bank([a, b], [ra, rb], [ma, mb], ledger)
    b['split'] = 'train'
    rb['judgments'][0].update(label='match', reason='direct_act')
    with pytest.raises(ValueError, match='conflicting labels'):
        prepare_membership_bank([a, b], [ra, rb], [ma, mb], ledger)


def test_source_components_cannot_cross_splits_even_with_different_inputs():
    a, ra, ma = annotated_case()
    b, rb, mb = annotated_case('translation', split='tuning')
    ledger = build_source_ledger([entry('example'), entry('translation', parents=['example'])])
    with pytest.raises(ValueError, match='component crosses'):
        prepare_membership_bank([a, b], [ra, rb], [ma, mb], ledger)
    ledger['components'][0]['allowed_partitions'] = ['train']
    with pytest.raises(ValueError, match='canonical'):
        prepare_membership_bank([a], [ra], [ma], ledger)


def test_case_cannot_bypass_history_by_pointing_to_an_unrelated_fresh_entry():
    case, ref, review = annotated_case()
    case['source_entry'] = 'unrelated'
    ledger = build_source_ledger([entry('unrelated'), entry(case['id'], exposure=['regression'])])
    with pytest.raises(ValueError, match='exposure'):
        prepare_membership_bank([case], [ref], [review], ledger)


def test_independent_qualification_requires_identity_evidence_and_complete_reference():
    case, ref, review = annotated_case(split='qualification')
    review.update(state='adjudicated', reviewers=['structural-fixture-author', 'structural-test-reviewer'],
                  independent_review=True)
    semantic, complete = prepare_membership_bank([case], [ref], [review], build_source_ledger([entry(case['id'])]))
    assert len(semantic) == len(complete) == 4
    ref['judgments'].pop()
    with pytest.raises(ValueError, match='Incomplete membership'):
        prepare_membership_bank([case], [ref], [review], build_source_ledger([entry(case['id'])]))


def test_raw_json_pin_cannot_be_bypassed(tmp_path):
    path = tmp_path / 'snapshot.json'
    digest = write_json(path, {'verified': True})
    assert pinned_json(path, digest) == {'verified': True}
    with pytest.raises(ValueError, match='Pinned source changed'): pinned_json(path, '0' * 64)
