"""Real source/recall regressions; transport controls never establish quality."""
import importlib.util
import json
from argparse import Namespace

import pytest

from psychology_base.eval.benchmarks import application_context as study
from psychology_base.applications.context import render_context_block
from psychology_base.llm import load_config
from tests.test_application_context_study import real_sources


@pytest.fixture(scope='module')
def diagnostic(real_sources):
    if importlib.util.find_spec('jieba') is None:
        pytest.skip('Real jieba backend required')
    _, cases, sources = real_sources
    return study.context_diagnostics([c for c in cases if c['split']=='development'], sources, load_config())


def test_actual_source_roles_are_eligible_and_both_applications_receive_changed_context(diagnostic):
    assert diagnostic['passed']
    assert diagnostic['pairs']==96 and len(diagnostic['records'])==96
    assert diagnostic['source_parse_status']=={'valid':diagnostic['unique_source_fragments']}
    assert diagnostic['unique_source_fragments']>0
    assert diagnostic['provider_requests']==diagnostic['generated_application_replies']==0
    for app in ('companion','tutor'):
        assert diagnostic['per_app'][app]['source_present_pairs']==24
        assert diagnostic['per_app'][app]['pairs_with_neighbors']>0
        assert diagnostic['per_app'][app]['changed_context_pairs']>0
    assert len({(r['case'],r['query_index'],r['absent']) for r in diagnostic['records']})==96
    for row in diagnostic['records']:
        a,b=(row['arms'][arm] for arm in study.ARMS)
        assert a['inventory_sha256']==b['inventory_sha256']
        assert a['retained_support_ids']==b['retained_support_ids']
        assert not row['absent'] or not (a['delivered_support_ids'] or b['delivered_support_ids'])
        assert row['context_changed']==(a['context_sha256']!=b['context_sha256'])


def test_identity_receipts_and_source_bytes_match_the_completed_run(diagnostic):
    path=study.ROOT/'.data/application_context/hybrid-screen-v3-development.json'
    if not path.exists(): pytest.skip('Archived real development report unavailable')
    original=study.read(path)
    rows={(r['case'],r['query_index'],r['absent']):r for r in original['trials'] if r['candidate']=='identity'}
    for row in diagnostic['records']:
        before=rows[row['case'],row['query_index'],row['absent']]
        for arm in study.ARMS:
            assert row['arms'][arm]['inventory_sha256']==study.shared.stable_key(before['evidence']['inventory'])
        identity=row['arms']['identity']
        assert identity['selected_ids']==[f['id'] for f in before['capture']['fragments']]
        fragments=study.pam_import({**study.pam_export(()),'fragments':before['capture']['fragments']})
        current=study.render(row['app'],fragments)
        assert identity['context_sha256']==study.shared.stable_key(current)
        if row['app']=='companion':
            assert render_context_block(fragments,assistant_speaker='companion',format_version=1)==before['capture']['context']
        else:
            # Historical tutor context omitted labels; source words/order stay exact.
            assert '\n'.join(f'- {f.content}' for f in fragments)==before['capture']['context']
        expected=[]
        for fragment in fragments:
            role=next(t.split(':',1)[1] for t in fragment.tags if t.startswith('speaker:'))
            expected.append({'speaker':role if role in ('user','tutor','companion') else None,
                             'session':'source','text':fragment.content})
        assert list(map(json.loads,current.splitlines()))==expected


def test_contrast_diagnostics_cannot_open_heldout_cases(real_sources):
    _, cases, sources=real_sources
    for selected in ([],[next(c for c in cases if c['split']=='evaluation')]):
        with pytest.raises(ValueError,match='development cases only'):
            study.context_diagnostics(selected,sources,load_config())


@pytest.mark.skipif(importlib.util.find_spec('jieba') is None,reason='Frozen runtime requires jieba')
def test_missing_treatment_contrast_stops_before_budget_or_provider_creation(real_sources,tmp_path,monkeypatch):
    monkeypatch.setattr(study,'context_diagnostics',lambda *a:{'passed':False})
    monkeypatch.setattr(study,'make_routers',lambda *a:pytest.fail('No contrast reached a provider'))
    monkeypatch.setattr(study,'study_budget',lambda *a:pytest.fail('No contrast opened the shared ledger'))
    with pytest.raises(ValueError,match='treatment contrast'):
        study.run(Namespace(command='application-preflight',application_profile=study.FIXTURES/'hybrid-screen-v2.json',
                            output=tmp_path/'preflight.json',llm_config=None,application_resume=None))


@pytest.mark.skipif(importlib.util.find_spec('jieba') is None,reason='Real jieba backend required')
def test_real_tutor_trial_pins_both_judges_and_preserves_proxy_status(real_sources,tmp_path):
    from tests.test_application_call_audit import audited
    _,cases,sources=real_sources
    case=next(c for c in cases if c['split']=='development' and '/mathdial/' in c['source'])
    generation,_,gb=audited(tmp_path/'g',calls=1)
    judge,_,jb=audited(tmp_path/'j',calls=2)
    generation.max_input=judge.max_input=12000
    try:
        row=study.trial(case,0,0,'scaled_focus',False,sources,load_config(),generation,judge,tmp_path)
        requests=[r for line in jb.writer.path.read_text(encoding='utf-8').splitlines()
                  if (r:=json.loads(line))['event']=='reservation']
        assert len(requests)==2 and all(r['temperature']==0 and r['max_tokens']==300 for r in requests)
        assert row['proxy'] and not row['scores']['resolved']
        assert row['tutor_judgment']['parse_failed']
    finally:
        gb.writer.close(); jb.writer.close()
