"""Source/qualification software checks; scripted judgments are not model results."""
import copy
import json

import pytest

from psychology_base.eval.benchmarks import application_context as study
from psychology_base.applications.measurement.memory_grounding import parse_runtime_memory, runtime_memory_protocol


@pytest.fixture
def controls():
    pin = study.read(study.FIXTURES / 'runtime-qualification-anchors-v1.json')['source_pin']
    if not (study.ROOT / pin['path']).exists():
        pytest.skip('Optional locally provisioned public source packet is unavailable')
    return study.runtime_qualification_items()


def scripted_rows(controls):
    """Invented schema controls for gate logic only; never saved as empirical data."""
    by_id = {row['id']: row for row in controls}
    rows = []
    for scheduled in study.runtime_qualification_schedule(controls):
        item = by_id[scheduled['control']]
        criteria = item['expected_criteria'] or dict(addresses_request=True, grounded=True,
            attribution_correct=True, abstains_appropriately=True, critical_error=False)
        raw = json.dumps(dict(criteria, decision=item['expected_decision'], rationale='SCRIPTED software control'))
        rows.append(dict(scheduled, judgment=parse_runtime_memory(raw, 'stop', True)))
    return rows


def test_real_source_controls_are_disjoint_balanced_and_disclosed(controls):
    from collections import Counter
    assert len(controls) == 40 and len({row['source'] for row in controls}) == 16
    assert Counter((r['language'], r['app']) for r in controls) == {
        ('en', 'companion'): 10, ('en', 'tutor'): 10, ('zh', 'companion'): 10, ('zh', 'tutor'): 10}
    assert Counter(r['control_type'] for r in controls) == {
        'useful': 12, 'needless_refusal': 4, 'unsupported': 16, 'ambiguous': 8}
    assert all(row['authored_control'] is True for row in controls)


def test_fixed_schedule_has_all_primary_cases_and_twelve_three_run_sentinels(controls):
    schedule = study.runtime_qualification_schedule(controls)
    assert len(schedule) == 64
    assert {r['control'] for r in schedule[:40]} == {r['id'] for r in controls}
    assert len({r['control'] for r in schedule[40:52]}) == 12
    assert [r['control'] for r in schedule[52:]] == [r['control'] for r in reversed(schedule[40:52])]
    assert schedule == study.runtime_qualification_schedule(controls)
    assert all(row['index'] == i for i, row in enumerate(schedule))


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'label', 'criteria', 'stratum'])
def test_bad_control_catalog_rejects_before_dispatch(controls, mutation):
    controls = copy.deepcopy(controls)
    if mutation == 'duplicate': controls[-1] = copy.deepcopy(controls[0])
    elif mutation == 'missing': controls.pop()
    elif mutation == 'label': controls[0]['expected_decision'] = 'inconclusive'
    elif mutation == 'criteria': controls[0]['expected_criteria']['grounded'] = False
    else: controls[0]['language'] = 'fr'
    with pytest.raises(ValueError):
        study.runtime_qualification_schedule(controls)


def test_scripted_perfect_gate_is_only_a_software_control(controls):
    rows = scripted_rows(controls)
    metrics = study.runtime_qualification_metrics(controls, rows)
    assert metrics['accuracy_passed'] and metrics['repeatability_passed']
    assert metrics['complete'] and metrics['false_accepts_on_unsupported'] == 0
    assert metrics['primary_completed'] == 40 and metrics['judgments_completed'] == 64
    assert {k:v['total'] for k,v in metrics['by_language'].items()}=={'en':20,'zh':20}
    assert {k:v['total'] for k,v in metrics['by_application'].items()}=={'companion':20,'tutor':20}
    assert len(metrics['by_source_family'])==16
    assert all(row['judgment']['proxy'] is True for row in rows)


def replace_decision(row, decision, **flags):
    raw = json.dumps(row['judgment']['criteria'] | flags | dict(decision=decision))
    row['judgment'] = parse_runtime_memory(raw, 'stop', True)


@pytest.mark.parametrize('failure', ['false_accept', 'useful_refused', 'refusal_useful', 'ambiguity_accept', 'parse_failure', 'repeat_flip'])
def test_irreversible_failures_stay_in_their_denominators(controls, failure):
    rows = scripted_rows(controls)
    by_id = {row['id']: row for row in controls}
    primary = lambda kind: next(r for r in rows[:40] if by_id[r['control']]['control_type'] == kind)
    if failure == 'false_accept':
        replace_decision(primary('unsupported'), 'accept', grounded=True, attribution_correct=True,
                         critical_error=False, abstains_appropriately=True)
    elif failure == 'useful_refused':
        replace_decision(primary('useful'), 'inconclusive')
    elif failure == 'refusal_useful':
        replace_decision(primary('needless_refusal'), 'accept', addresses_request=True)
    elif failure == 'ambiguity_accept':
        replace_decision(primary('ambiguous'), 'accept')
    elif failure == 'parse_failure':
        rows[0]['judgment'] = parse_runtime_memory('{}', 'stop', True)
    else:
        repeated = next(r for r in rows[40:] if by_id[r['control']]['expected_decision'] == 'accept')
        replace_decision(repeated, 'inconclusive')
    metrics = study.runtime_qualification_metrics(controls, rows)
    assert not metrics['passed'] and metrics['irreversible_failure']
    assert metrics['primary_total'] == 40 and metrics['primary_completed'] == 40


def test_expected_inconclusive_flags_are_provisional_but_decisions_must_repeat(controls):
    rows = scripted_rows(controls)
    by_id = {row['id']: row for row in controls}
    for row in rows[40:]:
        if by_id[row['control']]['expected_decision'] == 'inconclusive':
            replace_decision(row, 'inconclusive', addresses_request=False, abstains_appropriately=False)
    assert study.runtime_qualification_metrics(controls, rows)['passed']


@pytest.mark.parametrize('change', ['skip', 'duplicate', 'reorder'])
def test_result_schedule_drift_rejects(controls, change):
    rows = scripted_rows(controls)
    if change == 'skip': rows.pop(1)
    elif change == 'duplicate': rows[1] = copy.deepcopy(rows[0])
    else: rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match='schedule'):
        study.runtime_qualification_metrics(controls, rows)


def test_absent_review_blocks_before_any_judgment(controls):
    class Forbidden:
        controls = False
        def chat(self, *args, **kwargs): pytest.fail('No review must block before dispatch')
    with pytest.raises(ValueError, match='review'):
        study.runtime_qualification(controls, runtime_memory_protocol(), Forbidden(), review=None)


def test_request_builder_uses_existing_judge_payload_without_expected_labels(controls):
    for item in controls:
        messages = study.memory_judge_messages({'app': item['app']}, None, item['capture'],
                                               item['response'], None, protocol=runtime_memory_protocol())
        payload = json.loads(messages[1].content)
        assert set(payload) == {'APPLICATION', 'QUERY', 'VISIBLE_CONTEXT', 'REPLY'}
        assert payload['REPLY'] == item['response']
        from psychology_base.eval.benchmarks.application_calls import request_input_bound
        assert request_input_bound(messages) <= 12000


def scripted_review(controls):
    """A fabricated review receipt strictly for schema/gate tests."""
    return dict(schema='runtime-qualification-review-v1', controls_sha256=study.shared.stable_key(controls),
        contract_sha256=study.shared.stable_key(runtime_memory_protocol()),
        independent_of_preparer=True, blinded_to_proposed_labels=True, reviewer_kind='llm',
        reviewer='SCRIPTED-REVIEW-ONLY-NOT-ACTUAL-INDEPENDENT-REVIEW', judgments=[dict(
            control=row['id'], decision=row['expected_decision'], criteria=row['expected_criteria'],
            reason='SCRIPTED software control', evidence='SCRIPTED software control') for row in controls])


@pytest.mark.parametrize('mutation', ['author', 'unblinded', 'missing', 'duplicate', 'disagreement', 'wrong_hash', 'wrong_contract'])
def test_review_disagreement_and_missing_provenance_block(controls, mutation):
    review=scripted_review(controls)
    if mutation=='author': review['reviewer']='preparing-codex-root-runtime-qualification-v1'
    elif mutation=='unblinded': review['blinded_to_proposed_labels']=False
    elif mutation=='missing': review['judgments'].pop()
    elif mutation=='duplicate': review['judgments'][-1]=review['judgments'][0]
    elif mutation=='disagreement': review['judgments'][0]['decision']='inconclusive'
    elif mutation=='wrong_hash': review['controls_sha256']='wrong'
    else: review['contract_sha256']='wrong'
    with pytest.raises(ValueError, match='review'):
        study.validate_runtime_qualification_review(controls,review)


def test_partial_results_do_not_credit_unobserved_or_provisional_boolean_labels(controls):
    rows=scripted_rows(controls)
    empty=study.runtime_qualification_metrics(controls,[])
    assert empty['critical_agreement']==dict(grounded=0,attribution_correct=0,critical_error=0)
    assert not empty['irreversible_failure'] and not empty['passed']
    by_id={row['id']:row for row in controls}
    target=next(row for row in rows[:40] if by_id[row['control']]['control_type']=='unsupported')
    replace_decision(target,'inconclusive')
    metrics=study.runtime_qualification_metrics(controls,rows)
    assert metrics['critical_errors']==dict(grounded=1,attribution_correct=1,critical_error=1)
    # The old repeated resolved sentinel must now disagree with this new primary.
    assert metrics['critical_agreement']==dict(grounded=31/32,attribution_correct=31/32,critical_error=31/32)


@pytest.mark.parametrize('field', ['reference_sha256','initial_chain_sha256','source','source_pin'])
def test_actual_source_or_position_drift_is_rejected(tmp_path,controls,field):
    bank=copy.deepcopy(study.read(study.FIXTURES/'runtime-qualification-anchors-v1.json'))
    if field=='source_pin': bank['source_pin']['sha256']='wrong'
    elif field=='source': bank['anchors'][0]['source']='msc_2'
    else: bank['anchors'][0][field]='wrong'
    path=tmp_path/'drift.json'
    path.write_text(json.dumps(bank),encoding='utf-8')
    with pytest.raises(ValueError): study.runtime_qualification_items(path)


def simulated_run(monkeypatch,controls,*,malformed_first=False,qualification_version=1):
    """Exercise the loop/replay with invented transport, entirely inside pytest."""
    from decimal import Decimal
    protocol=(runtime_memory_protocol() if qualification_version==1 else
              runtime_memory_protocol(contract_version=2,prompt_revision=4))
    if qualification_version!=1: protocol['runtime_qualification_version']=qualification_version
    protocol['judge']=study.read(study.FIXTURES/'hybrid-screen-v4.json')['judge']
    judge=object.__new__(study.RecordingRouter)
    judge.controls=False; judge.provider=protocol['judge']['provider']; judge.model=protocol['judge']['model']
    judge.remote_spec=protocol['judge']; judge.local_spec=None; judge.max_input=12000
    ledger=[]; invocations=[]
    def assess(kind,item,protocol_arg,router):
        index=len(invocations)+1; invocations.append(item['id'])
        criteria=item['expected_criteria'] or dict(addresses_request=True,grounded=True,
            attribution_correct=True,abstains_appropriately=True,critical_error=False)
        output=(dict(criteria,decision=item['expected_decision'],rationale='SCRIPTED') if qualification_version==1 else
                dict(criteria,semantic_status='inconclusive' if item['expected_decision']=='inconclusive' else 'resolved',rationale='SCRIPTED'))
        raw=('{}' if malformed_first and index==1 else json.dumps(output))
        result=parse_runtime_memory(raw,'stop',False,contract_version=protocol['runtime_memory_contract_version'])
        spec=protocol['judge']
        call=dict(event='completion',dispatch=index,cached=False,proxy=False,usage_valid=True,wall_ms=0,
            accounted_usd=str((Decimal(spec['input_rate'])+Decimal(spec['output_rate']))/1_000_000),
            completion=dict(provider=spec['provider'],model_id=spec['model'],tier=study.TaskTier.ADVANCED.value,
                prompt_tokens=1,completion_tokens=1,content=raw,finish_reason='stop',metadata=dict(
                    response_model=spec['model'],completion_id='SCRIPTED-'+str(index),reasoning_effort=spec['reasoning_effort'],
                    reasoning_tokens=0,service_tier='default',json_mode=True,test_control=True)))
        messages=study.memory_judge_messages({'app':item['app']},None,item['capture'],item['response'],None,protocol=protocol)
        ledger.extend([dict(event='reservation',dispatch=index,reserved_usd='1',
            provider=spec['provider'],model=spec['model'],tier=study.TaskTier.ADVANCED.value,
            messages=[study.asdict(m) for m in messages],max_tokens=300,temperature=0,proxy=False),call])
        row=dict(judgment=result,observed_success=study.runtime_memory_resolved(result) and result['decision']=='accept',call=call)
        if qualification_version==2: row['disposition']=study.runtime_memory_action(result)
        return row
    original_validate=study.validate_runtime_qualification
    monkeypatch.setattr(study,'assess_calibration_control',assess)
    monkeypatch.setattr(study,'validate_runtime_qualification',lambda report,items,proto,**kwargs:
        original_validate(report,items,proto,ledger_rows=ledger,**kwargs))
    review=scripted_review(controls)
    report=study.runtime_qualification(controls,protocol,judge,review=review)
    return report,protocol,review,ledger,invocations,original_validate


def test_scripted_adapter_runs_exactly_64_assessments_and_replays_all_raw_requests(controls,monkeypatch):
    report,protocol,review,ledger,invocations,validate=simulated_run(monkeypatch,controls)
    assert len(invocations)==64 and len(ledger)==128
    assert report['passed'] and report['generated_application_replies']==0
    assert validate(report,controls,protocol,review=review,ledger_rows=ledger)
    # Neither this invented review nor these invented calls are persisted as model evidence.
    assert all(row['call']['completion']['metadata']['test_control'] for row in report['rows'])


def test_scripted_adapter_stops_at_first_irreversible_failure_without_retry(controls,monkeypatch):
    report,protocol,review,ledger,invocations,validate=simulated_run(monkeypatch,controls,malformed_first=True)
    assert len(invocations)==1 and not report['passed']
    assert report['metrics']['remaining_judgments']==63
    assert not validate(report,controls,protocol,review=review,ledger_rows=ledger)


@pytest.mark.parametrize('mutation', ['payload','completion_id','raw','proxy','code','metrics','source_identity'])
def test_saved_report_cannot_override_real_request_or_result_provenance(controls,monkeypatch,mutation):
    report,protocol,review,ledger,_,validate=simulated_run(monkeypatch,controls)
    report=copy.deepcopy(report); ledger=copy.deepcopy(ledger)
    if mutation=='payload': ledger[0]['messages'][1]['content']='{}'
    elif mutation=='completion_id':
        report['rows'][1]['call']['completion']['metadata']['completion_id']=report['rows'][0]['call']['completion']['metadata']['completion_id']
    elif mutation=='raw': report['rows'][0]['judgment']['raw']='{}'
    elif mutation=='proxy': report['proxy']=True
    elif mutation=='code': report['code_sha256']={}
    elif mutation=='metrics': report['metrics']['passed']=False
    else: report['source_identity']={}
    with pytest.raises(ValueError): validate(report,controls,protocol,review=review,ledger_rows=ledger)


def test_frozen_review_packet_contains_every_case_without_proposed_labels(controls):
    packet_path=study.ROOT/'docs/experiments/phase-9-application-context/runtime-qualification-packet-v1.json'
    if not packet_path.exists(): pytest.skip('Optional local frozen qualification packet unavailable')
    frozen=study.read(packet_path)
    blind_path=study.ROOT/frozen['blinded_review']['path']
    assert study.shared.digest(blind_path)==frozen['blinded_review']['sha256']
    blind=study.read(blind_path)
    assert blind['controls_sha256']==study.shared.stable_key(controls)
    current = study.fingerprints()
    # Historical identity is fixed; the current source tree can gain owners.
    # Pin both archived names and hashes instead of maintaining an ever-growing
    # exclusion list or rewriting the consumed packet to match the checkout.
    later_owners = {'src/psychology_base/companion/navigation.py',
                    'src/psychology_base/llm/accounting.py',
                        'src/psychology_base/llm/execution.py',
                        'src/psychology_base/llm/providers/local_pair.py'}
    assert later_owners <= set(current)
    assert study.shared.stable_key(frozen['source_fingerprints']) == '53c783d3f588b3e5d195dadee002c2255a0f0dcbcb4c64bee62324be465425a1'
    assert set(frozen['source_fingerprints']) <= set(current) - later_owners
    assert frozen['sources']==study.runtime_qualification_identity()
    assert len(blind['cases'])==40
    for case in blind['cases']:
        assert set(case)=={'control','APPLICATION','QUERY','VISIBLE_CONTEXT','REPLY'}
    assert frozen['provider_calls']==0 and not frozen['verifier_qualified']


def test_message_refactor_preserves_all_previous_protocol_wire_bytes(controls):
    import importlib.util
    from types import SimpleNamespace
    path=study.ROOT/'.data/application_context/runtime-qualification-v1/before-application-context.py'
    if not path.exists(): pytest.skip('Optional pre-change owner snapshot unavailable')
    spec=importlib.util.spec_from_file_location('runtime_qualification_previous_owner',path)
    previous=importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
    protocols=[{},study.revised_grading_protocol({}),study.revised_grading_protocol({},current_query_evidence=True)]
    protocols.extend(study.revised_grading_protocol({},speaker_evidence_version=v) for v in (1,2,3,4))
    protocols.append(runtime_memory_protocol())
    class Capture:
        controls=False
        def chat(self,messages,**options):
            self.wire=json.dumps([study.asdict(m) for m in messages],ensure_ascii=False)
            self.options=options
            return SimpleNamespace(content=json.dumps(dict(addresses_request=True,grounded=True,
                attribution_correct=True,abstains_appropriately=True,critical_error=False,decision='accept')),
                finish_reason='stop')
    for protocol in protocols:
        for item in controls[:4]:
            old=Capture(); new=Capture()
            args=({'app':item['app']},False,item['capture'],item['response'],'visible only to evaluation')
            assert previous.judge_memory(old,*args,protocol=protocol)==study.judge_memory(new,*args,protocol=protocol)
            assert old.wire==new.wire and old.options==new.options


def test_wrong_judge_model_is_rejected_before_the_first_assessment(controls,monkeypatch):
    protocol=runtime_memory_protocol(); protocol['judge']=study.read(study.FIXTURES/'hybrid-screen-v4.json')['judge']
    judge=object.__new__(study.RecordingRouter)
    judge.controls=False; judge.provider='openai'; judge.model='wrong'; judge.max_input=12000
    judge.remote_spec=protocol['judge']; judge.local_spec=None
    def forbidden(*args,**kwargs): pytest.fail('Model drift must block before dispatch')
    monkeypatch.setattr(study,'assess_calibration_control',forbidden)
    with pytest.raises(ValueError,match='pre-dispatch'):
        study.runtime_qualification(controls,protocol,judge,review=scripted_review(controls))


def test_replay_rejects_calls_after_an_irreversible_failure(controls,monkeypatch):
    report,protocol,review,ledger,_,validate=simulated_run(monkeypatch,controls)
    first=report['rows'][0]
    first['judgment']=parse_runtime_memory('{}','stop',False)
    first['observed_success']=False
    first['call']['completion']['content']='{}'
    report['metrics']=study.runtime_qualification_metrics(controls,report['rows'])
    report['passed']=False
    with pytest.raises(ValueError,match='stopped schedule'):
        validate(report,controls,protocol,review=review,ledger_rows=ledger)
