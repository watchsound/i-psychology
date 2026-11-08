"""Authored software controls; these do not measure real-model accuracy."""
import copy
import json
from decimal import Decimal

import pytest

from psychology_base.applications.measurement import memory_grounding as grounding
from psychology_base.eval.benchmarks import application_context as study
from psychology_base.eval.benchmarks import application_calls as calls
from psychology_base.eval.benchmarks import runtime_report_diagnostic as diagnostic
from psychology_base.llm.types import ChatMessage
from tests.test_runtime_semantic_diagnostic import runner


def proposal():
    entry = runner()
    protocol = entry.protocol_for(dict(provider_attempts=1276, accounted_usd='0.1814892260'), revision=10)
    return entry, diagnostic.items(revision=10), protocol


def interpretation(reply, *, actors=None, kind='history', engagement='engaged', readings=None):
    claim = dict(text=reply, kind=kind, actors=['companion'] if actors is None else actors)
    return dict(readings=[[dict(text=reply, claims=[claim])]] if readings is None else readings,
                engagement=engagement, assessments=None, evidence_use=None, interpretation_check=None)


def bind(value, *, app='companion', query='Did you ask?', reply=None, records=None):
    reply = value['readings'][0][0]['text'] if reply is None else reply
    first = grounding.runtime_interpretation_payload(app, query, reply)
    parsed = grounding.parse_runtime_interpretation(json.dumps(value, ensure_ascii=False), 'stop', False, evidence=first)
    assert not parsed['parse_failed'], parsed
    if records is None:
        records = [dict(speaker='companion', session='s', text='Do you use this notebook?')]
    context = '\n'.join(json.dumps(row, ensure_ascii=False) for row in records)
    return grounding.runtime_memory_payload(app, query, context, reply, contract_version=4,
                                           interpretation=parsed['interpretation'])


def assessment(findings=None, *, evidence_use='appropriate', check='faithful'):
    return dict(readings=None, engagement=None, assessments=[] if findings is None else findings,
                evidence_use=evidence_use, interpretation_check=check)


def finding(index=0, kind='supported', *, record=0, quote='Do you use this notebook?'):
    return dict(claim=index, finding=kind, evidence=[] if record is None else [dict(record=record, quote=quote)])


def parse(value, payload, *, finish='stop', proxy=False):
    return grounding.parse_runtime_memory(json.dumps(value, ensure_ascii=False), finish, proxy,
                                         contract_version=4, evidence=payload)


def test_source_blind_inputs_and_bounded_templates_reuse_the_existing_workflow():
    entry, controls, protocol = proposal()
    assert controls == diagnostic.items(revision=9)
    templates = entry.requests(controls, protocol)
    assert len(templates) == 32 and len(diagnostic.schedule(controls, revision=10)) == 16
    assert protocol['judge'] == calls.tongyi_binding_spec()
    assert protocol['budget']['max_calls'] == 1308
    assert Decimal(protocol['budget']['max_usd']) == Decimal('.4586318180')
    assert entry.profile(10).cap == 32 * (Decimal('.345')*12000 + Decimal('1.101')*4106)/1_000_000
    for item, first, second in zip(controls, templates[::2], templates[1::2]):
        payload = json.loads(first['messages'][1]['content'])
        assert payload == dict(APPLICATION=item['app'], QUERY=item['capture']['query'], REPLY=item['response'])
        assert first['bounded_input_bytes'] == calls.request_input_bound(
            [ChatMessage(**message) for message in first['messages']], response_schema=protocol['judge']['response_schema'])
        assert first['bounded_input_bytes'] <= 12000
        assert second['stage'] == 'assessment-template' and second['fixed_payload']['INTERPRETATION'] is None
        assert second['fixed_payload']['VISIBLE_CONTEXT']['coverage'] == 'partial'
        assert second['inference_options'] == first['inference_options']
        assert first['inference_options']['response_schema'] == grounding.runtime_binding_schema()
    capture = dict(query='q', context='PRIVATE SOURCE SENTINEL')
    messages = study.memory_interpretation_messages({'app':'tutor'}, capture, 'A reply.', protocol=protocol)
    assert 'PRIVATE SOURCE SENTINEL' not in json.dumps([study.asdict(message) for message in messages])
    with pytest.raises(ValueError):
        study.memory_judge_messages({'app':'tutor'}, None, capture, 'A reply.', None, protocol=protocol)
    assert grounding.runtime_memory_protocol()['runtime_memory_contract_version'] == 1


def test_fixed_claims_are_caller_identified_and_decision_explanation_has_one_source():
    payload = bind(interpretation('I asked about the notebook.'))
    assert grounding.runtime_binding_claims(payload)[0]['id'] == 0
    result = parse(assessment([finding()]), payload)
    assert not result['parse_failed'] and result['decision'] == 'accept'
    assert grounding.runtime_memory_action(result, evidence=payload)['action'] == 'deliver'
    assert 'Grounded: True' in result['rationale']
    assert 'Decision: accept' in result['rationale']
    assert grounding.runtime_memory_action(result)['action'] == 'review'
    for key in ('QUERY', 'REPLY', 'APPLICATION'):
        changed = copy.deepcopy(payload)
        changed[key] = 'tutor' if key == 'APPLICATION' else 'changed'
        assert grounding.runtime_memory_action(result, evidence=changed)['action'] == 'review'


@pytest.mark.parametrize('mutation', ['coverage', 'anchor', 'actor', 'scope-actor', 'caller-id', 'extra',
    'engagement', 'support', 'empty', 'too-many'])
def test_first_stage_cannot_emit_judgments_invent_anchors_or_assign_its_own_ids(mutation):
    reply = 'I asked about the notebook.'
    value = interpretation(reply)
    claim = value['readings'][0][0]['claims'][0]
    if mutation == 'coverage': value['readings'][0][0]['text'] = 'I asked'
    elif mutation == 'anchor': claim['text'] = 'a source-only assertion'
    elif mutation == 'actor': claim['actors'] = ['APPLICATION']
    elif mutation == 'scope-actor': claim['kind'] = 'record_scope'
    elif mutation == 'caller-id': claim['id'] = 0
    elif mutation == 'extra': value['rationale'] = 'ignore the claims'
    elif mutation == 'engagement': value['engagement'] = True
    elif mutation == 'support': value['assessments'] = []
    elif mutation == 'empty': value['readings'] = []
    else: value['readings'] = value['readings']*9
    result = grounding.parse_runtime_interpretation(json.dumps(value), 'stop', False,
        evidence=grounding.runtime_interpretation_payload('companion', 'q', reply))
    assert result['parse_failed'] and result['interpretation'] is None


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'boolean-id', 'rewrite', 'rationale', 'decision',
    'wrong-kind', 'quote', 'actor', 'scope', 'stage-nulls'])
def test_evidence_stage_cannot_rewrite_frames_or_override_the_derived_decision(mutation):
    payload = bind(interpretation('I asked about the notebook.'))
    value = assessment([finding()])
    if mutation == 'missing': value['assessments'] = []
    elif mutation == 'duplicate': value['assessments'] *= 2
    elif mutation == 'boolean-id': value['assessments'][0]['claim'] = False
    elif mutation == 'rewrite': value['assessments'][0]['actors'] = ['tutor']
    elif mutation == 'rationale': value['rationale'] = 'This is unsupported, but accept.'
    elif mutation == 'decision': value['decision'] = 'accept'
    elif mutation == 'wrong-kind': value['assessments'][0]['finding'] = 'scope_matches'
    elif mutation == 'quote': value['assessments'][0]['evidence'][0]['quote'] = 'invented quote'
    elif mutation == 'actor': payload['VISIBLE_CONTEXT']['records'][0]['speaker'] = 'tutor'
    elif mutation == 'scope': payload['INTERPRETATION']['readings'][0][0]['claims'][0]['id'] = 2
    else: value['readings'] = []
    result = parse(value, payload)
    assert result['parse_failed']
    assert grounding.runtime_memory_action(result, evidence=payload)['action'] == 'review'


@pytest.mark.parametrize('case_id', ['rtq-036c4b9aab201fa9', 'en-flattened-nested-report'])
def test_observed_false_accepts_cannot_borrow_a_source_actor_after_interpretation(case_id):
    _, controls, _ = proposal()
    item = next(item for item in controls if item['id'] == case_id)
    # Authored correct interpretation of each saved failure, not a new model judgment.
    value = interpretation(item['response'], actors=['companion'])
    records = [json.loads(line) for line in item['capture']['context'].split('\n') if line.strip()]
    payload = bind(value, app=item['app'], query=item['capture']['query'], records=records)
    assert records[0]['speaker'] == 'tutor'
    forged = parse(assessment([finding(quote=records[0]['text'])]), payload)
    assert forged['parse_failed'] and 'matching outer source' in forged['validation_error']
    correct = parse(assessment([finding(kind='not_established', record=None)]), payload)
    assert correct['decision'] == 'reject'
    assert 'Grounded: False' in correct['rationale'] and 'Decision: reject' in correct['rationale']


def test_independent_own_evidence_and_nested_reporting_are_preserved():
    reply = 'The companion asked.'
    records = [dict(speaker='tutor', session='s', text='The companion asked.'),
               dict(speaker='companion', session='s', text='Do you use this notebook?')]
    payload = bind(interpretation(reply), records=records)
    assert parse(assessment([finding(record=1)]), payload)['decision'] == 'accept'
    scoped = 'The tutor reports that the companion asked.'
    payload = bind(interpretation(scoped, actors=['tutor','companion']), records=records[:1])
    assert parse(assessment([finding(quote=records[0]['text'])]), payload)['decision'] == 'accept'


@pytest.mark.parametrize('reply', ['These records do not show I asked.', '这些记录没有显示我问过。'])
def test_record_scope_is_distinct_from_history_and_wrong_scope_requires_improvement(reply):
    payload = bind(interpretation(reply, actors=[], kind='record_scope'), records=[])
    correct = parse(assessment([finding(kind='scope_matches', record=None)]), payload)
    assert correct['decision'] == 'accept' and correct['criteria']['grounded']
    incorrect = parse(assessment([finding(kind='scope_mismatch', record=None)]), payload)
    assert incorrect['decision'] == 'accept' and incorrect['criteria']['grounded']
    assert not incorrect['criteria']['abstains_appropriately']
    assert 'Appropriate evidence use: False' in incorrect['rationale']
    assert grounding.runtime_memory_action(incorrect, evidence=payload)['action'] == 'improve'
    denial = bind(interpretation('I never asked.'), records=[])
    assert parse(assessment([finding(record=None)]), denial)['parse_failed']
    assert parse(assessment([finding(kind='not_established', record=None)]), denial)['decision'] == 'reject'


def test_competing_readings_are_not_flattened_and_definite_unsupported_tail_overrides_ambiguity():
    reply = 'Yes. The tutor asked. You confirmed it.'
    own = dict(text='Yes.', kind='history', actors=['companion'])
    other = dict(text='The tutor asked.', kind='history', actors=['tutor'])
    tail = dict(text='You confirmed it.', kind='history', actors=['user'])
    readings = [[dict(text=reply, claims=[own, other])], [dict(text=reply, claims=[other])]]
    records = [dict(speaker='tutor', session='s', text='Asked about the notebook.')]
    payload = bind(interpretation(reply, readings=readings), records=records)
    values = [finding(kind='not_established', record=None), finding(1, quote=records[0]['text']),
              finding(2, quote=records[0]['text'])]
    result = parse(assessment(values), payload)
    assert result['decision'] == 'inconclusive'
    assert grounding.runtime_memory_action(result, evidence=payload)['action'] == 'review'
    for reading in readings:
        reading[0]['claims'].append(tail)
    payload = bind(interpretation(reply, readings=readings), records=records)
    values = [finding(0, 'not_established', record=None), finding(1, quote=records[0]['text']),
              finding(2, 'not_established', record=None), finding(3, quote=records[0]['text']),
              finding(4, 'not_established', record=None)]
    result = parse(assessment(values), payload)
    assert result['decision'] == 'reject' and result['reading_outcomes'] == ['unsupported','unsupported']


def test_transfer_requires_contrary_actor_evidence_and_refusal_is_not_fabrication():
    records = [dict(speaker='tutor', session='s', text='I like hiking.')]
    payload = bind(interpretation('You said this.', actors=['user']), records=records)
    result = parse(assessment([finding(kind='speaker_transfer', quote=records[0]['text'])]), payload)
    assert result['decision'] == 'reject' and not result['criteria']['attribution_correct']
    assert parse(assessment([finding(kind='speaker_transfer', record=None)]), payload)['parse_failed']
    reply = 'I will not discuss that.'
    payload = bind(interpretation(reply, readings=[[dict(text=reply, claims=[])]], engagement='not_engaged'))
    result = parse(assessment(evidence_use='needless_refusal'), payload)
    assert result['decision'] == 'accept' and not result['criteria']['critical_error']
    assert grounding.runtime_memory_action(result, evidence=payload)['action'] == 'improve'
    payload['INTERPRETATION']['engagement'] = 'engaged'
    assert parse(assessment(evidence_use='needless_refusal'), payload)['parse_failed']


@pytest.mark.parametrize('check', ['not_faithful','unresolved',None])
def test_wrong_inventory_requires_review_without_credit_for_semantic_ambiguity(check):
    payload = bind(interpretation('I asked about the notebook.'))
    result = parse(assessment([finding()], check=check), payload)
    assert result['parse_failed'] and result['decision'] is None and not result['inconclusive']
    assert grounding.runtime_memory_action(result, evidence=payload)['reason'] == 'invalid_assessment'


def test_source_separation_is_not_a_proof_of_semantic_extraction_or_entailment():
    reply = 'You told me last winter that you loved skiing.'
    payload = bind(interpretation(reply, readings=[[dict(text=reply, claims=[])]]), records=[])
    # Two semantically wrong model outputs can still pass structural checks.
    # Preserve this limitation explicitly rather than claiming all history is found.
    result = parse(assessment(), payload)
    assert result['decision'] == 'accept'
    assert result['evidence_validation_scope'] == 'fixed_claims_structure_and_citations_only'


def wire_binding(tmp_path, *, invalid_first=False, transport_failure=False, oversized=False):
    from tests.test_runtime_tongyi_judge import wire_judge
    entry, controls, protocol = proposal()
    def change(data):
        request = seen[-1]
        actual = json.loads(request['messages'][1]['content'])
        if set(actual) == {'APPLICATION','QUERY','REPLY'}:
            value = interpretation(actual['REPLY'], readings=[[dict(text=actual['REPLY'], claims=[])]])
            if invalid_first: value['decision'] = 'accept'
            if oversized:
                value['readings'] = [[dict(text=actual['REPLY'], claims=[dict(text=actual['REPLY'],
                    kind='history', actors=['user']) for _ in range(32)])] for _ in range(4)]
        else:
            if transport_failure:
                raise RuntimeError('AUTHORED TRANSPORT FAILURE AFTER FIRST COMPLETION')
            value = assessment()
        data['model'] = protocol['judge']['model']
        data['choices'][0]['message']['content'] = json.dumps(value, ensure_ascii=False)
    judge, budget, seen = wire_judge(tmp_path, protocol, response_change=change)
    return entry, controls, protocol, judge, budget, seen


@pytest.mark.parametrize('invalid_first', [False,True])
def test_both_sdk_calls_have_exact_input_binding_and_shared_audit(tmp_path, monkeypatch, invalid_first):
    from tests.test_runtime_tongyi_judge import ledger
    _, controls, protocol, judge, budget, seen = wire_binding(tmp_path, invalid_first=invalid_first)
    original = diagnostic.validate
    monkeypatch.setattr(diagnostic, 'validate', lambda report, items, current:
        original(report, items, current, ledger_rows=ledger(budget)))
    try:
        report = diagnostic.run(controls, protocol, judge)
        count = 16 if invalid_first else 32
        assert len(seen) == budget.attempts == count and not budget.pending
        assert not report['runtime_enabled'] and not report['qualification_eligible'] and not report['passed']
        assert not report['metrics']['candidate_reference_screen_passed']
        assert report['metrics']['arms']['8']['provider_calls'] == count
        assert report['metrics']['arms']['8']['interpretation_calls'] == 16
        assert report['metrics']['arms']['8']['assessment_calls'] == (0 if invalid_first else 16)
        assert report['metrics']['arms']['8']['skipped_assessments'] == (16 if invalid_first else 0)
        assert report['metrics']['arms']['8']['invalid_interpretations'] == (16 if invalid_first else 0)
        assert Decimal(report['metrics']['arms']['8']['accounted_usd']) == budget.spent
        for row in report['rows']:
            physical = study.runtime_assessment_calls(row, protocol)
            assert len(physical) == (1 if invalid_first else 2)
            assert row['interpretation']['parse_failed'] is invalid_first
        for index, request in enumerate(seen):
            assert request['model'] == protocol['judge']['model']
            assert request['enable_thinking'] is False and request['max_completion_tokens'] == 4096
            assert request['response_format']['json_schema']['schema'] == grounding.runtime_binding_schema()
            actual = json.loads(request['messages'][1]['content'])
            if invalid_first or index%2 == 0:
                assert set(actual) == {'APPLICATION','QUERY','REPLY'}
            else:
                assert actual['INTERPRETATION'] == report['rows'][index//2]['interpretation']['interpretation']
        for mutation in ('packet','first-raw','call-alias','duplicate','missing','schema','wire','cost','action'):
            changed = copy.deepcopy(report); events = ledger(budget)
            row = changed['rows'][0]
            if mutation == 'packet': row['interpretation']['validation_error'] = 'forged'
            elif mutation == 'first-raw': row['calls'][0]['completion']['content'] = '{}'
            elif mutation == 'call-alias': row['call']['accounted_usd'] = '0'
            elif mutation == 'duplicate': changed['rows'][1]['calls'][0] = copy.deepcopy(row['calls'][0])
            elif mutation == 'missing': row['calls'] = []
            elif mutation == 'schema': row['calls'][0]['completion']['metadata']['response_schema_sha256'] = 'forged'
            elif mutation == 'wire':
                next(event for event in events if event['event']=='reservation')['messages'][1]['content'] = '{}'
            elif mutation == 'cost': row['calls'][0]['accounted_usd'] = '0'
            else: row['disposition']['action'] = 'forged'
            with pytest.raises(ValueError): original(changed, controls, protocol, ledger_rows=events)
    finally:
        judge.router.providers['tongyi']._client.close()
        budget.writer.close()


@pytest.mark.parametrize('failure', ['transport','oversized'])
def test_second_stage_failure_never_retries_and_retains_first_call_cost(tmp_path, failure):
    from tests.test_runtime_tongyi_judge import ledger
    _, controls, protocol, judge, budget, seen = wire_binding(tmp_path,
        transport_failure=failure=='transport', oversized=failure=='oversized')
    try:
        error = pytest.importorskip('openai').APIConnectionError if failure=='transport' else ValueError
        with pytest.raises(error):
            study.assess_calibration_control('memory', controls[0], protocol, judge)
        expected = 2 if failure=='transport' else 1
        assert len(seen) == budget.attempts == expected and not budget.pending
        events = ledger(budget)
        assert sum(event['event']=='completion' for event in events) == 1
        assert budget.spent > 0
    finally:
        judge.router.providers['tongyi']._client.close()
        budget.writer.close()


@pytest.mark.parametrize('version,revision', [(4,7),(3,8),(True,8),(4,True)])
def test_implicit_or_mismatched_contracts_are_rejected(version, revision):
    with pytest.raises(ValueError): grounding.runtime_memory_protocol(contract_version=version,prompt_revision=revision)


@pytest.mark.parametrize('invalid_first,allowed_calls', [(False,2),(True,2),(False,1)])
def test_shared_executor_counts_physical_calls_instead_of_case_rows(tmp_path, monkeypatch, invalid_first, allowed_calls):
    from contextlib import contextmanager
    _, controls, protocol, judge, budget, seen = wire_binding(tmp_path, invalid_first=invalid_first)
    budget.writer.path.touch()
    plan = dict(endpoint=calls.TONGYI_JUDGE_ENDPOINT, baseline_ledger_sha256=study.shared.digest(budget.writer.path),
        baseline=dict(provider_attempts=0, dispatches=0, cache_hits=0, accounted_usd='0'),
        additional_attempts=allowed_calls, additional_conservative_usd_cap='.277142592')
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps(plan), encoding='utf-8')
    approval=tmp_path/'approval.json';approval.write_text(json.dumps(dict(approved=True,
        plan_sha256=study.shared.digest(plan_path), basis='AUTHORED SOFTWARE APPROVAL')), encoding='utf-8')
    @contextmanager
    def local_budget(current):
        assert current == protocol
        yield budget
    def single_assessment(items,current,router,*,review,on_assessment):
        row=dict(index=0,control=items[0]['id'],prompt_revision=8,
                 **study.assess_calibration_control('memory',items[0],current,router))
        on_assessment(row)
        return dict(rows=[row],passed=False)
    monkeypatch.setattr(study,'study_budget',local_budget)
    try:
        options=dict(
            validator=lambda path:(plan,study.load_config(),controls,None,protocol),
            run_directory=tmp_path/'run',assess=single_assessment,ledger=budget.writer.path,
            judge_factory=lambda *args:judge)
        if allowed_calls==1:
            with pytest.raises(ValueError,match='accounting'):
                study.execute_runtime_study(approval,plan_path,**options)
            assert len(seen)==2 and not budget.pending
            assert (tmp_path/'run/abort.json').exists()
            assert not (tmp_path/'run/result.json').exists()
            assert len((tmp_path/'run/judgments.jsonl').read_text(encoding='utf-8').splitlines())==1
            return
        study.execute_runtime_study(approval,plan_path,**options)
        result=study.read(tmp_path/'run/result.json')
        assert result['judgments']==1 and result['provider_calls']==len(seen)==(1 if invalid_first else 2)
        assert not (tmp_path/'run/abort.json').exists()
    finally:
        judge.router.providers['tongyi']._client.close()
        budget.writer.close()


@pytest.mark.parametrize('key', ['model','enable_thinking','response_schema','call_policy'])
def test_binding_transport_spec_cannot_drift_silently(key):
    spec=calls.tongyi_binding_spec()
    if key=='model': spec[key]='another-model'
    elif key=='enable_thinking': spec[key]=True
    elif key=='response_schema': spec[key]['additionalProperties']=True
    else: spec[key]['version']=4
    with pytest.raises(ValueError): calls.hosted_request_options(spec)
