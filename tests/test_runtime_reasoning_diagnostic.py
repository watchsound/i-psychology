"""Authored transport/metric controls; these tests make no model-accuracy claim."""
import copy
import json
from decimal import Decimal

import pytest

from psychology_base.eval.benchmarks import application_context as study
from psychology_base.eval.benchmarks import application_calls as calls
from psychology_base.eval.benchmarks import runtime_report_diagnostic as diagnostic
from tests.test_runtime_semantic_diagnostic import runner
from tests.test_runtime_decision_separation import authored_audit_row


BASELINE_SHA = 'a08a572062dcb3d5d5cdf1735b6c4c7f3cf04b019090f939d2b78287cd672d6b'


@pytest.fixture(scope='module')
def proposal(tmp_path_factory):
    entry = runner()
    path = tmp_path_factory.mktemp('reasoning-historical-ledger') / 'ledger.jsonl'
    prefix = b''.join(line for line in entry.LEDGER.read_bytes().splitlines(keepends=True)
                      if json.loads(line)['dispatch'] <= 996)
    path.write_bytes(prefix)
    assert study.shared.digest(path) == BASELINE_SHA
    entry.LEDGER = path
    return entry, diagnostic.items(revision=4), entry.protocol_for(entry.baseline(revision=4), revision=4)


def metric_rows(controls):
    rows = []
    by_id = {item['id']: item for item in controls}
    for entry in diagnostic.schedule(controls, revision=4):
        item = by_id[entry['control']]
        raw = json.dumps(dict(item['expected_criteria'] or dict(addresses_request=True, grounded=True,
            attribution_correct=True, abstains_appropriately=True, critical_error=False),
            semantic_status='inconclusive' if item['expected_decision'] == 'inconclusive' else 'resolved',
            rationale='AUTHORED METRIC CONTROL; NOT A MODEL RESPONSE'))
        judgment = study.parse_runtime_memory(raw, 'stop', False, contract_version=2)
        row, _, _, _ = authored_audit_row()
        row.update(entry, judgment=judgment)
        row['call']['completion'].update(content=raw, finish_reason='stop', prompt_tokens=100, completion_tokens=70)
        row['call']['completion']['metadata'].update(reasoning_tokens=30 if entry['inference_arm'] == 'low' else 0)
        rows.append(row)
    return rows


def test_new_comparison_reuses_all_exposed_cases_and_fixes_every_other_setting(proposal):
    entry, controls, protocol = proposal
    assert controls == diagnostic.items(revision=3)
    requests = entry.requests(controls, protocol)
    assert len(requests) == 98
    for index in range(49):
        first, second = requests[2*index:2*index+2]
        assert first['messages'] == second['messages']
        assert first['prompt_revision'] == second['prompt_revision'] == 4
        assert [first['inference_arm'], second['inference_arm']] == (
            ['none', 'low'] if index % 2 == 0 else ['low', 'none'])
        for request in (first, second):
            assert request['max_tokens'] == 2048 and request['temperature'] is None
            assert set(json.loads(request['messages'][1]['content'])) == {'APPLICATION','QUERY','VISIBLE_CONTEXT','REPLY'}
            assert request['bounded_input_bytes'] <= 12000
    assert entry.profile(4).cap == Decimal('0.247352')
    assert protocol['budget'] == dict(max_calls=1094, max_usd='0.374602525', local_only=False)


def test_metrics_compare_reasoning_arms_and_separate_budget_limited_output(proposal):
    _, controls, _ = proposal
    rows = metric_rows(controls)
    result = diagnostic.metrics(controls, rows, revision=4)
    assert set(result['arms']) == {'none','low'}
    assert result['candidate_reference_screen_passed'] and result['paired']['both_match'] == 49
    assert result['arms']['low']['reasoning_tokens'] == 49 * 30
    assert result['arms']['none']['reasoning_tokens'] == 0
    candidate = next(row for row in rows if row['inference_arm'] == 'low')
    candidate['call']['completion'].update(content='', finish_reason='length', completion_tokens=2048)
    candidate['judgment'] = study.parse_runtime_memory('', 'length', False, contract_version=2)
    result = diagnostic.metrics(controls, rows, revision=4)
    assert result['arms']['low']['budget_limited'] == 1
    assert result['arms']['low']['empty_outputs'] == 1
    assert not result['candidate_reference_screen_passed'] and result['paired']['regressed'] == 1
    assert not diagnostic.metrics(controls, [], revision=4)['candidate_reference_screen_passed']


def test_wrong_arm_or_duplicate_pair_is_rejected(proposal):
    _, controls, protocol = proposal
    rows = metric_rows(controls)
    rows[1]['inference_arm'] = rows[0]['inference_arm']
    with pytest.raises(ValueError, match='schedule'): diagnostic.metrics(controls, rows, revision=4)
    with pytest.raises(ValueError): diagnostic.condition_protocol(protocol, dict(prompt_revision=5, inference_arm='low'))
    with pytest.raises(ValueError): diagnostic.condition_protocol(protocol, dict(prompt_revision=4, inference_arm='medium'))


@pytest.mark.parametrize('change', ['max_tokens','temperature','version','extra','effort','model','null'])
def test_widening_output_requires_the_exact_declared_policy(proposal, change):
    _, _, protocol = proposal
    spec = copy.deepcopy(protocol['judge'])
    if change == 'extra': spec['call_policy']['extra'] = True
    elif change == 'effort': spec['reasoning_effort'] = 'high'
    elif change == 'model': spec['model'] = 'another-model'
    elif change == 'null': spec['call_policy'] = None
    else: spec['call_policy'][change] = {'max_tokens':2049,'temperature':0,'version':True}[change]
    with pytest.raises(ValueError): calls.inference_call_options(spec)


@pytest.mark.parametrize('effort', ['none','low'])
def test_real_adapter_sdk_wire_and_total_token_billing(tmp_path, proposal, effort):
    """SDK transport is injected; the real adapter/recorder/accounting still run."""
    from tests.test_application_hybrid_screen import hybrid_judge
    _, _, protocol = proposal
    def usage(data):
        data['usage'].update(completion_tokens=100)
        data['usage']['completion_tokens_details']['reasoning_tokens'] = 60 if effort == 'low' else 0
    _, judge, budget, requests, _ = hybrid_judge(tmp_path, response_change=usage)
    spec = diagnostic.condition_protocol(protocol, dict(prompt_revision=4, inference_arm=effort))['judge']
    judge.remote_spec = spec
    judge.router.providers['openai'].reasoning_effort = effort
    judge.chat([study.ChatMessage('user','JSON authored wire control')], tier=study.TaskTier.ADVANCED,
               max_tokens=2048, temperature=None, use_cache=False)
    assert len(requests) == budget.attempts == 1
    request = requests[0]
    assert request['reasoning_effort'] == effort and request['max_completion_tokens'] == 2048
    assert request['response_format'] == {'type':'json_object'} and request['service_tier'] == 'default'
    assert 'max_tokens' not in request and 'temperature' not in request
    assert budget.spent == (Decimal('.125')*12+Decimal('.50')*100)/1_000_000
    calls.validate_screening_call(judge.last_call, spec)
    budget.writer.close()


@pytest.mark.parametrize('reasoning,completion', [(True,10),(-1,10),(11,10),(None,10),(1,2049)])
def test_low_reasoning_usage_must_be_valid_and_within_total_completion(proposal, reasoning, completion):
    _, _, protocol = proposal
    spec = diagnostic.condition_protocol(protocol, dict(prompt_revision=4, inference_arm='low'))['judge']
    row, _, _, _ = authored_audit_row()
    call = row['call']
    call['completion']['metadata'].update(reasoning_effort='low', reasoning_tokens=reasoning)
    call['completion']['completion_tokens'] = completion
    call['accounted_usd'] = str((Decimal(spec['input_rate'])*call['completion']['prompt_tokens']+
                                Decimal(spec['output_rate'])*completion)/1_000_000)
    with pytest.raises(ValueError): calls.validate_screening_call(call,spec)


def test_legacy_call_limits_and_zero_reasoning_attestation_are_preserved(tmp_path):
    from tests.test_application_hybrid_screen import hybrid_judge
    _, judge, budget, requests, protocol = hybrid_judge(tmp_path)
    assert calls.inference_call_options(protocol['judge']) is None
    with pytest.raises(ValueError):
        judge.chat([study.ChatMessage('user','JSON unit')],tier=study.TaskTier.ADVANCED,max_tokens=2048)
    assert not requests and budget.attempts == 0
    row, _, _, _ = authored_audit_row()
    row['call']['completion']['metadata']['reasoning_tokens'] = 1
    with pytest.raises(ValueError): calls.validate_screening_call(row['call'],protocol['judge'])
    budget.writer.close()


def test_exact_output_and_temperature_checks_precede_network_and_reservation(tmp_path, proposal):
    from tests.test_application_hybrid_screen import hybrid_judge
    _, _, protocol = proposal
    _, judge, budget, requests, _ = hybrid_judge(tmp_path)
    judge.remote_spec = protocol['judge']
    for options in [dict(max_tokens=300,temperature=None), dict(max_tokens=2048,temperature=0)]:
        with pytest.raises(ValueError): judge.chat([],tier=study.TaskTier.ADVANCED,**options)
    assert not requests and budget.attempts == 0
    budget.writer.close()


def test_full_98_request_wire_loop_uses_two_stable_routers_one_budget_and_raw_audit(tmp_path, proposal, monkeypatch):
    """Invented SDK responses test plumbing end to end; never persisted as empirical results."""
    from psychology_base.applications.pilot.session_runner import SessionLogWriter
    openai = pytest.importorskip('openai')
    httpx = pytest.importorskip('httpx')
    _, controls, protocol = proposal
    budget = calls.Budget(SessionLogWriter(tmp_path/'events.jsonl'), max_calls=98, max_usd='.247352')
    judges = diagnostic.make_reasoning_judges(protocol, study.load_config(), budget)
    assert judges['none'] is not judges['low'] and all(j.budget is budget for j in judges.values())
    assert all(j.router.cache is None and set(j.router.providers) == {'openai'} for j in judges.values())
    expected = diagnostic.schedule(controls, revision=4)
    by_id = {item['id']: item for item in controls}
    wire = []
    def respond(request):
        payload = json.loads(request.content)
        entry = expected[len(wire)]
        assert payload['reasoning_effort'] == entry['inference_arm']
        assert payload['max_completion_tokens'] == 2048 and 'temperature' not in payload
        item = by_id[entry['control']]
        wanted = study.runtime_memory_payload(item['app'], item['capture']['query'], item['capture']['context'], item['response'])
        assert json.loads(payload['messages'][1]['content']) == wanted
        raw = json.dumps(dict(item['expected_criteria'] or dict(addresses_request=True, grounded=True,
            attribution_correct=True, abstains_appropriately=True, critical_error=False),
            semantic_status='inconclusive' if item['expected_decision'] == 'inconclusive' else 'resolved',
            rationale='AUTHORED SDK CONTROL; NOT MODEL EVIDENCE'))
        wire.append(payload)
        return httpx.Response(200,json=dict(id=f'AUTHORED-SDK-{len(wire)}',object='chat.completion',created=1,
            model='gpt-6-luna',service_tier='default',
            choices=[dict(index=0,message=dict(role='assistant',content=raw),finish_reason='stop')],
            usage=dict(prompt_tokens=100,completion_tokens=100,total_tokens=200,
                       completion_tokens_details=dict(reasoning_tokens=60 if entry['inference_arm']=='low' else 0))))
    for judge in judges.values():
        judge.router.providers['openai']._client = openai.OpenAI(api_key='authored-unit-key',max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    validate = diagnostic.validate
    def ledger_rows(): return [json.loads(line) for line in budget.writer.path.read_text(encoding='utf-8').splitlines()]
    monkeypatch.setattr(diagnostic,'validate',lambda report,controls,proto: validate(report,controls,proto,ledger_rows=ledger_rows()))
    progress = []
    report = diagnostic.run(controls,protocol,judges,on_assessment=progress.append)
    assert len(wire) == len(progress) == budget.attempts == budget.dispatches == 98
    assert not budget.pending and budget.cache_hits == 0 and not budget.uncertain
    assert budget.spent == Decimal('.006125')
    assert report['metrics']['candidate_reference_screen_passed']
    assert report['passed'] is report['runtime_enabled'] is report['qualification_eligible'] is False
    progress[0]['judgment']['raw'] = 'callback mutation'
    assert report['rows'][0]['judgment']['raw'] != 'callback mutation'
    for mutate in ('temperature','output_limit','effort','reused_id','disposition'):
        changed = copy.deepcopy(report)
        ledger = ledger_rows()
        if mutate == 'temperature': ledger[0]['temperature'] = 0
        elif mutate == 'output_limit': ledger[0]['max_tokens'] = 300
        elif mutate == 'effort': changed['rows'][0]['call']['completion']['metadata']['reasoning_effort'] = 'low'
        elif mutate == 'reused_id': changed['rows'][1]['call']['completion']['metadata']['completion_id'] = 'AUTHORED-SDK-1'
        else: changed['rows'][0]['disposition']['action'] = 'invented_action'
        with pytest.raises(ValueError): validate(changed,controls,protocol,ledger_rows=ledger)
    budget.writer.close()


def test_reasoning_reservation_blocks_overspend_before_sdk(tmp_path, proposal):
    from tests.test_application_hybrid_screen import hybrid_judge
    _, _, protocol = proposal
    _, judge, budget, requests, _ = hybrid_judge(tmp_path,dollars='.002523')
    judge.remote_spec = protocol['judge']
    with pytest.raises(ValueError,match='before dispatch'):
        judge.chat([study.ChatMessage('user','JSON unit')],tier=study.TaskTier.ADVANCED,max_tokens=2048)
    assert not requests and budget.attempts == 0
    budget.writer.close()


def test_new_cli_passes_only_the_judge_factory_to_the_existing_executor(proposal, monkeypatch):
    entry, _, _ = proposal
    captured = []
    monkeypatch.setattr(study,'execute_runtime_study',lambda *args,**kwargs: captured.append((args,kwargs)))
    entry.run('authored-approval.json',revision=4)
    assert len(captured) == 1
    args, kwargs = captured[0]
    assert kwargs['judge_factory'] is diagnostic.make_reasoning_judges
    assert kwargs['assess'] is diagnostic.run and kwargs['ledger'] == entry.LEDGER
    assert args[1] == entry.profile(4).plan


@pytest.mark.parametrize('revision',[True,4.0,'4',0,11])
def test_invalid_revision_stops_before_selecting_a_scope(revision):
    with pytest.raises(ValueError): runner().profile(revision)
    with pytest.raises(ValueError): diagnostic.items(revision=revision)


def test_prior_approval_cannot_authorize_the_new_reasoning_scope(tmp_path, monkeypatch):
    entry = runner()
    plan = tmp_path/'unapproved-new-scope.json'
    plan.write_text('{}',encoding='utf-8')
    prior = entry.profile(3).plan.with_name('runtime-semantic-resolution-diagnostic-hosted-approval-v1.json')
    def forbidden(*args,**kwargs): pytest.fail('Different approved scope must stop before validation/provider access')
    monkeypatch.setattr(entry,'validate',forbidden)
    with pytest.raises(ValueError,match='approval'): entry.run(prior,plan,revision=4)


@pytest.mark.parametrize('abort',[False,True])
def test_shared_executor_passes_locked_budget_and_first_party_config_to_factory(tmp_path, proposal, monkeypatch, abort):
    """Exercise the new factory hook and existing durable success/abort boundary locally."""
    from contextlib import contextmanager
    from types import SimpleNamespace
    _, controls, protocol = proposal
    ledger = tmp_path/'events.jsonl'
    ledger.write_bytes(b'')
    plan = dict(baseline_ledger_sha256=study.shared.digest(ledger),
        baseline=dict(provider_attempts=0, dispatches=0, accounted_usd='0',cache_hits=0),
        endpoint='https://api.openai.com/v1',additional_conservative_usd_cap='.247352')
    plan_path = tmp_path/'authored-plan.json'
    plan_path.write_text(json.dumps(plan),encoding='utf-8')
    approval = tmp_path/'authored-approval.json'
    approval.write_text(json.dumps(dict(approved=True,plan_sha256=study.shared.digest(plan_path))),encoding='utf-8')
    budget = SimpleNamespace(attempts=0,dispatches=0,spent=Decimal(0),cache_hits=0,pending=set())
    budget.summary = lambda: dict(authored_control=True,provider_attempts=budget.attempts)
    @contextmanager
    def locked(current):
        assert current is protocol
        yield budget
    monkeypatch.setattr(study,'study_budget',locked)
    sentinel = object()
    factories = []
    def factory(current,cfg,shared):
        assert current is protocol and shared is budget
        assert set(cfg.providers) == {'openai'}
        assert cfg.get_provider('openai').base_url == 'https://api.openai.com/v1'
        factories.append(True)
        return sentinel
    def assess(current_controls,current,judge,**kwargs):
        assert current_controls is controls and current is protocol and judge is sentinel
        row = dict(authored_control=True)
        kwargs['on_assessment'](row)
        budget.attempts += 1; budget.dispatches += 1
        if abort: raise RuntimeError('Authored injected failure; never retry')
        return dict(rows=[row],passed=False)
    directory = tmp_path/'authored-run'
    options = dict(validator=lambda path:(plan,study.load_config(),controls,None,protocol),
        run_directory=directory,assess=assess,ledger=ledger,judge_factory=factory)
    if abort:
        with pytest.raises(RuntimeError): study.execute_runtime_study(approval,plan_path,**options)
        assert study.read(directory/'abort.json')['automatic_retry'] is False
    else:
        study.execute_runtime_study(approval,plan_path,**options)
        assert study.read(directory/'result.json')['judgments'] == 1
    assert len(factories) == 1
    with pytest.raises(ValueError,match='already exists'): study.execute_runtime_study(approval,plan_path,**options)
    assert len(factories) == 1


@pytest.fixture(scope='module')
def prospective_packet(tmp_path_factory, proposal):
    """Rebuild local plans after execution using the pinned real 996-attempt prefix."""
    original_entry, controls, protocol = proposal
    entry = runner()
    entry.LEDGER = original_entry.LEDGER
    directory = tmp_path_factory.mktemp('reasoning-prospective-packet')
    selected = entry.profile(4)
    if not selected.plan.exists(): pytest.skip('Prepared after pre-freeze integration checks pass')
    summary = entry.baseline(revision=4)
    plan = dict(entry.fields(controls,protocol,study.load_config(),summary,revision=4),timestamp_utc='LOCAL-ONLY-PROSPECTIVE-PLAN')
    path = directory/'plan.json'
    entry.save(path,plan)
    assert entry.validate(path,revision=4)[0] == plan
    return entry, plan


@pytest.mark.parametrize('field,value',[
    ('additional_attempts',99),('shared_attempt_cap',1095),('prompt_revisions',[5]),
    ('additional_conservative_usd_cap','1'),('endpoint','https://example.com/v1'),
    ('qualification_eligible',True),('fixture_sha256','wrong'),('requests_sha256','wrong'),
    ('inference_conditions',{}),('previous_diagnostic_report',{})])
def test_frozen_reasoning_scope_and_budget_drift_stop_before_provider(prospective_packet,tmp_path,field,value):
    entry, original = prospective_packet
    changed = copy.deepcopy(original)
    changed[field] = value
    path = tmp_path/'drift.json'
    path.write_text(json.dumps(changed),encoding='utf-8')
    with pytest.raises(ValueError): entry.validate(path,revision=4)
