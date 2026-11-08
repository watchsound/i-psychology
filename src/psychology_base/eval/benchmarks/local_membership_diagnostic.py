"""Shared bounded local focus executor; packet producers never own dispatch."""
from dataclasses import asdict, replace
from contextlib import nullcontext
import json
import math
import socket
import time
from urllib.request import Request

from psychology_base.applications.measurement.local_query_binding import LocalChatMembershipAssessor, MembershipOutputError
from psychology_base.eval.benchmarks.application_calls import LocalNativeCapture
from psychology_base.eval.benchmarks.local_membership_audit import membership_wire_payload, assessment_output_field
from psychology_base.llm.config import default_config
from psychology_base.llm.execution import ExecutionPolicy
from psychology_base.llm.providers.ollama import OllamaProvider
from psychology_base.llm.providers.llama_cpp import LlamaCppProvider
from psychology_base.llm.local_backend import local_memory_admission, native_spec_from_record, OwnedLlamaServer
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.accounting import request_input_bound
from psychology_base.llm.types import ChatMessage


def _error_chain(exc):
    """Retain provider causes hidden by the router's availability envelope."""
    values, seen = [], set()
    while exc is not None and id(exc) not in seen and len(values) < 8:
        seen.add(id(exc))
        values.append(repr(exc))
        exc = exc.__cause__
    return values


def diagnostic_router(plan, *, out=None):
    """Construct the real local stack for offline compatibility and execution."""
    limits = plan['limits']
    if type(plan.get('stop_on_label_error', False)) is not bool:
        raise ValueError('Diagnostic label-error stop must be explicit boolean')
    if type(plan.get('stop_on_semantic_error', False)) is not bool:
        raise ValueError('Diagnostic semantic-error stop must be explicit boolean')
    if plan.get('stop_on_label_error', False) and plan.get('assessment_kind', 'membership') != 'membership':
        raise ValueError('Label-error stop requires the membership grading protocol')
    if type(plan.get('resident_memory_admission',False)) is not bool:
        raise ValueError('Resident memory admission must be an explicit boolean')
    if plan.get('resident_memory_admission',False):
        local_memory_admission(plan,dict(models=[]),0)
    cfg = default_config()
    cfg = replace(cfg, cache=replace(cfg.cache, enabled=False), execution=ExecutionPolicy(mode='local',
        request_timeout_s=limits['request_timeout_s'], operation_timeout_s=limits['operation_timeout_s'],
        offline_local_diagnostic=plan.get('offline_local_diagnostic', False)))
    transport = plan.get('transport', 'ollama')
    if transport not in ('ollama', 'llama_cpp'):
        raise ValueError('Unknown local diagnostic transport')
    native_operation_scope = plan.get('native_operation_scope', 'scope')
    if (native_operation_scope not in ('scope', 'condition')
            or (native_operation_scope == 'condition' and transport != 'llama_cpp')):
        raise ValueError('Per-condition native operations require the explicit native transport')
    if ('native_backend' in plan) != (transport == 'llama_cpp'):
        raise ValueError('Native backend controls require the explicit transport')
    if transport == 'llama_cpp':
        if (out is None or plan.get('assessment_kind','membership') not in ('membership', 'interpretation-selections-v2')
                or plan.get('think') is not True or plan.get('reasoning_budget') is not True):
            raise ValueError('Native diagnostic requires its explicit thinking membership scope')
        spec = native_spec_from_record(plan['native_backend'])
        if (spec.model.model_id != plan['model'] or spec.model.source_digest != plan['digest']
                or spec.context_length != limits['context_length'] or spec.max_tokens != limits['max_tokens']
                or type(plan['seed']) is not int or spec.seed != plan['seed']):
            raise ValueError('Native backend differs from the frozen assessment scope')
        provider = LlamaCppProvider(OwnedLlamaServer(spec, log_path=out/'backend.log'),
            request_timeout_s=limits['request_timeout_s'], model_metadata_timeout_s=limits['metadata_timeout_s'])
        return LLMRouter(cfg, providers={}, local_chat_provider=provider)
    provider = OllamaProvider(base_url='http://127.0.0.1:11434', simple_model=plan['model'], advanced_model=plan['model'],
        request_timeout_s=limits['request_timeout_s'], model_metadata_timeout_s=limits['metadata_timeout_s'],
        model_metadata_ttl_s=limits['metadata_ttl_s'], local_only=True, expected_digest=plan['digest'],
        context_length=limits['context_length'], seed=plan['seed'], json_mode=True, think=plan.get('think'))
    provider.validate_metadata_settings(ttl=provider.model_metadata_ttl_s)
    if plan.get('assessment_kind') == 'reference-extraction-v2':
        from psychology_base.applications.measurement.reference_readings import reference_response_schema
        provider.response_schema = reference_response_schema()
    return LLMRouter(cfg, providers={'ollama': provider})


def diagnostic_assessor(router, plan, entry, *, reference_scope=None):
    limits = plan['limits']
    return LocalChatMembershipAssessor(router, prompt_profile=entry['prompt_profile'],
        format_version=plan['contract_version'], descriptions_per_call=entry['descriptions_per_call'],
        max_input=limits['max_input'], max_tokens=limits['max_tokens'],
        assessment_kind=plan.get('assessment_kind', 'membership'), reference_scope=reference_scope,
        reasoning_budget=plan.get('reasoning_budget', False))


def run_local_membership_diagnostic(*, out, plan_path, result_path, archive, rows_for, grade):
    """Execute a frozen schedule; rows_for receives exactly one case and entry.

    The historical single-call runner is preserved with its consumed artifacts.
    This parameterized owner supports one/two calls without hiding either call.
    References are read only after inference and never supplied to the assessor.
    """
    base = archive.base
    if result_path.exists() or (out/'started.json').exists():
        raise FileExistsError('Never replace a consumed diagnostic')
    plan = base.read(plan_path)
    archive.validate_frozen_sources(plan, out)
    limits = plan['limits']
    if plan.get('think') is not None and type(plan['think']) is not bool:
        raise ValueError('Frozen thinking control must be a boolean or absent')
    assessment_kind = plan.get('assessment_kind', 'membership')
    output_field = assessment_output_field(assessment_kind)
    if (limits['paid_api_usd'] != '0' or plan['paid_calls'] is not False
            or base.sha(out/'inputs.json') != plan['input_sha256']
            or base.sha(out/'references.json') != plan['references_sha256']):
        raise ValueError('Frozen input or local-only limits changed')
    inputs = base.read(out/'inputs.json')
    cases, schedule = inputs['cases'], inputs['schedule']
    if len({e['id'] for e in schedule}) != len(schedule):
        raise ValueError('Duplicate scheduled condition')
    prepared, subsets = {}, {}
    for entry in schedule:
        size = entry['descriptions_per_call']
        if type(size) is not int or size not in (1, 2):
            raise ValueError('Invalid scheduled grouping')
        rows = rows_for(cases[entry['case']], entry)
        active = [i for i, row in enumerate(rows) if row['constraint'] is None]
        prepared[entry['id']] = rows
        subsets[entry['id']] = [active[i:i+size] for i in range(0, len(active), size)]
    if sum(map(len, subsets.values())) != limits['max_native_attempts']:
        raise ValueError('Native request allowance differs from the complete frozen schedule')
    native_started = time.monotonic() if plan.get('transport') == 'llama_cpp' else None
    router = diagnostic_router(plan, out=out) if native_started is not None else diagnostic_router(plan)
    provider = getattr(router, '_diagnostic_chat', None) or router.providers['ollama']
    native_spec = provider.configuration() if isinstance(provider, LlamaCppProvider) else None
    process_record = dict(spec=native_spec) if native_spec is not None else None

    def assessor_for(entry):
        scope = None
        if assessment_kind == 'reference-selections-v1':
            from psychology_base.applications.measurement.reference_readings import reference_selection_schema
            scope = json.loads(prepared[entry['id']][0]['pair'][0])
            provider.response_schema = reference_selection_schema(scope)
        elif assessment_kind == 'interpretation-selections-v2':
            scope = json.loads(prepared[entry['id']][0]['pair'][0])
        return diagnostic_assessor(router, plan, entry, reference_scope=scope)

    for entry in schedule:
        assessor_for(entry)
        for indices in subsets[entry['id']]:
            payload = membership_wire_payload([prepared[entry['id']][i] for i in indices], plan['model'],
                prompt_profile=entry['prompt_profile'], context_length=limits['context_length'],
                max_tokens=limits['max_tokens'], seed=plan['seed'],
                assessment_kind=assessment_kind, think=plan.get('think'), native_spec=native_spec)
            if request_input_bound([ChatMessage(**m) for m in payload['messages']],
                    response_schema=provider.response_schema) > limits['max_input']:
                raise ValueError('Frozen assessment input exceeds its bound')
    started = native_started if native_started is not None else time.monotonic()
    deadline = started + limits['max_wall_s'] - limits['finalization_reserve_s']
    shared_deadline = plan.get('shared_deadline_monotonic')
    if shared_deadline is not None:
        if (type(shared_deadline) not in (int,float) or not math.isfinite(shared_deadline)
                or shared_deadline <= started + limits['finalization_reserve_s']):
            raise ValueError('Shared diagnostic deadline leaves no inference allowance')
        deadline = min(deadline, shared_deadline - limits['finalization_reserve_s'])
    base.save(out/'started.json', dict(date=plan['date'], paid_api_usd='0', monotonic=started, wall_ns=time.time_ns()))
    observations = []
    result = dict(schema='local-membership-diagnostic-results-v1', date=plan['date'], qualified=False,
        defaults_changed=False, paid_api_usd='0', training_steps=0, holdout_calls=0)
    if assessment_kind != 'membership':
        result['assessment_kind'] = assessment_kind
    if shared_deadline is not None:
        result['shared_deadline_monotonic'] = shared_deadline
    active = entry = None

    def validate_chat(payload):
        if active is None or entry is None or not active.last_calls:
            raise ValueError('Only a scheduled assessor may dispatch')
        index = len(active.last_calls)-1
        previous = [e for e in capture.events if e['path']==capture.chat_path and e['condition']==entry['id']]
        if index != len(previous) or index >= len(subsets[entry['id']]):
            raise ValueError('Unscheduled or repeated call within the focus')
        indices = subsets[entry['id']][index]
        expected = membership_wire_payload([prepared[entry['id']][i] for i in indices], plan['model'],
            prompt_profile=entry['prompt_profile'], context_length=limits['context_length'],
            max_tokens=limits['max_tokens'], seed=plan['seed'], assessment_kind=assessment_kind,
            think=plan.get('think'), native_spec=native_spec)
        call = active.last_calls[index]
        if (call['row_indices'] != indices or payload != expected
                or [asdict(m) for m in call['messages']] != expected['messages']):
            raise ValueError('Native request differs from its scheduled canonical subset')

    capture = LocalNativeCapture(provider, max_attempts=limits['max_native_attempts'], deadline=deadline,
        max_metadata=limits['max_metadata_attempts'], validate_chat=validate_chat,
        observer=lambda event, phase:base.save(out/f"native-{event['sequence']:03d}-{phase}.json", event))
    original_connect = socket.socket.connect
    port = native_spec['port'] if native_spec is not None else 11434
    def connect(sock, address):
        if not isinstance(address, tuple) or address[:2] != ('127.0.0.1', port):
            raise ValueError('Only the pinned local diagnostic endpoint is allowed')
        return original_connect(sock, address)
    socket.socket.connect = connect
    try:
        # Historical native sentinels share startup and generation in one
        # operation. An explicit batch scope bounds startup separately; the
        # assessor already owns one operation for each complete condition.
        scope_operation = native_spec is not None and plan.get('native_operation_scope', 'scope') == 'scope'
        with capture, (router.operation('native_membership_diagnostic') if scope_operation else nullcontext()):
            capture.condition = 'preflight'
            with router.operation('membership_diagnostic_preflight'):
                if native_spec is not None:
                    provider.server.start()
                    process_record['identity'] = provider.server.assert_owned(listener=False)
                    process_record['argv'] = provider.server.spec.argv()
                    base.save(out/'native-process-started.json', process_record)
                    base.save(out/'memory-admission.json', dict(decision=provider.server.memory_admission,
                        monotonic=time.monotonic(), wall_ns=time.time_ns()))
                    for poll in range(9):
                        try:
                            if provider.metadata('/health') == {'status':'ok'}:
                                break
                        except (OSError, ValueError):
                            provider.server.assert_owned(listener=False)
                        if poll == 8:
                            raise TimeoutError('Native backend did not become ready within its metadata allowance')
                        from psychology_base.llm.execution import remaining_time
                        time.sleep(min(30, remaining_time(30)))
                result['attestation'] = provider.attest_model(plan['model'])
                if native_spec is None:
                    with provider._open(Request(provider.base_url+'/api/ps'), timeout=limits['metadata_timeout_s']) as response:
                        result['residency_preflight'] = json.load(response)
                if native_spec is None and plan.get('resident_memory_admission',False):
                    import psutil
                    result['memory_admission'] = local_memory_admission(plan,result['residency_preflight'],
                        psutil.virtual_memory().available)
                    base.save(out/'memory-admission.json',dict(decision=result['memory_admission'],
                        monotonic=time.monotonic(),wall_ns=time.time_ns()))
                    if not result['memory_admission']['admitted']:
                        raise MemoryError('Insufficient available memory for the observed local model residency')
            for entry in schedule:
                if deadline-time.monotonic() < limits['operation_timeout_s']:
                    raise TimeoutError('Insufficient allowance for a complete bounded condition')
                capture.condition = entry['id']
                active = assessor_for(entry)
                observation = dict(id=entry['id'], assessor_identity=active.identity, attempt_before=capture.attempts)
                observations.append(observation)
                at = time.monotonic()
                try:
                    observation[output_field] = active(prepared[entry['id']])
                except MembershipOutputError as exc:
                    observation.update(error=repr(exc), format_failure=True)
                except Exception as exc:
                    observation.update(error=repr(exc), format_failure=False)
                    if native_spec is not None:
                        observation['error_chain'] = _error_chain(exc)
                    raise
                finally:
                    observation.update(elapsed_ms=(time.monotonic()-at)*1000, attempt_after=capture.attempts,
                        native_calls=[dict(messages=[asdict(m) for m in c['messages']], row_indices=c['row_indices'],
                            completion=asdict(c['completion']) if 'completion' in c else None) for c in active.last_calls])
                    base.save(out/f'observation-{len(observations):02d}.json', observation)
                print(json.dumps(dict(condition=entry['id'], completed=output_field in observation,
                    attempts=capture.attempts, elapsed_s=time.monotonic()-started)), flush=True)
                if plan.get('stop_on_label_error', False) or plan.get('stop_on_semantic_error', False):
                    scored = grade(cases, base.read(out/'references.json'), schedule, observations)
                    current = next(r for r in scored['records'] if r['id'] == entry['id'])
                    if not current['correct']:
                        result.update(completed=False, stopped_at=entry['id'], stop_reason='semantic_gate')
                        break
            else:
                result['completed'] = True
    except Exception as exc:
        result.update(completed=False, stopped_at=capture.condition, error=repr(exc))
        if native_spec is not None:
            result['error_chain'] = _error_chain(exc)
    finally:
        try:
            if native_spec is not None:
                result['memory_admission'] = provider.server.memory_admission
                try:
                    provider.server.close()
                except Exception as exc:
                    result.update(completed=False, cleanup_error=repr(exc))
        finally:
            socket.socket.connect = original_connect
        if native_spec is not None:
            process_record['cleanup'] = provider.server.cleanup
            if not (out/'memory-admission.json').exists():
                base.save(out/'memory-admission.json', dict(decision=provider.server.memory_admission,
                    monotonic=time.monotonic(), wall_ns=time.time_ns()))
            base.save(out/'native-process.json', process_record)
        base.save(out/'observations.json', observations)
        base.save(out/'transport.json', capture.events)
        result.update(native_attempts=capture.attempts, metadata_attempts=capture.metadata_attempts,
            graded=grade(cases, base.read(out/'references.json'), schedule, observations), wall_s=time.monotonic()-started)
        archive.validate_frozen_sources(plan, out)
        base.save(out/'result.json', result)
        base.save(result_path, dict(result, plan_sha256=base.sha(plan_path), output_pins={
            p.relative_to(out).as_posix():base.sha(p) for p in out.rglob('*')
            if p.is_file() and 'executed-source' not in p.parts}))
        print(json.dumps(dict(completed=result['completed'], native_attempts=capture.attempts,
            summary=result['graded']['summary'], wall_s=result['wall_s'])), flush=True)
