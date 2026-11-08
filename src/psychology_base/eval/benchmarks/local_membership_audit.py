"""Shared read-only replay of local membership captures; never grades semantics."""
import hashlib
import json
import math

from psychology_base.applications.measurement.local_query_binding import membership_prompt, REFERENCE_KINDS
from psychology_base.applications.measurement.memory_grounding import _unique_json_object
from psychology_base.llm.accounting import request_input_bound
from psychology_base.llm.types import ChatMessage


def paired_final_write_wall_s(started, anchor, final_wall_ns):
    """Bridge a pre-preparation monotonic start with a recorded paired clock.

    The anchor must be an unchanged journal containing adjacent monotonic/wall
    readings. A wall reading written after preparation is not the start clock.
    """
    start, at, wall = started.get('monotonic'), anchor.get('monotonic'), anchor.get('wall_ns')
    if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (start,at))
            or type(wall) is not int or type(final_wall_ns) is not int
            or not 0 < wall <= final_wall_ns or at < start):
        raise ValueError('Invalid paired experiment clock evidence')
    return at - start + (final_wall_ns - wall) / 1e9


def assessment_output_field(kind):
    if kind == 'membership':
        return 'labels'
    if kind in REFERENCE_KINDS:
        return 'analysis'
    raise ValueError('Unsupported local assessment kind')


def membership_wire_payload(rows, model, *, prompt_profile='baseline-v1', context_length=8192, max_tokens=64, seed=0,
                            assessment_kind='membership', think=None, native_spec=None):
    """Independent audit reconstruction; never composes an application request."""
    active = [r for r in rows if r['constraint'] is None]
    if not active or len({r['pair'][0] for r in rows}) != 1:
        raise ValueError('Noncanonical focus scope')
    assessment_output_field(assessment_kind)
    payload = dict(focus_input=json.loads(active[0]['pair'][0]))
    prompt = membership_prompt(prompt_profile)
    wire_format = 'json'
    if assessment_kind in REFERENCE_KINDS:
        from psychology_base.applications.measurement.local_query_binding import REFERENCE_PROMPT, REFERENCE_IDS_PROMPT
        from psychology_base.applications.measurement.reference_readings import (
            reference_turns, reference_response_schema, PROJECTION_VERSION,
        )
        if len(rows) != 2 or len(active) != 2 or prompt_profile != 'baseline-v1':
            raise ValueError('Extraction requires one complete active focus')
        focus = payload['focus_input']
        turns = reference_turns(focus)
        prompt = REFERENCE_PROMPT
        if assessment_kind == 'reference-extraction-v2':
            # Independently reconstruct IDs from the canonical window, never
            # trust the application projection or stored request as an oracle.
            payload = dict(schema=PROJECTION_VERSION, context_scope=focus['context_scope'],
                turns=[dict(id=i, speaker=t['speaker'], text=t['text']) for i,t in enumerate(turns)],
                focus_turn=len(turns)-1)
            prompt, wire_format = REFERENCE_IDS_PROMPT, reference_response_schema()
        if assessment_kind == 'reference-selections-v1':
            from psychology_base.applications.measurement.local_query_binding import REFERENCE_SELECTIONS_PROMPT
            from psychology_base.applications.measurement.reference_readings import reference_candidate_view, reference_selection_schema
            payload = reference_candidate_view(focus)
            prompt, wire_format = REFERENCE_SELECTIONS_PROMPT, reference_selection_schema(focus)
        if assessment_kind == 'interpretation-selections-v2':
            from psychology_base.applications.measurement.local_query_binding import INTERPRETATION_PROMPT
            from psychology_base.applications.measurement.reference_readings import interpretation_candidate_view
            payload = interpretation_candidate_view(focus)
            prompt = INTERPRETATION_PROMPT
    else:
        payload['descriptions'] = [r['pair'][1] for r in active]
    content = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    wire = dict(model=model, stream=False, format=wire_format,
        options=dict(temperature=0, num_predict=max_tokens, num_ctx=context_length, seed=seed),
        messages=[dict(role='system', content=prompt),dict(role='user', content=content)])
    if think is not None:
        if type(think) is not bool:
            raise ValueError('Thinking wire control must be a boolean or None')
        wire['think'] = think
    if native_spec is not None:
        if assessment_kind not in ('membership', 'interpretation-selections-v2') or think is not True:
            raise ValueError('Native transport requires the explicit thinking membership protocol')
        # Independently serialize the native boundary; canonical message scope
        # remains shared, while transport reconstruction is an audit cross-check.
        wire = dict(model=model, messages=wire['messages'], stream=False, max_tokens=max_tokens,
            temperature=0, seed=seed, response_format=dict(type='json_object'),
            chat_template_kwargs=dict(enable_thinking=True), reasoning_format='deepseek',
            reasoning_budget_tokens=native_spec['thinking_tokens'], cache_prompt=False, return_tokens=True, verbose=True)
    return wire


def audit_membership_capture(events, observations, schedule, *, rows_for, snapshot,
                             model, digest, max_native, max_metadata,
                             context_length=8192, max_tokens=64, max_input=8000, seed=0,
                             assessment_kind='membership', think=None, native_spec=None):
    """Reconstruct native payloads and derive labels only from captured responses.

    Scenario-specific references, contrasts and qualification remain outside
    this function. Failed or missing responses never acquire inferred labels.
    """
    def require(value, message):
        if not value:
            raise ValueError(message)

    output_field = assessment_output_field(assessment_kind)
    chat_path = '/v1/chat/completions' if native_spec is not None else '/api/chat'
    paths = ('/health', '/props', '/apply-template', chat_path) if native_spec is not None else (
        '/api/tags', '/api/show', '/api/ps', chat_path)
    process_record = None
    if native_spec is not None:
        from psychology_base.llm.providers.llama_cpp import validate_native_props, decode_native_response
        require(native_spec['model']['model_id'] == model and native_spec['model']['source_digest'] == digest
            and type(native_spec['context_length']) is int and native_spec['context_length'] == context_length
            and type(native_spec['max_tokens']) is int and native_spec['max_tokens'] == max_tokens
            and type(native_spec['seed']) is int and native_spec['seed'] == seed,
            'Native audit settings differ from the frozen scope')
        if events:
            process_record = snapshot('native-process.json')
            from psychology_base.llm.local_backend import native_argv
            started = snapshot('native-process-started.json')
            require(process_record.get('spec') == native_spec and process_record.get('cleanup', {}).get('verified_exit') is True
                and process_record['cleanup'].get('job_verified_empty') is True
                and process_record.get('argv') == native_argv(native_spec)
                and all(process_record.get(k) == v for k,v in started.items()),
                'Native process lacks frozen ownership or verified cleanup')
            expected_identity = hashlib.sha256(json.dumps(native_spec,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
            require(process_record.get('identity', {}).get('spec_sha256') == expected_identity
                and type(process_record['identity'].get('pid')) is int
                and type(process_record['identity'].get('created')) in (int,float), 'Native process identity changed')
    entries = {e['id']: e for e in schedule}
    require(len(entries) == len(schedule), 'Duplicate scheduled condition')
    native = [e for e in events if e['path'] == chat_path]
    require(len(native) <= max_native and len(events)-len(native) <= max_metadata, 'Capture exceeds request caps')
    rows_by_id = {key: rows_for(entry) for key, entry in entries.items()}
    chunks, payloads = {}, {}
    for key, entry in entries.items():
        size = entry.get('descriptions_per_call', 2)
        require(type(size) is int and size in (1, 2), 'Invalid scheduled description grouping')
        rows = rows_by_id[key]
        active = [i for i, row in enumerate(rows) if row['constraint'] is None]
        chunks[key] = [active[i:i+size] for i in range(0, len(active), size)]
        payloads[key] = [membership_wire_payload([rows[i] for i in subset], model,
            prompt_profile=entry.get('prompt_profile', 'baseline-v1'), context_length=context_length,
            max_tokens=max_tokens, seed=seed, assessment_kind=assessment_kind, think=think,
            native_spec=native_spec) for subset in chunks[key]]
    by_condition = {}
    positions = {key: i for i, key in enumerate(entries)}
    previous = -1
    for event in native:
        key = event['condition']
        require(key in entries, 'Unscheduled native condition')
        require(positions[key] >= previous, 'Reordered native conditions')
        previous = positions[key]
        by_condition.setdefault(key, []).append(event)
        require(len(by_condition[key]) <= len(chunks[key]), 'Repeated native dispatch')
    offsets, thinking_sources, last_thinking_source = {}, {}, None
    rendered, native_props = {}, None
    for sequence, event in enumerate(events, 1):
        require(event['sequence'] == sequence, 'Native event sequence changed')
        require(snapshot(f'native-{sequence:03d}-finished.json') == event, 'Finished snapshot changed')
        start = snapshot(f'native-{sequence:03d}-started.json')
        require(all(event.get(k) == v for k, v in start.items()), 'Started snapshot changed')
        require(event['path'] in paths, 'Undeclared endpoint')
        if native_spec is not None:
            require(event.get('endpoint') == f"http://127.0.0.1:{native_spec['port']}", 'Native endpoint changed')
        if 'http_error' in event:
            import base64
            failure = event['http_error']
            require(type(failure) is dict and type(failure.get('status')) is int and 400 <= failure['status'] <= 599
                and 'error' in event and 'response' not in event, 'Malformed captured HTTP failure')
            if set(failure) == {'status', 'body_base64', 'body_sha256'}:
                raw = base64.b64decode(failure['body_base64'], validate=True)
                require(hashlib.sha256(raw).hexdigest() == failure['body_sha256'], 'Captured HTTP error body changed')
            else:
                require(set(failure) == {'status','body_read_error'} and type(failure['body_read_error']) is str,
                    'Missing captured HTTP error body')
        if 'response_text' in event:
            require(hashlib.sha256(event['response_text'].encode()).hexdigest() == event['response_sha256'],
                    'Native response hash changed')
            if 'response' in event:
                require(json.loads(event['response_text']) == event['response'], 'Decoded native response changed')
            else:
                require('error' in event, 'Undecoded response lacks recorded failure')
        if native_spec is not None and event['path'] != chat_path:
            if event['path'] == '/apply-template':
                key, offset = event['condition'], offsets.get(event['condition'], 0)
                require(key in payloads and offset < len(payloads[key])
                    and event['request'] == payloads[key][offset], 'Native template request changed')
                if 'response' in event:
                    prompt = event['response'].get('prompt') if type(event['response']) is dict else None
                    require((key,offset) not in rendered, 'Repeated native rendering')
                    rendered[key,offset] = prompt
            else:
                require(event['request'] is None, 'Native metadata request changed')
                if event['path'] == '/props' and 'response' in event:
                    native_props = event['response']
            continue
        if event['path'] != chat_path:
            expected = {'model': model} if event['path'] == '/api/show' else None
            require(event['request'] == expected, 'Metadata request changed')
            if event['path'] == '/api/show' and 'response' in event:
                details = event['response']
                last_thinking_source = details.get('thinking') if type(details) is dict else None
            continue
        thinking_sources[event['sequence']] = native_props if native_spec is not None else last_thinking_source
        key = event['condition']
        offset = offsets.get(key, 0)
        offsets[key] = offset + 1
        expected_payload = payloads[key][offset]
        require(event['request'] == expected_payload,
            'Native payload differs from canonical rows')
        require(request_input_bound([ChatMessage(**m) for m in expected_payload['messages']],
                response_schema=expected_payload.get('format') if type(expected_payload.get('format')) is dict else None) <= max_input,
                'Native input exceeds bound')
    seen, predictions = set(), {}
    for obs in observations:
        key = obs['id']
        require(key in entries and key not in seen, 'Unscheduled or duplicate observation')
        seen.add(key)
        calls, events_for = obs['native_calls'], by_condition.get(key, [])
        require(len(events_for) <= len(calls) <= min(len(events_for)+1, len(chunks[key])),
                'Native call prefix changed')
        accepted = output_field in obs
        if accepted:
            require(len(calls) == len(events_for) == len(chunks[key]) and bool(events_for),
                    'Incomplete matrix cannot be accepted')
        labels = [row['constraint'] for row in rows_by_id[key]]
        analysis = None
        for index, call in enumerate(calls):
            require(call['row_indices'] == chunks[key][index], 'Native row binding changed')
            require(call['messages'] == payloads[key][index]['messages'], 'Observed request changed')
            completion = call['completion']
            if index == len(events_for):
                require(not accepted and completion is None and 'error' in obs, 'Uncaptured completion')
                continue
            event = events_for[index]
            if completion is None:
                require(not accepted and 'error' in obs, 'Missing completion lacks observed failure')
                continue
            raw = event.get('response', {})
            require(type(raw) is dict, 'Malformed native completion')
            if native_spec is not None:
                require((key,index) in rendered, 'Native generation lacks its canonical render')
                require(type(rendered[key,index]) is str and rendered[key,index].endswith('<think>\n')
                    and len(rendered[key,index].encode())+1+max_tokens <= context_length, 'Invalid native rendered input')
                decoded = decode_native_response(raw, native_spec, rendered_prompt=rendered[key,index])
                native_content, input_count, output_count = decoded['content'], decoded['prompt_tokens'], decoded['completion_tokens']
            else:
                native_content, input_count, output_count = raw['message']['content'], raw['prompt_eval_count'], raw['eval_count']
            require(completion['content'] == native_content and completion['prompt_tokens'] == input_count
                and completion['completion_tokens'] == output_count, 'Observed completion changed')
            require(completion['provider'] == ('llama_cpp' if native_spec is not None else 'ollama') and completion['model_id'] == model
                and completion['metadata']['model_attestation']['digest'] == digest, 'Completion identity changed')
            if not accepted:
                require('error' in obs, 'Rejected completion lacks an observed failure')
                continue
            metadata = completion['metadata']
            if native_spec is not None:
                require(thinking_sources[event['sequence']] is not None, 'Native response lacks observed process metadata')
                validate_native_props(thinking_sources[event['sequence']], native_spec)
                control = metadata['model_attestation']
                require(control.get('weight_digest') == native_spec['model']['weight']['sha256']
                    and control.get('template_sha256') == native_spec['model']['template_sha256']
                    and control.get('process') == process_record['identity']
                    and type(control.get('requested_thinking_tokens')) is int
                    and control['requested_thinking_tokens'] == native_spec['thinking_tokens']
                    and metadata.get('native_configuration_sha256') == hashlib.sha256(json.dumps(native_spec,sort_keys=True).encode()).hexdigest()
                    and metadata.get('thinking_closure') == decoded['closure']
                    and decoded['closure']['observed'] is True and decoded['closure']['budget_forcing_verified'] is False
                    and metadata.get('think') is True, 'Accepted native evidence differs from captured/frozen controls')
            elif think is not None:
                control = metadata.get('model_attestation', {}).get('thinking_control', {})
                supported = control.get('supported')
                native_control = thinking_sources.get(event['sequence'])
                native_values = native_control.get('values') if type(native_control) is dict else None
                require(metadata.get('think') is think and control.get('requested') is think
                    and type(supported) is list and bool(supported)
                    and all(type(value) is bool for value in supported)
                    and type(native_values) is list and all(type(value) is bool for value in native_values)
                    and supported == native_values and any(value is think for value in supported),
                    'Accepted completion lacks frozen thinking control')
                trace = raw.get('message', {}).get('thinking')
                require(trace is None or type(trace) is str and (think is True or trace == ''),
                    'Native thinking output contradicts frozen control')
            if assessment_kind in ('reference-extraction-v2', 'reference-selections-v1'):
                expected_schema_hash = hashlib.sha256(json.dumps(payloads[key][index]['format'],
                    sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
                require(metadata.get('response_schema_sha256') == expected_schema_hash,
                    'Accepted completion schema differs from the frozen request')
            require(raw.get('model') == model and (native_spec is not None or raw.get('done') is True), 'Unattested native completion')
            require(completion.get('finish_reason') == 'stop' and metadata.get('local_only') is True
                and metadata.get('done') is True and metadata.get('response_model') == model
                and type(metadata.get('context_length')) is int and metadata['context_length'] == context_length
                and type(metadata.get('seed')) is int and metadata['seed'] == seed and metadata.get('json_mode') is True,
                'Accepted completion lacks frozen local settings')
            require((decoded['finish_reason'] == 'stop' if native_spec is not None else raw.get('done_reason') == 'stop')
                and type(input_count) is int and type(output_count) is int
                and type(completion['prompt_tokens']) is int and type(completion['completion_tokens']) is int
                and 0 < output_count <= max_tokens and input_count > 0 and input_count+max_tokens <= context_length,
                'Incomplete native token result')
            if assessment_kind in REFERENCE_KINDS:
                from psychology_base.applications.measurement.reference_readings import (
                    decode_reference_readings, decode_reference_selections, decode_interpretation_response,
                )
                require(len(calls) == 1, 'Extraction cannot accept a partial or multi-window result')
                decoder = (decode_interpretation_response if assessment_kind == 'interpretation-selections-v2'
                    else decode_reference_selections if assessment_kind == 'reference-selections-v1' else decode_reference_readings)
                analysis = decoder(native_content,
                    json.loads(rows_by_id[key][0]['pair'][0]))
                continue
            value = json.loads(native_content, object_pairs_hook=_unique_json_object)
            require(type(value) is dict and set(value) == {'labels'} and type(value['labels']) is list
                and len(value['labels']) == len(chunks[key][index])
                and all(type(v) is str and v in ('match', 'nonmatch', 'uncertain') for v in value['labels']),
                'Accepted malformed native labels')
            for i, label in zip(chunks[key][index], value['labels']):
                labels[i] = label
        if accepted:
            prediction = analysis if assessment_kind != 'membership' else labels
            require(prediction == obs[output_field], 'Observed assessment differs from native response')
            predictions[key] = prediction
    require(set(by_condition) <= seen, 'Native dispatch lacks observation')
    return dict(native_attempts=len(native), native_responses=sum('response' in e for e in native),
                metadata_attempts=len(events)-len(native), predictions=predictions)
