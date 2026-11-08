"""Native llama.cpp diagnostic transport; shared router owns all dispatch.

The owned process and raw token evidence are mandatory. No OpenAI SDK, paid
provider, model acquisition, response repair, retry or implicit server launch.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.request import Request

from psychology_base.llm.local_backend import OwnedLlamaServer
from psychology_base.llm.execution import MAX_OFFLINE_LOCAL_TIMEOUT_S
from psychology_base.llm.loopback import open_loopback
from psychology_base.llm.types import Completion, TaskTier


def native_chat_payload(messages, *, model, max_tokens, thinking_tokens, seed):
    return dict(model=model, messages=[dict(role=m.role, content=m.content) for m in messages],
        stream=False, max_tokens=max_tokens, temperature=0, seed=seed,
        response_format=dict(type='json_object'), chat_template_kwargs=dict(enable_thinking=True),
        reasoning_format='deepseek', reasoning_budget_tokens=thinking_tokens,
        cache_prompt=False, return_tokens=True, verbose=True)


def validate_native_props(value, record):
    """Compare real backend settings to offline-installed assets."""
    if type(value) is not dict:
        raise ValueError('Malformed native backend metadata')
    model = record['model']
    settings = value.get('default_generation_settings', {})
    template = value.get('chat_template')
    if (value.get('model_alias') != model['model_id']
            or type(value.get('model_path')) is not str
            or Path(value['model_path']).resolve() != Path(model['weight']['path']).resolve()
            or type(settings.get('n_ctx')) is not int or settings['n_ctx'] != record['context_length']
            or type(value.get('total_slots')) is not int or value['total_slots'] != 1
            or type(template) is not str or hashlib.sha256(template.encode()).hexdigest() != model['template_sha256']
            or value.get('is_sleeping') is not False or value.get('ui') is not False
            or value.get('cors_proxy_enabled') is not False
            or type(value.get('build_info')) is not str or '39febd2ad' not in value['build_info']):
        raise ValueError('Native backend metadata differs from the pinned process/model/template')


def decode_native_response(value, record, *, rendered_prompt):
    """Validate native evidence without turning rejected traces into answers.

    The native verbose settings do not echo the reasoning limit. We report
    requested controls and observed closure separately; forcing remains unproven.
    Length-terminated results remain representable for failed-run accounting.
    """
    if type(value) is not dict:
        raise ValueError('Malformed native completion envelope')
    choices, usage, detail = value.get('choices'), value.get('usage'), value.get('__verbose')
    if (value.get('model') != record['model']['model_id'] or type(choices) is not list or len(choices) != 1
            or type(usage) is not dict or type(detail) is not dict or value.get('object') != 'chat.completion'
            or type(value.get('system_fingerprint')) is not str or '39febd2ad' not in value['system_fingerprint']):
        raise ValueError('Missing native completion identity or evidence')
    choice = choices[0]
    if type(choice) is not dict or type(choice.get('message')) is not dict:
        raise ValueError('Malformed native completion choice')
    message, settings, tokens = choice['message'], detail.get('generation_settings'), detail.get('tokens')
    prompt_count, count = usage.get('prompt_tokens'), usage.get('completion_tokens')
    model = record['model']
    if (message.get('role') != 'assistant' or type(message.get('content')) is not str
            or message.get('tool_calls') not in (None, [])
            or type(settings) is not dict or type(tokens) is not list
            or type(prompt_count) is not int or prompt_count <= 0
            or type(count) is not int or not 0 < count <= record['max_tokens']
            or type(usage.get('total_tokens')) is not int or usage['total_tokens'] != prompt_count + count
            or prompt_count + record['max_tokens'] > record['context_length']
            or len(tokens) != count or any(type(t) is not int or not 0 <= t < model['token_count'] for t in tokens)
            or type(settings.get('seed')) is not int or settings['seed'] != record['seed']
            or type(settings.get('temperature')) not in (int, float) or settings['temperature'] != 0
            or type(settings.get('n_predict')) is not int or settings['n_predict'] != record['max_tokens']
            or settings.get('stream') is not False or settings.get('reasoning_format') != 'deepseek'
            or type(settings.get('grammar')) is not str or not settings['grammar']
            or type(settings.get('generation_prompt')) is not str
            or not settings['generation_prompt'].endswith('<think>\n')
            or detail.get('prompt') != rendered_prompt
            or detail.get('stop') is not True or detail.get('truncated') is not False
            or type(detail.get('tokens_predicted')) is not int or detail['tokens_predicted'] != count
            or type(detail.get('tokens_evaluated')) is not int or detail['tokens_evaluated'] != prompt_count):
        raise ValueError('Native completion settings, token counts or rendered input differ')
    end = model['thinking_end']
    close = tokens.index(end) if end in tokens else None
    finish = choice.get('finish_reason')
    closure = dict(observed=close is not None, generated_tokens_before_close=close,
        requested_thinking_tokens=record['thinking_tokens'], budget_forcing_verified=False)
    if finish == 'stop':
        raw = detail.get('content')
        if (detail.get('stop_type') != 'eos' or tokens[-1] != model['eos']
                or model['eos'] in tokens[:-1] or model['thinking_start'] in tokens
                or tokens.count(end) != 1 or close > record['thinking_tokens'] + 8
                or close >= len(tokens)-2 or not message['content'].strip()
                or type(message.get('reasoning_content')) is not str
                or type(raw) is not str or raw.count('</think>') != 1
                or raw.split('</think>', 1)[1].strip() != message['content'].strip()
                or raw.split('</think>', 1)[0].strip() != message['reasoning_content'].strip()):
            raise ValueError('Native final answer lacks a bounded single thinking closure')
    elif finish != 'length' or detail.get('stop_type') != 'limit':
        raise ValueError('Unrecognized native completion termination')
    return dict(content=message['content'], finish_reason=finish, prompt_tokens=prompt_count,
                completion_tokens=count, closure=closure)


@dataclass(frozen=True)
class LlamaCppProvider:
    server: OwnedLlamaServer
    request_timeout_s: float = 1200
    model_metadata_timeout_s: float = 60
    name: str = field(default='llama_cpp', init=False)
    local_only: bool = field(default=True, init=False)
    json_mode: bool = field(default=True, init=False)
    think: bool = field(default=True, init=False)
    response_schema: None = field(default=None, init=False)

    def __post_init__(self):
        if type(self.server) is not OwnedLlamaServer:
            raise ValueError('Native diagnostic requires its explicit process owner')
        for value, maximum in ((self.request_timeout_s,MAX_OFFLINE_LOCAL_TIMEOUT_S), (self.model_metadata_timeout_s,60)):
            if type(value) not in (int,float) or not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError('Invalid native diagnostic transport deadline')

    @property
    def base_url(self): return self.server.spec.base_url
    @property
    def advanced_model(self): return self.server.spec.model.model_id
    @property
    def expected_digest(self): return self.server.spec.model.source_digest
    @property
    def context_length(self): return self.server.spec.context_length
    @property
    def seed(self): return self.server.spec.seed

    def model_for(self, tier):
        return self.advanced_model

    def configuration(self):
        return self.server.spec.record()

    def require_local_model(self, model, *, capability):
        if model != self.advanced_model or capability != 'completion':
            raise ValueError('Native diagnostic supports only its pinned chat model')
        self.server.assert_owned()

    def _open(self, request, *, timeout):
        self.server.assert_owned()
        return open_loopback(request, base_url=self.base_url, timeout=timeout)

    def metadata(self, path, payload=None):
        if path not in ('/health', '/props', '/apply-template'):
            raise ValueError('Undeclared native diagnostic metadata endpoint')
        request = Request(self.base_url + path, data=None if payload is None else json.dumps(payload).encode(),
                          headers={'Content-Type':'application/json'})
        with self._open(request, timeout=self.model_metadata_timeout_s) as response:
            return json.load(response)

    def attest_model(self, model, *, capability='completion'):
        self.require_local_model(model, capability=capability)
        props = self.metadata('/props')
        record = self.configuration()
        validate_native_props(props, record)
        return dict(digest=self.expected_digest, weight_digest=record['model']['weight']['sha256'],
            template_sha256=record['model']['template_sha256'], process=self.server.assert_owned(),
            thinking_control=dict(requested=True, supported=[True]),
            requested_thinking_tokens=self.server.spec.thinking_tokens)

    def chat(self, messages, *, tier=TaskTier.SIMPLE, max_tokens=None, temperature=None):
        if (type(max_tokens) is not int or max_tokens != self.server.spec.max_tokens
                or type(temperature) not in (int,float) or temperature != 0
                or not messages or any(m.role not in ('system','user','assistant') or type(m.content) is not str for m in messages)):
            raise ValueError('Native request differs from its bounded diagnostic controls')
        attestation = self.attest_model(self.advanced_model)
        payload = native_chat_payload(messages, model=self.advanced_model, max_tokens=max_tokens,
            thinking_tokens=self.server.spec.thinking_tokens, seed=self.seed)
        rendered = self.metadata('/apply-template', payload).get('prompt')
        if (type(rendered) is not str or not rendered.endswith('<think>\n')
                or len(rendered.encode()) + 1 + max_tokens > self.context_length):
            raise ValueError('Rendered native request exceeds context or lacks thinking prefill')
        started = time.monotonic()
        request = Request(self.base_url + '/v1/chat/completions', data=json.dumps(payload).encode(),
                          headers={'Content-Type':'application/json'})
        with self._open(request, timeout=self.request_timeout_s) as response:
            raw = json.load(response)
        decoded = decode_native_response(raw, self.configuration(), rendered_prompt=rendered)
        self.server.assert_owned()
        return Completion(content=decoded['content'], provider=self.name, model_id=self.advanced_model, tier=tier,
            prompt_tokens=decoded['prompt_tokens'], completion_tokens=decoded['completion_tokens'],
            finish_reason=decoded['finish_reason'], latency_ms=(time.monotonic()-started)*1000,
            metadata=dict(local_only=True, done=True, response_model=raw['model'], context_length=self.context_length,
                seed=self.seed, json_mode=True, think=True, model_attestation=attestation,
                native_configuration_sha256=hashlib.sha256(json.dumps(self.configuration(), sort_keys=True).encode()).hexdigest(),
                thinking_closure=decoded['closure']))
