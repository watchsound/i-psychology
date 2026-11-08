"""One evaluation-only completion audit; routing and caches remain in LLMRouter."""
from dataclasses import asdict
from decimal import Decimal
import time
import json

from psychology_base.llm.types import TaskTier


class LocalNativeCapture:
    """Evaluation-only native capture; dispatch remains in the real provider.

    The observer receives independent started/finished snapshots. Attempts are
    recorded before transport, including requests that later fail. No retries.
    """

    def __init__(self, provider, *, max_attempts, deadline, observer,
                 validate_chat, max_metadata=256, clock=time.monotonic):
        from psychology_base.llm.providers.ollama import OllamaProvider
        from psychology_base.llm.providers.llama_cpp import LlamaCppProvider
        import math
        native = type(provider) is LlamaCppProvider
        if (not isinstance(provider, OllamaProvider) and not native or not provider.local_only
                or not native and provider.base_url != 'http://127.0.0.1:11434'
                or type(max_attempts) is not int or max_attempts <= 0
                or type(max_metadata) is not int or max_metadata <= 0
                or type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError('Native capture requires pinned local transport and finite bounds')
        self.provider, self.max_attempts, self.deadline = provider, max_attempts, deadline
        self.observer, self.validate_chat, self.clock = observer, validate_chat, clock
        self.max_metadata = max_metadata
        self.attempts = self.metadata_attempts = 0
        self.events = []
        self.condition = None
        self._opened = None
        self.base_url = provider.base_url
        self.chat_path = '/v1/chat/completions' if native else '/api/chat'
        self.paths = ('/health', '/props', '/apply-template', self.chat_path) if native else (
            '/api/tags', '/api/show', '/api/ps', self.chat_path)

    def __enter__(self):
        if self._opened is not None:
            raise ValueError('Capture is already installed')
        self._opened = self.provider._open
        object.__setattr__(self.provider, '_open', self._open)
        return self

    def __exit__(self, *exc):
        object.__setattr__(self.provider, '_open', self._opened)
        self._opened = None

    def _emit(self, event, phase):
        self.observer(json.loads(json.dumps(event)), phase)

    def _open(self, request, *, timeout):
        import hashlib
        import io
        from urllib.parse import urlsplit
        url = urlsplit(request.full_url)
        if (url.scheme != 'http' or url.netloc != urlsplit(self.base_url).netloc
                or self.provider.base_url != self.base_url
                or url.username or url.password or url.query or url.fragment or url.path not in self.paths):
            raise ValueError('Undeclared local diagnostic endpoint')
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError('Local diagnostic inference deadline')
        payload = None if request.data is None else json.loads(request.data)
        if url.path == self.chat_path:
            self.validate_chat(payload)
            if self.attempts >= self.max_attempts:
                raise ValueError('Local diagnostic native attempt cap')
            self.attempts += 1
        else:
            if url.path == '/apply-template':
                self.validate_chat(payload)
            if self.metadata_attempts >= self.max_metadata:
                raise ValueError('Local diagnostic metadata attempt cap')
            self.metadata_attempts += 1
        started = self.clock()
        event = dict(sequence=len(self.events)+1, condition=self.condition,
                     path=url.path, request=payload, start_monotonic=started)
        if self.chat_path != '/api/chat':
            event['endpoint'] = self.base_url
        self.events.append(event)
        self._emit(event, 'started')
        try:
            with self._opened(request, timeout=min(timeout, remaining)) as response:
                raw = response.read()
            event['response_text'] = raw.decode('utf-8')
            event['response_sha256'] = hashlib.sha256(raw).hexdigest()
            event['response'] = json.loads(event['response_text'])
            return io.BytesIO(raw)
        except Exception as exc:
            event['error'] = repr(exc)
            from urllib.error import HTTPError
            if isinstance(exc, HTTPError):
                import base64
                failure = dict(status=exc.code)
                try:
                    raw = exc.read()
                    failure.update(body_base64=base64.b64encode(raw).decode('ascii'),
                                   body_sha256=hashlib.sha256(raw).hexdigest())
                except Exception as read_error:
                    failure['body_read_error'] = repr(read_error)
                finally:
                    exc.close()
                event['http_error'] = failure
            raise
        finally:
            event['elapsed_ms'] = (self.clock()-started)*1000
            self._emit(event, 'finished')

TONGYI_JUDGE_MODEL = 'qwen3.7-flash-2026-07-15'
TONGYI_JUDGE_ENDPOINT = 'https://dashscope.aliyuncs.com/compatible-mode/v1'


def tongyi_judge_spec():
    """Fixed cheap development profile; return fresh state for every study."""
    return dict(provider='tongyi',model=TONGYI_JUDGE_MODEL,transport='openai_compatible',
        region='beijing',base_url=TONGYI_JUDGE_ENDPOINT,enable_thinking=False,json_mode=True,
        chat_timeout_s=60,max_input=12000,input_rate='.0375',output_rate='.130',
        call_policy=dict(version=2,max_tokens=2048,temperature=0))


def tongyi_reasoning_spec(thinking):
    """Paired profile with a total-output limit and documented billing margin."""
    if type(thinking) is not bool:
        raise ValueError('Thinking mode must be explicitly boolean')
    return dict(tongyi_judge_spec(),enable_thinking=thinking,total_completion_limit=True,
        billable_output_ceiling=4106,call_policy=dict(version=3,max_tokens=4096,temperature=0))


def tongyi_reliability_spec():
    """Opt-in pinned Plus assessment candidate; neither a default nor qualification."""
    from psychology_base.applications.measurement.memory_grounding import runtime_evidence_schema
    return dict(tongyi_reasoning_spec(False), model='qwen3.7-plus-2026-05-26',
        input_rate='.345', output_rate='1.101', response_schema=runtime_evidence_schema(),
        call_policy=dict(version=4,max_tokens=4096,temperature=0))


def tongyi_binding_spec():
    """One existing compatible router serves two bounded, separately audited stages."""
    from psychology_base.applications.measurement.memory_grounding import runtime_binding_schema
    return dict(tongyi_reliability_spec(),response_schema=runtime_binding_schema(),
        call_policy=dict(version=5,max_tokens=4096,temperature=0))


from psychology_base.llm.accounting import Budget, positive_decimal, request_input_bound


def inference_call_options(spec):
    """Opt in to bounded study options; legacy call limits stay intact."""
    if 'call_policy' not in spec:
        return None
    if spec.get('provider') == 'tongyi':
        policy = spec.get('call_policy')
        paired=isinstance(policy,dict) and policy.get('version')==3
        structured=isinstance(policy,dict) and policy.get('version') in (4,5)
        expected=(tongyi_binding_spec() if structured and policy['version']==5 else tongyi_reliability_spec() if structured else
                  tongyi_reasoning_spec(spec.get('enable_thinking')) if paired else tongyi_judge_spec())
        if (spec != expected or not isinstance(policy, dict)
                or type(policy.get('version')) is not int
                or type(policy.get('max_tokens')) is not int
                or type(policy.get('temperature')) is not int or policy['temperature'] != 0
                or type(spec.get('enable_thinking')) is not bool or spec.get('json_mode') is not True
                or type(spec.get('chat_timeout_s')) is not int or type(spec.get('max_input')) is not int
                or ((paired or structured) and (spec.get('total_completion_limit') is not True
                    or type(spec.get('billable_output_ceiling')) is not int))):
            raise ValueError('Alibaba screening requires its frozen model/endpoint/decoding cost policy')
        return dict(max_tokens=policy['max_tokens'], temperature=policy['temperature'])
    policy = spec['call_policy']
    if (not isinstance(policy, dict) or set(policy) != {'version', 'max_tokens', 'temperature'}
            or type(policy.get('version')) is not int or policy['version'] != 1
            or type(policy.get('max_tokens')) is not int or policy['max_tokens'] != 2048
            or policy.get('temperature') is not None
            or spec.get('provider') != 'openai' or spec.get('model') != 'gpt-6-luna'
            or spec.get('reasoning_effort') not in ('none', 'low') or spec.get('json_mode') is not True):
        raise ValueError('Hosted inference requires the exact bounded reasoning comparison policy')
    return dict(max_tokens=policy['max_tokens'], temperature=policy['temperature'])


def billable_output_limit(spec, max_tokens):
    """Keep configured output and the provider's bounded billing tolerance separate."""
    if spec is not None and spec.get('provider')=='tongyi' and 'call_policy' in spec:
        options=inference_call_options(spec)
        if spec['call_policy']['version'] in (3,4,5):
            if type(max_tokens) is not int or max_tokens!=options['max_tokens']:
                raise ValueError('Configured output limit differs from the frozen policy')
            return spec['billable_output_ceiling']
    return max_tokens


def hosted_request_options(spec):
    """One request-policy record shared by preparation, dispatch and raw replay."""
    if spec.get('provider') != 'tongyi':
        return {}
    if inference_call_options(spec) is None:
        raise ValueError('Alibaba screening requires an explicit call policy')
    options=dict(transport=spec['transport'], endpoint=spec['base_url'],
        enable_thinking=spec['enable_thinking'], json_mode=True, max_retries=0, follow_redirects=False)
    if spec['call_policy']['version'] in (3,4,5):
        options.update(total_completion_limit=True,
            configured_max_completion_tokens=spec['call_policy']['max_tokens'],
            billable_output_ceiling=spec['billable_output_ceiling'])
    if 'response_schema' in spec:
        # Include the full requested schema in the frozen request/ledger scope.
        from copy import deepcopy
        options['response_schema']=deepcopy(spec['response_schema'])
    return dict(inference_options=options)


def validate_local_call(call, spec):
    completion=call['completion']; metadata=completion['metadata']
    if (call.get('local_only') is not True or call.get('proxy') is not False or call.get('cached') is not False
            or call.get('usage_valid') is not True or Decimal(call['accounted_usd']) != 0
            or completion['provider'] != 'ollama' or completion['model_id'] != spec['model']
            or completion['tier'] != TaskTier.ADVANCED.value
            or metadata.get('local_only') is not True or metadata.get('done') is not True
            or metadata.get('response_model') != spec['model']
            or metadata.get('model_attestation',{}).get('digest') != spec['digest']
            or metadata.get('model_attestation',{}).get('family') != spec['family']
            or metadata.get('context_length') != spec['context_length']
            or metadata.get('seed') != spec['seed'] or metadata.get('json_mode') is not spec['json_mode']):
        raise ValueError('Local completion lacks the frozen model/decoding attestation')


def validate_screening_call(call, spec):
    """Validate either role of a screening run against its own provider contract."""
    options = inference_call_options(spec)
    if spec['provider'] == 'ollama':
        validate_local_call(call, spec)
        return
    completion = call['completion']; metadata = completion['metadata']
    usage = (completion['prompt_tokens'], completion['completion_tokens'])
    if spec['provider'] == 'tongyi':
        hosted_request_options(spec)
        reasoning = metadata.get('reasoning_tokens')
        valid_reasoning=(reasoning is None or (type(reasoning) is int and reasoning==0))
        if spec['enable_thinking']:
            valid_reasoning=(type(reasoning) is int and type(usage[1]) is int and 0<reasoning<=usage[1]
                             and metadata.get('reasoning_content_present') is True)
        else:
            valid_reasoning=valid_reasoning and metadata.get('reasoning_content_present') is False
        if spec.get('total_completion_limit') is True and (
                metadata.get('total_completion_limit') is not True
                or metadata.get('reported_usage_types_preserved') is not True
                or type(metadata.get('configured_max_completion_tokens')) is not int
                or metadata['configured_max_completion_tokens']!=options['max_tokens']):
            raise ValueError('Alibaba completion lacks the frozen total-output limit evidence')
        if 'response_schema' in spec:
            import hashlib
            identity=hashlib.sha256(json.dumps(spec['response_schema'],sort_keys=True,
                separators=(',', ':'),ensure_ascii=False).encode('utf-8')).hexdigest()
            if metadata.get('response_format')!='json_schema' or metadata.get('response_schema_sha256')!=identity:
                raise ValueError('Alibaba completion lacks the frozen schema request evidence')
        if (call.get('local_only', False) is not False or call.get('proxy') is not False
                or call.get('cached') is not False or call.get('usage_valid') is not True
                or completion['provider'] != 'tongyi' or completion['model_id'] != spec['model']
                or completion['tier'] != TaskTier.ADVANCED.value
                or metadata.get('response_model') != spec['model'] or not metadata.get('completion_id')
                or metadata.get('enable_thinking') is not spec['enable_thinking'] or metadata.get('json_mode') is not True
                or not valid_reasoning
                or not all(type(n) is int for n in usage) or not 0 < usage[0] <= spec['max_input']
                or not 0 <= usage[1] <= billable_output_limit(spec,options['max_tokens'])):
            raise ValueError('Alibaba completion lacks frozen model/decoding/usage evidence')
        expected = (Decimal(spec['input_rate'])*usage[0]+Decimal(spec['output_rate'])*usage[1])/1_000_000
        if Decimal(call['accounted_usd']) != expected:
            raise ValueError('Alibaba completion accounting differs from frozen token rates')
        return
    reasoning = metadata.get('reasoning_tokens')
    valid_reasoning = (type(reasoning) is int and reasoning == 0)
    if options is not None and spec['reasoning_effort'] == 'low':
        valid_reasoning = (type(reasoning) is int and type(usage[1]) is int and 0 <= reasoning <= usage[1])
    if (call.get('local_only', False) is not False or call.get('proxy') is not False
            or call.get('cached') is not False or call.get('usage_valid') is not True
            or completion['provider'] != spec['provider'] or completion['model_id'] != spec['model']
            or completion['tier'] != TaskTier.ADVANCED.value
            or metadata.get('response_model') != spec['model'] or not metadata.get('completion_id')
            or metadata.get('reasoning_effort') != spec['reasoning_effort']
            or not valid_reasoning
            or metadata.get('service_tier') != 'default'
            or metadata.get('json_mode') is not spec['json_mode']
            or not all(type(n) is int for n in usage) or not 0 < usage[0] <= spec['max_input']
            or not 0 <= usage[1] <= (options['max_tokens'] if options is not None else 300)):
        raise ValueError('Hosted completion lacks the frozen model/decoding/usage attestation')
    expected = (Decimal(spec['input_rate'])*usage[0]+Decimal(spec['output_rate'])*usage[1])/1_000_000
    if Decimal(call['accounted_usd']) != expected:
        raise ValueError('Hosted completion accounting differs from frozen token rates')


class RecordingRouter:
    """Delegate once, retaining completions lost by text-only application adapters.

    All study calls are serial. Failed/unknown-usage requests retain the full
    reservation; budget accounting never relies on a failed response being free.
    """

    def __init__(self, router, budget, *, provider, model, input_rate, output_rate,
                 max_input=12000, controls=False, local_spec=None, remote_spec=None):
        self.router, self.budget = router, budget
        self.provider, self.model = provider, model
        self.local_spec = local_spec
        self.remote_spec = remote_spec
        if local_spec is not None and remote_spec is not None:
            raise ValueError("A completion cannot be both local and hosted")
        if local_spec is not None:
            if Decimal(str(input_rate)) != 0 or Decimal(str(output_rate)) != 0:
                raise ValueError("Local inference requires explicit local accounting and zero API rates")
            self.input_rate = self.output_rate = Decimal(0)
        else:
            self.input_rate, self.output_rate = positive_decimal(input_rate), positive_decimal(output_rate)
        if type(max_input) is not int or not 0 < max_input <= 12000:
            raise ValueError("Invalid frozen input token ceiling")
        self.max_input, self.controls = max_input, controls
        self.last_completion = self.last_call = None

    def verify_provider(self, tier):
        provider, decision = self.router.route(tier)
        if decision.provider != self.provider or decision.model_id != self.model:
            raise ValueError("Actual provider/model differs from the frozen study")
        if not self.controls:
            from psychology_base.llm.providers import AlibabaTongyiProvider, OpenAIProvider, AnthropicProvider, OllamaProvider
            if self.local_spec is not None:
                if (type(provider) is not OllamaProvider or not provider.local_only
                        or set(self.router.providers) != {'ollama'} or self.router.cache is not None
                        or provider.expected_digest != self.local_spec['digest']
                        or provider.base_url != self.local_spec['base_url']
                        or provider.context_length != self.local_spec['context_length']
                        or provider.seed != self.local_spec['seed'] or provider.json_mode != self.local_spec['json_mode']
                        or provider.request_timeout_s != self.local_spec.get('chat_timeout_s',provider.request_timeout_s)
                        or provider.model_metadata_timeout_s != self.local_spec.get('model_metadata_timeout_s',provider.model_metadata_timeout_s)):
                    raise ValueError("Local evaluation requires the pinned isolated Ollama provider without cache/fallback")
                if provider.attest_model(self.model, _reuse_metadata=True)['family'] != self.local_spec['family']:
                    raise ValueError("Installed model family differs from the local profile")
            elif type(provider) not in (AlibabaTongyiProvider, OpenAIProvider, AnthropicProvider):
                raise ValueError("Quality runs require an existing real provider")
            if isinstance(provider, (OpenAIProvider, AnthropicProvider)):
                client=provider._ensure_client()
                if provider.max_retries != 0 or client.max_retries != 0:
                    raise ValueError("Study provider must explicitly disable SDK retries")
                from urllib.parse import urlsplit
                expected=('dashscope.aliyuncs.com' if self.remote_spec is not None and self.provider=='tongyi'
                          else 'api.openai.com' if isinstance(provider,OpenAIProvider) else 'api.anthropic.com')
                if urlsplit(str(client.base_url)).hostname!=expected:
                    raise ValueError('Frozen first-party prices do not qualify a different API endpoint')
            if self.remote_spec is not None and self.provider == 'tongyi':
                hosted_request_options(self.remote_spec)
                if (type(provider) is not OpenAIProvider or provider.name != 'tongyi'
                        or self.router.cache is not None or set(self.router.providers) != {'tongyi'}
                        or str(provider._ensure_client().base_url).rstrip('/') != self.remote_spec['base_url']
                        or provider.enable_thinking is not self.remote_spec['enable_thinking'] or provider.json_mode is not True
                        or provider.response_schema != self.remote_spec.get('response_schema')
                        or provider.total_completion_limit is not self.remote_spec.get('total_completion_limit',False)
                        or provider.reasoning_effort is not None or provider.request_timeout_s != 60
                        or provider._ensure_client().timeout != 60
                        or provider.follow_redirects is not False
                        or provider._ensure_client()._client.follow_redirects is not False):
                    raise ValueError('Alibaba screening requires isolated compatible transport and frozen controls')
            elif self.remote_spec is not None and (
                    type(provider) is not OpenAIProvider or self.router.cache is not None
                    or set(self.router.providers) != {'openai'}
                    or str(provider._ensure_client().base_url).rstrip('/') != 'https://api.openai.com/v1'
                    or provider.reasoning_effort != self.remote_spec['reasoning_effort']
                    or provider.json_mode is not self.remote_spec['json_mode']):
                raise ValueError('Hosted screening requires its isolated provider and frozen decoding without cache')
        return provider

    def chat(self, messages, *, tier=TaskTier.SIMPLE, max_tokens=None, temperature=None, use_cache=True):
        options = inference_call_options(self.remote_spec) if self.remote_spec is not None else None
        bounded = (type(max_tokens) is int and 0 < max_tokens <= 400) if options is None else (
            type(max_tokens) is int and max_tokens == options['max_tokens']
            and type(temperature) is type(options['temperature']) and temperature == options['temperature'])
        if tier != TaskTier.ADVANCED or not bounded:
            raise ValueError("Study chat needs the frozen ADVANCED tier and bounded output")
        messages = list(messages)
        # Conservative byte-based input bound plus framing allowance; verify
        # actual returned token usage against the frozen ceiling as well.
        if request_input_bound(messages,response_schema=(self.remote_spec or {}).get('response_schema')) > self.max_input:
            raise ValueError("Request exceeds conservative frozen input bound")
        start = time.perf_counter()
        self.verify_provider(tier)
        output_limit=billable_output_limit(self.remote_spec,max_tokens)
        ceiling = (self.input_rate * self.max_input + self.output_rate * output_limit) / 1_000_000
        request = {"provider": self.provider, "model": self.model, "tier": tier.value,
                   "messages": [asdict(m) for m in messages], "max_tokens": max_tokens,
                   "temperature": temperature, "proxy": self.controls}
        if self.local_spec is not None:
            request.update(local_only=True, model_digest=self.local_spec['digest'])
        if self.remote_spec is not None:
            request.update(hosted_request_options(self.remote_spec))
        dispatch = self.budget.reserve(ceiling, request)
        self.last_completion = self.last_call = None
        cache = self.router.cache
        hits = cache.hits if cache else 0
        try:
            completion = self.router.chat(messages, tier=tier, max_tokens=max_tokens,
                                          temperature=temperature, use_cache=use_cache)
        except Exception as exc:
            self.budget.pending.discard(dispatch)
            self.budget.uncertain = True
            self.budget.writer.write({"event": "failure", "dispatch": dispatch,
                                      "exception_type": type(exc).__name__, "retained_usd": str(ceiling)})
            raise
        cached = bool(cache and cache.hits > hits)
        self.budget.pending.discard(dispatch)
        self.budget.attempts -= int(cached)
        self.budget.cache_hits += int(cached)
        usage = (completion.prompt_tokens, completion.completion_tokens)
        valid_usage = (all(type(n) is int and n >= 0 for n in usage)
                       and usage[0] > 0 and usage[0] <= self.max_input and usage[1] <= output_limit)
        if self.remote_spec is not None and self.provider == 'tongyi' and completion.content.strip():
            valid_usage = valid_usage and usage[1] > 0
        charged = Decimal(0) if cached else (
            (self.input_rate * usage[0] + self.output_rate * usage[1]) / 1_000_000 if valid_usage else ceiling)
        self.budget.spent += charged - ceiling
        self.budget.uncertain |= not valid_usage
        self.last_completion = completion
        record = {"event": "completion", "dispatch": dispatch, "cached": cached,
                  "accounted_usd": str(charged), "usage_valid": valid_usage,
                  "wall_ms": (time.perf_counter() - start) * 1000,
                  "completion": {**asdict(completion), "tier": completion.tier.value},
                  "proxy": self.controls}
        if self.local_spec is not None:
            record['local_only'] = True
        self.last_call = record
        self.budget.writer.write(record)
        if (completion.provider != self.provider or completion.model_id != self.model
                or completion.tier != tier or not valid_usage):
            raise ValueError("Completion identity or usage violates the frozen study")
        if not self.controls and self.provider=='openai' and (
                completion.metadata.get('response_model')!=self.model or not completion.metadata.get('completion_id')):
            raise ValueError('OpenAI response must attest the frozen model and completion ID')
        if self.local_spec is not None:
            validate_local_call(record,self.local_spec)
        if self.remote_spec is not None:
            validate_screening_call(record,self.remote_spec)
        return completion
