"""OpenAI provider — lazy-imports the ``openai`` SDK on first call.

Importing this module does not require ``openai`` to be installed; only
calling ``chat()`` does. Keeps tests and Ollama-only installs clean.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from psychology_base.llm.types import ChatMessage, Completion, TaskTier
from psychology_base.llm.execution import current_operation, remaining_time


@dataclass
class OpenAIProvider:
    name: str = "openai"
    simple_model: str = "gpt-4o-mini"
    advanced_model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # for OpenAI-compatible proxies
    _client: Any = field(default=None, repr=False)
    max_retries: Optional[int] = None
    reasoning_effort: Optional[str] = None
    json_mode: bool = False
    enable_thinking: Optional[bool] = None  # Alibaba's compatible API; omitted for OpenAI.
    request_timeout_s: Optional[float] = None
    follow_redirects: Optional[bool] = None
    total_completion_limit: bool = False
    response_schema: Optional[dict] = None
    api_dialect: str = "openai"  # Explicit compatible API; serving defaults stay OpenAI.

    def __post_init__(self):
        if self.api_dialect not in ("openai", "deepseek"):
            raise ValueError("Unsupported compatible API dialect")
        if self.api_dialect == "deepseek" and self.enable_thinking is not None:
            raise ValueError("DeepSeek thinking uses reasoning_effort, not Alibaba controls")
        if self.reasoning_effort not in (None, "none", "minimal", "low", "medium", "high", "xhigh", "max"):
            raise ValueError("Unsupported reasoning effort")
        if type(self.json_mode) is not bool:
            raise ValueError("json_mode must be boolean")
        if self.enable_thinking is not None and type(self.enable_thinking) is not bool:
            raise ValueError("enable_thinking must be boolean or null")
        if self.enable_thinking is not None and self.reasoning_effort is not None:
            raise ValueError("Provider thinking controls cannot be combined")
        if self.request_timeout_s is not None and (
                type(self.request_timeout_s) not in (int, float) or not 0 < self.request_timeout_s <= 300):
            raise ValueError("request_timeout_s must be positive and at most 300")
        if self.follow_redirects is not None and type(self.follow_redirects) is not bool:
            raise ValueError("follow_redirects must be boolean or null")
        if type(self.total_completion_limit) is not bool:
            raise ValueError("total_completion_limit must be boolean")
        if self.response_schema is not None:
            if (not self.json_mode or type(self.response_schema) is not dict
                    or self.response_schema.get('type') != 'object'):
                raise ValueError('response_schema requires JSON mode and an object schema')
            json.dumps(self.response_schema, allow_nan=False)
            self.response_schema = copy.deepcopy(self.response_schema)

    def model_for(self, tier: TaskTier) -> str:
        return self.simple_model if tier == TaskTier.SIMPLE else self.advanced_model

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover — exercised in integration only
            raise RuntimeError(
                "OpenAIProvider.chat() requires the `openai` package. "
                "Install with `pip install openai`."
            ) from exc
        kwargs: dict = {}
        if self.max_retries is not None:
            kwargs["max_retries"] = self.max_retries
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.request_timeout_s is not None:
            kwargs["timeout"] = self.request_timeout_s
        if self.follow_redirects is not None:
            import httpx
            kwargs["http_client"] = httpx.Client(follow_redirects=self.follow_redirects)
        self._client = openai.OpenAI(**kwargs)
        return self._client

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tier: TaskTier = TaskTier.SIMPLE,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Completion:
        if temperature is not None and self.reasoning_effort not in (None, "none"):
            raise ValueError("Temperature requires reasoning effort none")
        client = self._ensure_client()
        model = self.model_for(tier)
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
        }
        if max_tokens is not None:
            kwargs["max_completion_tokens" if self.api_dialect != "deepseek" and (
                   self.reasoning_effort is not None or self.total_completion_limit)
                   else "max_tokens"] = max_tokens
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
            if self.api_dialect != "deepseek":
                kwargs["service_tier"] = "default"
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.response_schema is not None:
            kwargs['response_format'] = dict(type='json_schema', json_schema=dict(
                name='conversation_assessment', strict=True, schema=copy.deepcopy(self.response_schema)))
        if self.enable_thinking is not None:
            kwargs["extra_body"] = {"enable_thinking": self.enable_thinking}
        if temperature is not None:
            kwargs["temperature"] = temperature

        t0 = time.monotonic()
        if current_operation.get() is not None and current_operation.get().deadline is not None:
            kwargs["timeout"] = remaining_time(self.request_timeout_s)
        raw_usage = None
        if self.total_completion_limit:
            raw_response = client.chat.completions.with_raw_response.create(**kwargs)
            body = raw_response.http_response.json()
            reported = body.get('usage') if isinstance(body, dict) else None
            raw_usage = reported if isinstance(reported, dict) else {}
            response = raw_response.parse()
        else:
            response = client.chat.completions.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000.0

        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        metadata = {key: value for key, value in (
            ("completion_id", getattr(response, "id", None)),
            ("request_id", getattr(response, "_request_id", None)),
            ("response_model", getattr(response, "model", None)),
            ("system_fingerprint", getattr(response, "system_fingerprint", None)),
        ) if value is not None}
        if self.api_dialect == "deepseek":
            metadata.update(api_dialect="deepseek", output_limit_parameter="max_tokens")
        if self.reasoning_effort is not None:
            details = getattr(usage, "completion_tokens_details", None)
            metadata.update(reasoning_effort=self.reasoning_effort,
                            reasoning_tokens=getattr(details, "reasoning_tokens", None),
                            service_tier=getattr(response, "service_tier", None))
        if self.json_mode:
            metadata["json_mode"] = True
        if self.response_schema is not None:
            metadata.update(response_format='json_schema', response_schema_sha256=hashlib.sha256(
                json.dumps(self.response_schema, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest())
        if self.enable_thinking is not None:
            details = getattr(usage, "completion_tokens_details", None)
            metadata.update(enable_thinking=self.enable_thinking,
                reasoning_tokens=getattr(details, "reasoning_tokens", None),
                reasoning_content_present=bool(getattr(choice.message, "reasoning_content", None)))
        if self.total_completion_limit:
            # SDK model parsing can coerce booleans/strings to integers. Billing
            # and reasoning evidence must retain the provider's original types.
            details = raw_usage.get('completion_tokens_details')
            choices = body.get('choices') if isinstance(body, dict) else None
            first = choices[0] if isinstance(choices, list) and choices else None
            message = first.get('message') if isinstance(first, dict) else None
            reasoning_text = message.get('reasoning_content') if isinstance(message, dict) else None
            metadata.update(total_completion_limit=True, configured_max_completion_tokens=max_tokens,
                reported_usage_types_preserved=True,
                reasoning_content_present=isinstance(reasoning_text, str) and bool(reasoning_text.strip()),
                reasoning_tokens=details.get('reasoning_tokens') if isinstance(details, dict) else None)
        return Completion(
            content=choice.message.content or "",
            provider=self.name,
            model_id=model,
            tier=tier,
            prompt_tokens=(raw_usage.get('prompt_tokens', 0) if self.total_completion_limit else
                           getattr(usage, "prompt_tokens", 0) if usage else 0),
            completion_tokens=(raw_usage.get('completion_tokens', 0) if self.total_completion_limit else
                               getattr(usage, "completion_tokens", 0) if usage else 0),
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason or "",
            metadata=metadata,
        )
