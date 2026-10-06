"""OpenAI provider — lazy-imports the ``openai`` SDK on first call.

Importing this module does not require ``openai`` to be installed; only
calling ``chat()`` does. Keeps tests and Ollama-only installs clean.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from psychology_base.llm.types import ChatMessage, Completion, TaskTier


@dataclass
class OpenAIProvider:
    name: str = "openai"
    simple_model: str = "gpt-4o-mini"
    advanced_model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # for OpenAI-compatible proxies
    _client: Any = field(default=None, repr=False)

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
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
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
        client = self._ensure_client()
        model = self.model_for(tier)
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature

        t0 = time.monotonic()
        response = client.chat.completions.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000.0

        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return Completion(
            content=choice.message.content or "",
            provider=self.name,
            model_id=model,
            tier=tier,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason or "",
        )
