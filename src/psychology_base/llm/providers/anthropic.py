"""Anthropic provider — lazy-imports the ``anthropic`` SDK on first call."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from psychology_base.llm.types import ChatMessage, Completion, TaskTier


@dataclass
class AnthropicProvider:
    name: str = "anthropic"
    simple_model: str = "claude-haiku-4-5-20251001"
    advanced_model: str = "claude-opus-4-7"
    api_key: Optional[str] = None
    _client: Any = field(default=None, repr=False)

    def model_for(self, tier: TaskTier) -> str:
        return self.simple_model if tier == TaskTier.SIMPLE else self.advanced_model

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "AnthropicProvider.chat() requires the `anthropic` package. "
                "Install with `pip install anthropic`."
            ) from exc
        kwargs: dict = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        self._client = anthropic.Anthropic(**kwargs)
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

        # Anthropic takes system prompt as a top-level field, not as a
        # role in messages. Split them.
        system_parts: list[str] = []
        chat_messages: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        kwargs: dict = {
            "model": model,
            "messages": chat_messages,
            # Anthropic requires max_tokens; use a sensible default when
            # caller doesn't supply one.
            "max_tokens": max_tokens if max_tokens is not None else 4096,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if temperature is not None:
            kwargs["temperature"] = temperature

        t0 = time.monotonic()
        response = client.messages.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000.0

        # Anthropic returns content as a list of blocks; concatenate text blocks.
        text_parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
        content = "".join(text_parts)

        usage = getattr(response, "usage", None)
        return Completion(
            content=content,
            provider=self.name,
            model_id=model,
            tier=tier,
            prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            latency_ms=latency_ms,
            finish_reason=getattr(response, "stop_reason", "") or "",
        )
