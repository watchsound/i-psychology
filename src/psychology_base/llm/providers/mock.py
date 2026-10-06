"""Mock provider for tests.

Deterministic. Records every call so tests can assert routing + batching
behaviour without a real backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from psychology_base.llm.types import ChatMessage, Completion, TaskTier


Responder = Callable[[list[ChatMessage], TaskTier], str]


@dataclass
class MockProvider:
    """Returns scripted responses; records invocations for test assertions.

    ``responder`` lets tests shape the response based on messages/tier.
    Default echoes the last user message, which is enough for routing +
    cache tests.
    """

    name: str = "mock"
    simple_model: str = "mock-simple"
    advanced_model: str = "mock-advanced"
    simple_embed_dim: int = 8
    responder: Optional[Responder] = None
    calls: list[tuple[list[ChatMessage], TaskTier]] = field(default_factory=list)
    embed_calls: list[str] = field(default_factory=list)

    def model_for(self, tier: TaskTier) -> str:
        return self.simple_model if tier == TaskTier.SIMPLE else self.advanced_model

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tier: TaskTier = TaskTier.SIMPLE,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Completion:
        self.calls.append((list(messages), tier))
        content = (
            self.responder(messages, tier)
            if self.responder is not None
            else self._echo(messages)
        )
        return Completion(
            content=content,
            provider=self.name,
            model_id=self.model_for(tier),
            tier=tier,
            prompt_tokens=sum(len(m.content) for m in messages),
            completion_tokens=len(content),
            finish_reason="stop",
        )

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        # Deterministic cheap pseudo-embedding keyed on text hash — not a
        # real semantic vector, just useful for cache + batching tests.
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.simple_embed_dim]]

    @staticmethod
    def _echo(messages: list[ChatMessage]) -> str:
        for m in reversed(messages):
            if m.role == "user":
                return f"echo: {m.content}"
        return "echo: (no user message)"
