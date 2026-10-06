"""Core types and the LLMProvider protocol.

Pattern follows the reference project at
``ai-business/Stock-AI/ai-stock-market-analysis/backend/app/services/llm/`` —
``TaskTier`` routes requests (SIMPLE vs. ADVANCED), each provider exposes
both a simple and an advanced model slot, a unified router makes the
selection. Simplified for a research library:

- synchronous (no asyncio) — callers are CLI tools and batch benchmark
  runs, not a FastAPI server;
- no cost tracking (not needed at research stage);
- no streaming surface (no call site needs it yet);
- vendor SDKs lazy-imported inside each provider so the project never
  forces an ``openai`` or ``anthropic`` install.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, Protocol, runtime_checkable


class TaskTier(str, Enum):
    """Which model slot to route to.

    SIMPLE  — cheap/fast; semantic classification, labelling, judge-style
              yes/no scoring. Prefers local Ollama when reachable.
    ADVANCED — higher-quality reasoning; multi-step planning, free-form
              judge calls where a wrong answer invalidates downstream work.
    """

    SIMPLE = "simple"
    ADVANCED = "advanced"


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class Completion:
    """Response envelope from a provider.

    ``model_id`` is the concrete model string the provider actually
    dispatched to — needed as part of cache keys so a model swap
    invalidates stale cached verdicts (see
    ``eval/benchmarks/judge.py::JudgmentKey.judge_id``).
    """

    content: str
    provider: str
    model_id: str
    tier: TaskTier
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Minimum surface a concrete provider must implement.

    Providers must be side-effect-free at import time — SDK clients are
    constructed lazily on first ``chat`` call. This keeps tests cheap
    (no vendor SDK installed) and startup fast.
    """

    name: str

    def model_for(self, tier: TaskTier) -> str:
        """Return the concrete model string this provider uses for ``tier``."""
        ...

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tier: TaskTier = TaskTier.SIMPLE,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Completion:
        """Dispatch a chat completion. Must not retry silently — raise on
        provider errors so the router / caller can decide retry policy."""
        ...
