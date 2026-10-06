"""Concrete LLM provider implementations.

Each provider module exposes a class that satisfies ``LLMProvider`` (see
``../types.py``) by lazy-importing its vendor SDK on first ``chat()``
call. Importing this package must not import any vendor SDK — tests and
Ollama-only deployments work without them.
"""

from psychology_base.llm.providers.alibaba_tongyi import AlibabaTongyiProvider
from psychology_base.llm.providers.anthropic import AnthropicProvider
from psychology_base.llm.providers.mock import MockProvider
from psychology_base.llm.providers.ollama import OllamaProvider
from psychology_base.llm.providers.openai import OpenAIProvider

__all__ = [
    "AlibabaTongyiProvider",
    "AnthropicProvider",
    "MockProvider",
    "OllamaProvider",
    "OpenAIProvider",
]
