"""Ollama local provider — chat + embeddings over stdlib urllib.

Uses stdlib so the project doesn't acquire an httpx/requests dependency
just for Ollama. Ollama is critical to the project's cost model:
semantic-analysis calls are high-frequency (per-fragment summarization,
per-recall re-ranking, Witness divergence scoring), and running them
against a local model keeps the hot path at zero marginal cost.

The ``probe()`` method verifies BOTH that the server is reachable AND
that the configured ``simple_model`` is actually pulled. A server that's
up but missing the model would pass a plain liveness ping and then 404
every ``/api/chat`` call — exactly the silent-failure mode the reference
project at Stock-AI learned to guard against.

API refs:
  chat:       POST {base_url}/api/chat
  embeddings: POST {base_url}/api/embeddings
  tags:       GET  {base_url}/api/tags
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from psychology_base.llm.types import ChatMessage, Completion, TaskTier


@dataclass
class OllamaProvider:
    name: str = "ollama"
    base_url: str = "http://localhost:11434"
    simple_model: str = "qwen2.5:7b"
    advanced_model: str = "qwen2.5:32b"
    request_timeout_s: float = 120.0
    # Dedicated embedding model. Falls back to ``simple_model`` for
    # backward compatibility, but chat LLMs produce non-retrieval-tuned
    # vectors — set this to bge-m3 / mxbai-embed-large / bge-small-zh.
    embedding_model: Optional[str] = None
    # Populated by ``probe`` and consulted by the router before dispatch
    # so a cold cache doesn't retry per-call when Ollama is down.
    _server_up: Optional[bool] = field(default=None, repr=False)
    _available_models: Optional[frozenset[str]] = field(default=None, repr=False)

    def model_for(self, tier: TaskTier) -> str:
        return self.simple_model if tier == TaskTier.SIMPLE else self.advanced_model

    # ---------- health probe ----------

    def probe(self, *, timeout_s: float = 1.0) -> bool:
        """Check that Ollama is reachable AND the simple_model is pulled.

        Only ``simple_model`` is probed because that's the hot-path tier.
        If a caller uses ``tier=ADVANCED`` while the advanced model isn't
        pulled, the chat call itself raises — correct behaviour for a
        rare advanced-tier miss.
        """
        try:
            req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = {m["name"] for m in data.get("models", [])}
            self._available_models = frozenset(models)
            self._server_up = True
            return self.simple_model in models
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            self._server_up = False
            self._available_models = frozenset()
            return False

    # ---------- chat ----------

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tier: TaskTier = TaskTier.SIMPLE,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Completion:
        model = self.model_for(tier)
        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": options,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        latency_ms = (time.monotonic() - t0) * 1000.0

        data = json.loads(raw)
        content = (data.get("message") or {}).get("content", "")
        return Completion(
            content=content,
            provider=self.name,
            model_id=model,
            tier=tier,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=latency_ms,
            finish_reason=data.get("done_reason", ""),
            metadata={"total_duration_ns": data.get("total_duration", 0)},
        )

    # ---------- embeddings ----------

    def embed(self, text: str, *, model: Optional[str] = None) -> list[float]:
        """Return an embedding vector via Ollama's /api/embeddings endpoint.

        Model precedence: explicit kwarg > ``embedding_model`` > ``simple_model``.
        The ``simple_model`` fallback preserves backward compatibility but
        yields degenerate vectors for chat-tuned LLMs — Phase 1.5 E1 v1
        confirmed this (cos(cue, correct_fact) ≈ 0.25 vs cos(cue,
        distractor) ≈ 0.46). Configure a real embedder.
        """
        model = model or self.embedding_model or self.simple_model
        payload = {"model": model, "prompt": text}
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return list(data.get("embedding", []))
