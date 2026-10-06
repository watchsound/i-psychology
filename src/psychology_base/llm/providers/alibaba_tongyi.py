"""Alibaba Tongyi (通义千问) provider — DashScope native API over stdlib urllib.

DashScope has an OpenAI-compatible endpoint, but we use the native API
because the reference project (``Stock-AI/.../services/llm/alibaba_tongyi.py``)
proved it works, it returns ``request_id`` in the metadata (useful for
debugging stuck calls), and keeping the shape explicit documents what
Tongyi actually expects.

API ref: https://help.aliyun.com/zh/dashscope/

Model slots:
  SIMPLE   → qwen-turbo    (fast, cheap, strong Chinese + English)
  ADVANCED → qwen-max      (most capable; the judge/reader tier)

Both are OpenAI-chat-shaped under ``input.messages``. Returns token
counts under ``usage.input_tokens`` / ``usage.output_tokens`` — note the
distinct naming vs. OpenAI's ``prompt_tokens`` / ``completion_tokens``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from psychology_base.llm.types import ChatMessage, Completion, TaskTier


DASHSCOPE_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
)
DASHSCOPE_EMBED_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
)
DASHSCOPE_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)


@dataclass
class AlibabaTongyiProvider:
    name: str = "tongyi"
    simple_model: str = "qwen-turbo"
    advanced_model: str = "qwen-max"
    api_key: Optional[str] = None
    request_timeout_s: float = 60.0
    base_url: str = DASHSCOPE_URL
    # DashScope text-embedding-v3 is the SOTA retrieval-tuned embedder on
    # the same account that hosts qwen-turbo/qwen-max. 1024-dim; outranks
    # mxbai-embed-large on Chinese and matches on English (Phase 1.5 E1
    # v3 showed mxbai inverted "origin Sweden" vs "experience transition"
    # — the semantic-nearest-neighbor miss that drove the swap).
    embedding_model: str = "text-embedding-v3"
    embedding_url: str = DASHSCOPE_EMBED_URL
    # Cross-encoder reranker. Takes (query, [documents]) and returns ranked
    # [{index, relevance_score}]. Phase 1.5-prime P3 uses this to re-score
    # cosine top-K fragments jointly against the cue — closes paraphrase-
    # cue failures that bi-encoder cosine ranking buries.
    rerank_model: str = "gte-rerank-v2"
    rerank_url: str = DASHSCOPE_RERANK_URL
    _opened: bool = field(default=False, repr=False)

    def model_for(self, tier: TaskTier) -> str:
        return self.simple_model if tier == TaskTier.SIMPLE else self.advanced_model

    def _ensure_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "AlibabaTongyiProvider.chat() requires an API key. Set the "
                "env var named by `api_key_env` in configs/llm.yaml "
                "(default: ALIBABA_TONGYI_API_KEY)."
            )
        return self.api_key

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tier: TaskTier = TaskTier.SIMPLE,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Completion:
        key = self._ensure_key()
        model = self.model_for(tier)

        parameters: dict[str, Any] = {"result_format": "message"}
        if max_tokens is not None:
            parameters["max_tokens"] = max_tokens
        if temperature is not None:
            parameters["temperature"] = temperature

        payload = {
            "model": model,
            "input": {
                "messages": [
                    {"role": m.role, "content": m.content} for m in messages
                ],
            },
            "parameters": parameters,
        }

        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"Tongyi HTTP {exc.code}: {body[:500]}"
            ) from exc
        latency_ms = (time.monotonic() - t0) * 1000.0

        data = json.loads(raw)
        output = data.get("output") or {}
        choices = output.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content", "")
        else:
            # Legacy result_format=text path — some models still return
            # content under output.text.
            content = output.get("text", "")

        if not content:
            raise RuntimeError(
                f"Tongyi returned empty content. request_id="
                f"{data.get('request_id')!r} full={json.dumps(data, ensure_ascii=False)[:500]}"
            )

        usage = data.get("usage") or {}
        return Completion(
            content=content,
            provider=self.name,
            model_id=model,
            tier=tier,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            latency_ms=latency_ms,
            finish_reason=(choices[0].get("finish_reason", "") if choices else ""),
            metadata={"request_id": data.get("request_id", "")},
        )

    def embed(self, text: str) -> list[float]:
        """Return a 1024-dim embedding via DashScope text-embedding-v3.

        Shape:
          POST {embedding_url}
          { "model": "text-embedding-v3",
            "input": {"texts": ["..."]},
            "parameters": {} }
        Response: output.embeddings[0].embedding is the vector.

        Single-text per call: the batch surface is handled by
        ``LLMRouter.embed_batch`` running concurrent single calls, so the
        router's SHA-keyed cache stays per-text. If this becomes a
        throughput bottleneck we can switch to native batch here.
        """
        key = self._ensure_key()
        payload = {
            "model": self.embedding_model,
            "input": {"texts": [text]},
            "parameters": {},
        }
        req = urllib.request.Request(
            self.embedding_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"Tongyi embedding HTTP {exc.code}: {body[:500]}"
            ) from exc

        data = json.loads(raw)
        output = data.get("output") or {}
        embeddings = output.get("embeddings") or []
        if not embeddings:
            raise RuntimeError(
                f"Tongyi embedding returned no vectors. request_id="
                f"{data.get('request_id')!r} full={json.dumps(data, ensure_ascii=False)[:500]}"
            )
        vec = embeddings[0].get("embedding")
        if not isinstance(vec, list) or not vec:
            raise RuntimeError(
                f"Tongyi embedding returned malformed vector. full="
                f"{json.dumps(embeddings[0], ensure_ascii=False)[:300]}"
            )
        return [float(x) for x in vec]

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """Return ``[(original_index, relevance_score)]`` sorted desc.

        Shape:
          POST {rerank_url}
          { "model": "gte-rerank-v2",
            "input": {"query": "...", "documents": [...]},
            "parameters": {"top_n": N, "return_documents": false} }
        Response: ``output.results = [{"index": int, "relevance_score": float}]``.

        ``top_n`` caps the response size on the server side; passing None
        returns scores for every input document (useful when the caller
        wants to re-sort externally).
        """
        key = self._ensure_key()
        parameters: dict[str, Any] = {"return_documents": False}
        if top_n is not None:
            parameters["top_n"] = top_n
        payload = {
            "model": self.rerank_model,
            "input": {"query": query, "documents": documents},
            "parameters": parameters,
        }
        req = urllib.request.Request(
            self.rerank_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"Tongyi rerank HTTP {exc.code}: {body[:500]}"
            ) from exc
        data = json.loads(raw)
        output = data.get("output") or {}
        results = output.get("results") or []
        if not results:
            raise RuntimeError(
                f"Tongyi rerank returned no results. request_id="
                f"{data.get('request_id')!r} full={json.dumps(data, ensure_ascii=False)[:500]}"
            )
        out: list[tuple[int, float]] = []
        for r in results:
            idx = r.get("index")
            score = r.get("relevance_score")
            if idx is None or score is None:
                continue
            out.append((int(idx), float(score)))
        out.sort(key=lambda it: -it[1])
        return out

    def model_for_embedding(self) -> str:
        """Hook used by LLMRouter to label cache keys. The router already
        reads ``embedding_model`` directly; this method is for
        symmetry with providers that compute it dynamically."""
        return self.embedding_model
