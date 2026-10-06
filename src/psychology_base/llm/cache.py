"""Disk-backed response cache for LLM calls.

Semantic analysis is a high-frequency operation across this project
(per-fragment summarization, per-recall relevance scoring, reconsolidation
delta detection, Witness divergence). Deterministic re-runs over the same
inputs — especially during benchmark iteration — should hit disk, not the
model. The cache is sharded (first 2 hex chars as subdirectory) to keep
any one directory under ~65k files.

Key composition (SHA-256 of canonical JSON):
    {provider, model_id, tier, temperature, max_tokens, messages,
     purpose='chat' | 'embed'}

A different model, tier, or temperature yields a different key — so a
model swap invalidates the cache without a manual flush.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from psychology_base.llm.types import ChatMessage, Completion, TaskTier


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def chat_cache_key(
    *,
    provider: str,
    model_id: str,
    tier: TaskTier,
    messages: list[ChatMessage],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> str:
    payload = {
        "purpose": "chat",
        "provider": provider,
        "model_id": model_id,
        "tier": tier.value,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def embedding_cache_key(
    *,
    provider: str,
    model_id: str,
    text: str,
) -> str:
    payload = {
        "purpose": "embed",
        "provider": provider,
        "model_id": model_id,
        "text": text,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class LLMCache:
    """SHA-sharded on-disk cache for chat and embedding responses.

    Writes are atomic (``.tmp`` then rename). A failed read deletes the
    corrupt entry silently so the next call re-populates — cache misses
    on cold load are preferable to crashing the research run.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "psychology_base" / "llm"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    # ---------- chat ----------

    def get_chat(self, key: str) -> Optional[Completion]:
        path = self._path_for(key, kind="chat")
        if not path.exists():
            self.misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            completion = Completion(
                content=data["content"],
                provider=data["provider"],
                model_id=data["model_id"],
                tier=TaskTier(data["tier"]),
                prompt_tokens=data.get("prompt_tokens", 0),
                completion_tokens=data.get("completion_tokens", 0),
                latency_ms=data.get("latency_ms", 0.0),
                finish_reason=data.get("finish_reason", ""),
                metadata=data.get("metadata", {}),
            )
            self.hits += 1
            return completion
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            try:
                path.unlink()
            except OSError:
                pass
            self.misses += 1
            return None

    def put_chat(self, key: str, completion: Completion) -> None:
        path = self._path_for(key, kind="chat")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(completion) if dataclasses.is_dataclass(completion) else {
            "content": completion.content,
            "provider": completion.provider,
            "model_id": completion.model_id,
            "tier": completion.tier.value,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "latency_ms": completion.latency_ms,
            "finish_reason": completion.finish_reason,
            "metadata": completion.metadata,
        }
        payload["tier"] = completion.tier.value  # enum → str for JSON
        self._atomic_write(path, _canonical_json(payload))

    # ---------- embeddings ----------

    def get_embedding(self, key: str) -> Optional[list[float]]:
        path = self._path_for(key, kind="embed")
        if not path.exists():
            self.misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.hits += 1
            return list(data["vector"])
        except (OSError, json.JSONDecodeError, KeyError):
            try:
                path.unlink()
            except OSError:
                pass
            self.misses += 1
            return None

    def put_embedding(self, key: str, vector: list[float], *, model_id: str) -> None:
        path = self._path_for(key, kind="embed")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"vector": list(vector), "model_id": model_id}
        self._atomic_write(path, _canonical_json(payload))

    # ---------- internals ----------

    def _path_for(self, key: str, *, kind: str) -> Path:
        shard = key[:2]
        return self.cache_dir / kind / shard / f"{key}.json"

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        # Per-writer tmp name so concurrent writers on the same key
        # (chat_batch over duplicate prompts) don't fight for a shared
        # ``.tmp``. Windows raises PermissionError on the second
        # ``os.replace`` otherwise.
        suffix = f".{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        tmp = path.with_name(path.name + suffix)
        tmp.write_text(text, encoding="utf-8")
        try:
            os.replace(tmp, path)
        except PermissionError:
            # Another writer already finalized this key — our content
            # is semantically identical (SHA-keyed), so dropping is fine.
            try:
                tmp.unlink()
            except OSError:
                pass
