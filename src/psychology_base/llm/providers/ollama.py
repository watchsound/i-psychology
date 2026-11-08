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

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit

from psychology_base.llm.types import ChatMessage, Completion, TaskTier, validate_embedding
from psychology_base.llm.execution import remaining_time
from psychology_base.llm.loopback import open_loopback, RejectRedirects as _RejectRedirects

EMBEDDING_BATCH_SIZE = 32

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
    # Multilingual slot — Phase 9+ i18n reserved knob. ``LLMRouter.embed``
    # routes here instead of ``embedding_model`` when
    # ``i18n.embedder_multilingual`` is true in the top-level config.
    # Default None preserves single-slot behavior for pre-i18n configs.
    # Set to ``bge-m3`` (1024-dim, 100+ languages, retrieval-tuned) for
    # Chinese-first application deployments.
    embedding_model_multilingual: Optional[str] = None
    # Opt-in isolation for local evaluations; ordinary callers keep their
    # existing transport and decoding defaults.
    local_only: bool = False
    expected_digest: Optional[str] = None
    context_length: Optional[int] = None
    seed: Optional[int] = None
    json_mode: bool = False
    # Populated by ``probe`` and consulted by the router before dispatch
    # so a cold cache doesn't retry per-call when Ollama is down.
    _server_up: Optional[bool] = field(default=None, repr=False)
    _available_models: Optional[frozenset[str]] = field(default=None, repr=False)
    model_metadata_timeout_s: float = 10.0
    # Opt-in reuse of artifact-derived embedding metadata, never of live tags.
    # One process-local entry expires without extending its lifetime on reads.
    embedding_metadata_ttl_s: float = 0.0
    # Runtime transport isolation does not imply a research digest attestation.
    loopback_only: bool = False
    # Reuse only digest-addressed /show details. Tags remain live on every check.
    model_metadata_ttl_s: float = 30.0
    _model_metadata_cache: Optional[tuple] = field(default=None, init=False, repr=False, compare=False)
    # Opt-in native structured output. Ordinary JSON-mode payloads stay exact.
    response_schema: Optional[dict] = None
    # Explicit controls are limited to pinned local experiments. None keeps
    # the historical request and model-default behavior byte-identical.
    think: Optional[bool] = None

    def enforce_loopback(self):
        url = urlsplit(self.base_url)
        if (url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1")
                or url.port != 11434 or url.username or url.password
                or url.path not in ("", "/") or url.query or url.fragment):
            raise ValueError("Local execution requires the loopback Ollama endpoint")
        self.base_url = "http://127.0.0.1:11434"
        self.loopback_only = True

    def require_local_model(self, model, *, capability):
        """Check installed model and reject Ollama cloud forwarding on each miss.

        Recovery is checked per attempted operation, independently of the cached
        chat probe. Historical pinned evaluations retain their stronger checks.
        """
        self.enforce_loopback()
        self._model_metadata(model, capability=capability, ttl=self.model_metadata_ttl_s, reuse=True)

    def model_for(self, tier: TaskTier) -> str:
        return self.simple_model if tier == TaskTier.SIMPLE else self.advanced_model

    def _open(self, request, *, timeout):
        timeout = remaining_time(timeout)
        if not (self.local_only or self.loopback_only):
            return urllib.request.urlopen(request, timeout=timeout)
        url = urlsplit(request.full_url)
        if (url.scheme != "http" or url.hostname != "127.0.0.1" or url.username or url.password
                or url.query or url.fragment or url.port != 11434):
            raise ValueError("Local-only Ollama requires http://127.0.0.1:11434")
        return open_loopback(request, base_url='http://127.0.0.1:11434', timeout=timeout)

    def validate_metadata_settings(self, *, ttl):
        """Validate metadata limits without network access or cache mutation."""
        timeout = self.model_metadata_timeout_s
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Model metadata timeout must be finite and positive")
        if type(ttl) not in (int, float) or not math.isfinite(ttl) or not 0 <= ttl <= 60:
            raise ValueError("Model metadata TTL must be finite and within [0, 60] seconds")
        return timeout

    def _model_metadata(self, model, *, capability, ttl, reuse, pinned=False):
        """Fresh tags plus one bounded, immutable artifact-details entry.

        No digest means no reuse. Changes to any tag field, endpoint or context,
        failures, expiry and explicit refresh require /show again. A cache read
        never extends its lifetime; callers receive independent decoded objects.
        """
        try:
            if capability not in ("completion", "embedding"):
                raise ValueError("Unsupported model attestation capability")
            timeout = self.validate_metadata_settings(ttl=ttl)
            request = urllib.request.Request(f"{self.base_url.rstrip('/')}/api/tags")
            with self._open(request, timeout=timeout) as response:
                models = json.load(response).get("models", [])
            aliases = {model, model + ":latest"} if not pinned and ":" not in model else {model}
            matches = [m for m in models if m.get("name") in aliases]
            if (len(matches) != 1 or type(matches[0].get("size")) not in (int, float)
                    or not math.isfinite(matches[0]["size"]) or matches[0]["size"] <= 0
                    or (pinned and matches[0].get("digest") != self.expected_digest)):
                raise ValueError("Pinned Ollama model is missing or its digest changed" if pinned
                                 else "Configured local model is not installed unambiguously")
            tag = matches[0]
            digest = tag.get("digest")
            content_addressed = (isinstance(digest, str) and len(digest) == 64
                                 and all(c in "0123456789abcdef" for c in digest))
            key = (self.base_url, model, self.expected_digest, self.context_length,
                   json.dumps(tag, sort_keys=True, allow_nan=False))
            cached = self._model_metadata_cache
            hit = (content_addressed and reuse and ttl > 0 and cached is not None and cached[0] == key
                    and 0 <= time.monotonic() - cached[1] < min(ttl, cached[2]))
            if hit:
                details = json.loads(cached[3])
            else:
                self._model_metadata_cache = None
                request = urllib.request.Request(f"{self.base_url.rstrip('/')}/api/show",
                    data=json.dumps({"model": model}).encode(), headers={"Content-Type": "application/json"})
                with self._open(request, timeout=timeout) as response:
                    details = json.load(response)
            if details.get("remote_model") or details.get("remote_host") or capability not in details.get("capabilities", []):
                raise ValueError(f"Local model requires installed {capability} capability without cloud forwarding")
            if not hit and content_addressed and ttl > 0:
                # Cache lifetime starts when validated immutable details arrive.
                # Fresh tags above still guard every read; hits never renew it.
                self._model_metadata_cache = (key, time.monotonic(), ttl, json.dumps(details, allow_nan=False))
            return tag, details
        except Exception:
            self._model_metadata_cache = None
            raise

    def attest_model(self, model, *, capability="completion", _reuse_metadata=False):
        """Verify installed bytes; explicit calls refresh details by default."""
        if capability not in ("completion", "embedding"):
            raise ValueError("Unsupported model attestation capability")
        if not self.local_only or not self.expected_digest:
            raise ValueError("Model attestation requires local-only transport and a pinned digest")
        requested_think = self.think if capability == 'completion' else None
        if requested_think is not None and type(requested_think) is not bool:
            raise ValueError('Thinking control must be an explicit boolean or None')
        ttl = self.embedding_metadata_ttl_s if capability == "embedding" else self.model_metadata_ttl_s
        try:
            tag, details = self._model_metadata(model, capability=capability, ttl=ttl,
                                                reuse=_reuse_metadata, pinned=True)
            lengths = [v for k, v in details.get("model_info", {}).items() if k.endswith(".context_length")]
            if (not lengths or any(type(v) is not int or v <= 0 for v in lengths)
                    or type(self.context_length) is not int or self.context_length <= 0
                    or self.context_length > min(lengths)):
                raise ValueError("Frozen context length exceeds the installed model's capacity")
            result = {"model": model, "digest": self.expected_digest,
                      "family": details.get("details", {}).get("family"), "size": tag["size"]}
            if requested_think is not None:
                controls = details.get('thinking')
                values = controls.get('values') if type(controls) is dict else None
                if (type(values) is not list or not values
                        or any(type(value) is not bool for value in values)
                        or not any(value is requested_think for value in values)):
                    raise ValueError('Installed model does not attest the requested thinking control')
                result['thinking_control'] = dict(requested=requested_think, supported=list(values))
            if capability == "embedding":
                dimensions = [v for k, v in details.get("model_info", {}).items() if k.endswith(".embedding_length")]
                if (not dimensions or any(type(v) is not int or v <= 0 for v in dimensions)
                        or len(set(dimensions)) != 1):
                    raise ValueError("Installed embedding dimension is missing or inconsistent")
                if self.context_length != min(lengths):
                    raise ValueError("Pinned embedding context must match the installed model's capacity")
                result.update(dimension=dimensions[0], context_length=self.context_length)
            return result
        except Exception:
            self._model_metadata_cache = None
            raise

    def embedding_identity(self, model, *, refresh=False):
        """Fresh tags and bounded artifact metadata for no-truncation inputs.

        ``refresh`` forces full metadata verification even inside the opt-in
        reuse window. Failed checks invalidate the process-local entry.
        """
        try:
            attested = self.attest_model(model, capability="embedding", _reuse_metadata=not refresh)
        except Exception:
            self._model_metadata_cache = None
            raise
        return {key: attested[key] for key in ("model", "digest", "dimension", "context_length")} | {
            "input_policy": "ollama-embed-raw-no-truncate-v1"}

    # ---------- health probe ----------

    def probe(self, *, timeout_s: float = 1.0, model: Optional[str] = None) -> bool:
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
            with self._open(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = {m["name"] for m in data.get("models", [])}
            self._available_models = frozenset(models)
            self._server_up = True
            return (model if model is not None else self.simple_model) in models
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            self._server_up = False
            self._available_models = frozenset()
            self._model_metadata_cache = None
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
        think = self.think
        if think is not None and (type(think) is not bool or not self.local_only):
            raise ValueError('Explicit thinking control requires a boolean and pinned local transport')
        schema = self.response_schema
        if schema is not None:
            if type(schema) is not dict or not schema or any(type(k) is not str for k in schema):
                raise ValueError('Ollama response schema must be a nonempty JSON object')
            # Snapshot before attestation/transport; never mutate a caller's tree.
            schema = json.loads(json.dumps(schema, allow_nan=False))
        model = self.model_for(tier)
        attestation = self.attest_model(model, _reuse_metadata=True) if self.local_only else None
        if think is not None and attestation.get('thinking_control', {}).get('requested') is not think:
            raise ValueError('Thinking control changed during model attestation')
        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if self.context_length is not None:
            options["num_ctx"] = self.context_length
        if self.seed is not None:
            options["seed"] = self.seed

        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": options,
        }
        if think is not None:
            payload['think'] = think
        if schema is not None:
            payload["format"] = schema
        elif self.json_mode:
            payload["format"] = "json"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.monotonic()
        try:
            with self._open(req, timeout=self.request_timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
        except Exception:
            self._model_metadata_cache = None
            raise
        latency_ms = (time.monotonic() - t0) * 1000.0

        content = (data.get("message") or {}).get("content", "")
        if think is not None:
            trace = (data.get('message') or {}).get('thinking')
            if trace is not None and (type(trace) is not str or think is False and trace != ''):
                raise ValueError('Native thinking output contradicts the requested control')
        metadata = {"total_duration_ns": data.get("total_duration", 0)}
        if think is not None:
            metadata['think'] = think
        if schema is not None:
            metadata['response_schema_sha256'] = hashlib.sha256(json.dumps(schema, sort_keys=True,
                ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
        if attestation is not None:
            metadata.update(local_only=True, model_attestation=attestation,
                            response_model=data.get("model"), done=data.get("done"),
                            context_length=self.context_length, seed=self.seed, json_mode=self.json_mode)
        return Completion(
            content=content,
            provider=self.name,
            model_id=model,
            tier=tier,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=latency_ms,
            finish_reason=data.get("done_reason", ""),
            metadata=metadata,
        )

    # ---------- embeddings ----------

    @contextmanager
    def verified_embedding_batch(self, model):
        """One fresh identity boundary around cache reads and native requests.

        Consumers publish results/cache writes only after this scope exits.
        The encoder is local to this scope, never a reusable attestation token.
        """
        identity = self.embedding_identity(model)
        active = True
        inferred = False

        def encode(texts):
            nonlocal inferred
            if not active:
                raise ValueError("Embedding verification scope has closed")
            texts = list(texts)
            if any(not isinstance(text, str) for text in texts) or len(texts) > EMBEDDING_BATCH_SIZE:
                raise ValueError("Native embedding batch requires at most 32 text inputs")
            if not texts:
                return []
            inferred = True
            req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/api/embed",
                data=json.dumps({"model": model, "input": texts[0] if len(texts) == 1 else texts,
                                 "truncate": False}).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            with self._open(req, timeout=self.request_timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("model") != model:
                raise ValueError("Embedding response model differs from the pinned model")
            rows = data.get("embeddings")
            if not isinstance(rows, list) or len(rows) != len(texts):
                raise ValueError("Embedding response must contain one vector per input")
            return [validate_embedding(row, dimension=identity["dimension"]) for row in rows]

        try:
            yield dict(identity), encode
            if self.embedding_identity(model, refresh=inferred) != identity:
                raise ValueError("Embedding model identity changed during the batch")
        except Exception:
            self._model_metadata_cache = None
            raise
        finally:
            active = False

    def embed(self, text: str, *, model: Optional[str] = None) -> list[float]:
        """Return an embedding vector via Ollama's /api/embeddings endpoint.

        Model precedence: explicit kwarg > ``embedding_model`` > ``simple_model``.
        The ``simple_model`` fallback preserves backward compatibility but
        yields degenerate vectors for chat-tuned LLMs — Phase 1.5 E1 v1
        confirmed this (cos(cue, correct_fact) ≈ 0.25 vs cos(cue,
        distractor) ≈ 0.46). Configure a real embedder.
        """
        model = model or self.embedding_model or self.simple_model
        if self.local_only:
            with self.verified_embedding_batch(model) as (_, encode):
                return encode([text])[0]
        payload = {"model": model, "prompt": text}
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._open(req, timeout=self.request_timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return list(data.get("embedding", []))
