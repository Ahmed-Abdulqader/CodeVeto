"""
embedding.py

Replaces the old `OllamaEmbedder` (local Ollama runtime + nomic-embed-text,
~2GB combined download) with an interface that has no required download at
all. `search_context` works fully on `search/lexical.py`'s BM25 index with
`NullEmbeddingProvider` (the default) or no provider configured.

If dense/semantic search is wanted later, `RemoteEmbeddingProvider` calls
any OpenAI-embeddings-compatible HTTP endpoint (OpenAI, OpenRouter, a
self-hosted server, ...) using only `urllib.request` from the stdlib — no
new dependency, and nothing downloaded to disk. `FallbackEmbeddingProvider`
chains several of those, in the same spirit as CodeVeto's multi-model LLM
routing: if one provider is unreachable or out of quota, the next is
tried, and if all of them fail the engine logs it and falls back to
lexical-only search rather than stopping.

For fully offline semantic search, `LocalOnnxEmbeddingProvider` runs a
quantized ONNX model (e.g. jina-embeddings-v2-base-code's INT8 export,
~160MB) on-device via `onnxruntime`, with no network call at index time or
query time. Unlike `RemoteEmbeddingProvider`, this one does need two extra
packages — `onnxruntime` and `tokenizers` — imported lazily inside
`LocalOnnxEmbeddingProvider.__init__` rather than at module load, so
importing this file (and using BM25 or the remote provider) still needs
neither of them installed.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from context_engine.logging_config import get_logger

logger = get_logger("context_engine.embedding")


class EmbeddingProviderError(RuntimeError):
    """Raised by a provider when it can't produce an embedding. Callers
    (FallbackEmbeddingProvider, HybridSearch) catch this and move on."""


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class NullEmbeddingProvider:
    """The default. Signals "semantic search is off" without every caller
    needing an `if embedding_provider is not None` check — call it, get an
    `EmbeddingProviderError`, and the caller's existing fallback path
    (skip to lexical-only) handles it the same as a real provider failing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingProviderError("No embedding provider configured.")

    def embed_query(self, text: str) -> list[float]:
        raise EmbeddingProviderError("No embedding provider configured.")


class RemoteEmbeddingProvider:
    """Calls a remote OpenAI-embeddings-compatible endpoint
    (`POST {base_url}/embeddings`, body `{"model": ..., "input": [...]}`).

    The API key is read from an environment variable at call time (never
    stored in config, the database, or a log line) so this works the same
    way with OpenAI, OpenRouter, or any compatible self-hosted server —
    swap `base_url` and `api_key_env`, nothing else changes.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str,
        dimension: int | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.dimension = dimension
        self.timeout_seconds = timeout_seconds

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise EmbeddingProviderError(
                f"Environment variable {self.api_key_env!r} is not set."
            )
        request_body: dict = {"model": self.model, "input": texts}
        if self.dimension is not None:
            # Several providers behind OpenAI-compatible endpoints (OpenAI's
            # text-embedding-3-*, Voyage 4, Gemini Embedding 2) accept a
            # `dimensions` field to truncate the output vector. `dimension`
            # was previously stored on this class but never sent — without
            # it, passing a value here silently did nothing.
            request_body["dimensions"] = self.dimension
        body = json.dumps(request_body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise EmbeddingProviderError(f"Embedding request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise EmbeddingProviderError(
                f"Embedding response was not valid JSON: {exc}"
            ) from exc

        try:
            return [item["embedding"] for item in payload["data"]]
        except (KeyError, TypeError) as exc:
            raise EmbeddingProviderError(
                f"Unexpected embedding response shape: {payload!r}"
            ) from exc


class LocalOnnxEmbeddingProvider:
    """Runs a local ONNX embedding model fully offline — no network call
    at index time or query time, ever. This class never downloads
    anything itself; point `model_path`/`tokenizer_path` at files you
    already have on disk (that's the whole point of "offline").

    Written generically rather than hardcoded to one model — any ONNX
    export that takes `input_ids`/`attention_mask` (optionally
    `token_type_ids`) and returns per-token hidden states works here — but
    it's sized and documented around the model it's meant for:

        jinaai/jina-embeddings-v2-base-code, INT8-quantized ONNX export
        (~160MB model + ~1.2MB tokenizer.json). Code-specialized, up to
        8192 tokens of context, no task-prefix required on the input text.

    Needs two packages nothing else in this file requires:
    `onnxruntime` and `tokenizers` (numpy comes along as a transitive
    dependency of both). Measured together, that's ~50MB of packages on
    top of the model file — budget for the model *and* the runtime, not
    just the ~160MB quoted for the model alone. Both are imported lazily
    here, inside `__init__`, so `NullEmbeddingProvider` and
    `RemoteEmbeddingProvider` keep working with neither installed — this
    is deliberately the opposite of eagerly importing a heavy optional
    dependency at module load, which is what breaks every caller of a
    module the moment that dependency isn't installed.

    I haven't been able to run this against the actual jina ONNX file to
    confirm its exact input/output names — I don't have network access to
    huggingface.co from where I'm building this. So instead of hardcoding
    input names, `__init__` reads them off the loaded ONNX session itself
    (`session.get_inputs()`) and only feeds the ones the graph actually
    declares. Test this against the real model file before relying on it;
    if the model's output shape isn't `(batch, sequence, hidden)` for
    output index 0, `_mean_pool` will need adjusting.
    """

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        max_length: int = 8192,
        normalize: bool = True,
    ) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise EmbeddingProviderError(
                "LocalOnnxEmbeddingProvider needs the 'onnxruntime' and "
                "'tokenizers' packages installed (pip install onnxruntime "
                "tokenizers)."
            ) from exc

        self._np = np
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        if not self.model_path.exists():
            raise EmbeddingProviderError(f"ONNX model not found at {self.model_path}")
        if not self.tokenizer_path.exists():
            raise EmbeddingProviderError(
                f"Tokenizer not found at {self.tokenizer_path}"
            )

        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        pad_id = self.tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            pad_id = 0
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")

        self.max_length = max_length
        self.normalize = normalize

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        np = self._np
        try:
            encodings = self.tokenizer.encode_batch(texts)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            )

            feed = {}
            if "input_ids" in self._input_names:
                feed["input_ids"] = input_ids
            if "attention_mask" in self._input_names:
                feed["attention_mask"] = attention_mask
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.zeros_like(input_ids)

            hidden_states = self.session.run(None, feed)[0]
            pooled = self._mean_pool(hidden_states, attention_mask)
            if self.normalize:
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                pooled = pooled / norms
            return pooled.tolist()
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError(f"Local ONNX embedding failed: {exc}") from exc

    def _mean_pool(self, hidden_states, attention_mask):
        """Standard mean-pooling over token embeddings, masked so padding
        tokens don't drag the average down — the pooling strategy
        jina-embeddings-v2 (and most sentence-embedding BERT exports) is
        trained to be read with."""
        np = self._np
        mask = attention_mask[..., None].astype(hidden_states.dtype)
        summed = (hidden_states * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        return summed / counts


class FallbackEmbeddingProvider:
    """Tries each provider in order; the first one that succeeds wins.
    Mirrors CodeVeto's multi-provider LLM routing, applied to embeddings:
    one provider running out of quota shouldn't take semantic search down
    with it. If every provider fails, the failure propagates as
    `EmbeddingProviderError` and the caller (`search/hybrid.py`) falls
    back to lexical-only search."""

    def __init__(self, providers: list[EmbeddingProvider]) -> None:
        if not providers:
            raise ValueError("FallbackEmbeddingProvider needs at least one provider.")
        self.providers = providers

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._try_each(lambda p: p.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._try_each(lambda p: p.embed_query(text))

    def _try_each(self, call):
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                return call(provider)
            except EmbeddingProviderError as exc:
                logger.warning(
                    "Embedding provider %s failed: %s", type(provider).__name__, exc
                )
                last_error = exc
        raise EmbeddingProviderError(
            f"All {len(self.providers)} embedding provider(s) failed."
        ) from last_error
