"""
embedding.py

The context engine's semantic-search layer, fully offline by design.

`search_context` works completely on `search/lexical.py`'s BM25 index
with `NullEmbeddingProvider` (the default, no model installed at all).
When semantic search is wanted, `LocalOnnxEmbeddingProvider` runs a
quantized ONNX model (jina-embeddings-v2-base-code's INT8 export, ~162MB)
on-device via `onnxruntime` -- no network call at index time or query
time, ever. There is deliberately no remote/API-based provider here
anymore: this package used to also support calling out to a remote
embeddings endpoint (OpenAI/OpenRouter/etc.), but the project's direction
is "runs fully offline," and code leaving the machine to be embedded
contradicts that regardless of how convenient it is. If that ever changes,
a remote provider is a small, separate class behind the same
`EmbeddingProvider` protocol -- it doesn't belong mixed in with the
offline path.

`onnxruntime` and `tokenizers` are imported lazily inside
`LocalOnnxEmbeddingProvider.__init__` rather than at module load, so
importing this file -- and using `NullEmbeddingProvider` / plain BM25 --
still needs neither of them installed, even though both are now listed as
real dependencies in `pyproject.toml` (see the module docstring's sibling,
`model_downloader.py`, for where the actual model file comes from).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from context_engine.logging_config import get_logger

logger = get_logger("context_engine.embedding")


class EmbeddingProviderError(RuntimeError):
    """Raised by a provider when it can't produce an embedding. Callers
    (`search/hybrid.py`) catch this and fall back to lexical-only search."""


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class NullEmbeddingProvider:
    """The default. Signals "semantic search is off" without every caller
    needing an `if embedding_provider is not None` check -- call it, get
    an `EmbeddingProviderError`, and the caller's existing fallback path
    (skip to lexical-only) handles it the same as a real provider
    failing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingProviderError("No embedding provider configured.")

    def embed_query(self, text: str) -> list[float]:
        raise EmbeddingProviderError("No embedding provider configured.")


class LocalOnnxEmbeddingProvider:
    """Runs a local ONNX embedding model fully offline -- no network call
    at index time or query time, ever. This class never downloads
    anything itself (see `model_downloader.py` for that); point
    `model_path`/`tokenizer_path` at files already on disk, or use
    `from_global_install()` to point at the standard `~/.codeveto/models/`
    location.

    Written generically rather than hardcoded to one model -- any ONNX
    export that takes `input_ids`/`attention_mask` (optionally
    `token_type_ids`) and returns per-token hidden states works here --
    but it's sized and documented around the model it's meant for:

        jinaai/jina-embeddings-v2-base-code, INT8-quantized ONNX export
        (~162MB model + a few MB for tokenizer.json). Code-specialized,
        up to 8192 tokens of context, no task-prefix required on input
        text. Confirmed from the model's own card: mean pooling is the
        pooling strategy it's meant to be read with (matches
        `_mean_pool` below), hidden size 768, pad_token_id 0.

    Needs two packages nothing else in this file requires: `onnxruntime`
    and `tokenizers` (numpy comes along as a transitive dependency of
    both). Measured together, that's ~50MB of packages on top of the
    model file -- budget for the model *and* the runtime, not just the
    ~162MB quoted for the model alone. Both are imported lazily here,
    inside `__init__`, rather than at module load -- the opposite of
    eagerly importing a heavy dependency at module load, which is what
    breaks every caller of a module the instant that dependency isn't
    installed.

    I haven't been able to run this against the actual jina ONNX file
    myself to confirm its exact input/output names -- I don't have
    network access to huggingface.co from where this is built. So instead
    of hardcoding input names, `__init__` reads them off the loaded ONNX
    session itself (`session.get_inputs()`) and only feeds the ones the
    graph actually declares -- tested against synthetic ONNX graphs both
    with and without a `token_type_ids` input, both work. Please run this
    against the real model file before relying on it; if the model's
    output shape isn't `(batch, sequence, hidden)` at output index 0,
    `_mean_pool` will need adjusting.
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

    @classmethod
    def from_global_install(
        cls, max_length: int = 8192, normalize: bool = True
    ) -> LocalOnnxEmbeddingProvider:
        """Points at the model `model_downloader.ensure_model_installed()`
        downloads into `~/.codeveto/models/` -- the path every project
        should use unless deliberately testing against a different local
        model. Raises `EmbeddingProviderError` (not a raw
        FileNotFoundError) if the model hasn't been installed yet, so
        callers can catch one exception type either way."""
        from context_engine.paths import jina_code_model_path, jina_code_tokenizer_path

        model_path = jina_code_model_path()
        tokenizer_path = jina_code_tokenizer_path()
        if not model_path.exists() or not tokenizer_path.exists():
            raise EmbeddingProviderError(
                f"Embedding model not found under {model_path.parent}. "
                "Run model_downloader.ensure_model_installed() first."
            )
        return cls(
            model_path, tokenizer_path, max_length=max_length, normalize=normalize
        )

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
        """Masked mean-pooling over token embeddings -- padding tokens
        don't drag the average down. This matches jina-embeddings-v2's own
        published reference pooling code (mask-weighted sum divided by a
        clamped mask sum), not just a generic sentence-embedding
        convention."""
        np = self._np
        mask = attention_mask[..., None].astype(hidden_states.dtype)
        summed = (hidden_states * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        return summed / counts
