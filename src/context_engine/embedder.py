"""
embedder.py

Embedding component for the Code Context Engine.

Converts parsed `Chunk` objects (see `src/models/context_engine_models.py`) into
dense vector embeddings using a local Ollama instance running the
`nomic-embed-text` model, so that code snippets can be stored in a vector
database and retrieved via semantic similarity search.

Usage:
    from src.context_engine.embedder import OllamaEmbedder

    embedder = OllamaEmbedder()  # verifies Ollama + pulls the model if needed
    embedded_chunks = embedder.embed_batch(chunks)
    query_vector = embedder.embed_query("how does auth middleware work?")
"""

from __future__ import annotations

import logging
import time

import requests

from models.context_engine_models import Chunk, EmbeddedChunk

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL_NAME = "nomic-embed-text"
EXPECTED_EMBEDDING_DIM = 768

# nomic-embed-text requires task-specific prefixes on the raw text so the
# model can distinguish between indexing content and querying it.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class OllamaConnectionError(RuntimeError):
    """Raised when the local Ollama service cannot be reached."""


class OllamaModelError(RuntimeError):
    """Raised when the embedding model cannot be found or pulled."""


# --------------------------------------------------------------------------- #
# Embedder
# --------------------------------------------------------------------------- #

class OllamaEmbedder:
    """
    Generates embeddings for code `Chunk` objects (and free-text search
    queries) using a local Ollama instance running `nomic-embed-text`.

    On initialization, this class:
      1. Verifies that Ollama is reachable at `base_url`.
      2. Verifies that `model` is available locally, pulling it automatically
         if it is not.

    Both steps raise a descriptive `RuntimeError` (specifically
    `OllamaConnectionError` / `OllamaModelError`) if they fail, prompting the
    caller to start `ollama serve` and/or check their network connection.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL_NAME,
        auto_pull: bool = True,
        request_timeout: float = 30.0,
        pull_timeout: float = 600.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.request_timeout = request_timeout
        self.pull_timeout = pull_timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

        self._verify_connection()
        if not self._model_is_available():
            if auto_pull:
                logger.info("Model '%s' not found locally. Pulling...", self.model)
                self._pull_model()
            else:
                raise OllamaModelError(
                    f"Model '{self.model}' is not available locally and "
                    f"auto_pull=False. Run `ollama pull {self.model}` "
                    "and try again."
                )

    # ------------------------------------------------------------------ #
    # Connectivity / model management
    # ------------------------------------------------------------------ #

    def _verify_connection(self) -> None:
        """Confirm the Ollama service is reachable, raising a clear error if not."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", timeout=self.request_timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at '{self.base_url}'. "
                "Is it running? Start it with `ollama serve` and try again."
            ) from exc

    def _model_is_available(self) -> bool:
        """Check whether `self.model` is already pulled locally."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", timeout=self.request_timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at '{self.base_url}' while checking "
                "installed models. Start it with `ollama serve` and try again."
            ) from exc

        installed = response.json().get("models", [])
        installed_names = {m.get("name", "") for m in installed}
        # Ollama tag names may include a ":latest" (or other) suffix, so match
        # either the exact name or the bare model name before the colon.
        for name in installed_names:
            if name == self.model or name.split(":")[0] == self.model:
                return True
        return False

    def _pull_model(self) -> None:
        """Pull `self.model` from the Ollama registry."""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model, "stream": False},
                timeout=self.pull_timeout,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise OllamaModelError(
                f"Failed to pull model '{self.model}' from Ollama. "
                "Check your network connection and that Ollama is running "
                "(`ollama serve`), then try again."
            ) from exc

        if not self._model_is_available():
            raise OllamaModelError(
                f"Pull for model '{self.model}' completed but the model is "
                "still not listed as installed. Try running "
                f"`ollama pull {self.model}` manually."
            )
        logger.info("Model '%s' pulled successfully.", self.model)

    # ------------------------------------------------------------------ #
    # Text formatting
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_document_text(chunk: Chunk) -> str:
        """
        Format a `Chunk` for embedding as a document: prefix it with
        `search_document: ` plus file path / signature context so the model
        has structural information beyond the raw code content.
        """
        context_bits = [f"file: {chunk.file_metadata.path}"]
        if chunk.signature:
            context_bits.append(f"signature: {chunk.signature}")
        if chunk.docstring:
            context_bits.append(f"docstring: {chunk.docstring}")
        context_line = " | ".join(context_bits)

        return f"{DOCUMENT_PREFIX}{context_line}\n{chunk.content}"

    @staticmethod
    def _build_query_text(query: str) -> str:
        """Format a free-text search query with the `search_query: ` prefix."""
        return f"{QUERY_PREFIX}{query}"

    # ------------------------------------------------------------------ #
    # Embedding calls
    # ------------------------------------------------------------------ #

    def _embed_text(self, text: str) -> list[float]:
        """Call the Ollama embeddings endpoint for a single piece of text."""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding")
                if not embedding:
                    raise OllamaModelError(
                        f"Ollama returned no embedding for model '{self.model}'. "
                        f"Response: {data}"
                    )
                return embedding
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "Embedding request failed (attempt %d/%d): %s. Retrying...",
                        attempt,
                        self.max_retries,
                        exc,
                    )
                    time.sleep(self.retry_backoff_seconds * attempt)

        raise OllamaConnectionError(
            f"Failed to generate embedding via Ollama at '{self.base_url}' "
            f"after {self.max_retries} attempts. Ensure Ollama is running "
            "(`ollama serve`) and reachable, then try again."
        ) from last_exc

    def embed_batch(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """
        Embed a list of code `Chunk` objects.

        Each chunk's text is formatted with file path / signature context and
        the `search_document: ` prefix before being sent to Ollama. Returns a
        list of `EmbeddedChunk` objects pairing each original chunk with its
        resulting vector.
        """
        embedded_chunks: list[EmbeddedChunk] = []
        for chunk in chunks:
            text = self._build_document_text(chunk)
            embedding = self._embed_text(text)
            embedded_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding))
        return embedded_chunks

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a natural language search query for retrieval.

        The query is formatted with the `search_query: ` prefix before being
        sent to Ollama, so the resulting vector lives in the same embedding
        space as document chunks but is optimized for retrieval matching.
        """
        text = self._build_query_text(query)
        return self._embed_text(text)