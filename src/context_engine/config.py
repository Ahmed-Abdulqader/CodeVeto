"""
config.py

One place to configure a `ContextEngine` instance: where its SQLite file
and logs live, what to skip while indexing, how big a memory brief can
get, and whether to layer semantic search on top of the default lexical
search.

`EmbeddingSettings` defaults to `"local_onnx"` -- meaning "use the model
under `~/.codeveto/models/` if it's there." It's `main.py`'s job to make
sure it's there (see `model_downloader.ensure_model_installed()`) before
constructing a `ContextEngine`; if it isn't there yet, `build()` below
doesn't raise or block startup, it just returns `None` and the engine runs
on lexical search alone. That fallback is what makes `EngineConfig()`
with no arguments a fully working configuration regardless of whether the
model has been downloaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from context_engine.ignore import DEFAULT_IGNORE_PATTERNS
from context_engine.logging_config import get_logger
from context_engine.search.embedding import (
    EmbeddingProvider,
    EmbeddingProviderError,
    LocalOnnxEmbeddingProvider,
    NullEmbeddingProvider,
)

logger = get_logger("context_engine.config")


class EmbeddingSettings(BaseModel):
    # "local_onnx" (default): use the model at ~/.codeveto/models/ if
    # main.py has already downloaded it; silently fall back to
    # lexical-only if it hasn't. "none": never even try -- for anyone who
    # wants to guarantee onnxruntime is never touched.
    provider: Literal["local_onnx", "none"] = "local_onnx"
    max_length: int = 8192
    normalize: bool = True

    def build(self) -> EmbeddingProvider | None:
        if self.provider == "none":
            return None
        try:
            return LocalOnnxEmbeddingProvider.from_global_install(
                max_length=self.max_length, normalize=self.normalize
            )
        except EmbeddingProviderError as exc:
            logger.info(
                "Local embedding model not available yet, running on "
                "lexical search alone: %s",
                exc,
            )
            return None


class EngineConfig(BaseModel):
    project_root: Path
    db_path: Path | None = None
    log_dir: Path | None = None
    ignore_patterns: tuple[str, ...] = DEFAULT_IGNORE_PATTERNS
    max_file_size_bytes: int = 1_000_000
    brief_budget_chars: int = 2000
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)

    def model_post_init(self, __context) -> None:
        if self.db_path is None:
            self.db_path = self.project_root / ".codeveto" / "context.db"
        if self.log_dir is None:
            self.log_dir = self.project_root / ".codeveto" / "logs"

    def build_embedding_provider(self) -> EmbeddingProvider:
        return self.embedding.build() or NullEmbeddingProvider()
