"""
config.py

One place to configure a `ContextEngine` instance: where its SQLite file
and logs live, what to skip while indexing, how big a memory brief can get,
and — optionally — which embedding provider(s) to layer on top of the
default lexical search. Nothing here requires an embedding provider; the
`embedding.provider == "none"` default is a fully working configuration.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from context_engine.ignore import DEFAULT_IGNORE_PATTERNS
from context_engine.search.embedding import (
    EmbeddingProvider,
    FallbackEmbeddingProvider,
    NullEmbeddingProvider,
    RemoteEmbeddingProvider,
)


class RemoteProviderSettings(BaseModel):
    """One entry in `EmbeddingSettings.providers`. `api_key_env` names an
    environment variable to read at call time — never put a raw key in
    this config or in a checked-in file."""

    base_url: str
    model: str
    api_key_env: str
    dimension: int | None = None
    timeout_seconds: float = 20.0


class EmbeddingSettings(BaseModel):
    # "none" (default): lexical-only search, zero downloads, zero API calls.
    # "remote_api": embed via one or more OpenAI-compatible HTTP endpoints,
    # tried in order (see FallbackEmbeddingProvider) with automatic
    # fallback to lexical-only if every provider fails.
    provider: str = "none"
    providers: list[RemoteProviderSettings] = Field(default_factory=list)

    def build(self) -> EmbeddingProvider | None:
        if self.provider == "none" or not self.providers:
            return None
        if len(self.providers) == 1:
            return _build_one(self.providers[0])
        return FallbackEmbeddingProvider([_build_one(p) for p in self.providers])


def _build_one(settings: RemoteProviderSettings) -> RemoteEmbeddingProvider:
    return RemoteEmbeddingProvider(
        base_url=settings.base_url,
        model=settings.model,
        api_key_env=settings.api_key_env,
        dimension=settings.dimension,
        timeout_seconds=settings.timeout_seconds,
    )


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
