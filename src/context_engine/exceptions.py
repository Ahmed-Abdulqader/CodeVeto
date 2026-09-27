"""Custom exceptions for the context engine.

`ContextEngine` and `tools.context_tools.ContextTools` catch these (and
only these, plus `EmbeddingProviderError` where relevant) at the boundary
and turn them into `{"ok": False, "error": ...}` results rather than
letting them propagate — a tool call failing shouldn't crash whatever
agent loop is calling it.
"""

from __future__ import annotations


class ContextEngineError(Exception):
    """Base class for every error this package raises on purpose."""


class UnsupportedLanguageError(ContextEngineError):
    """Raised when a file's extension isn't in `chunker.EXTENSION_MAP`."""


class IndexingError(ContextEngineError):
    """Raised when a file can't be parsed/indexed (e.g. a syntax error bad
    enough that tree-sitter can't recover, or an unreadable file)."""


class StorageError(ContextEngineError):
    """Raised for storage-layer failures that aren't plain SQLite errors
    (e.g. a chunk_id referenced by a tool call that no longer exists)."""


class SessionNotFoundError(ContextEngineError):
    """Raised when a tool call references a session_id that was never
    created with `ContextEngine.new_session()`."""
