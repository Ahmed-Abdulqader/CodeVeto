"""
indexer.py

Nothing in `main`/`embedder`/`feature/graph-builder` currently connects
`chunker.py`, `graph_builder.py`, and storage to each other — each is a
standalone module you can run directly but nothing calls in sequence. This
is that missing wiring: walk the repo, parse each supported file once, and
write its chunks, graph nodes/edges, and lexical postings from that single
parse.

Re-indexing a file is a full delete-then-reinsert of that file's rows
(chunks, postings, graph nodes/edges, embeddings all cascade or are cleared
explicitly) rather than a diff — simpler to reason about, and cheap enough
at the scale of one file, which is the unit `index_file` is meant to be
called at (e.g. right after the developer lets the agent write to one
file).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from context_engine.chunker import EXTENSION_MAP, Chunks
from context_engine.exceptions import IndexingError
from context_engine.graph.graph_builder import GraphBuilder
from context_engine.ignore import should_ignore
from context_engine.logging_config import get_logger
from context_engine.search.embedding import EmbeddingProvider, EmbeddingProviderError
from context_engine.search.lexical import LexicalIndex
from context_engine.storage.chunk_store import ChunkStore
from context_engine.storage.graph_store import GraphStore
from context_engine.storage.vector_store import VectorStore
from models.context_engine_models import FileMetadata

logger = get_logger("context_engine.indexer")


class RepoIndexer:
    def __init__(
        self,
        project_root: Path,
        chunk_store: ChunkStore,
        graph_store: GraphStore,
        lexical_index: LexicalIndex,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        ignore_patterns: tuple[str, ...],
        max_file_size_bytes: int,
    ) -> None:
        self.project_root = project_root
        self.chunk_store = chunk_store
        self.graph_store = graph_store
        self.lexical_index = lexical_index
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.ignore_patterns = ignore_patterns
        self.max_file_size_bytes = max_file_size_bytes

    # -- whole-repo indexing ---------------------------------------------

    def index_repository(self, force: bool = False) -> dict[str, int]:
        indexed, skipped, failed = 0, 0, 0
        for path in self._walk():
            try:
                if force or self._file_changed(path):
                    self.index_file(path)
                    indexed += 1
                else:
                    skipped += 1
            except IndexingError as exc:
                logger.warning("Skipping %s: %s", path, exc)
                failed += 1
        return {"indexed": indexed, "unchanged": skipped, "failed": failed}

    def _walk(self):
        for dirpath, dirnames, filenames in os.walk(self.project_root):
            dirpath_p = Path(dirpath)
            dirnames[:] = [
                d
                for d in dirnames
                if not should_ignore(
                    dirpath_p / d, self.project_root, self.ignore_patterns
                )
            ]
            for filename in filenames:
                path = dirpath_p / filename
                if should_ignore(path, self.project_root, self.ignore_patterns):
                    continue
                if path.suffix.lower() not in EXTENSION_MAP:
                    continue
                # Normalized here so every downstream lookup (file-hash
                # comparison, chunk_store keys, graph node ids) agrees with
                # chunker.py's own `FileMetadata.path`, which is always
                # `os.path.abspath(...)` regardless of how the path was
                # first reached.
                yield Path(os.path.abspath(path))

    def _file_changed(self, path: Path) -> bool:
        stored_hash = self.chunk_store.get_file_hash(str(path))
        if stored_hash is None:
            return True
        return stored_hash != _hash_file(path)

    # -- single-file indexing ---------------------------------------------

    def index_file(self, path: str | Path) -> int:
        """(Re)index one file. Returns the number of chunks produced.
        Safe to call repeatedly on the same file — old rows for that file
        are cleared first."""
        path_str = os.path.abspath(str(path))
        path = Path(path_str)
        if not path.exists():
            self.remove_file(path_str)
            return 0

        if path.stat().st_size > self.max_file_size_bytes:
            raise IndexingError(f"{path} exceeds max_file_size_bytes, skipped.")

        try:
            parsed = Chunks(path_str)
        except (ValueError, OSError) as exc:
            raise IndexingError(f"Could not parse {path}: {exc}") from exc

        self.remove_file(path_str)  # clear any previous version of this file's rows

        file_metadata = FileMetadata(
            name=path.name, path=path_str, modified_ts=os.path.getmtime(path_str)
        )
        self.chunk_store.upsert_file(file_metadata, parsed.language, _hash_file(path))
        self.chunk_store.insert_chunks(list(parsed.chunks))

        graph = GraphBuilder().extract_graph(
            path_str, parsed.tree.root_node, parsed.source_bytes, parsed.language
        )
        self.graph_store.save_graph(graph)

        for chunk in parsed.chunks:
            self.lexical_index.index_chunk(chunk.id, _searchable_text(chunk))

        self._embed_chunks(list(parsed.chunks))
        return len(parsed.chunks)

    def remove_file(self, path: str | Path) -> None:
        path_str = str(path)
        for chunk in self.chunk_store.get_chunks_by_file(path_str):
            self.lexical_index.remove_chunk(chunk.id)
            self.vector_store.remove(chunk.id)
        self.graph_store.delete_file_nodes(path_str)
        self.chunk_store.delete_file(path_str)  # cascades remaining chunk rows

    def _embed_chunks(self, chunks: list) -> None:
        if not chunks:
            return
        texts = [_searchable_text(c) for c in chunks]
        try:
            vectors = self.embedding_provider.embed_documents(texts)
        except EmbeddingProviderError as exc:
            # Best-effort: indexing still succeeds on lexical search alone.
            logger.info(
                "Embedding skipped for this batch (%d chunks): %s", len(chunks), exc
            )
            return
        provider_name = type(self.embedding_provider).__name__
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.vector_store.upsert(chunk.id, provider_name, "unspecified", vector)


def _searchable_text(chunk) -> str:
    parts = [chunk.signature or "", chunk.docstring or "", chunk.content]
    return "\n".join(p for p in parts if p)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
