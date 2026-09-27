from __future__ import annotations

import os

from context_engine.indexer import RepoIndexer
from context_engine.search.embedding import NullEmbeddingProvider
from context_engine.search.lexical import LexicalIndex
from context_engine.storage.chunk_store import ChunkStore
from context_engine.storage.graph_store import GraphStore
from context_engine.storage.vector_store import VectorStore


def _indexer(db, project_root):
    return RepoIndexer(
        project_root=project_root,
        chunk_store=ChunkStore(db),
        graph_store=GraphStore(db),
        lexical_index=LexicalIndex(db),
        vector_store=VectorStore(db),
        embedding_provider=NullEmbeddingProvider(),
        ignore_patterns=("node_modules", ".git"),
        max_file_size_bytes=1_000_000,
    )


class TestIndexRepository:
    def test_indexes_supported_files_and_skips_ignored_dirs(self, db, sample_repo):
        indexer = _indexer(db, sample_repo)
        stats = indexer.index_repository(force=True)

        assert stats["indexed"] == 3  # greeter.py, greeter.js, greeter.rs
        assert stats["failed"] == 0

        files = indexer.chunk_store.list_files()
        paths = {os.path.basename(f["path"]) for f in files}
        assert paths == {"greeter.py", "greeter.js", "greeter.rs"}
        assert "ignored.js" not in paths

    def test_unchanged_files_are_skipped_on_second_pass(self, db, sample_repo):
        indexer = _indexer(db, sample_repo)
        indexer.index_repository(force=True)

        stats = indexer.index_repository(force=False)
        assert stats["indexed"] == 0
        assert stats["unchanged"] == 3

    def test_modified_file_is_reindexed(self, db, sample_repo):
        indexer = _indexer(db, sample_repo)
        indexer.index_repository(force=True)

        py_file = sample_repo / "greeter.py"
        py_file.write_text(py_file.read_text() + "\n\ndef extra():\n    return 1\n")

        stats = indexer.index_repository(force=False)
        assert stats["indexed"] == 1
        assert stats["unchanged"] == 2

        chunks = indexer.chunk_store.get_chunks_by_file(str(py_file.resolve()))
        assert any(c.signature == "def extra()" for c in chunks)


class TestIndexFile:
    def test_index_file_populates_chunks_and_graph(self, db, sample_repo):
        indexer = _indexer(db, sample_repo)
        py_file = sample_repo / "greeter.py"

        count = indexer.index_file(str(py_file))
        assert count > 0

        abspath = str(py_file.resolve())
        assert indexer.chunk_store.get_chunks_by_file(abspath)
        assert indexer.graph_store.nodes_for_file(abspath)

    def test_reindexing_same_file_does_not_duplicate_rows(self, db, sample_repo):
        indexer = _indexer(db, sample_repo)
        py_file = str(sample_repo / "greeter.py")

        indexer.index_file(py_file)
        first_count = len(
            indexer.chunk_store.get_chunks_by_file(
                str((sample_repo / "greeter.py").resolve())
            )
        )
        indexer.index_file(py_file)
        second_count = len(
            indexer.chunk_store.get_chunks_by_file(
                str((sample_repo / "greeter.py").resolve())
            )
        )

        assert first_count == second_count

    def test_removing_file_clears_its_rows(self, db, sample_repo):
        indexer = _indexer(db, sample_repo)
        py_file = sample_repo / "greeter.py"
        indexer.index_file(str(py_file))
        abspath = str(py_file.resolve())
        assert indexer.chunk_store.get_chunks_by_file(abspath)

        py_file.unlink()
        indexer.index_file(
            str(py_file)
        )  # file no longer exists -> should clear, not error

        assert indexer.chunk_store.get_chunks_by_file(abspath) == []
        assert indexer.graph_store.nodes_for_file(abspath) == []

    def test_file_is_searchable_after_indexing(self, db, sample_repo):
        indexer = _indexer(db, sample_repo)
        indexer.index_file(str(sample_repo / "greeter.py"))

        results = indexer.lexical_index.search("say hello")
        assert results
