"""
chunk_store.py

Persists `FileMetadata` and `Chunk` rows. Deliberately knows nothing about
search ranking or the symbol graph — `search/lexical.py` and
`storage/graph_store.py` are updated separately by the indexer, so each
store stays small and independently testable.
"""

from __future__ import annotations

from context_engine.storage.database import Database
from models.context_engine_models import Chunk, FileMetadata


class ChunkStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    # -- files ---------------------------------------------------------

    def upsert_file(
        self, file_metadata: FileMetadata, language: str, content_hash: str
    ) -> None:
        # `language` is passed separately rather than read off `file_metadata`
        # because `FileMetadata` (chunker.py's contract: name/path/modified_ts
        # only) doesn't carry it — chunks do. The indexer knows the language
        # from `Chunks.language` even for a file that produced zero chunks.
        self.db.execute(
            """
            INSERT INTO files (path, language, modified_ts, content_hash)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                language = excluded.language,
                modified_ts = excluded.modified_ts,
                content_hash = excluded.content_hash
            """,
            (file_metadata.path, language, file_metadata.modified_ts, content_hash),
        )

    def get_file_hash(self, path: str) -> str | None:
        row = self.db.query_one(
            "SELECT content_hash FROM files WHERE path = ?", (path,)
        )
        return row["content_hash"] if row else None

    def delete_file(self, path: str) -> None:
        # ON DELETE CASCADE removes the file's chunks, postings, doc stats,
        # and embeddings in one go.
        self.db.execute("DELETE FROM files WHERE path = ?", (path,))

    def list_files(self, path_prefix: str | None = None) -> list[dict]:
        if path_prefix:
            rows = self.db.query(
                "SELECT path, language, modified_ts FROM files "
                "WHERE path LIKE ? ORDER BY path",
                (f"{path_prefix}%",),
            )
        else:
            rows = self.db.query(
                "SELECT path, language, modified_ts FROM files ORDER BY path"
            )
        return [dict(r) for r in rows]

    # -- chunks ----------------------------------------------------------

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.db.executemany(
            """
            INSERT OR REPLACE INTO chunks
                (id, file_path, language, chunk_type, signature, docstring,
                 content, start_line, end_line)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c.id,
                    c.file_metadata.path,
                    c.language,
                    c.chunk_type,
                    c.signature,
                    c.docstring,
                    c.content,
                    c.start_line,
                    c.end_line,
                )
                for c in chunks
            ],
        )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self.db.query_one(
            """
            SELECT c.*, f.modified_ts, f.path AS file_path_full
            FROM chunks c JOIN files f ON f.path = c.file_path
            WHERE c.id = ?
            """,
            (chunk_id,),
        )
        return _row_to_chunk(row) if row else None

    def get_chunks_by_file(self, file_path: str) -> list[Chunk]:
        rows = self.db.query(
            """
            SELECT c.*, f.modified_ts, f.path AS file_path_full
            FROM chunks c JOIN files f ON f.path = c.file_path
            WHERE c.file_path = ?
            ORDER BY c.start_line
            """,
            (file_path,),
        )
        return [_row_to_chunk(r) for r in rows]

    def chunk_count(self) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM chunks")
        return row["n"] if row else 0


def _row_to_chunk(row) -> Chunk:
    return Chunk(
        id=row["id"],
        language=row["language"],
        chunk_type=row["chunk_type"],
        docstring=row["docstring"],
        signature=row["signature"],
        content=row["content"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        file_metadata=FileMetadata(
            name=row["file_path"].rsplit("/", 1)[-1],
            path=row["file_path"],
            modified_ts=row["modified_ts"],
        ),
    )
