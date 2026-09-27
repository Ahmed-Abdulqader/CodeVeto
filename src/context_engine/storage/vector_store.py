"""
vector_store.py

Persists dense vectors for chunks that were embedded via an optional
`search.embedding.EmbeddingProvider`, and answers nearest-neighbour queries
with plain-Python cosine similarity. This table stays empty for anyone not
using an embedding provider — `search.lexical.LexicalIndex` alone is a
complete search backend.

A hand-rolled scan is fine at the scale this is built for (a single
developer's project, thousands of chunks, not millions) and keeps this
dependency-free; if that ever stops being true, this is the file to swap
for `sqlite-vec` or a real vector database, behind the same
`search_similar` signature.
"""

from __future__ import annotations

import json
import math

from context_engine.storage.database import Database


class VectorStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(
        self, chunk_id: str, provider: str, model: str, vector: list[float]
    ) -> None:
        self.db.execute(
            """
            INSERT OR REPLACE INTO embeddings
                (chunk_id, provider, model, dimension, vector)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chunk_id, provider, model, len(vector), json.dumps(vector)),
        )

    def remove(self, chunk_id: str) -> None:
        self.db.execute("DELETE FROM embeddings WHERE chunk_id = ?", (chunk_id,))

    def is_empty(self) -> bool:
        row = self.db.query_one("SELECT 1 FROM embeddings LIMIT 1")
        return row is None

    def search_similar(
        self, query_vector: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        rows = self.db.query("SELECT chunk_id, vector FROM embeddings")
        scored = []
        for row in rows:
            vector = json.loads(row["vector"])
            if len(vector) != len(query_vector):
                continue
            scored.append((row["chunk_id"], _cosine_similarity(query_vector, vector)))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
