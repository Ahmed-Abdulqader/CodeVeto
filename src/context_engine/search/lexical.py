"""
lexical.py

BM25 ranking over the chunks table, implemented in plain Python against
the `bm25_postings` / `bm25_doc_stats` tables. No embedding model, no
model download, no extra dependency beyond the stdlib — this is what
`search_context` uses by default, and what it falls back to if an
embedding provider isn't configured or fails (see `search/hybrid.py`).

This exists because the original plan (Ollama + nomic-embed-text, ~2GB
between the Ollama runtime and the model) doesn't fit a mobile-data
budget. BM25 needs zero bytes downloaded and is a well-established
baseline for code search — semantic embeddings can be layered on top
later via `search/embedding.py` without touching this module.
"""

from __future__ import annotations

import math
from collections import Counter

from context_engine.search.tokenizer import tokenize
from context_engine.storage.database import Database
from models.context_engine_models import SearchHit

# Standard BM25 constants (Robertson/Sparck Jones defaults). Not exposed as
# config for now — they rarely need tuning before there's a real relevance
# problem to tune them against.
_K1 = 1.5
_B = 0.75


class LexicalIndex:
    def __init__(self, db: Database) -> None:
        self.db = db

    def index_chunk(self, chunk_id: str, text: str) -> None:
        """(Re)index one chunk's searchable text. Call `remove_chunk` first
        if the chunk was previously indexed with different content."""
        terms = Counter(tokenize(text))
        if not terms:
            self.db.execute(
                "INSERT OR REPLACE INTO bm25_doc_stats (chunk_id, doc_length) "
                "VALUES (?, ?)",
                (chunk_id, 0),
            )
            return
        self.db.executemany(
            "INSERT OR REPLACE INTO bm25_postings (term, chunk_id, term_freq) "
            "VALUES (?, ?, ?)",
            [(term, chunk_id, freq) for term, freq in terms.items()],
        )
        self.db.execute(
            "INSERT OR REPLACE INTO bm25_doc_stats (chunk_id, doc_length) "
            "VALUES (?, ?)",
            (chunk_id, sum(terms.values())),
        )

    def remove_chunk(self, chunk_id: str) -> None:
        self.db.execute("DELETE FROM bm25_postings WHERE chunk_id = ?", (chunk_id,))
        self.db.execute("DELETE FROM bm25_doc_stats WHERE chunk_id = ?", (chunk_id,))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return `[(chunk_id, bm25_score), ...]`, highest score first."""
        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        stats_row = self.db.query_one(
            "SELECT COUNT(*) AS n, AVG(doc_length) AS avg_len FROM bm25_doc_stats"
        )
        total_docs = stats_row["n"] if stats_row else 0
        avg_doc_len = (stats_row["avg_len"] or 0.0) if stats_row else 0.0
        if total_docs == 0 or not avg_doc_len:
            return []

        scores: dict[str, float] = {}
        doc_lengths: dict[str, int] = {}

        for term in query_terms:
            postings = self.db.query(
                """
                SELECT p.chunk_id, p.term_freq, s.doc_length
                FROM bm25_postings p JOIN bm25_doc_stats s ON s.chunk_id = p.chunk_id
                WHERE p.term = ?
                """,
                (term,),
            )
            if not postings:
                continue
            doc_freq = len(postings)
            idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))

            for row in postings:
                chunk_id = row["chunk_id"]
                tf = row["term_freq"]
                doc_len = row["doc_length"]
                doc_lengths[chunk_id] = doc_len
                denom = tf + _K1 * (1 - _B + _B * doc_len / avg_doc_len)
                term_score = idf * (tf * (_K1 + 1)) / denom if denom else 0.0
                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]


def build_preview(
    signature: str | None, docstring: str | None, content: str, max_chars: int = 240
) -> str:
    """Compact, token-saving preview for a `SearchHit` — a signature and a
    docstring first line beat dumping the whole chunk body into a search
    result the caller has to pay to read before deciding it's not the one
    they need. Call `expand_chunk` for the full body."""
    parts = []
    if signature:
        parts.append(signature.strip())
    if docstring:
        parts.append(docstring.strip().splitlines()[0])
    preview = (
        " — ".join(parts)
        if parts
        else content.strip().splitlines()[0]
        if content.strip()
        else ""
    )
    return preview[:max_chars]


def hit_from_chunk(chunk, score: float, matched_by: list[str]) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.id,
        file_path=chunk.file_metadata.path,
        chunk_type=chunk.chunk_type,
        signature=chunk.signature,
        preview=build_preview(chunk.signature, chunk.docstring, chunk.content),
        score=score,
        matched_by=matched_by,
    )
