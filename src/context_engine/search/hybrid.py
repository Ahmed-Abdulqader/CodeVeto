"""
hybrid.py

`ContextSearch.search()` is what `tools.context_tools.search_context`
actually calls. It always runs the BM25 lexical search; if an embedding
provider is configured, it also embeds the query and searches stored
vectors, then merges the two rankings with Reciprocal Rank Fusion (RRF) —
a simple, well-known way to combine rankings without needing the two
scores to be on the same scale.

If no embedding provider is configured, or the provider call fails for any
reason (network, quota, bad response), this silently degrades to
lexical-only results and logs the failure — it never raises, and it never
returns an empty result just because semantic search wasn't available.
"""

from __future__ import annotations

from context_engine.logging_config import get_logger
from context_engine.search.embedding import EmbeddingProvider, EmbeddingProviderError
from context_engine.search.lexical import LexicalIndex, hit_from_chunk
from context_engine.storage.chunk_store import ChunkStore
from context_engine.storage.vector_store import VectorStore
from models.context_engine_models import SearchHit

logger = get_logger("context_engine.search")

_RRF_K = 60  # standard constant from the original RRF paper; rarely tuned


class ContextSearch:
    def __init__(
        self,
        chunk_store: ChunkStore,
        lexical_index: LexicalIndex,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.chunk_store = chunk_store
        self.lexical_index = lexical_index
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        widened = max(top_k * 3, top_k)
        lexical_ranked = self.lexical_index.search(query, top_k=widened)
        lexical_ids = [chunk_id for chunk_id, _ in lexical_ranked]

        vector_ids = self._vector_search(query, widened)

        if vector_ids:
            fused_ids = _reciprocal_rank_fusion([lexical_ids, vector_ids])
            matched_by = {
                chunk_id: [
                    label
                    for label, ids in (("lexical", lexical_ids), ("vector", vector_ids))
                    if chunk_id in ids
                ]
                for chunk_id in fused_ids
            }
            ranked_ids = fused_ids
        else:
            ranked_ids = lexical_ids
            matched_by = {chunk_id: ["lexical"] for chunk_id in lexical_ids}

        hits: list[SearchHit] = []
        for rank, chunk_id in enumerate(ranked_ids[:top_k]):
            chunk = self.chunk_store.get_chunk(chunk_id)
            if chunk is None:
                continue  # stale index entry; shouldn't happen, cheap to skip
            score = 1.0 / (rank + 1)
            hits.append(
                hit_from_chunk(
                    chunk, score=score, matched_by=matched_by.get(chunk_id, [])
                )
            )
        return hits

    def _vector_search(self, query: str, top_k: int) -> list[str]:
        if self.embedding_provider is None or self.vector_store.is_empty():
            return []
        try:
            query_vector = self.embedding_provider.embed_query(query)
        except EmbeddingProviderError as exc:
            logger.warning(
                "Vector search skipped, falling back to lexical only: %s", exc
            )
            return []
        return [
            chunk_id
            for chunk_id, _ in self.vector_store.search_similar(
                query_vector, top_k=top_k
            )
        ]


def _reciprocal_rank_fusion(rankings: list[list[str]]) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return [
        chunk_id
        for chunk_id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]
