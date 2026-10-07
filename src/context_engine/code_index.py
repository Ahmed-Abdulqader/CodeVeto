from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from models.embedding_models import (
    CodeItemIn,
    CodeItemOut,
    EntityType,
    SearchQuery,
    SearchResult,
)



_SCHEMA = """
CREATE TABLE IF NOT EXISTS code_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (entity_type, source_id)
);

CREATE INDEX IF NOT EXISTS idx_code_entities_entity_type
    ON code_entities (entity_type);

CREATE INDEX IF NOT EXISTS idx_code_entities_name
    ON code_entities (name);
"""


class OnnxTextEmbedder:
    """
    Embeds text using an ONNX model and a HuggingFace tokenizers tokenizer.

    Works without torch.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)

        self.max_length = max_length
        self.batch_size = batch_size

        input_names = [inp.name for inp in self.session.get_inputs()]

        self.input_ids_name = self._pick_input_name(
            input_names,
            ("input_ids", "input_ids:0", "input", "inputs"),
            required=True,
        )

        self.attention_mask_name = self._pick_input_name(
            input_names,
            ("attention_mask", "attention_mask:0", "attention"),
            required=False,
        )

        self.token_type_ids_name = self._pick_input_name(
            input_names,
            ("token_type_ids", "token_type_ids:0", "token_type"),
            required=False,
        )

        self.pad_token_id = self._infer_pad_id()

    @staticmethod
    def _pick_input_name(
        names: Sequence[str],
        keywords: Sequence[str],
        required: bool,
    ) -> Optional[str]:
        """
        Attempts to find the correct ONNX input name.

        Different ONNX exports can use slightly different input names.
        """
        lowered = [name.lower() for name in names]

        # Exact match first.
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for name, low in zip(names, lowered):
                if low == keyword_lower:
                    return name

        # Fuzzy match second.
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for name, low in zip(names, lowered):
                if keyword_lower in low:
                    return name

        if required:
            raise ValueError(
                f"Could not find required ONNX input. "
                f"Available inputs: {list(names)}"
            )

        return None

    def _infer_pad_id(self) -> int:
        """
        Try to infer the tokenizer pad token id.
        If not found, default to 0.
        """
        candidates = (
            "[PAD]",
            "<pad>",
            "pad",
            "<|pad|>",
        )

        for token in candidates:
            pad_id = self.tokenizer.token_to_id(token)
            if pad_id is not None:
                return pad_id

        return 0

    def embed(self, texts: Union[str, Sequence[str]]) -> np.ndarray:
        """
        Embeds one or more texts and returns normalized float32 embeddings.

        Output shape:
            (batch_size, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        texts = [text if isinstance(text, str) else str(text) for text in texts]

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_vectors: List[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]

            encodings = self.tokenizer.encode_batch(batch)

            max_len = max([len(enc.ids) for enc in encodings] + [1])

            input_ids = np.full(
                (len(batch), max_len),
                self.pad_token_id,
                dtype=np.int64,
            )

            attention_mask = np.zeros(
                (len(batch), max_len),
                dtype=np.int64,
            )

            for row, encoding in enumerate(encodings):
                ids = encoding.ids[:max_len]
                attn = encoding.attention_mask[:max_len]

                if ids:
                    input_ids[row, : len(ids)] = ids
                    attention_mask[row, : len(ids)] = attn

            feed = {
                self.input_ids_name: input_ids,
            }

            if self.attention_mask_name is not None:
                feed[self.attention_mask_name] = attention_mask

            if self.token_type_ids_name is not None:
                feed[self.token_type_ids_name] = np.zeros_like(input_ids)

            outputs = self.session.run(None, feed)

            tensor = np.asarray(outputs[0], dtype=np.float32)

            # Some ONNX exports return:
            #   (batch_size, sequence_length, hidden_size)
            # Some already return:
            #   (batch_size, hidden_size)
            if tensor.ndim == 3:
                mask = attention_mask[:, :, None].astype(np.float32)

                summed = np.sum(tensor * mask, axis=1)
                counts = np.maximum(np.sum(mask, axis=1), 1.0)

                vectors = summed / counts

            elif tensor.ndim == 2:
                vectors = tensor

            else:
                raise ValueError(
                    f"Unexpected embedding tensor shape: {tensor.shape}"
                )

            # L2 normalize for cosine similarity.
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)

            all_vectors.append(vectors.astype(np.float32))

        return np.vstack(all_vectors)


class CodeEntityStore:
    """
    SQLite-backed semantic index for functions/classes/methods.

    Important:
        source_id is not globally unique.
        The real unique key is:
            entity_type + source_id
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        embedder: OnnxTextEmbedder,
        preload: bool = True,
    ):
        self.db_path = str(db_path)

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.embedder = embedder
        self._lock = threading.RLock()

        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row

        self._create_schema()

        # In-memory cache for fast cosine search.
        self._row_ids: List[int] = []
        self._entity_types: List[str] = []
        self._source_ids: List[int] = []
        self._names: List[str] = []
        self._type_array = np.array([], dtype=object)
        self._embeddings = np.empty((0, 0), dtype=np.float32)

        if preload:
            self.reload_cache()

    def _create_schema(self) -> None:
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    def reload_cache(self) -> None:
        """
        Loads all embeddings from SQLite into memory.

        This makes search very fast because it avoids decoding BLOBs
        on every query.
        """
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT
                    id,
                    entity_type,
                    source_id,
                    name,
                    embedding
                FROM code_entities
                ORDER BY id
                """
            ).fetchall()

            row_ids: List[int] = []
            entity_types: List[str] = []
            source_ids: List[int] = []
            names: List[str] = []
            vectors: List[np.ndarray] = []

            for row in rows:
                vector = np.frombuffer(row["embedding"], dtype=np.float32)

                # Skip corrupted/empty embeddings instead of crashing.
                if vector.size == 0:
                    continue

                row_ids.append(row["id"])
                entity_types.append(row["entity_type"])
                source_ids.append(row["source_id"])
                names.append(row["name"])
                vectors.append(vector)

            self._row_ids = row_ids
            self._entity_types = entity_types
            self._source_ids = source_ids
            self._names = names
            self._type_array = np.array(entity_types, dtype=object)

            if vectors:
                self._embeddings = np.vstack(vectors).astype(
                    np.float32,
                    copy=False,
                )
            else:
                self._embeddings = np.empty((0, 0), dtype=np.float32)

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> "CodeEntityStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def count(self) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM code_entities"
            ).fetchone()
            return int(row[0])

    def add_item(self, item: CodeItemIn) -> None:
        """
        Inserts or updates one item.
        """
        self.add_items([item])

    def add_items(self, items: Iterable[CodeItemIn]) -> int:
        """
        Inserts or updates many items.

        Upsert key:
            entity_type + source_id
        """
        item_list = list(items)

        if not item_list:
            return 0

        texts = [item.embedding_text() for item in item_list]
        vectors = self.embedder.embed(texts)

        if vectors.shape[0] != len(item_list):
            raise RuntimeError(
                "Embedding count does not match item count. "
                f"Expected {len(item_list)}, got {vectors.shape[0]}."
            )

        rows = []

        for item, vector in zip(item_list, vectors):
            blob = vector.astype(np.float32, copy=False).tobytes()

            rows.append(
                (
                    item.entity_type.value,
                    int(item.source_id),
                    item.name.strip(),
                    blob,
                )
            )

        sql = """
        INSERT INTO code_entities (
            entity_type,
            source_id,
            name,
            embedding
        )
        VALUES (?, ?, ?, ?)
        ON CONFLICT(entity_type, source_id) DO UPDATE SET
            name = excluded.name,
            embedding = excluded.embedding,
            created_at = datetime('now');
        """

        with self._lock, self.conn:
            self.conn.executemany(sql, rows)

        self.reload_cache()

        return len(rows)

    def add_function(
        self,
        source_id: int,
        name: str,
        extra_text: Optional[str] = None,
    ) -> None:
        self.add_item(
            CodeItemIn(
                entity_type=EntityType.FUNCTION,
                source_id=source_id,
                name=name,
                extra_text=extra_text,
            )
        )

    def add_class(
        self,
        source_id: int,
        name: str,
        extra_text: Optional[str] = None,
    ) -> None:
        self.add_item(
            CodeItemIn(
                entity_type=EntityType.CLASS,
                source_id=source_id,
                name=name,
                extra_text=extra_text,
            )
        )

    def add_method(
        self,
        source_id: int,
        name: str,
        extra_text: Optional[str] = None,
    ) -> None:
        self.add_item(
            CodeItemIn(
                entity_type=EntityType.METHOD,
                source_id=source_id,
                name=name,
                extra_text=extra_text,
            )
        )

    def delete_item(
        self,
        entity_type: Union[EntityType, str],
        source_id: int,
    ) -> bool:
        """
        Deletes an item from the index.
        """
        entity_type_value = (
            entity_type.value
            if isinstance(entity_type, EntityType)
            else str(entity_type)
        )

        with self._lock, self.conn:
            cursor = self.conn.execute(
                """
                DELETE FROM code_entities
                WHERE entity_type = ?
                  AND source_id = ?
                """,
                (entity_type_value, int(source_id)),
            )

        deleted = cursor.rowcount > 0

        if deleted:
            self.reload_cache()

        return deleted

    def get_item(
        self,
        entity_type: Union[EntityType, str],
        source_id: int,
    ) -> Optional[CodeItemOut]:
        """
        Retrieves one stored item by its composite original key.
        """
        entity_type_value = (
            entity_type.value
            if isinstance(entity_type, EntityType)
            else str(entity_type)
        )

        with self._lock:
            row = self.conn.execute(
                """
                SELECT
                    id,
                    entity_type,
                    source_id,
                    name
                FROM code_entities
                WHERE entity_type = ?
                  AND source_id = ?
                """,
                (entity_type_value, int(source_id)),
            ).fetchone()

        if row is None:
            return None

        return CodeItemOut(
            row_id=row["id"],
            entity_type=EntityType(row["entity_type"]),
            source_id=row["source_id"],
            name=row["name"],
        )

    def search(
        self,
        query: Union[str, SearchQuery],
        limit: Optional[int] = None,
        entity_types: Optional[Sequence[Union[EntityType, str]]] = None,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Semantic search.

        Returns:
            entity_type, source_id, name, score

        You can then use entity_type + source_id to query your original DB.
        """
        if isinstance(query, SearchQuery):
            query_text = query.query
            limit = query.limit if limit is None else limit
            entity_types = query.entity_types if entity_types is None else entity_types
            min_score = query.min_score if min_score is None else min_score
        else:
            query_text = query

        if not query_text or not query_text.strip():
            return []

        limit = 10 if limit is None else int(limit)
        min_score = -1.0 if min_score is None else float(min_score)

        if limit <= 0:
            return []

        # Grab cache references under lock.
        with self._lock:
            if self._embeddings.size == 0:
                return []

            embeddings = self._embeddings
            type_array = self._type_array
            row_ids = self._row_ids
            entity_type_values = self._entity_types
            source_ids = self._source_ids
            names = self._names

        query_vector = self.embedder.embed([query_text])[0]

        if embeddings.ndim != 2:
            raise RuntimeError("Stored embeddings cache is invalid.")

        if embeddings.shape[1] != query_vector.shape[0]:
            raise ValueError(
                "Stored embedding dimension does not match query embedding dimension. "
                "This usually means the DB was created with a different embedding model."
            )

        # Cosine similarity.
        # Embeddings are normalized, so dot product == cosine similarity.
        scores = embeddings @ query_vector
        scores = np.asarray(scores, dtype=np.float32)

        # Optional entity type filter.
        if entity_types:
            wanted = {
                item.value
                if isinstance(item, EntityType)
                else EntityType(str(item)).value
                for item in entity_types
            }

            mask = np.array(
                [entity_type in wanted for entity_type in type_array],
                dtype=bool,
            )

            if not mask.any():
                return []

            scores = np.where(mask, scores, -np.inf)

        # Remove non-finite scores.
        scores = np.where(np.isfinite(scores), scores, -np.inf)

        if scores.size == 0:
            return []

        if scores.size <= limit:
            candidate_indices = np.argsort(-scores)
        else:
            candidate_indices = np.argpartition(-scores, limit)[:limit]
            candidate_indices = candidate_indices[
                np.argsort(-scores[candidate_indices])
            ]

        results: List[SearchResult] = []

        for idx in candidate_indices:
            score = float(scores[idx])

            if not np.isfinite(score):
                continue

            if score < min_score:
                continue

            results.append(
                SearchResult(
                    entity_type=EntityType(entity_type_values[idx]),
                    source_id=source_ids[idx],
                    name=names[idx],
                    score=score,
                    row_id=row_ids[idx],
                )
            )

        return results