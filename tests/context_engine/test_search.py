from __future__ import annotations

import pytest

from context_engine.search.embedding import (
    EmbeddingProviderError,
    FallbackEmbeddingProvider,
    LocalOnnxEmbeddingProvider,
    NullEmbeddingProvider,
    RemoteEmbeddingProvider,
)
from context_engine.search.hybrid import ContextSearch
from context_engine.search.lexical import LexicalIndex
from context_engine.search.tokenizer import tokenize
from context_engine.storage.chunk_store import ChunkStore
from context_engine.storage.vector_store import VectorStore
from models.context_engine_models import Chunk, FileMetadata


def _make_chunk(content: str) -> Chunk:
    return Chunk(
        id="c1",
        language="python",
        chunk_type="function",
        docstring=None,
        signature="def hash_password(pw)",
        content=content,
        start_line=1,
        end_line=1,
        file_metadata=FileMetadata(name="a.py", path="a.py", modified_ts=1.0),
    )


class TestTokenizer:
    def test_splits_snake_case(self):
        assert tokenize("get_user_by_id") == ["get", "user", "by", "id"]

    def test_splits_camel_case(self):
        assert tokenize("getUserById") == ["get", "user", "by", "id"]

    def test_lowercases_and_drops_stopwords(self):
        assert tokenize("The Quick FOX") == ["quick", "fox"]

    def test_empty_input(self):
        assert tokenize("") == []
        assert tokenize(None) == []


class TestLexicalIndex:
    def test_search_ranks_more_relevant_doc_higher(self, db):
        index = LexicalIndex(db)
        index.index_chunk(
            "c1", "def hash_password(password): use bcrypt to hash the password"
        )
        index.index_chunk("c2", "def read_config(path): load a yaml config file")

        results = index.search("hash password")
        assert results
        assert results[0][0] == "c1"

    def test_no_match_returns_empty(self, db):
        index = LexicalIndex(db)
        index.index_chunk("c1", "def read_config(path): load yaml")
        assert index.search("completely unrelated query xyz") == []

    def test_remove_chunk_drops_it_from_results(self, db):
        index = LexicalIndex(db)
        index.index_chunk("c1", "hash password with bcrypt")
        index.remove_chunk("c1")
        assert index.search("hash password") == []


class TestEmbeddingProviders:
    def test_null_provider_always_errors(self):
        provider = NullEmbeddingProvider()
        with pytest.raises(EmbeddingProviderError):
            provider.embed_query("hello")

    def test_remote_provider_requires_api_key_env(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        provider = RemoteEmbeddingProvider(
            base_url="https://example.invalid", model="m", api_key_env="MISSING_KEY"
        )
        with pytest.raises(EmbeddingProviderError, match="MISSING_KEY"):
            provider.embed_query("hello")

    def test_fallback_tries_next_provider_on_failure(self):
        class AlwaysFails:
            def embed_query(self, text):
                raise EmbeddingProviderError("nope")

            def embed_documents(self, texts):
                raise EmbeddingProviderError("nope")

        class Succeeds:
            def embed_query(self, text):
                return [0.1, 0.2]

            def embed_documents(self, texts):
                return [[0.1, 0.2] for _ in texts]

        provider = FallbackEmbeddingProvider([AlwaysFails(), Succeeds()])
        assert provider.embed_query("hello") == [0.1, 0.2]

    def test_fallback_raises_when_everything_fails(self):
        class AlwaysFails:
            def embed_query(self, text):
                raise EmbeddingProviderError("nope")

            def embed_documents(self, texts):
                raise EmbeddingProviderError("nope")

        provider = FallbackEmbeddingProvider([AlwaysFails(), AlwaysFails()])
        with pytest.raises(EmbeddingProviderError):
            provider.embed_query("hello")

    def test_fallback_requires_at_least_one_provider(self):
        with pytest.raises(ValueError):
            FallbackEmbeddingProvider([])


class TestContextSearch:
    def _seeded_store(self, db):
        store = ChunkStore(db)
        store.upsert_file(
            FileMetadata(name="a.py", path="a.py", modified_ts=1.0), "python", "h"
        )
        store.insert_chunks(
            [_make_chunk(content="def hash_password(pw): return bcrypt.hash(pw)")]
        )
        return store

    def test_lexical_only_when_no_embedding_provider(self, db):
        store = self._seeded_store(db)
        lexical = LexicalIndex(db)
        lexical.index_chunk("c1", "hash password with bcrypt")
        search = ContextSearch(store, lexical, VectorStore(db), embedding_provider=None)

        hits = search.search("hash password")
        assert hits
        assert hits[0].matched_by == ["lexical"]

    def test_falls_back_to_lexical_when_embedding_provider_fails(self, db):
        store = self._seeded_store(db)
        lexical = LexicalIndex(db)
        lexical.index_chunk("c1", "hash password with bcrypt")
        vector_store = VectorStore(db)
        vector_store.upsert(
            "c1", "p", "m", [1.0, 0.0]
        )  # non-empty, so vector path is attempted

        search = ContextSearch(
            store, lexical, vector_store, embedding_provider=NullEmbeddingProvider()
        )

        hits = search.search("hash password")
        assert hits
        assert hits[0].matched_by == ["lexical"]


@pytest.fixture
def onnx_embedding_model(tmp_path):
    """Builds a tiny, self-contained ONNX 'embedding model' (a plain
    embedding-table lookup, no real semantics) plus a matching WordLevel
    tokenizer -- enough to exercise LocalOnnxEmbeddingProvider's actual
    tokenize -> run -> mean-pool -> normalize pipeline against a real
    ONNX Runtime session, without needing network access or the real
    ~160MB jina model file."""
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import TensorProto, helper
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"[PAD]": 0, "[UNK]": 1, "hash": 2, "password": 3, "def": 4, "bcrypt": 5}
    tokenizer_path = tmp_path / "tokenizer.json"
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(tokenizer_path))

    hidden = 8
    rng = np.random.default_rng(0)
    table = rng.normal(size=(len(vocab), hidden)).astype(np.float32)

    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT64, ["b", "s"]
    )
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["b", "s"]
    )
    token_type_ids = helper.make_tensor_value_info(
        "token_type_ids", TensorProto.INT64, ["b", "s"]
    )
    output = helper.make_tensor_value_info(
        "last_hidden_state", TensorProto.FLOAT, ["b", "s", hidden]
    )
    table_init = helper.make_tensor(
        "embedding_table", TensorProto.FLOAT, table.shape, table.flatten().tolist()
    )
    gather = helper.make_node(
        "Gather",
        inputs=["embedding_table", "input_ids"],
        outputs=["last_hidden_state"],
        axis=0,
    )
    graph = helper.make_graph(
        [gather],
        "fake_embedding_model",
        [input_ids, attention_mask, token_type_ids],
        [output],
        initializer=[table_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    model_path = tmp_path / "model.onnx"
    onnx.save(model, str(model_path))

    return model_path, tokenizer_path


class TestLocalOnnxEmbeddingProvider:
    def test_missing_onnxruntime_or_tokenizers_raises_embedding_provider_error(
        self, tmp_path, monkeypatch
    ):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in ("onnxruntime", "tokenizers"):
                raise ImportError(f"no {name} here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(EmbeddingProviderError, match="onnxruntime"):
            LocalOnnxEmbeddingProvider(tmp_path / "m.onnx", tmp_path / "t.json")

    def test_missing_model_file_raises(self, tmp_path):
        pytest.importorskip("onnxruntime")
        pytest.importorskip("tokenizers")
        (tmp_path / "tokenizer.json").write_text("{}")
        with pytest.raises(EmbeddingProviderError, match="not found"):
            LocalOnnxEmbeddingProvider(
                tmp_path / "missing.onnx", tmp_path / "tokenizer.json"
            )

    def test_embed_query_returns_normalized_vector(self, onnx_embedding_model):
        pytest.importorskip("onnxruntime")
        model_path, tokenizer_path = onnx_embedding_model
        provider = LocalOnnxEmbeddingProvider(model_path, tokenizer_path, max_length=16)

        vector = provider.embed_query("hash password")
        assert len(vector) == 8
        assert abs(sum(v * v for v in vector) ** 0.5 - 1.0) < 1e-5

    def test_embed_documents_batch_and_determinism(self, onnx_embedding_model):
        pytest.importorskip("onnxruntime")
        model_path, tokenizer_path = onnx_embedding_model
        provider = LocalOnnxEmbeddingProvider(model_path, tokenizer_path, max_length=16)

        vectors = provider.embed_documents(
            ["def", "hash password", "bcrypt hash password"]
        )
        assert len(vectors) == 3
        assert provider.embed_query("hash password") == provider.embed_query(
            "hash password"
        )

    def test_padding_does_not_change_a_short_text_embedding(self, onnx_embedding_model):
        """The whole point of masking in mean-pooling: batching a short
        text alongside a longer one (which pads the short one out) must
        not change the short text's own embedding."""
        pytest.importorskip("onnxruntime")
        model_path, tokenizer_path = onnx_embedding_model
        provider = LocalOnnxEmbeddingProvider(model_path, tokenizer_path, max_length=16)

        alone = provider.embed_query("hash")
        padded = provider.embed_documents(["hash", "bcrypt hash password unknownword"])[
            0
        ]
        assert alone == pytest.approx(padded, abs=1e-5)

    def test_works_without_token_type_ids_input(self, tmp_path):
        """Some ONNX exports don't declare a token_type_ids input at all --
        the provider should auto-detect that from the session rather than
        always feeding it."""
        onnx = pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        import numpy as np
        from onnx import TensorProto, helper
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        vocab = {"[PAD]": 0, "[UNK]": 1, "hash": 2}
        tokenizer_path = tmp_path / "tokenizer.json"
        tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        tok.save(str(tokenizer_path))

        hidden = 4
        table = (
            np.random.default_rng(0)
            .normal(size=(len(vocab), hidden))
            .astype(np.float32)
        )
        input_ids = helper.make_tensor_value_info(
            "input_ids", TensorProto.INT64, ["b", "s"]
        )
        attention_mask = helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, ["b", "s"]
        )
        output = helper.make_tensor_value_info(
            "last_hidden_state", TensorProto.FLOAT, ["b", "s", hidden]
        )
        table_init = helper.make_tensor(
            "embedding_table", TensorProto.FLOAT, table.shape, table.flatten().tolist()
        )
        gather = helper.make_node(
            "Gather",
            inputs=["embedding_table", "input_ids"],
            outputs=["last_hidden_state"],
            axis=0,
        )
        graph = helper.make_graph(
            [gather],
            "no_ttid",
            [input_ids, attention_mask],
            [output],
            initializer=[table_init],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        provider = LocalOnnxEmbeddingProvider(model_path, tokenizer_path)
        assert "token_type_ids" not in provider._input_names
        vector = provider.embed_query("hash")
        assert len(vector) == 4
