from context_engine.embedder import OllamaEmbedder
from models.context_engine_models import Chunk


def make_chunk() -> Chunk:
    return Chunk(
        file_metadata={
            "path": "src/auth/middleware.py",
        },
        signature="authenticate(request)",
        docstring="Authenticate an incoming request.",
        content=(
            "def authenticate(request):\n"
            "    token = request.headers.get('Authorization')\n"
            "    return verify_token(token)"
        ),
    )


def test_embed_query():
    embedder = OllamaEmbedder()

    vector = embedder.embed_query(
        "How does authentication middleware work?"
    )

    assert isinstance(vector, list)
    assert len(vector) == 768
    assert all(isinstance(value, float) for value in vector)


def test_embed_batch():
    embedder = OllamaEmbedder()

    chunk = make_chunk()

    result = embedder.embed_batch([chunk])

    assert len(result) == 1

    embedded_chunk = result[0]

    assert embedded_chunk.chunk == chunk
    assert isinstance(embedded_chunk.embedding, list)
    assert len(embedded_chunk.embedding) == 768
    assert all(
        isinstance(value, float)
        for value in embedded_chunk.embedding
    )


def test_document_text_contains_required_context():
    chunk = make_chunk()

    text = OllamaEmbedder._build_document_text(chunk)

    assert text.startswith("search_document: ")
    assert "file: src/auth/middleware.py" in text
    assert "signature: authenticate(request)" in text
    assert "docstring: Authenticate an incoming request." in text
    assert "def authenticate(request):" in text


def test_query_text_contains_required_prefix():
    query = "How does authentication work?"

    text = OllamaEmbedder._build_query_text(query)

    assert text == "search_query: How does authentication work?"
