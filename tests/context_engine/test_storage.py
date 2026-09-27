from __future__ import annotations

from context_engine.storage.chunk_store import ChunkStore
from context_engine.storage.graph_store import GraphStore
from context_engine.storage.session_store import SessionStore
from context_engine.storage.vector_store import VectorStore
from models.context_engine_models import (
    Chunk,
    CodeArea,
    Decision,
    EdgeType,
    FileMetadata,
    GraphEdge,
    GraphNode,
    NodeType,
    Session,
    SessionEvent,
    SymbolGraph,
)


def _make_chunk(path="a.py", chunk_id="c1", content="def f(): pass"):
    return Chunk(
        id=chunk_id,
        language="python",
        chunk_type="function",
        docstring=None,
        signature="def f()",
        content=content,
        start_line=1,
        end_line=1,
        file_metadata=FileMetadata(name=path, path=path, modified_ts=1.0),
    )


class TestChunkStore:
    def test_upsert_file_and_chunks_round_trip(self, db):
        store = ChunkStore(db)
        meta = FileMetadata(name="a.py", path="a.py", modified_ts=1.0)
        store.upsert_file(meta, "python", "hash1")
        store.insert_chunks([_make_chunk()])

        chunk = store.get_chunk("c1")
        assert chunk is not None
        assert chunk.signature == "def f()"
        assert chunk.file_metadata.path == "a.py"

    def test_delete_file_cascades_chunks(self, db):
        store = ChunkStore(db)
        meta = FileMetadata(name="a.py", path="a.py", modified_ts=1.0)
        store.upsert_file(meta, "python", "hash1")
        store.insert_chunks([_make_chunk()])

        store.delete_file("a.py")

        assert store.get_chunk("c1") is None
        assert store.list_files() == []

    def test_get_file_hash_reflects_upsert(self, db):
        store = ChunkStore(db)
        meta = FileMetadata(name="a.py", path="a.py", modified_ts=1.0)
        assert store.get_file_hash("a.py") is None
        store.upsert_file(meta, "python", "hash1")
        assert store.get_file_hash("a.py") == "hash1"
        store.upsert_file(meta, "python", "hash2")
        assert store.get_file_hash("a.py") == "hash2"

    def test_list_files_prefix_filter(self, db):
        store = ChunkStore(db)
        store.upsert_file(
            FileMetadata(name="a.py", path="src/a.py", modified_ts=1.0), "python", "h1"
        )
        store.upsert_file(
            FileMetadata(name="b.py", path="tests/b.py", modified_ts=1.0),
            "python",
            "h2",
        )

        assert [f["path"] for f in store.list_files("src/")] == ["src/a.py"]


class TestGraphStore:
    def test_save_and_query_callers_callees(self, db):
        store = GraphStore(db)
        graph = SymbolGraph()
        graph.add_node(
            GraphNode(
                id="f1",
                type=NodeType.FUNCTION,
                name="outer",
                file_path="a.py",
                start_line=1,
                end_line=2,
            )
        )
        graph.add_node(
            GraphNode(
                id="f2",
                type=NodeType.FUNCTION,
                name="inner",
                file_path="a.py",
                start_line=4,
                end_line=5,
            )
        )
        graph.add_edge(GraphEdge(source_id="f1", target_id="f2", type=EdgeType.CALLS))
        store.save_graph(graph)

        assert [n.name for n in store.callees_of("f1")] == ["inner"]
        assert [n.name for n in store.callers_of("f2")] == ["outer"]

    def test_delete_file_nodes_removes_dangling_edges(self, db):
        store = GraphStore(db)
        graph = SymbolGraph()
        graph.add_node(
            GraphNode(id="f1", type=NodeType.FUNCTION, name="outer", file_path="a.py")
        )
        graph.add_node(
            GraphNode(id="f2", type=NodeType.FUNCTION, name="inner", file_path="a.py")
        )
        graph.add_edge(GraphEdge(source_id="f1", target_id="f2", type=EdgeType.CALLS))
        store.save_graph(graph)

        store.delete_file_nodes("a.py")

        assert store.nodes_for_file("a.py") == []
        assert store.callees_of("f1") == []

    def test_node_for_chunk_span_matches_by_file_and_start_line(self, db):
        store = GraphStore(db)
        graph = SymbolGraph()
        graph.add_node(
            GraphNode(
                id="f1",
                type=NodeType.FUNCTION,
                name="outer",
                file_path="a.py",
                start_line=7,
                end_line=9,
            )
        )
        store.save_graph(graph)

        found = store.node_for_chunk_span("a.py", 7)
        assert found is not None
        assert found.name == "outer"
        assert store.node_for_chunk_span("a.py", 999) is None


class TestSessionStore:
    def test_session_lifecycle(self, db):
        store = SessionStore(db)
        session = Session(label="demo")
        store.create_session(session)
        assert store.session_exists(session.id)
        assert not store.session_exists("nonexistent")

    def test_events_decisions_and_areas_round_trip(self, db):
        store = SessionStore(db)
        session = store.create_session(Session())

        store.add_event(
            SessionEvent(session_id=session.id, event_type="search", data={"q": "auth"})
        )
        store.add_decision(
            Decision(
                session_id=session.id, decision="use bcrypt", related_chunk_ids=["c1"]
            )
        )
        store.add_code_area(
            CodeArea(session_id=session.id, file_path="auth.py", note="entry point")
        )

        events = store.recent_events(session.id)
        decisions = store.recent_decisions(session.id)
        areas = store.recent_code_areas(session.id)

        assert events[0].event_type == "search"
        assert events[0].data == {"q": "auth"}
        assert decisions[0].decision == "use bcrypt"
        assert decisions[0].related_chunk_ids == ["c1"]
        assert areas[0].file_path == "auth.py"

        counts = store.counts(session.id)
        assert counts == {"events": 1, "decisions": 1, "code_areas": 1}

    def test_events_scoped_per_session(self, db):
        store = SessionStore(db)
        s1 = store.create_session(Session())
        s2 = store.create_session(Session())
        store.add_event(SessionEvent(session_id=s1.id, event_type="a"))
        store.add_event(SessionEvent(session_id=s2.id, event_type="b"))

        assert [e.event_type for e in store.recent_events(s1.id)] == ["a"]
        assert [e.event_type for e in store.recent_events(s2.id)] == ["b"]


class TestVectorStore:
    def test_upsert_and_search_similar(self, db):
        store = VectorStore(db)
        store.upsert("c1", "test-provider", "test-model", [1.0, 0.0])
        store.upsert("c2", "test-provider", "test-model", [0.0, 1.0])

        results = store.search_similar([1.0, 0.0], top_k=2)
        assert results[0][0] == "c1"
        assert results[0][1] > results[1][1]

    def test_is_empty(self, db):
        store = VectorStore(db)
        assert store.is_empty()
        store.upsert("c1", "p", "m", [1.0])
        assert not store.is_empty()

    def test_remove(self, db):
        store = VectorStore(db)
        store.upsert("c1", "p", "m", [1.0, 0.0])
        store.remove("c1")
        assert store.is_empty()
