from __future__ import annotations

import json

from context_engine.memory.session_manager import SessionManager
from context_engine.search.hybrid import ContextSearch
from context_engine.search.lexical import LexicalIndex
from context_engine.storage.chunk_store import ChunkStore
from context_engine.storage.graph_store import GraphStore
from context_engine.storage.session_store import SessionStore
from context_engine.storage.vector_store import VectorStore
from context_engine.tools.context_tools import ContextTools
from context_engine.tools.schemas import TOOL_SPECS
from models.context_engine_models import Chunk, FileMetadata


def _make_chunk(path: str, chunk_id: str, content: str) -> Chunk:
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


def _tools(db):
    chunk_store = ChunkStore(db)
    graph_store = GraphStore(db)
    lexical = LexicalIndex(db)
    search = ContextSearch(
        chunk_store, lexical, VectorStore(db), embedding_provider=None
    )
    session_manager = SessionManager(SessionStore(db))
    return (
        ContextTools(chunk_store, graph_store, search, session_manager),
        chunk_store,
        lexical,
        session_manager,
    )


class TestToolContract:
    def test_search_context_returns_ok_with_results(self, db):
        tools, chunk_store, lexical, _ = _tools(db)
        chunk_store.upsert_file(
            FileMetadata(name="a.py", path="a.py", modified_ts=1.0), "python", "h"
        )
        chunk_store.insert_chunks(
            [_make_chunk("a.py", "c1", "def hash_password(pw): return pw")]
        )
        lexical.index_chunk("c1", "hash password")

        result = tools.search_context("hash password")
        assert result["ok"] is True
        assert result["results"][0]["chunk_id"] == "c1"

    def test_expand_chunk_unknown_id_returns_ok_false(self, db):
        tools, *_ = _tools(db)
        result = tools.expand_chunk("does-not-exist")
        assert result == {"ok": False, "error": "No chunk with id 'does-not-exist'."}

    def test_expand_chunk_includes_graph_neighbors(self, db):
        tools, chunk_store, lexical, _ = _tools(db)
        chunk_store.upsert_file(
            FileMetadata(name="a.py", path="a.py", modified_ts=1.0), "python", "h"
        )
        chunk_store.insert_chunks([_make_chunk("a.py", "c1", "def f(): pass")])

        # Seed a matching graph node + a caller/callee so the enrichment
        # path in expand_chunk (node_for_chunk_span -> callers_of/callees_of)
        # has something real to find. _make_chunk's chunk always starts on
        # line 1.
        from models.context_engine_models import (
            EdgeType,
            GraphEdge,
            GraphNode,
            NodeType,
            SymbolGraph,
        )

        graph = SymbolGraph()
        graph.add_node(
            GraphNode(
                id="fn:f",
                type=NodeType.FUNCTION,
                name="f",
                file_path="a.py",
                start_line=1,
                end_line=1,
            )
        )
        graph.add_node(
            GraphNode(
                id="fn:caller",
                type=NodeType.FUNCTION,
                name="caller",
                file_path="a.py",
                start_line=5,
                end_line=6,
            )
        )
        graph.add_edge(
            GraphEdge(source_id="fn:caller", target_id="fn:f", type=EdgeType.CALLS)
        )
        GraphStore(db).save_graph(graph)

        result = tools.expand_chunk("c1")
        assert result["ok"] is True
        assert result["chunk"]["callers"] == ["caller"]
        assert result["chunk"]["callees"] == []

    def test_list_files(self, db):
        tools, chunk_store, *_ = _tools(db)
        chunk_store.upsert_file(
            FileMetadata(name="a.py", path="a.py", modified_ts=1.0), "python", "h"
        )
        result = tools.list_files()
        assert result["ok"] is True
        assert result["files"][0]["path"] == "a.py"

    def test_record_decision_and_area_require_a_real_session(self, db):
        tools, *_ = _tools(db)
        result = tools.record_decision("nonexistent-session", "some decision")
        assert result["ok"] is False
        assert "nonexistent-session" in result["error"]

        result = tools.record_code_area("nonexistent-session", "a.py")
        assert result["ok"] is False

    def test_full_session_flow(self, db):
        tools, *_, session_manager = _tools(db)
        session = session_manager.start_session()

        dec = tools.record_decision(session.id, "use bcrypt")
        assert dec["ok"] is True
        area = tools.record_code_area(session.id, "auth.py")
        assert area["ok"] is True
        ev = tools.session_event(session.id, "ran_tests", {"passed": True})
        assert ev["ok"] is True

        brief = tools.get_context_brief(session.id)
        assert brief["ok"] is True
        assert "use bcrypt" in brief["brief"]["text"]

    def test_search_context_with_bad_session_id_still_succeeds(self, db):
        """A bogus session_id passed to a read-only tool must not turn a
        successful search into a failure -- only the best-effort memory
        logging for it is skipped."""
        tools, chunk_store, lexical, _ = _tools(db)
        chunk_store.upsert_file(
            FileMetadata(name="a.py", path="a.py", modified_ts=1.0), "python", "h"
        )
        chunk_store.insert_chunks(
            [_make_chunk("a.py", "c1", "def hash_password(pw): return pw")]
        )
        lexical.index_chunk("c1", "hash password")

        result = tools.search_context("hash password", session_id="does-not-exist")
        assert result["ok"] is True


class TestToolSpecs:
    def test_specs_are_json_serializable_and_named(self):
        names = {spec["name"] for spec in TOOL_SPECS}
        assert names == {
            "search_context",
            "expand_chunk",
            "list_files",
            "session_event",
            "record_decision",
            "record_code_area",
            "get_context_brief",
        }
        json.dumps(TOOL_SPECS)  # must not raise

    def test_no_write_or_execute_tools_are_exposed(self):
        names = {spec["name"] for spec in TOOL_SPECS}
        assert "write_file" not in names
        assert "run_code" not in names
