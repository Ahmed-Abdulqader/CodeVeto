from __future__ import annotations

from pathlib import Path

from context_engine.config import EngineConfig
from context_engine.engine import ContextEngine


class TestContextEngineIntegration:
    def test_index_then_search_then_expand(self, engine: ContextEngine):
        stats = engine.index(force=True)
        assert stats["indexed"] == 3

        session = engine.new_session(label="integration-test")
        result = engine.tools.search_context("say hello", session_id=session.id)
        assert result["ok"] is True
        assert result["results"]

        chunk_id = result["results"][0]["chunk_id"]
        expanded = engine.tools.expand_chunk(chunk_id, session_id=session.id)
        assert expanded["ok"] is True
        assert expanded["chunk"]["id"] == chunk_id

    def test_decisions_and_areas_surface_in_brief(self, engine: ContextEngine):
        engine.index(force=True)
        session = engine.new_session()

        engine.tools.record_decision(session.id, "reuse say_hello helper")
        engine.tools.record_code_area(
            session.id, "greeter.py", note="entry point for greetings"
        )

        brief = engine.tools.get_context_brief(session.id)
        assert brief["ok"] is True
        assert "reuse say_hello helper" in brief["brief"]["text"]
        assert "entry point for greetings" in brief["brief"]["text"]

    def test_reindexing_one_file_after_edit_is_reflected_in_search(
        self, engine: ContextEngine, sample_repo: Path
    ):
        engine.index(force=True)

        py_file = sample_repo / "greeter.py"
        py_file.write_text(
            py_file.read_text()
            + "\n\ndef farewell(name):\n    return f'Bye, {name}!'\n"
        )
        engine.index_file(str(py_file))

        result = engine.tools.search_context("farewell bye")
        assert result["ok"] is True
        assert any("farewell" in (r["signature"] or "") for r in result["results"])

    def test_engine_works_with_embedding_disabled_by_default(self, sample_repo: Path):
        config = EngineConfig(project_root=sample_repo)
        assert config.embedding.provider == "none"
        with ContextEngine(config) as engine:
            engine.index(force=True)
            result = engine.tools.search_context("say hello")
            assert result["ok"] is True
            assert all(r["matched_by"] == ["lexical"] for r in result["results"])

    def test_tool_specs_exposed_on_engine(self, engine: ContextEngine):
        names = {spec["name"] for spec in engine.tool_specs}
        assert "search_context" in names
        assert "write_file" not in names
