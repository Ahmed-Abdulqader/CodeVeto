from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import Agent, Tool
from pydantic_ai.models.test import TestModel

from agents.tools import (
    CONTEXT_ENGINE_TOOLS,
    ContextEngineDeps,
    expand_chunk,
    get_context_brief,
    list_files,
    record_code_area,
    record_decision,
    search_context,
    session_event,
)
from context_engine import ContextEngine, EngineConfig
from context_engine.config import EmbeddingSettings

PYTHON_SAMPLE = "def say_hello(name):\n    return f'Hello, {name}!'\n"


class _FakeRunContext:
    """Stand-in for pydantic_ai.RunContext. Every function in agents/tools.py
    only ever reads `ctx.deps` -- nothing else on the real RunContext -- so
    a bare object with just that attribute is a faithful, much lighter
    substitute for unit-testing the tools' own logic. `test_full_agent_run`
    below still exercises the *real* RunContext, via a real Agent."""

    def __init__(self, deps: ContextEngineDeps) -> None:
        self.deps = deps


@pytest.fixture
def engine(tmp_path: Path):
    (tmp_path / "greeter.py").write_text(PYTHON_SAMPLE)
    config = EngineConfig(
        project_root=tmp_path,
        db_path=tmp_path / ".codeveto" / "context.db",
        log_dir=tmp_path / ".codeveto" / "logs",
        embedding=EmbeddingSettings(provider="none"),
    )
    eng = ContextEngine(config)
    eng.index(force=True)
    yield eng
    eng.close()


@pytest.fixture
def deps(engine: ContextEngine) -> ContextEngineDeps:
    session = engine.new_session(label="test")
    return ContextEngineDeps(engine=engine, session_id=session.id)


@pytest.fixture
def ctx(deps: ContextEngineDeps) -> _FakeRunContext:
    return _FakeRunContext(deps)


class TestContextEngineDeps:
    def test_stores_engine_and_session_id(self, engine: ContextEngine):
        d = ContextEngineDeps(engine=engine, session_id="sess_123")
        assert d.engine is engine
        assert d.session_id == "sess_123"


class TestToolFunctionsDelegateCorrectly:
    """Each function should do exactly one thing: call the matching
    ContextEngine.tools.* method with ctx.deps.session_id threaded
    through -- never asking the model to supply a session_id itself."""

    def test_search_context(self, ctx: _FakeRunContext):
        result = search_context(ctx, "say hello", top_k=3)
        assert result["ok"] is True
        assert result["results"]
        assert result["results"][0]["signature"]

    def test_expand_chunk(self, ctx: _FakeRunContext):
        found = search_context(ctx, "say hello")["results"][0]
        result = expand_chunk(ctx, found["chunk_id"])
        assert result["ok"] is True
        assert result["chunk"]["id"] == found["chunk_id"]

    def test_expand_chunk_unknown_id(self, ctx: _FakeRunContext):
        result = expand_chunk(ctx, "does-not-exist")
        assert result["ok"] is False

    def test_list_files(self, ctx: _FakeRunContext):
        result = list_files(ctx)
        assert result["ok"] is True
        assert any("greeter.py" in f["path"] for f in result["files"])

    def test_session_event_lands_in_the_right_session(
        self, ctx: _FakeRunContext, deps: ContextEngineDeps
    ):
        result = session_event(ctx, "ran_tests", {"passed": True})
        assert result["ok"] is True
        brief = deps.engine.tools.get_context_brief(deps.session_id)
        assert "ran_tests" in brief["brief"]["text"]

    def test_record_decision(self, ctx: _FakeRunContext, deps: ContextEngineDeps):
        result = record_decision(ctx, "use bcrypt", rationale="already a dependency")
        assert result["ok"] is True
        brief = deps.engine.tools.get_context_brief(deps.session_id)
        assert "use bcrypt" in brief["brief"]["text"]

    def test_record_code_area(self, ctx: _FakeRunContext, deps: ContextEngineDeps):
        result = record_code_area(ctx, "greeter.py", note="entry point")
        assert result["ok"] is True
        brief = deps.engine.tools.get_context_brief(deps.session_id)
        assert "greeter.py" in brief["brief"]["text"]

    def test_get_context_brief(self, ctx: _FakeRunContext):
        record_decision(ctx, "use bcrypt")
        result = get_context_brief(ctx)
        assert result["ok"] is True
        assert "use bcrypt" in result["brief"]["text"]

    def test_different_sessions_do_not_see_each_others_decisions(
        self, engine: ContextEngine
    ):
        session_a = engine.new_session()
        session_b = engine.new_session()
        record_decision(
            _FakeRunContext(ContextEngineDeps(engine, session_a.id)), "A's decision"
        )

        brief_b = get_context_brief(
            _FakeRunContext(ContextEngineDeps(engine, session_b.id))
        )
        assert "A's decision" not in brief_b["brief"]["text"]


class TestContextEngineTools:
    def test_seven_tools_registered(self):
        assert len(CONTEXT_ENGINE_TOOLS) == 7

    def test_expected_names(self):
        names = {
            tool.function_schema.function.__name__ for tool in CONTEXT_ENGINE_TOOLS
        }
        assert names == {
            "search_context",
            "expand_chunk",
            "list_files",
            "session_event",
            "record_decision",
            "record_code_area",
            "get_context_brief",
        }

    def test_all_take_context(self):
        assert all(isinstance(tool, Tool) for tool in CONTEXT_ENGINE_TOOLS)

    def test_no_write_or_execute_tools(self):
        names = {
            tool.function_schema.function.__name__ for tool in CONTEXT_ENGINE_TOOLS
        }
        assert "write_file" not in names
        assert "run_code" not in names

    def test_docstrings_produce_parameter_descriptions(self):
        """Sanity-checks that pydantic-ai's docstring parsing actually
        picked up the Args: entries -- if a docstring format ever silently
        stops parsing (e.g. after an unrelated pydantic-ai upgrade), this
        is what would catch it."""
        search_tool = next(
            t
            for t in CONTEXT_ENGINE_TOOLS
            if t.function_schema.function.__name__ == "search_context"
        )
        schema = search_tool.function_schema.json_schema
        assert "description" in schema["properties"]["query"]
        assert len(schema["properties"]["query"]["description"]) > 0

    def test_every_tool_and_parameter_has_a_description(self):
        for tool in CONTEXT_ENGINE_TOOLS:
            name = tool.function_schema.function.__name__
            assert tool.function_schema.description, f"{name} has no description"
            props = tool.function_schema.json_schema.get("properties", {})
            for param, spec in props.items():
                assert spec.get("description"), f"{name}.{param} has no description"

    def test_session_id_is_never_exposed_to_the_model(self):
        for tool in CONTEXT_ENGINE_TOOLS:
            props = tool.function_schema.json_schema.get("properties", {})
            assert "session_id" not in props


class TestFullAgentRun:
    """The pieces above are tested in isolation; this confirms they still
    work wired into a real pydantic-ai Agent and driven by the real tool
    manager, which is a different code path than calling the functions
    directly (schema validation, the real RunContext, argument coercion)."""

    def test_agent_can_call_every_tool_without_error(
        self, engine: ContextEngine, deps: ContextEngineDeps
    ):
        agent = Agent(
            TestModel(call_tools="all"),
            deps_type=ContextEngineDeps,
            tools=CONTEXT_ENGINE_TOOLS,
        )
        result = agent.run_sync("look into the greeting code", deps=deps)
        assert result.output  # ran to completion without raising

        # every tool call actually reached the real engine/session
        brief = engine.tools.get_context_brief(deps.session_id)
        assert brief["ok"] is True
        assert brief["brief"]["event_count"] >= 1
