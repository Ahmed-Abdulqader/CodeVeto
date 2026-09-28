from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_ai.models.test import TestModel

from context_engine import paths
from context_engine.model_downloader import ModelDownloadError

# `codeveto/__init__.py` does `from codeveto.main import main` (so the
# `codeveto:main` entry point in pyproject.toml resolves to a callable) --
# which means `codeveto.main` the *attribute* is the function, not the
# submodule, by the time `codeveto` finishes importing. `importlib` is the
# standard way to reach the actual submodule regardless: it looks it up in
# `sys.modules` rather than doing attribute access on the parent package.
codeveto_main = importlib.import_module("codeveto.main")


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.GLOBAL_CONFIG_DIR_ENV, str(tmp_path / "codeveto_home"))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "greeter.py").write_text("def say_hello(name):\n    return name\n")
    return project_dir


class TestEnsureGlobalConfigDir:
    def test_creates_directory_when_missing(self):
        config_dir = paths.global_config_dir()
        assert not config_dir.exists()
        codeveto_main.ensure_global_config_dir()
        assert config_dir.exists()

    def test_noop_when_already_present(self):
        config_dir = paths.global_config_dir()
        config_dir.mkdir(parents=True)
        marker = config_dir / "marker.txt"
        marker.write_text("keep me")
        codeveto_main.ensure_global_config_dir()
        assert marker.exists()  # untouched, not recreated


class TestEnsureEmbeddingModel:
    def test_returns_true_without_downloading_when_already_installed(self):
        paths.jina_code_model_dir().mkdir(parents=True)
        paths.jina_code_model_path().write_bytes(b"x")
        paths.jina_code_tokenizer_path().write_text("{}")

        with patch.object(codeveto_main, "ensure_model_installed") as mock_ensure:
            result = codeveto_main.ensure_embedding_model()

        assert result is True
        mock_ensure.assert_not_called()

    def test_returns_false_and_does_not_raise_when_download_fails(self, capsys):
        with patch.object(
            codeveto_main,
            "ensure_model_installed",
            side_effect=ModelDownloadError("network is down", "go to this url instead"),
        ):
            result = codeveto_main.ensure_embedding_model()

        assert result is False
        captured = capsys.readouterr()
        assert "go to this url instead" in captured.out
        assert "Continuing without semantic search" in captured.out

    def test_returns_true_when_download_succeeds(self):
        def fake_ensure(progress_callback=None):
            if progress_callback:
                from context_engine.model_downloader import DownloadProgress

                progress_callback(
                    DownloadProgress(
                        label="embedding model", downloaded_bytes=10, total_bytes=100
                    )
                )
            paths.jina_code_model_dir().mkdir(parents=True, exist_ok=True)
            paths.jina_code_model_path().write_bytes(b"x")
            paths.jina_code_tokenizer_path().write_text("{}")

        with patch.object(
            codeveto_main, "ensure_model_installed", side_effect=fake_ensure
        ):
            result = codeveto_main.ensure_embedding_model()

        assert result is True


class TestBuildEngine:
    def test_indexes_the_given_project(self, project: Path, capsys):
        engine = codeveto_main.build_engine(project)
        try:
            result = engine.tools.list_files()
            assert result["ok"] is True
            assert any("greeter.py" in f["path"] for f in result["files"])
        finally:
            engine.close()

        captured = capsys.readouterr()
        assert "Indexing" in captured.out


class TestBuildAgent:
    def _make_deps(self, project: Path):
        from agents.tools import ContextEngineDeps

        engine = codeveto_main.build_engine(project)
        session = engine.new_session()
        return ContextEngineDeps(engine=engine, session_id=session.id), engine

    def test_defaults_to_test_model_with_no_env_var(self, monkeypatch, project: Path):
        monkeypatch.delenv(codeveto_main.MODEL_ENV_VAR, raising=False)
        deps, engine = self._make_deps(project)
        try:
            agent = codeveto_main.build_agent()
            result = agent.run_sync("hello", deps=deps)
            assert result.output  # ran without needing any API key
        finally:
            engine.close()

    def test_falls_back_to_test_model_when_configured_model_is_unusable(
        self, monkeypatch, capsys, project: Path
    ):
        monkeypatch.setenv(codeveto_main.MODEL_ENV_VAR, "openai:gpt-4o")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        deps, engine = self._make_deps(project)

        try:
            agent = codeveto_main.build_agent()
            result = agent.run_sync("hello", deps=deps)  # must not raise
            assert result.output
        finally:
            engine.close()

        captured = capsys.readouterr()
        assert "Falling back to the built-in test model" in captured.out

    def test_uses_explicit_test_model_string_successfully(
        self, monkeypatch, project: Path
    ):
        monkeypatch.setenv(codeveto_main.MODEL_ENV_VAR, "test")
        deps, engine = self._make_deps(project)
        try:
            agent = codeveto_main.build_agent()
            assert agent.run_sync("hello", deps=deps).output
        finally:
            engine.close()


class TestRunRepl:
    def test_exits_on_exit_command(self, monkeypatch, project: Path):
        engine = codeveto_main.build_engine(project)
        try:
            session = engine.new_session()
            from pydantic_ai import Agent

            from agents.tools import CONTEXT_ENGINE_TOOLS, ContextEngineDeps

            deps = ContextEngineDeps(engine=engine, session_id=session.id)
            agent = Agent(
                TestModel(), deps_type=ContextEngineDeps, tools=CONTEXT_ENGINE_TOOLS
            )

            inputs = iter(["exit"])
            monkeypatch.setattr("builtins.input", lambda _: next(inputs))
            codeveto_main.run_repl(agent, deps)  # returns instead of looping forever
        finally:
            engine.close()

    def test_runs_a_turn_and_keeps_conversation_history(
        self, monkeypatch, project: Path, capsys
    ):
        engine = codeveto_main.build_engine(project)
        try:
            session = engine.new_session()
            from pydantic_ai import Agent

            from agents.tools import CONTEXT_ENGINE_TOOLS, ContextEngineDeps

            deps = ContextEngineDeps(engine=engine, session_id=session.id)
            agent = Agent(
                TestModel(), deps_type=ContextEngineDeps, tools=CONTEXT_ENGINE_TOOLS
            )

            inputs = iter(["find the greeting function", "exit"])
            monkeypatch.setattr("builtins.input", lambda _: next(inputs))
            codeveto_main.run_repl(agent, deps)
        finally:
            engine.close()

        captured = capsys.readouterr()
        assert "CodeVeto is ready" in captured.out

    def test_eof_ends_the_repl_without_crashing(self, monkeypatch, project: Path):
        engine = codeveto_main.build_engine(project)
        try:
            session = engine.new_session()
            from pydantic_ai import Agent

            from agents.tools import CONTEXT_ENGINE_TOOLS, ContextEngineDeps

            deps = ContextEngineDeps(engine=engine, session_id=session.id)
            agent = Agent(
                TestModel(), deps_type=ContextEngineDeps, tools=CONTEXT_ENGINE_TOOLS
            )

            def raise_eof(_):
                raise EOFError

            monkeypatch.setattr("builtins.input", raise_eof)
            codeveto_main.run_repl(agent, deps)  # must return, not raise
        finally:
            engine.close()
