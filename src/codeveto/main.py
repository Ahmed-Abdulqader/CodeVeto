"""
main.py

CodeVeto's entry point.  building an actual
agent means editing `DEFAULT_SYSTEM_PROMPT` and `build_agent()` below, not
wiring up storage, indexing, or tool registration first.

Run it with `uv run codeveto` (see `pyproject.toml`'s `[project.scripts]`)
or `python -m codeveto.main [project_path]`. With no argument it indexes
the current directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic_ai import Agent, UserError

from agents.tools import CONTEXT_ENGINE_TOOLS, ContextEngineDeps
from context_engine import (
    ContextEngine,
    EngineConfig,
    ModelDownloadError,
    ensure_model_installed,
    global_config_dir,
    is_jina_code_model_installed,
)
from context_engine.model_downloader import DownloadProgress

DEFAULT_SYSTEM_PROMPT = (
    "You are CodeVeto, an AppSec-focused coding assistant. You have tools "
    "to search, read, and remember context about the developer's codebase: "
    "search_context, expand_chunk, list_files, session_event, "
    "record_decision, record_code_area, get_context_brief. You do NOT have "
    "tools to edit files or run code -- the developer controls that "
    "separately and will tell you which file to work on. Use "
    "search_context and expand_chunk before making claims about the code "
    "rather than guessing. Use record_decision and record_code_area when "
    "something worth remembering for later turns comes up."
)

# Override to use a real model once you're ready, e.g.
# CODEVETO_MODEL="anthropic:claude-sonnet-4-5" (plus the matching API key
# env var). Left unset, this runs on pydantic-ai's built-in TestModel --
# no API key, no real LLM calls, just enough to confirm the whole pipeline
# (engine -> tools -> agent) is wired correctly end to end.
MODEL_ENV_VAR = "CODEVETO_MODEL"
DEFAULT_MODEL = "test"


def ensure_global_config_dir() -> None:
    """The literal ask: make sure ~/.codeveto exists before anything else
    tries to read from or write to it."""
    config_dir = global_config_dir()
    if not config_dir.exists():
        print(f"Setting up CodeVeto's config directory at {config_dir} ...")
        config_dir.mkdir(parents=True, exist_ok=True)


def ensure_embedding_model() -> bool:
    """Downloads the local embedding model into ~/.codeveto/models/ if
    it's not already there. Returns whether semantic search is available
    afterwards -- either way, the program keeps going: a failed or
    skipped download only changes search_context's result quality, it
    never blocks startup."""
    if is_jina_code_model_installed():
        return True

    print(
        "Local embedding model not found -- downloading "
        "jina-embeddings-v2-base-code (~162MB, plus a few MB for its "
        f"tokenizer) into {global_config_dir() / 'models'}\n"
        "This happens once; every other CodeVeto project reuses this copy."
    )

    def report(progress: DownloadProgress) -> None:
        if progress.total_bytes:
            pct = 100 * progress.downloaded_bytes / progress.total_bytes
            print(f"\r  {progress.label}: {pct:5.1f}%", end="", flush=True)
        else:
            mb = progress.downloaded_bytes / (1024 * 1024)
            print(f"\r  {progress.label}: {mb:6.1f} MB downloaded", end="", flush=True)

    try:
        ensure_model_installed(progress_callback=report)
        print("\nDone -- semantic search is available.")
        return True
    except ModelDownloadError as exc:
        print(f"\n\nCouldn't download the embedding model automatically: {exc}\n")
        print(exc.manual_instructions)
        print(
            "\nContinuing without semantic search for now -- search_context "
            "still works on keyword search alone. Re-run this program after "
            "installing the files above and it'll pick them up."
        )
        return False


def build_engine(project_root: Path) -> ContextEngine:
    config = EngineConfig(project_root=project_root)
    engine = ContextEngine(config)
    print(f"Indexing {project_root} ...")
    stats = engine.index()
    print(
        f"  indexed {stats['indexed']}, unchanged {stats['unchanged']}, "
        f"failed {stats['failed']}"
    )
    return engine


def build_agent() -> Agent:
    """Resolves CODEVETO_MODEL into a real model if it's set and usable,
    falling back to the test model if it's unset or the provider rejects
    construction (e.g. a missing API key) -- pydantic-ai validates a
    model string eagerly at Agent(...) construction time, not lazily on
    first run, so that failure has to be caught here rather than around
    the first agent.run_sync() call."""
    model_name = os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL)
    try:
        return Agent(
            model_name,
            deps_type=ContextEngineDeps,
            tools=CONTEXT_ENGINE_TOOLS,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
    except UserError as exc:
        print(f"Could not set up model {model_name!r}: {exc}")
        print(
            f"Falling back to the built-in test model (no API key needed, "
            f"no real LLM calls). Set {MODEL_ENV_VAR} (e.g. "
            "'anthropic:claude-sonnet-4-5') plus its API key env var to "
            "use a real model."
        )
        return Agent(
            DEFAULT_MODEL,
            deps_type=ContextEngineDeps,
            tools=CONTEXT_ENGINE_TOOLS,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )


def run_repl(agent: Agent, deps: ContextEngineDeps) -> None:
    print("\nCodeVeto is ready. Type a message, or 'exit' to quit.\n")
    message_history = None
    while True:
        try:
            user_input = input("> ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        try:
            result = agent.run_sync(
                user_input, deps=deps, message_history=message_history
            )
        except Exception as exc:
            # A bad turn (model error, tool exception that somehow escaped
            # the ok/error contract in context_tools.py) shouldn't kill the
            # whole REPL.
            print(f"(error: {exc})")
            continue

        print(result.output)
        message_history = result.all_messages()


def main() -> None:
    ensure_global_config_dir()
    ensure_embedding_model()

    project_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    engine = build_engine(project_root)
    session = engine.new_session()
    agent = build_agent()
    deps = ContextEngineDeps(engine=engine, session_id=session.id)

    try:
        run_repl(agent, deps)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
