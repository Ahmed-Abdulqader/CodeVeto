"""
tools.py

The context engine's tools, wired for pydantic-ai. Each function below is
a plain, type-annotated function taking `ctx: RunContext[ContextEngineDeps]`
as its first argument -- pydantic-ai's pattern for a context-aware tool --
wrapped in a `Tool(...)` so an `Agent` can be built with:

    from agents.tools import CONTEXT_ENGINE_TOOLS, ContextEngineDeps

    agent = Agent(model, deps_type=ContextEngineDeps, tools=CONTEXT_ENGINE_TOOLS)
    agent.run_sync("...", deps=ContextEngineDeps(engine=engine, session_id=session.id))

`session_id` is deliberately NOT a parameter the model fills in on any of
these -- it lives on `ContextEngineDeps` instead, set once when a session
starts (see `codeveto/main.py`). The model shouldn't have to know or
invent session ids; that's infrastructure, not something worth spending
its attention on.

No `write_file` / `run_code` here -- those are agent-execution tools
gated by which file the developer named, a decision for whatever agent
gets built on top of this, not the context engine itself.

Docstrings below are Google-style and parsed by pydantic-ai
(`docstring_format="auto"`, the `Tool` default) to build each tool's JSON
schema: the summary line becomes the tool's description, and each `Args:`
entry becomes that parameter's description, with `ctx` excluded
automatically. This was verified against a real `Tool` object and a real
`Agent` run with `pydantic_ai.models.test.TestModel` while building this
file, not assumed from documentation.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext, Tool

from context_engine import ContextEngine


class ContextEngineDeps:
    """Dependency object every tool below receives via `ctx.deps`.
    `session_id` is created once per agent run and reused across every
    tool call in that run, so decisions/code areas/events all land in the
    same session and `get_context_brief` sees all of them together."""

    def __init__(self, engine: ContextEngine, session_id: str) -> None:
        self.engine = engine
        self.session_id = session_id


def search_context(
    ctx: RunContext[ContextEngineDeps], query: str, top_k: int = 5
) -> dict[str, Any]:
    """Search the indexed codebase for code relevant to a query.

    Returns short previews, not full code -- call expand_chunk on a
    result's chunk_id to get the full body before reasoning about or
    changing it.

    Args:
        query: What to search for, in natural language, e.g.
            "password hashing" or "the retry loop for HTTP calls".
        top_k: Maximum number of results to return.
    """
    return ctx.deps.engine.tools.search_context(
        query, top_k=top_k, session_id=ctx.deps.session_id
    )


def expand_chunk(ctx: RunContext[ContextEngineDeps], chunk_id: str) -> dict[str, Any]:
    """Get the full content of a chunk found via search_context or list_files.

    Includes its callers and callees from the symbol graph where
    available, so what depends on this code is visible before it changes.

    Args:
        chunk_id: The chunk_id from a previous search_context or expand_chunk result.
    """
    return ctx.deps.engine.tools.expand_chunk(chunk_id, session_id=ctx.deps.session_id)


def list_files(
    ctx: RunContext[ContextEngineDeps], path_prefix: str | None = None
) -> dict[str, Any]:
    """List indexed files in the project, optionally filtered by a path prefix.

    Args:
        path_prefix: Only list files whose path starts with this, e.g.
            "src/auth/". Omit to list every indexed file.
    """
    return ctx.deps.engine.tools.list_files(path_prefix, session_id=ctx.deps.session_id)


def session_event(
    ctx: RunContext[ContextEngineDeps],
    event_type: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record that something happened in this session, e.g. a test ran.

    This is a lightweight activity log, not a place for conclusions -- use
    record_decision for those.

    Args:
        event_type: A short label for what happened, e.g. "ran_tests" or "read_file".
        data: Optional freeform details about the event.
    """
    return ctx.deps.engine.tools.session_event(ctx.deps.session_id, event_type, data)


def record_decision(
    ctx: RunContext[ContextEngineDeps],
    decision: str,
    rationale: str | None = None,
    related_chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Record a decision made this session so later turns need not re-derive it.

    Use this for choices that matter to how the work proceeds, e.g. an
    approach picked while fixing a bug -- not for routine activity, which
    is what session_event is for.

    Args:
        decision: The decision, stated plainly, e.g. "use bcrypt for password hashing".
        rationale: Optional -- why this was chosen.
        related_chunk_ids: Optional chunk_ids this decision relates to.
    """
    return ctx.deps.engine.tools.record_decision(
        ctx.deps.session_id,
        decision,
        rationale=rationale,
        related_chunk_ids=related_chunk_ids,
    )


def record_code_area(
    ctx: RunContext[ContextEngineDeps],
    file_path: str,
    chunk_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Flag a file, and optionally a specific chunk, as relevant to this session.

    Flagged areas surface in future context briefs, so re-reading a whole
    file to remember why it mattered isn't necessary.

    Args:
        file_path: Path of the file to flag.
        chunk_id: Optional -- a specific chunk within that file, if the
            whole file isn't the point.
        note: Optional -- why this area matters.
    """
    return ctx.deps.engine.tools.record_code_area(
        ctx.deps.session_id, file_path, chunk_id=chunk_id, note=note
    )


def get_context_brief(
    ctx: RunContext[ContextEngineDeps], budget_chars: int | None = None
) -> dict[str, Any]:
    """Get a condensed summary of this session: decisions, flagged areas, activity.

    Use this at the start of a turn to re-establish context instead of
    relying on raw conversation history.

    Args:
        budget_chars: Optional override for how many characters the summary can use.
    """
    return ctx.deps.engine.tools.get_context_brief(
        ctx.deps.session_id, budget_chars=budget_chars
    )


CONTEXT_ENGINE_TOOLS: list[Tool] = [
    Tool(search_context, takes_ctx=True),
    Tool(expand_chunk, takes_ctx=True),
    Tool(list_files, takes_ctx=True),
    Tool(session_event, takes_ctx=True),
    Tool(record_decision, takes_ctx=True),
    Tool(record_code_area, takes_ctx=True),
    Tool(get_context_brief, takes_ctx=True),
]
