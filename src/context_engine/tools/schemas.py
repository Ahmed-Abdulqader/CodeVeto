"""
schemas.py

Plain JSON-schema descriptions of every tool `ContextTools` exposes. This
is intentionally *not* tied to pydantic-ai (or any other agent framework)
— just the name/description/parameters shape most tool-calling APIs
(OpenAI, Anthropic, MCP) expect, so wiring these into whichever framework
CodeVeto's agent layer ends up using is a lookup, not a rewrite.

No `write_file` / `run_code` here on purpose — those are agent-execution
tools gated by which file the developer named, which is explicitly outside
what the context engine does.
"""

from __future__ import annotations

from typing import Any

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "search_context",
        "description": (
            "Search the indexed codebase for functions, methods, or classes "
            "relevant to a natural-language query. Returns short previews, "
            "not full code — call expand_chunk on a result to get the full "
            "body."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
                "session_id": {
                    "type": "string",
                    "description": (
                        "Optional. If given, this search is logged to session memory."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "expand_chunk",
        "description": (
            "Get the full content of a chunk returned by search_context or "
            "list_files, including its callers and callees from the symbol "
            "graph where available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "session_id": {"type": "string", "description": "Optional."},
            },
            "required": ["chunk_id"],
        },
    },
    {
        "name": "list_files",
        "description": "List indexed files, optionally filtered by a path prefix.",
        "parameters": {
            "type": "object",
            "properties": {
                "path_prefix": {
                    "type": "string",
                    "description": "Optional path prefix filter.",
                },
                "session_id": {"type": "string", "description": "Optional."},
            },
        },
    },
    {
        "name": "session_event",
        "description": (
            "Record that something happened in this session (e.g. a file was "
            "read, a search was run, a test was executed). Freeform data payload."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "event_type": {"type": "string"},
                "data": {
                    "type": "object",
                    "description": "Optional freeform metadata.",
                },
            },
            "required": ["session_id", "event_type"],
        },
    },
    {
        "name": "record_decision",
        "description": (
            "Record a decision made during this session, e.g. an approach "
            "chosen while fixing a bug, so future turns don't have to "
            "re-derive it from raw history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "decision": {"type": "string"},
                "rationale": {"type": "string", "description": "Optional."},
                "related_chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional chunk ids this decision relates to.",
                },
            },
            "required": ["session_id", "decision"],
        },
    },
    {
        "name": "record_code_area",
        "description": (
            "Flag a file (and optionally a specific chunk) as relevant to "
            "this session, so it surfaces in future context briefs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "file_path": {"type": "string"},
                "chunk_id": {"type": "string", "description": "Optional."},
                "note": {"type": "string", "description": "Optional."},
            },
            "required": ["session_id", "file_path"],
        },
    },
    {
        "name": "get_context_brief",
        "description": (
            "Get a condensed, budget-capped summary of this session's decisions, "
            "flagged code areas, and recent activity — use this instead of raw "
            "chat history to re-establish context at the start of a turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "budget_chars": {
                    "type": "integer",
                    "description": "Optional override.",
                },
            },
            "required": ["session_id"],
        },
    },
]
