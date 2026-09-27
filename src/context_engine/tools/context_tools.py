"""
context_tools.py

The tool surface an agent actually calls. Every method here:

- returns a plain dict, always `{"ok": True, ...}` or `{"ok": False, "error": ...}`
  — never raises, so a bad chunk_id or an unknown session_id can't crash
  whatever agent loop is calling these;
- is audit-logged via `logging_config.AuditLogger` (tool name, args,
  success, duration, a truncated result summary — never full chunk/file
  content, see logging_config's module docstring);
- only reads or records — nothing here can modify the developer's files.
  `write_file` / `run_code` are explicitly not part of this surface; those
  belong to whatever agent-execution layer the developer gates by which
  file they named, not to the context engine.

`search_context` and `list_files` never require a session; passing one
just also logs a `session_event` for that call, which is how a
"read-only, no session" mode (CodeVeto's "let the AI read the project
first" case) still works — those calls simply don't touch session memory.
"""

from __future__ import annotations

import time
from typing import Any

from context_engine.exceptions import ContextEngineError
from context_engine.logging_config import AuditLogger
from context_engine.memory.session_manager import SessionManager
from context_engine.search.hybrid import ContextSearch
from context_engine.storage.chunk_store import ChunkStore
from context_engine.storage.graph_store import GraphStore


class ContextTools:
    def __init__(
        self,
        chunk_store: ChunkStore,
        graph_store: GraphStore,
        search: ContextSearch,
        session_manager: SessionManager,
    ) -> None:
        self.chunk_store = chunk_store
        self.graph_store = graph_store
        self.search = search
        self.session_manager = session_manager
        self._audit = AuditLogger()

    # -- read-only context -------------------------------------------------

    def search_context(
        self, query: str, top_k: int = 5, session_id: str | None = None
    ) -> dict[str, Any]:
        def run():
            hits = self.search.search(query, top_k=top_k)
            return {"ok": True, "results": [h.model_dump() for h in hits]}

        result = self._call(
            "search_context", session_id, {"query": query, "top_k": top_k}, run
        )
        if session_id and result.get("ok"):
            self._log_event_quietly(
                session_id,
                "search_context",
                {"query": query, "result_count": len(result["results"])},
            )
        return result

    def expand_chunk(
        self, chunk_id: str, session_id: str | None = None
    ) -> dict[str, Any]:
        def run():
            chunk = self.chunk_store.get_chunk(chunk_id)
            if chunk is None:
                return {"ok": False, "error": f"No chunk with id {chunk_id!r}."}
            payload = chunk.to_dict()
            node = self.graph_store.node_for_chunk_span(
                chunk.file_metadata.path, chunk.start_line
            )
            if node is not None:
                payload["callers"] = [
                    n.name for n in self.graph_store.callers_of(node.id)
                ]
                payload["callees"] = [
                    n.name for n in self.graph_store.callees_of(node.id)
                ]
            return {"ok": True, "chunk": payload}

        result = self._call("expand_chunk", session_id, {"chunk_id": chunk_id}, run)
        if session_id and result.get("ok"):
            self._log_event_quietly(session_id, "expand_chunk", {"chunk_id": chunk_id})
        return result

    def list_files(
        self, path_prefix: str | None = None, session_id: str | None = None
    ) -> dict[str, Any]:
        def run():
            return {"ok": True, "files": self.chunk_store.list_files(path_prefix)}

        return self._call("list_files", session_id, {"path_prefix": path_prefix}, run)

    # -- session memory ----------------------------------------------------

    def session_event(
        self, session_id: str, event_type: str, data: dict | None = None
    ) -> dict[str, Any]:
        def run():
            event_id = self.session_manager.session_event(session_id, event_type, data)
            return {"ok": True, "event_id": event_id}

        return self._call(
            "session_event",
            session_id,
            {"event_type": event_type, "data": data or {}},
            run,
        )

    def record_decision(
        self,
        session_id: str,
        decision: str,
        rationale: str | None = None,
        related_chunk_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        def run():
            record = self.session_manager.record_decision(
                session_id, decision, rationale, related_chunk_ids
            )
            return {"ok": True, "decision_id": record.id}

        args = {
            "decision": decision,
            "rationale": rationale,
            "related_chunk_ids": related_chunk_ids,
        }
        return self._call("record_decision", session_id, args, run)

    def record_code_area(
        self,
        session_id: str,
        file_path: str,
        chunk_id: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        def run():
            record = self.session_manager.record_code_area(
                session_id, file_path, chunk_id, note
            )
            return {"ok": True, "area_id": record.id}

        args = {"file_path": file_path, "chunk_id": chunk_id, "note": note}
        return self._call("record_code_area", session_id, args, run)

    def get_context_brief(
        self, session_id: str, budget_chars: int | None = None
    ) -> dict[str, Any]:
        def run():
            brief = self.session_manager.get_context_brief(session_id, budget_chars)
            return {"ok": True, "brief": brief.model_dump()}

        return self._call(
            "get_context_brief", session_id, {"budget_chars": budget_chars}, run
        )

    # -- shared call wrapper: timing + audit logging + error containment ---

    def _log_event_quietly(self, session_id: str, event_type: str, data: dict) -> None:
        """Best-effort auto-logging of a read-only call into session
        memory (so it shows up in a later context brief). A bad/unknown
        session_id here must not turn an otherwise-successful
        search_context/expand_chunk call into a failure."""
        try:
            self.session_manager.session_event(session_id, event_type, data)
        except ContextEngineError:
            pass

    def _call(
        self, tool_name: str, session_id: str | None, args: dict, run
    ) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            result = run()
        except ContextEngineError as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            self._audit.log_call(
                tool_name,
                session_id,
                args,
                ok=False,
                duration_ms=duration_ms,
                error=str(exc),
            )
            return {"ok": False, "error": str(exc)}

        duration_ms = (time.perf_counter() - start) * 1000
        summary = _summarize(result)
        self._audit.log_call(
            tool_name,
            session_id,
            args,
            ok=result.get("ok", False),
            result_summary=summary,
            duration_ms=duration_ms,
        )
        return result


def _summarize(result: dict[str, Any]) -> dict[str, Any]:
    """What goes in the audit log for a successful call — counts and ids,
    never full chunk/file content."""
    if "results" in result:
        return {"result_count": len(result["results"])}
    if "files" in result:
        return {"file_count": len(result["files"])}
    if "chunk" in result:
        return {"chunk_id": result["chunk"].get("id")}
    if "brief" in result:
        return {"chars": len(result["brief"].get("text", ""))}
    return {k: v for k, v in result.items() if k != "ok"}
