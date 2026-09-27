"""
session_manager.py

The pieces CodeVeto's design asked for by name: `session_event`,
`record_decision`, `record_code_area`, plus the plumbing they need
(`start_session`) and the payoff (`get_context_brief`) — a condensed,
budget-capped summary handed to the agent instead of raw conversation
history. `tools.context_tools.ContextTools` is a thin wrapper around this
class; this is where the actual logic lives so it stays testable without
going through the tool-calling layer.
"""

from __future__ import annotations

from context_engine.exceptions import SessionNotFoundError
from context_engine.memory.summarizer import build_brief
from context_engine.storage.session_store import SessionStore
from models.context_engine_models import (
    CodeArea,
    ContextBrief,
    Decision,
    Session,
    SessionEvent,
)


class SessionManager:
    def __init__(
        self, session_store: SessionStore, brief_budget_chars: int = 2000
    ) -> None:
        self.store = session_store
        self.brief_budget_chars = brief_budget_chars

    def start_session(self, label: str | None = None) -> Session:
        session = Session(label=label)
        return self.store.create_session(session)

    def _require_session(self, session_id: str) -> None:
        if not self.store.session_exists(session_id):
            raise SessionNotFoundError(
                f"No session {session_id!r}. Call start_session() first."
            )

    def session_event(
        self, session_id: str, event_type: str, data: dict | None = None
    ) -> int:
        self._require_session(session_id)
        event = SessionEvent(
            session_id=session_id, event_type=event_type, data=data or {}
        )
        return self.store.add_event(event)

    def record_decision(
        self,
        session_id: str,
        decision: str,
        rationale: str | None = None,
        related_chunk_ids: list[str] | None = None,
    ) -> Decision:
        self._require_session(session_id)
        record = Decision(
            session_id=session_id,
            decision=decision,
            rationale=rationale,
            related_chunk_ids=related_chunk_ids or [],
        )
        return self.store.add_decision(record)

    def record_code_area(
        self,
        session_id: str,
        file_path: str,
        chunk_id: str | None = None,
        note: str | None = None,
    ) -> CodeArea:
        self._require_session(session_id)
        record = CodeArea(
            session_id=session_id, file_path=file_path, chunk_id=chunk_id, note=note
        )
        return self.store.add_code_area(record)

    def get_context_brief(
        self, session_id: str, budget_chars: int | None = None
    ) -> ContextBrief:
        self._require_session(session_id)
        budget = budget_chars if budget_chars is not None else self.brief_budget_chars
        decisions = self.store.recent_decisions(session_id)
        code_areas = self.store.recent_code_areas(session_id)
        events = self.store.recent_events(session_id)
        text = build_brief(decisions, code_areas, events, budget)
        counts = self.store.counts(session_id)
        return ContextBrief(
            session_id=session_id,
            text=text,
            decision_count=counts["decisions"],
            code_area_count=counts["code_areas"],
            event_count=counts["events"],
        )
