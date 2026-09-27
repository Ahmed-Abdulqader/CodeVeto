from __future__ import annotations

import pytest

from context_engine.exceptions import SessionNotFoundError
from context_engine.memory.session_manager import SessionManager
from context_engine.memory.summarizer import build_brief, compress_events
from context_engine.storage.session_store import SessionStore
from models.context_engine_models import CodeArea, Decision, SessionEvent


def _event(event_type: str) -> SessionEvent:
    return SessionEvent(session_id="s1", event_type=event_type)


class TestCompressEvents:
    def test_collapses_consecutive_duplicates(self):
        events = [
            _event("search"),
            _event("search"),
            _event("search"),
            _event("expand"),
        ]
        compressed = compress_events(events)
        assert [e.event_type for e in compressed] == ["search", "expand"]
        assert compressed[0].data["_repeated"] == 3

    def test_respects_max_items(self):
        events = [_event(f"type{i}") for i in range(20)]
        compressed = compress_events(events, max_items=5)
        assert len(compressed) == 5

    def test_empty_input(self):
        assert compress_events([]) == []


class TestBuildBrief:
    def test_empty_everything(self):
        assert (
            build_brief([], [], [], budget_chars=500)
            == "No prior activity recorded for this session."
        )

    def test_includes_decisions_and_areas(self):
        decisions = [
            Decision(session_id="s1", decision="use bcrypt", rationale="secure")
        ]
        areas = [CodeArea(session_id="s1", file_path="auth.py", note="entry point")]
        text = build_brief(decisions, areas, [], budget_chars=1000)
        assert "use bcrypt" in text
        assert "secure" in text
        assert "auth.py" in text
        assert "entry point" in text

    def test_truncates_to_budget(self):
        decisions = [Decision(session_id="s1", decision="x" * 500)]
        text = build_brief(decisions, [], [], budget_chars=50)
        assert len(text) <= 50
        assert text.endswith("(truncated to fit budget)")


class TestSessionManager:
    @pytest.fixture
    def manager(self, db) -> SessionManager:
        return SessionManager(SessionStore(db), brief_budget_chars=2000)

    def test_start_session_creates_row(self, manager):
        session = manager.start_session(label="demo")
        assert session.label == "demo"
        assert manager.store.session_exists(session.id)

    def test_operations_on_unknown_session_raise(self, manager):
        with pytest.raises(SessionNotFoundError):
            manager.session_event("nonexistent", "search")
        with pytest.raises(SessionNotFoundError):
            manager.record_decision("nonexistent", "some decision")
        with pytest.raises(SessionNotFoundError):
            manager.record_code_area("nonexistent", "a.py")
        with pytest.raises(SessionNotFoundError):
            manager.get_context_brief("nonexistent")

    def test_full_round_trip_into_brief(self, manager):
        session = manager.start_session()
        manager.session_event(session.id, "search_context", {"query": "auth"})
        manager.record_decision(session.id, "use bcrypt", rationale="industry standard")
        manager.record_code_area(
            session.id, "auth.py", note="password hashing lives here"
        )

        brief = manager.get_context_brief(session.id)
        assert brief.session_id == session.id
        assert brief.decision_count == 1
        assert brief.code_area_count == 1
        assert brief.event_count == 1
        assert "use bcrypt" in brief.text
        assert "auth.py" in brief.text

    def test_brief_budget_override(self, manager):
        session = manager.start_session()
        manager.record_decision(session.id, "x" * 1000)
        brief = manager.get_context_brief(session.id, budget_chars=30)
        assert len(brief.text) <= 30
