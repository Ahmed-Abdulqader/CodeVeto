"""
session_store.py

Raw persistence for session memory. This module just reads and writes rows
— it has no opinion about what's worth remembering or how to compress it
into a brief. That logic lives in `memory/session_manager.py` and
`memory/summarizer.py`, which is what `tools.context_tools` actually calls.
"""

from __future__ import annotations

import json

from context_engine.storage.database import Database
from models.context_engine_models import CodeArea, Decision, Session, SessionEvent


class SessionStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    # -- sessions --------------------------------------------------------

    def create_session(self, session: Session) -> Session:
        self.db.execute(
            "INSERT INTO sessions (id, label, created_ts) VALUES (?, ?, ?)",
            (session.id, session.label, session.created_ts),
        )
        return session

    def session_exists(self, session_id: str) -> bool:
        return (
            self.db.query_one("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
            is not None
        )

    # -- events ------------------------------------------------------------

    def add_event(self, event: SessionEvent) -> int:
        cur = self.db.execute(
            "INSERT INTO session_events (session_id, event_type, data, created_ts) "
            "VALUES (?, ?, ?, ?)",
            (
                event.session_id,
                event.event_type,
                json.dumps(event.data),
                event.created_ts,
            ),
        )
        return cur.lastrowid

    def recent_events(self, session_id: str, limit: int = 50) -> list[SessionEvent]:
        rows = self.db.query(
            "SELECT * FROM session_events WHERE session_id = ? "
            "ORDER BY created_ts DESC LIMIT ?",
            (session_id, limit),
        )
        return [
            SessionEvent(
                id=r["id"],
                session_id=r["session_id"],
                event_type=r["event_type"],
                data=json.loads(r["data"]) if r["data"] else {},
                created_ts=r["created_ts"],
            )
            for r in rows
        ]

    # -- decisions -----------------------------------------------------

    def add_decision(self, decision: Decision) -> Decision:
        self.db.execute(
            "INSERT INTO decisions "
            "(id, session_id, decision, rationale, related_chunk_ids, created_ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                decision.id,
                decision.session_id,
                decision.decision,
                decision.rationale,
                json.dumps(decision.related_chunk_ids),
                decision.created_ts,
            ),
        )
        return decision

    def recent_decisions(self, session_id: str, limit: int = 20) -> list[Decision]:
        rows = self.db.query(
            "SELECT * FROM decisions WHERE session_id = ? "
            "ORDER BY created_ts DESC LIMIT ?",
            (session_id, limit),
        )
        return [
            Decision(
                id=r["id"],
                session_id=r["session_id"],
                decision=r["decision"],
                rationale=r["rationale"],
                related_chunk_ids=json.loads(r["related_chunk_ids"])
                if r["related_chunk_ids"]
                else [],
                created_ts=r["created_ts"],
            )
            for r in rows
        ]

    # -- code areas --------------------------------------------------------

    def add_code_area(self, area: CodeArea) -> CodeArea:
        self.db.execute(
            "INSERT INTO code_areas "
            "(id, session_id, file_path, chunk_id, note, created_ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                area.id,
                area.session_id,
                area.file_path,
                area.chunk_id,
                area.note,
                area.created_ts,
            ),
        )
        return area

    def recent_code_areas(self, session_id: str, limit: int = 20) -> list[CodeArea]:
        rows = self.db.query(
            "SELECT * FROM code_areas WHERE session_id = ? "
            "ORDER BY created_ts DESC LIMIT ?",
            (session_id, limit),
        )
        return [
            CodeArea(
                id=r["id"],
                session_id=r["session_id"],
                file_path=r["file_path"],
                chunk_id=r["chunk_id"],
                note=r["note"],
                created_ts=r["created_ts"],
            )
            for r in rows
        ]

    # -- counts (cheap, used by ContextBrief) -----------------------------

    def counts(self, session_id: str) -> dict[str, int]:
        def _count(table: str) -> int:
            row = self.db.query_one(
                f"SELECT COUNT(*) AS n FROM {table} WHERE session_id = ?", (session_id,)
            )
            return row["n"] if row else 0

        return {
            "events": _count("session_events"),
            "decisions": _count("decisions"),
            "code_areas": _count("code_areas"),
        }
