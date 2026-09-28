"""
database.py

Thin SQLite connection wrapper. Every store (chunk_store, graph_store,
session_store) and the lexical index take a `Database` instance rather than
opening their own connection, so the whole engine shares one connection,
one set of pragmas, and one schema-creation path.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    """A single SQLite connection, schema applied on first use, shared
    safely across threads.

    `check_same_thread=False` plus the `threading.Lock` below is the
    standard safe pattern for one sqlite3 connection used from more than
    one thread: the lock serializes every actual access, so there's no
    concurrent use of the connection, just use from different threads over
    time. This matters here specifically because pydantic-ai runs
    synchronous tool functions (search_context, record_decision, ...) in
    a worker-thread pool by default, not the thread that constructed the
    `ContextEngine` -- with `check_same_thread` at its default (True),
    every tool call raised `sqlite3.ProgrammingError`. Found this by
    actually running an agent end to end against a real engine, not by
    reasoning about it in the abstract.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.lock = threading.Lock()
        self._apply_schema()

    def _apply_schema(self) -> None:
        schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        with self.lock:
            self.conn.executescript(schema_sql)
            self.conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def executemany(self, sql: str, seq_of_params) -> sqlite3.Cursor:
        with self.lock:
            cur = self.conn.executemany(sql, seq_of_params)
            self.conn.commit()
            return cur

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute(sql, params).fetchone()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
