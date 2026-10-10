"""Saves the chunks produced by `ast_chunker` into the SQLite index.

The index only holds what the agent's tools show: code, line ranges and (for
classes / functions / methods) a signature and a docstring.
"""

import ast
import contextlib
import sqlite3

from context_engine.ast_chunker import (
    ASTVisitor,
    TopLevelCodeVisitor,
    get_file_data,
    normalize_path,
)
from models.chunking_models import (
    Classes,
    Files,
    FunctionsAndMethods,
    Imports,
    Methods,
    TopLevelChunks,
)

# Bump when the layout below changes. The index is derived data (it can always be
# rebuilt from the source files), so an old layout is dropped, not migrated.
SCHEMA_VERSION = 2

# children first, so it is safe to drop them in this order
_ALL_TABLES = (
    "class_inheritance",  # removed in schema v2, still dropped from old databases
    "top_level_chunks",
    "imports",
    "methods",
    "functions",
    "classes",
    "files",
)

# Columns shared by classes, functions and methods
_DEF_COLS = "name, signature, docstring, start_line, end_line, code"
_DEF_VALS = ":name, :signature, :docstring, :start_line, :end_line, :code"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL
);

-- module-level import statements (one row per statement)
CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    code TEXT NOT NULL
);

-- every other module-level statement
CREATE TABLE IF NOT EXISTS top_level_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    code TEXT NOT NULL
);

-- signature = decorators + `class ...(bases):` header
CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    signature TEXT NOT NULL,
    docstring TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    code TEXT NOT NULL
);

-- top-level functions; signature = decorators + `def ...` header
CREATE TABLE IF NOT EXISTS functions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    signature TEXT NOT NULL,
    docstring TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    code TEXT NOT NULL
);

-- belongs to a class; file_id kept for fast per-file queries
CREATE TABLE IF NOT EXISTS methods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    signature TEXT NOT NULL,
    docstring TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    code TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_imports_file   ON imports(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_file    ON top_level_chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_classes_file   ON classes(file_id);
CREATE INDEX IF NOT EXISTS idx_functions_file ON functions(file_id);
CREATE INDEX IF NOT EXISTS idx_methods_class  ON methods(class_id);
CREATE INDEX IF NOT EXISTS idx_methods_file   ON methods(file_id);
"""


def init_database(db_path) -> None:
    """Create the index tables, rebuilding the database if it has an old layout."""
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            for table in _ALL_TABLES:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()


def get_indexed_hash(conn: sqlite3.Connection, file_path: str) -> str | None:
    """Content hash the file had when it was last indexed (None if never indexed)."""
    row = conn.execute(
        "SELECT content_hash FROM files WHERE file_path = ?",
        (normalize_path(file_path),),
    ).fetchone()
    return row[0] if row else None


def save_file(conn: sqlite3.Connection, file_path: str) -> bool:
    """Index one Python file.

    Returns False if the file is unchanged since it was last indexed (nothing is
    parsed or written). Raises SyntaxError / ValueError if the file cannot be
    parsed, in which case whatever was indexed before is left untouched.
    """
    # Needed on every connection for ON DELETE CASCADE to work; must run outside
    # a transaction.
    conn.execute("PRAGMA foreign_keys = ON")

    with open(file_path, encoding="utf-8") as f:
        source = f.read()

    file_data = Files.model_validate(get_file_data(file_path, source))

    row = conn.execute(
        "SELECT content_hash FROM files WHERE file_path = ?", (file_data.file_path,)
    ).fetchone()
    if row and row[0] == file_data.content_hash:
        return False  # unchanged

    # Parse before touching the database so a broken file never wipes the old rows.
    tree = ast.parse(source)
    visitor = ASTVisitor(source)
    visitor.visit(tree)
    top = TopLevelCodeVisitor(source)
    top.visit(tree)

    # `with conn` = commit on success, rollback if anything raises
    with conn:
        cur = conn.cursor()

        # Re-parse: deleting the file row cascades to all of its child rows
        cur.execute("DELETE FROM files WHERE file_path = ?", (file_data.file_path,))
        cur.execute(
            "INSERT INTO files (file_path, content_hash) "
            "VALUES (:file_path, :content_hash)",
            file_data.model_dump(),
        )
        file_id = cur.lastrowid

        for i in visitor.get_imports():
            imp = Imports.model_validate(i)
            cur.execute(
                "INSERT INTO imports (file_id, start_line, end_line, code) "
                "VALUES (:file_id, :start_line, :end_line, :code)",
                imp.to_db_params(file_id),
            )

        for ch in top.get_chunks():
            chunk = TopLevelChunks.model_validate(ch)
            cur.execute(
                "INSERT INTO top_level_chunks (file_id, start_line, end_line, code) "
                "VALUES (:file_id, :start_line, :end_line, :code)",
                chunk.to_db_params(file_id),
            )

        # classes -> remember (name, start_line) -> class_id for the methods
        class_ids: dict[tuple[str, int], int] = {}
        for c in visitor.get_classes():
            cls = Classes.model_validate(c)
            cur.execute(
                f"INSERT INTO classes (file_id, {_DEF_COLS}) "
                f"VALUES (:file_id, {_DEF_VALS})",
                cls.to_db_params(file_id),
            )
            class_ids[(cls.name, cls.start_line)] = cur.lastrowid

        for fn in visitor.get_functions():
            func = FunctionsAndMethods.model_validate(fn)
            cur.execute(
                f"INSERT INTO functions (file_id, {_DEF_COLS}) "
                f"VALUES (:file_id, {_DEF_VALS})",
                func.to_db_params(file_id),
            )

        for m in visitor.get_methods():
            meth = Methods.model_validate(m)
            class_id = class_ids[(meth.class_name, meth.class_start_line)]
            cur.execute(
                f"INSERT INTO methods (class_id, file_id, {_DEF_COLS}) "
                f"VALUES (:class_id, :file_id, {_DEF_VALS})",
                meth.to_db_params(file_id, class_id=class_id),
            )

    return True
