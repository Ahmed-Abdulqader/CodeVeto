import ast
import json
import sqlite3

from context_engine.ast_chunker import ASTVisitor, TopLevelCodeVisitor, get_file_data
from models.chunking_models import (
    Classes,
    ClassInheritance,
    Files,
    FunctionsAndMethods,
    Imports,
    Methods,
    TopLevelChunks,
)
from context_engine.config import (
    DATABASE_PATH,
    FUNC_COLS,
    FUNC_VALS
)



def init_database(db_path):
    

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executescript("""
PRAGMA foreign_keys = ON;  -- SQLite ignores ON DELETE rules without this (run on every connection)

-- 1. Files
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    content_hash TEXT,
    last_modified REAL
);

-- 2. Classes (belongs to a file)
CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    docstring TEXT,
    start_line INTEGER,
    end_line INTEGER,
    decorators TEXT,              -- JSON array, e.g. '["dataclass"]'
    code TEXT
);

-- 3. Class inheritance (one row per base class)
CREATE TABLE IF NOT EXISTS class_inheritance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    parent_class_name TEXT NOT NULL,
    parent_class_id INTEGER REFERENCES classes(id) ON DELETE SET NULL  -- NULL if parent is external/unresolved
);

-- 4. Functions (top-level only, belongs to a file)
CREATE TABLE IF NOT EXISTS functions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    type TEXT NOT NULL CHECK (type IN ('function', 'async_function')),
    name TEXT NOT NULL,
    docstring TEXT,
    returns TEXT,
    start_line INTEGER,
    end_line INTEGER,
    start_column INTEGER,
    end_column INTEGER,
    args TEXT,                   -- JSON: [{"name": "x", "type": "int"}, ...]
    calls TEXT,                  -- JSON: ["print", "append", ...]
    decorators TEXT ,             -- JSON array
    code TEXT
);

-- 5. Methods (belongs to a class; file_id kept for fast per-file queries)
CREATE TABLE IF NOT EXISTS methods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    type TEXT NOT NULL CHECK (type IN ('function', 'async_function')),
    name TEXT NOT NULL,
    docstring TEXT,
    returns TEXT,
    start_line INTEGER,
    end_line INTEGER,
    start_column INTEGER,
    end_column INTEGER,
    args TEXT,
    calls TEXT,
    decorators TEXT,
    code TEXT
);

-- 6. Imports
CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    import_type TEXT NOT NULL CHECK (import_type IN ('import', 'import_from')),
    module_name TEXT NOT NULL,   -- 'os.path', or '.utils' / '..pkg' for relative imports
    imported_name TEXT,          -- NULL for plain `import x`
    alias TEXT,
    target_file_id INTEGER REFERENCES files(id) ON DELETE SET NULL,
    line_number INTEGER
);

-- 7. (Optional) Top-level code chunks from TopLevelCodeVisitor
CREATE TABLE IF NOT EXISTS top_level_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    category TEXT,
    node_type TEXT,
    start_line INTEGER,
    end_line INTEGER,
    code TEXT
);

CREATE INDEX IF NOT EXISTS idx_classes_file        ON classes(file_id);
CREATE INDEX IF NOT EXISTS idx_inheritance_class   ON class_inheritance(class_id);
CREATE INDEX IF NOT EXISTS idx_inheritance_parent  ON class_inheritance(parent_class_id);
CREATE INDEX IF NOT EXISTS idx_functions_file      ON functions(file_id);
CREATE INDEX IF NOT EXISTS idx_methods_class       ON methods(class_id);
CREATE INDEX IF NOT EXISTS idx_methods_file        ON methods(file_id);
CREATE INDEX IF NOT EXISTS idx_imports_source      ON imports(source_file_id);
CREATE INDEX IF NOT EXISTS idx_imports_target      ON imports(target_file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_file         ON top_level_chunks(file_id);
    """)


def _j(value):
    return json.dumps(value)


def save_file(conn: sqlite3.Connection, file_path: str) -> bool:
    """Returns False if the file was unchanged and skipped."""
    conn.execute("PRAGMA foreign_keys = ON")  # must run outside a transaction

    with open(file_path, encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    visitor = ASTVisitor(source)
    visitor.visit(tree)
    top = TopLevelCodeVisitor(source)
    top.visit(tree)

    file_data = Files.model_validate(dict(get_file_data(file_path, source)))

    # `with conn` = commit on success, rollback if anything raises
    with conn:
        cur = conn.cursor()

        row = cur.execute(
            "SELECT content_hash FROM files WHERE file_path = ?", (file_data.file_path,)
        ).fetchone()
        if row and row[0] == file_data.content_hash:
            return False  # unchanged

        # Re-parse: deleting the file row cascades to all of its child rows
        cur.execute("DELETE FROM files WHERE file_path = ?", (file_data.file_path,))
        cur.execute(
            "INSERT INTO files (file_path, file_name, content_hash, last_modified) "
            "VALUES (:file_path, :file_name, :content_hash, :last_modified)",
            file_data.model_dump(),
        )
        file_id = cur.lastrowid

        # classes -> remember (name, start_line) -> class_id
        class_ids: dict[tuple[str, int], int] = {}
        for c in visitor.get_classes():
            cls = Classes.model_validate(c)
            cur.execute(
                "INSERT INTO classes (file_id, name, docstring, start_line, end_line, decorators, code) "
                "VALUES (:file_id, :name, :docstring, :start_line, :end_line, :decorators, :code)",
                cls.to_db_params(file_id),
            )
            class_ids[(cls.name, cls.start_line)] = cur.lastrowid

        # class_inheritance
        for r in visitor.get_class_inheritance():
            inh = ClassInheritance.model_validate(r)
            cur.execute(
                "INSERT INTO class_inheritance (class_id, parent_class_name, parent_class_id) "
                "VALUES (?, ?, ?)",
                (class_ids[(inh.class_name, inh.class_start_line)],
                 inh.parent_class_name, inh.parent_class_id),
            )

        # functions
        for fn in visitor.get_functions():
            func = FunctionsAndMethods.model_validate(fn)
            cur.execute(
                f"INSERT INTO functions (file_id, {FUNC_COLS}) VALUES (:file_id, {FUNC_VALS})",
                func.to_db_params(file_id),
            )

        # methods
        for m in visitor.get_methods():
            meth = Methods.model_validate(m)
            class_id = class_ids[(meth.class_name, meth.class_start_line)]
            cur.execute(
                f"INSERT INTO methods (class_id, file_id, {FUNC_COLS}) "
                f"VALUES (:class_id, :file_id, {FUNC_VALS})",
                meth.to_db_params(file_id, class_id),
            )

        # imports
        for i in visitor.get_imports():
            imp = Imports.model_validate(i)
            cur.execute(
                "INSERT INTO imports (source_file_id, import_type, module_name, imported_name, "
                "alias, target_file_id, line_number) "
                "VALUES (:source_file_id, :import_type, :module_name, :imported_name, "
                ":alias, :target_file_id, :line_number)",
                imp.to_db_params(file_id),
            )

        # top-level chunks
        for ch in top.get_chunks():
            chunk = TopLevelChunks.model_validate(ch)
            cur.execute(
                "INSERT INTO top_level_chunks (file_id, category, node_type, start_line, end_line, code) "
                "VALUES (:file_id, :category, :node_type, :start_line, :end_line, :code)",
                chunk.to_db_params(file_id),
            )

    return True


def resolve_parent_classes(conn: sqlite3.Connection):
    """Run after all files are saved. Prefers a parent defined in the same file."""
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("""
        UPDATE class_inheritance
        SET parent_class_id = (
            SELECT p.id FROM classes p
            WHERE p.name = class_inheritance.parent_class_name
            ORDER BY p.file_id = (SELECT file_id FROM classes
                                  WHERE id = class_inheritance.class_id) DESC
            LIMIT 1
        )
        WHERE parent_class_id IS NULL
    """)
    conn.commit()


if __name__ == "__main__":
    init_database(DATABASE_PATH)

    with sqlite3.connect(str(DATABASE_PATH)) as conn:
        save_file(conn, "chunking_models.py")
        resolve_parent_classes(conn)
