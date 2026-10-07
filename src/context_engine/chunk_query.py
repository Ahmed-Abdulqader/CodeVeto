import json
import os
import sqlite3
from collections import defaultdict

from models.chunking_models import (
    ClassWithParents,
    FileOut,
    FunctionOut,
    ImportOut,
    MethodOut,
    ParentRef,
    TopLevelChunkOut,
)
from context_engine.config import DATABASE_PATH


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _fetch(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    cur = conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _error(message: str) -> str:
    return json.dumps({"error": message})


def _dumps(models: list) -> str:
    return json.dumps([m.model_dump() for m in models], indent=4)


def _get_file_id(conn: sqlite3.Connection, file_path: str) -> int | None:
    """
    Look up a file's id. The saver stores os.path.abspath(file_path),
    so normalize here too or relative paths will never match.
    """
    rows = _fetch(
        conn,
        "SELECT id FROM files WHERE file_path = ?",
        (os.path.abspath(file_path),),
    )
    return rows[0]["id"] if rows else None


def _not_found(file_path: str) -> str:
    return _error(f"File '{file_path}' was not found in the database")


# ----------------------------------------------------------------------
# Queries. Every public function returns a JSON string:
#   - success: a JSON object / list (an empty list means "none found")
#   - failure: {"error": "..."}
# ----------------------------------------------------------------------
def get_file_info(conn: sqlite3.Connection, file_path: str) -> str:
    rows = _fetch(
        conn,
        """
        SELECT id, file_path, file_name, content_hash, last_modified
        FROM files
        WHERE file_path = ?
        """,
        (os.path.abspath(file_path),),
    )
    if not rows:
        return _not_found(file_path)

    return json.dumps(FileOut.model_validate(rows[0]).model_dump(), indent=4)

# needs to be chaged to return only the code
def get_imports(conn: sqlite3.Connection, file_path: str) -> str:
    file_id = _get_file_id(conn, file_path)
    if file_id is None:
        return _not_found(file_path)

    rows = _fetch(
        conn,
        """
        SELECT id, source_file_id, import_type, module_name, imported_name,
               alias, target_file_id, line_number
        FROM imports
        WHERE source_file_id = ?
        ORDER BY line_number
        """,
        (file_id,),
    )
    return _dumps([ImportOut.model_validate(r) for r in rows])


def get_top_level_chunks(conn: sqlite3.Connection, file_path: str) -> str:
    file_id = _get_file_id(conn, file_path)
    if file_id is None:
        return _not_found(file_path)

    rows = _fetch(
        conn,
        """
        SELECT id, file_id, category, node_type, start_line, end_line, code
        FROM top_level_chunks
        WHERE file_id = ?
        ORDER BY start_line
        """,
        (file_id,),
    )
    return _dumps([TopLevelChunkOut.model_validate(r) for r in rows])


# need to add a methods names that are in each clsses
def get_classes(conn: sqlite3.Connection, file_path: str) -> str:
    """All classes in a file, each with its list of parents."""
    file_id = _get_file_id(conn, file_path)
    if file_id is None:
        return _not_found(file_path)

    classes = _fetch(
        conn,
        """
        SELECT id, name, docstring, start_line, end_line, decorators, code
        FROM classes
        WHERE file_id = ?
        ORDER BY start_line
        """,
        (file_id,),
    )
    if not classes:
        return json.dumps([])

    # One query for every parent row in the file, grouped by child class id
    parent_rows = _fetch(
        conn,
        """
        SELECT ci.class_id, ci.parent_class_name, ci.parent_class_id
        FROM class_inheritance ci
        JOIN classes c ON c.id = ci.class_id
        WHERE c.file_id = ?
        """,
        (file_id,),
    )
    parents_by_class: dict[int, list[dict]] = defaultdict(list)
    for p in parent_rows:
        parents_by_class[p["class_id"]].append(
            ParentRef.model_validate(
                {
                    "parent_class_name": p["parent_class_name"],
                    "parent_class_id": p["parent_class_id"],
                }
            ).model_dump()
        )

    result = [
        ClassWithParents.model_validate(
            {**cls, "parents": parents_by_class.get(cls["id"], [])}
        )
        for cls in classes
    ]
    return _dumps(result)


def get_functions(conn: sqlite3.Connection, file_path: str) -> str:
    """Top-level functions of a file."""
    file_id = _get_file_id(conn, file_path)
    if file_id is None:
        return _not_found(file_path)

    rows = _fetch(
        conn,
        """
        SELECT id, type, name, docstring, returns, start_line, end_line,
               start_column, end_column, args, calls, decorators, code
        FROM functions
        WHERE file_id = ?
        ORDER BY start_line
        """,
        (file_id,),
    )
    return _dumps([FunctionOut.model_validate(r) for r in rows])


def get_methods(conn: sqlite3.Connection, class_id: int) -> str:
    """Methods of one class (use the `id` returned by get_classes)."""
    exists = _fetch(conn, "SELECT id FROM classes WHERE id = ?", (class_id,))
    if not exists:
        return _error(f"Class id {class_id} was not found in the database")

    rows = _fetch(
        conn,
        """
        SELECT id, class_id, file_id, type, name, docstring, returns,
               start_line, end_line, start_column, end_column,
               args, calls, decorators, code
        FROM methods
        WHERE class_id = ?
        ORDER BY start_line
        """,
        (class_id,),
    )
    return _dumps([MethodOut.model_validate(r) for r in rows])


if __name__ == "__main__":
    file_path = "/home/ahmed/projects/CodeVeto/minimal/CodeVeto/src/codeveto/test_chunker.py"

    with sqlite3.connect(DATABASE_PATH) as conn:
        print(get_file_info(conn, file_path))
        print(get_imports(conn, file_path))
        print(get_top_level_chunks(conn, file_path))
        print(get_classes(conn, file_path))
        print(get_functions(conn, file_path))
        print(get_methods(conn, 1))
