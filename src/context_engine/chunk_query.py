"""Read-only queries over the code index, rendered as Markdown for the AI agent.

Every public function returns a Markdown string:
  - success: code in fenced blocks, with a short header per item
  - failure / nothing found: a plain sentence ("**Error:** ..." for errors)

Nothing here returns JSON.
"""

import functools
import os
import re
import sqlite3
import textwrap
from collections import defaultdict

from context_engine.ast_chunker import normalize_path
from models.chunking_models import (
    ClassOut,
    Definition,
    FunctionsAndMethods,
    Imports,
    TopLevelChunks,
)


# ----------------------------------------------------------------------
# Token counting (only used to pick the cheaper of two renderings)
# ----------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception:  # not installed, or its vocabulary can't be downloaded offline
        return None


def count_tokens(text: str) -> int:
    """Token count of `text`; falls back to ~4 characters per token."""
    encoder = _encoder()
    if encoder is None:
        return len(text) // 4 + 1
    return len(encoder.encode(text))


# ----------------------------------------------------------------------
# Markdown helpers
# ----------------------------------------------------------------------
def error(message: str) -> str:
    return f"**Error:** {message}"


def _fence(code: str, lang: str = "python") -> str:
    """Fenced code block, with a fence long enough to survive backticks in `code`."""
    longest = max((len(run) for run in re.findall(r"`+", code)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{lang}\n{code}\n{ticks}"


def _lines_label(start: int, end: int) -> str:
    return f"line {start}" if start == end else f"lines {start}-{end}"


def _names(names: list[str], limit: int = 30) -> str:
    unique = list(dict.fromkeys(names))
    shown = ", ".join(f"`{n}`" for n in unique[:limit])
    if len(unique) > limit:
        shown += f" (and {len(unique) - limit} more)"
    return shown


def _heading(kind: str, label: str, d: Definition) -> str:
    return f"### {kind} `{label}` ({_lines_label(d.start_line, d.end_line)})"


def _outline(d: Definition) -> str:
    """Signature + docstring, with the body elided."""
    lines = [d.signature]
    if d.docstring:
        lines.append(textwrap.indent(f'"""{d.docstring}"""', "    "))
    lines.append("    ...")
    return "\n".join(lines)


def _item(heading: str, outline: str, code: str) -> str:
    """Render one definition as whichever form costs fewer tokens.

    For big definitions that is the outline; for tiny ones the full source is
    both shorter and more informative. A full-source item says so in its heading.
    """
    full = _fence(code)
    if count_tokens(outline) < count_tokens(full):
        return f"{heading}\n{outline}"
    return f"{heading} — full source\n{full}"


def _code_item(heading: str, d: Definition) -> str:
    return f"{heading}\n{_fence(d.code)}"


# ----------------------------------------------------------------------
# DB helpers
# ----------------------------------------------------------------------
def _fetch(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    cur = conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _get_file_id(conn: sqlite3.Connection, file_path: str) -> int | None:
    """The saver stores the normalized path, so normalize here too."""
    rows = _fetch(
        conn,
        "SELECT id FROM files WHERE file_path = ?",
        (normalize_path(file_path),),
    )
    return rows[0]["id"] if rows else None


def _not_found(file_path: str) -> str:
    return error(f"File `{file_path}` was not found in the index")


def _base(file_path: str) -> str:
    return os.path.basename(file_path)


_DEF_SELECT = "name, signature, docstring, start_line, end_line, code"


def _classes(
    conn: sqlite3.Connection, file_id: int, name: str | None = None
) -> list[ClassOut]:
    sql = f"SELECT id, {_DEF_SELECT} FROM classes WHERE file_id = ?"
    params: tuple = (file_id,)
    if name is not None:
        sql += " AND name = ?"
        params += (name,)
    rows = _fetch(conn, sql + " ORDER BY start_line", params)
    return [ClassOut.model_validate(r) for r in rows]


def _functions(
    conn: sqlite3.Connection, file_id: int, name: str | None = None
) -> list[FunctionsAndMethods]:
    sql = f"SELECT {_DEF_SELECT} FROM functions WHERE file_id = ?"
    params: tuple = (file_id,)
    if name is not None:
        sql += " AND name = ?"
        params += (name,)
    rows = _fetch(conn, sql + " ORDER BY start_line", params)
    return [FunctionsAndMethods.model_validate(r) for r in rows]


def _methods(
    conn: sqlite3.Connection, class_id: int, name: str | None = None
) -> list[FunctionsAndMethods]:
    sql = f"SELECT {_DEF_SELECT} FROM methods WHERE class_id = ?"
    params: tuple = (class_id,)
    if name is not None:
        sql += " AND name = ?"
        params += (name,)
    rows = _fetch(conn, sql + " ORDER BY start_line", params)
    return [FunctionsAndMethods.model_validate(r) for r in rows]


def _method_names_by_class(
    conn: sqlite3.Connection, file_id: int
) -> dict[int, list[str]]:
    rows = _fetch(
        conn,
        "SELECT class_id, name FROM methods WHERE file_id = ? ORDER BY start_line",
        (file_id,),
    )
    names: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        names[r["class_id"]].append(r["name"])
    return names


def _class_not_found(conn, file_id: int, file_path: str, class_name: str) -> str:
    available = [c.name for c in _classes(conn, file_id)]
    hint = (
        f" Classes in this file: {_names(available)}."
        if available
        else " This file has no classes."
    )
    return error(f"Class `{class_name}` was not found in `{_base(file_path)}`.{hint}")


# ----------------------------------------------------------------------
# Queries
# ----------------------------------------------------------------------
def get_imports(conn: sqlite3.Connection, file_path: str) -> str:
    """The module-level import statements of a file: code only."""
    file_id = _get_file_id(conn, file_path)
    if file_id is None:
        return _not_found(file_path)

    rows = _fetch(
        conn,
        "SELECT start_line, end_line, code FROM imports "
        "WHERE file_id = ? ORDER BY start_line",
        (file_id,),
    )
    if not rows:
        return f"`{_base(file_path)}` has no module-level imports."
    imports = [Imports.model_validate(r) for r in rows]
    return _fence("\n".join(i.code for i in imports))


def get_top_level_chunks(conn: sqlite3.Connection, file_path: str) -> str:
    """Module-level code that is not an import, function or class: code only."""
    file_id = _get_file_id(conn, file_path)
    if file_id is None:
        return _not_found(file_path)

    rows = _fetch(
        conn,
        "SELECT start_line, end_line, code FROM top_level_chunks "
        "WHERE file_id = ? ORDER BY start_line",
        (file_id,),
    )
    if not rows:
        return f"`{_base(file_path)}` has no other module-level code."
    chunks = [TopLevelChunks.model_validate(r) for r in rows]
    return _fence("\n\n".join(c.code for c in chunks))


def get_classes(conn: sqlite3.Connection, file_path: str) -> str:
    """Every class of a file: an outline (header, docstring, methods) or full code."""
    file_id = _get_file_id(conn, file_path)
    if file_id is None:
        return _not_found(file_path)

    classes = _classes(conn, file_id)
    if not classes:
        return f"`{_base(file_path)}` has no classes."

    method_names = _method_names_by_class(conn, file_id)
    parts = [f"## Classes in `{_base(file_path)}`"]
    for cls in classes:
        outline = _fence(_outline(cls))
        names = method_names.get(cls.id, [])
        if names:
            outline += f"\nMethods: {_names(names)}"
        parts.append(_item(_heading("class", cls.name, cls), outline, cls.code))
    return "\n\n".join(parts)


def get_functions(conn: sqlite3.Connection, file_path: str) -> str:
    """Every top-level function of a file: an outline or the full code."""
    file_id = _get_file_id(conn, file_path)
    if file_id is None:
        return _not_found(file_path)

    functions = _functions(conn, file_id)
    if not functions:
        return f"`{_base(file_path)}` has no top-level functions."

    parts = [f"## Functions in `{_base(file_path)}`"]
    for fn in functions:
        parts.append(
            _item(_heading("function", fn.name, fn), _fence(_outline(fn)), fn.code)
        )
    return "\n\n".join(parts)


def get_class_methods(
    conn: sqlite3.Connection, file_path: str, class_name: str
) -> str:
    """The methods of a class: outline (signature, docstring) or full source."""
    file_id = _get_file_id(conn, file_path)
    if file_id is None:
        return _not_found(file_path)

    classes = _classes(conn, file_id, class_name)
    if not classes:
        return _class_not_found(conn, file_id, file_path, class_name)

    parts = []
    for cls in classes:
        label = _lines_label(cls.start_line, cls.end_line)
        methods = _methods(conn, cls.id)
        if not methods:
            parts.append(f"Class `{cls.name}` ({label}) has no methods.")
            continue
        parts.append(f"## Methods of `{cls.name}` ({label})")
        for m in methods:
            heading = _heading("method", f"{cls.name}.{m.name}", m)
            parts.append(_item(heading, _fence(_outline(m)), m.code))
    return "\n\n".join(parts)


def get_function_code(
    conn: sqlite3.Connection, file_path: str, function_name: str
) -> str:
    """Full source of a top-level function (all definitions sharing that name)."""
    file_id = _get_file_id(conn, file_path)
    if file_id is None:
        return _not_found(file_path)

    functions = _functions(conn, file_id, function_name)
    if not functions:
        available = [f.name for f in _functions(conn, file_id)]
        hint = (
            f" Functions in this file: {_names(available)}."
            if available
            else " This file has no top-level functions."
        )
        return error(
            f"Function `{function_name}` was not found in `{_base(file_path)}`.{hint}"
        )
    return "\n\n".join(
        _code_item(_heading("function", f.name, f), f) for f in functions
    )


def get_class_code(conn: sqlite3.Connection, file_path: str, class_name: str) -> str:
    """Full source of a class, methods included."""
    file_id = _get_file_id(conn, file_path)
    if file_id is None:
        return _not_found(file_path)

    classes = _classes(conn, file_id, class_name)
    if not classes:
        return _class_not_found(conn, file_id, file_path, class_name)
    return "\n\n".join(_code_item(_heading("class", c.name, c), c) for c in classes)


def get_method_code(
    conn: sqlite3.Connection, file_path: str, class_name: str, method_name: str
) -> str:
    """Full source of one method. Getter/setter pairs return every definition."""
    file_id = _get_file_id(conn, file_path)
    if file_id is None:
        return _not_found(file_path)

    classes = _classes(conn, file_id, class_name)
    if not classes:
        return _class_not_found(conn, file_id, file_path, class_name)

    parts = []
    for cls in classes:
        parts.extend(
            _code_item(_heading("method", f"{cls.name}.{m.name}", m), m)
            for m in _methods(conn, cls.id, method_name)
        )
    if parts:
        return "\n\n".join(parts)

    available = [m.name for cls in classes for m in _methods(conn, cls.id)]
    hint = (
        f" Methods of this class: {_names(available)}."
        if available
        else " This class has no methods."
    )
    return error(f"Method `{method_name}` was not found in class `{class_name}`.{hint}")
