import ast
import contextlib
import os
import pathlib
import re
import shutil
import tempfile
import textwrap
from collections.abc import Callable

from pydantic_ai import RunContext

from context_engine import chunk_query
from context_engine.ast_chunker import ASTVisitor
from context_engine.chunks_saver import save_file
from models.agents_models import DatabaseDeps

_error = chunk_query.error


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _file_path_wrapper(file_name: str) -> pathlib.Path | None:
    """Resolve `file_name` to an existing .py file, or None."""
    path = pathlib.Path(file_name.strip()).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".py":
        return path
    return None


def _not_a_python_file(file_name: str) -> str:
    return _error(f"`{file_name}` is not an existing Python (.py) file.")


def _indexed_file(
    ctx: RunContext[DatabaseDeps], file_name: str
) -> tuple[str | None, str | None]:
    """Validate the path and bring the index up to date.

    Returns (resolved_path, None) on success or (None, markdown_error).
    """
    path = _file_path_wrapper(file_name)
    if path is None:
        return None, _not_a_python_file(file_name)
    try:
        save_file(ctx.deps.db, str(path))
    except (OSError, UnicodeDecodeError) as e:  # UnicodeDecodeError is a ValueError too
        return None, _error(f"`{file_name}` could not be read: {e}")
    except (SyntaxError, ValueError) as e:
        line = getattr(e, "lineno", None)
        where = f" (line {line})" if line else ""
        reason = getattr(e, "msg", None) or str(e)
        return None, _error(
            f"`{file_name}` cannot be parsed{where}: {reason}. "
            "Fix it before using the code tools."
        )
    return str(path), None


# ----------------------------------------------------------------------
# Reading tools
# ----------------------------------------------------------------------
def get_file_imports(ctx: RunContext[DatabaseDeps], file_name: str) -> str:
    """Show the module-level import statements of a Python file (code only).

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
    """
    path, err = _indexed_file(ctx, file_name)
    if err:
        return err
    return chunk_query.get_imports(ctx.deps.db, path)


def get_file_top_level_chunk(ctx: RunContext[DatabaseDeps], file_name: str) -> str:
    """Show the module-level code of a Python file that is not an import, function
    or class: constants, `if __name__ == "__main__":` blocks, top-level try/if
    blocks, and so on (code only).

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
    """
    path, err = _indexed_file(ctx, file_name)
    if err:
        return err
    return chunk_query.get_top_level_chunks(ctx.deps.db, path)


def get_file_classes(ctx: RunContext[DatabaseDeps], file_name: str) -> str:
    """List the classes of a Python file with their line ranges.

    Each class is shown as an outline (decorators, bases, docstring and the names
    of its methods) unless its full source is cheaper, in which case the full
    source is shown. Use `get_class_methods_signature` or `expand_class` for more.

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
    """
    path, err = _indexed_file(ctx, file_name)
    if err:
        return err
    return chunk_query.get_classes(ctx.deps.db, path)


def get_class_methods_signature(
    ctx: RunContext[DatabaseDeps], file_name: str, cls_name: str
) -> str:
    """List the methods of one class: signature, docstring and line range of each
    (or the full source when that is cheaper). Use `expand_method` for a method's body.

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
        cls_name: Name of the class, as listed by `get_file_classes`.
    """
    path, err = _indexed_file(ctx, file_name)
    if err:
        return err
    return chunk_query.get_class_methods(ctx.deps.db, path, cls_name)


def get_file_functions(ctx: RunContext[DatabaseDeps], file_name: str) -> str:
    """List the top-level functions of a Python file: signature, docstring and line
    range of each (or the full source when that is cheaper). Methods are not
    included; use `get_class_methods_signature` for those. Use `expand_function`
    for a function's body.

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
    """
    path, err = _indexed_file(ctx, file_name)
    if err:
        return err
    return chunk_query.get_functions(ctx.deps.db, path)


def expand_function(
    ctx: RunContext[DatabaseDeps], file_name: str, function_name: str
) -> str:
    """Show the full source of a top-level function, with its line range.

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
        function_name: Name of the function, as listed by `get_file_functions`.
    """
    path, err = _indexed_file(ctx, file_name)
    if err:
        return err
    return chunk_query.get_function_code(ctx.deps.db, path, function_name)


def expand_class(ctx: RunContext[DatabaseDeps], file_name: str, cls_name: str) -> str:
    """Show the full source of a class (methods included), with its line range.

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
        cls_name: Name of the class, as listed by `get_file_classes`.
    """
    path, err = _indexed_file(ctx, file_name)
    if err:
        return err
    return chunk_query.get_class_code(ctx.deps.db, path, cls_name)


def expand_method(
    ctx: RunContext[DatabaseDeps], file_name: str, cls_name: str, method_name: str
) -> str:
    """Show the full source of one method of a class, with its line range.

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
        cls_name: Name of the class that owns the method.
        method_name: Name of the method, as listed by `get_class_methods_signature`.
    """
    path, err = _indexed_file(ctx, file_name)
    if err:
        return err
    return chunk_query.get_method_code(ctx.deps.db, path, cls_name, method_name)


# ----------------------------------------------------------------------
# Writing: helpers
# ----------------------------------------------------------------------
class _EditError(Exception):
    """An edit that can't be applied. The message is shown to the agent."""


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_CLASSES = (ast.ClassDef,)
_DEFINITIONS = _FUNCTIONS + _CLASSES
_IMPORTS = (ast.Import, ast.ImportFrom)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _split(text: str) -> list[str]:
    """Lines of `text` without newline characters ("" -> [])."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _join(lines: list[str]) -> str:
    return "\n".join(lines) + "\n" if lines else ""


def _indent_of(line: str) -> str:
    return re.match(r"[ \t]*", line).group()


def _quoted(names: list[str], limit: int = 30) -> str:
    unique = list(dict.fromkeys(names))
    shown = ", ".join(f"`{n}`" for n in unique[:limit])
    return shown + (f" (and {len(unique) - limit} more)" if len(unique) > limit else "")


def _block(code: str, indent: str) -> list[str]:
    """`code` as lines: surrounding blank lines dropped, dedented, then indented.

    Dedenting first means the agent may pass code indented or not (a method
    exactly as `expand_method` shows it is fine).
    """
    lines = _normalize_newlines(code).split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return []
    return textwrap.indent(textwrap.dedent("\n".join(lines)), indent).split("\n")


def _describe(node: ast.stmt) -> str:
    if isinstance(node, _FUNCTIONS):
        return f"a function (`{node.name}`)"
    if isinstance(node, _CLASSES):
        return f"a class (`{node.name}`)"
    if isinstance(node, _IMPORTS):
        return "an import statement"
    return f"other code (`{type(node).__name__}`)"


def _parse_new(
    code: str,
    allowed: tuple[type, ...],
    label: str,
    *,
    single: bool = False,
    hint: str = "",
) -> list:
    """Check that `code` is made of `label`s only (e.g. "function definition")."""
    lines = _block(code, "")
    if not lines:
        raise _EditError("`new_code` is empty.")
    try:
        nodes = ast.parse("\n".join(lines)).body
    except SyntaxError as e:
        raise _EditError(
            f"`new_code` is not valid Python (line {e.lineno}: {e.msg})."
        ) from None
    for node in nodes:
        if not isinstance(node, allowed):
            raise _EditError(
                f"`new_code` must contain only {label}s, but it contains "
                f"{_describe(node)}.{hint}"
            )
    if single and len(nodes) != 1:
        raise _EditError(
            f"`new_code` must contain exactly one {label}, found {len(nodes)}."
        )
    return nodes


def _analyse(text: str) -> tuple[ast.Module, ASTVisitor]:
    """Parse the current file with the same visitor that builds the index."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as e:
        line = getattr(e, "lineno", None)
        where = f" (line {line})" if line else ""
        raise _EditError(
            f"The file cannot be parsed{where}: {getattr(e, 'msg', None) or e}. "
            "Fix it first with `replace_text`."
        ) from None
    visitor = ASTVisitor(text)
    visitor.visit(tree)
    return tree, visitor


def _pick(
    records: list[dict],
    kind: str,
    name: str,
    occurrence: int | None,
    where: str = "",
) -> dict:
    """The one record called `name` (or the `occurrence`-th if it appears twice)."""
    matches = [r for r in records if r["name"] == name]
    if not matches:
        known = _quoted([r["name"] for r in records])
        hint = f" Available: {known}." if known else ""
        raise _EditError(f"{kind} `{name}` was not found{where}.{hint}")
    if occurrence is None:
        if len(matches) > 1:
            spans = ", ".join(
                f"{i}: lines {m['start_line']}-{m['end_line']}"
                for i, m in enumerate(matches, 1)
            )
            raise _EditError(
                f"{kind} `{name}`{where} is defined {len(matches)} times ({spans}). "
                "Pass `occurrence` to choose which one."
            )
        return matches[0]
    if not 1 <= occurrence <= len(matches):
        raise _EditError(
            f"occurrence must be between 1 and {len(matches)} "
            f"({kind} `{name}`{where} is defined {len(matches)} time(s))."
        )
    return matches[occurrence - 1]


def _methods_of(visitor: ASTVisitor, cls: dict) -> list[dict]:
    key = (cls["name"], cls["start_line"])
    return [
        m
        for m in visitor.get_methods()
        if (m["class_name"], m["class_start_line"]) == key
    ]


def _class_node(tree: ast.Module, cls: dict) -> ast.ClassDef:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name == cls["name"]
            and node.end_lineno == cls["end_line"]
            and min([node.lineno, *(d.lineno for d in node.decorator_list)])
            == cls["start_line"]
        ):
            return node
    raise _EditError(f"Class `{cls['name']}` could not be located.")  # not expected


def _is_main_guard(node: ast.stmt) -> bool:
    """`if __name__ == "__main__":`"""
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def _replace_span(
    text: str, start: int, end: int, new_code: str
) -> tuple[str, int, int]:
    """Swap lines start..end for `new_code`, keeping the old first line's indent.

    Returns (new_text, new_first_line, new_last_line).
    """
    lines = _split(text)
    block = _block(new_code, _indent_of(lines[start - 1]))
    lines[start - 1 : end] = block
    return _join(lines), start, start + len(block) - 1


def _top_level_anchor(tree: ast.Module, after: str | None) -> int:
    """Line after which a new top-level definition goes (0 = top of the file).

    Default: the end of the file, but above a trailing `if __name__ == "__main__":`.
    """
    body = tree.body
    if after is not None:
        ends = [
            n.end_lineno
            for n in body
            if isinstance(n, _DEFINITIONS) and n.name == after
        ]
        if not ends:
            known = _quoted([n.name for n in body if isinstance(n, _DEFINITIONS)])
            hint = f" Top-level functions and classes: {known}." if known else ""
            raise _EditError(
                f"There is no top-level function or class named `{after}`.{hint}"
            )
        return ends[-1]
    if body and _is_main_guard(body[-1]):
        body = body[:-1]
    return body[-1].end_lineno if body else 0


def _insert_top_level(
    text: str, tree: ast.Module, new_code: str, after: str | None
) -> tuple[str, int, int]:
    """Insert `new_code` at module level (PEP 8: two blank lines around it)."""
    lines = _split(text)
    anchor = _top_level_anchor(tree, after)
    block = _block(new_code, "")
    before = ["", ""] if anchor else []
    trailing = ["", ""] if not anchor and lines else []
    lines[anchor:anchor] = before + block + trailing
    first = anchor + len(before) + 1
    return _join(lines), first, first + len(block) - 1


def _atomic_write(path: pathlib.Path, text: str) -> None:
    """Write via a temp file + rename so a crash never leaves a half-written file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _edit_file(
    ctx: RunContext[DatabaseDeps],
    file_name: str,
    edit: Callable[[str], tuple[str, str]],
) -> str:
    """Run one edit against the file on disk.

    `edit(text)` receives the current source (with "\\n" newlines) and returns
    (new_text, summary), or raises `_EditError`. The result is only written if it
    still parses, line endings and permissions are preserved, and the index is
    refreshed afterwards.
    """
    path = _file_path_wrapper(file_name)
    if path is None:
        return _not_a_python_file(file_name)

    try:
        with open(path, encoding="utf-8", newline="") as f:
            raw = f.read()
    except (OSError, UnicodeDecodeError) as e:
        return _error(f"`{file_name}` could not be read: {e}")

    eol = "\r\n" if "\r\n" in raw else "\n"
    text = _normalize_newlines(raw)

    try:
        new_text, summary = edit(text)
    except _EditError as e:
        return _error(f"{e} Nothing was written.")

    if new_text == text:
        return f"{summary} `{os.path.basename(path)}` was left unchanged."

    try:
        ast.parse(new_text)
    except (SyntaxError, ValueError) as e:
        line = getattr(e, "lineno", None)
        where = f" at line {line}" if line else ""
        reason = getattr(e, "msg", None) or str(e)
        return _error(
            f"This edit would leave `{file_name}` with a syntax error{where}: "
            f"{reason}. Nothing was written. Fix the code (keep brackets, "
            "indentation and the surrounding code consistent) and try again."
        )

    _atomic_write(path, new_text.replace("\n", eol) if eol != "\n" else new_text)
    save_file(ctx.deps.db, str(path))  # keep the index in sync with the file
    return f"{summary} in `{os.path.basename(path)}`. The file was re-indexed."


# ----------------------------------------------------------------------
# Writing tools: change something that already exists
# ----------------------------------------------------------------------
def replace_function(
    ctx: RunContext[DatabaseDeps],
    file_name: str,
    function_name: str,
    new_code: str,
    occurrence: int | None = None,
) -> str:
    """Replace an existing top-level function with new code.

    Use this to change a function you have looked at with `expand_function`.
    `new_code` is the COMPLETE new function (decorators, signature, docstring and
    body) and replaces the old one entirely, so repeat the parts you keep. It may
    use a different name. To change just a few lines inside a function, use
    `replace_text` instead. To add a function that doesn't exist yet, use
    `add_function`.

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
        function_name: Name of the function to replace.
        new_code: The complete new function (exactly one function definition).
        occurrence: Only needed if the name is defined more than once (1 = first).
    """

    def edit(text: str) -> tuple[str, str]:
        _, visitor = _analyse(text)
        rec = _pick(visitor.get_functions(), "Function", function_name, occurrence)
        _parse_new(
            new_code,
            _FUNCTIONS,
            "function definition",
            single=True,
            hint=" Use replace_class for classes.",
        )
        new_text, first, last = _replace_span(
            text, rec["start_line"], rec["end_line"], new_code
        )
        return new_text, (
            f"Replaced function `{function_name}` (was lines {rec['start_line']}-"
            f"{rec['end_line']}, now lines {first}-{last})"
        )

    return _edit_file(ctx, file_name, edit)


def replace_method(
    ctx: RunContext[DatabaseDeps],
    file_name: str,
    cls_name: str,
    method_name: str,
    new_code: str,
    occurrence: int | None = None,
) -> str:
    """Replace an existing method of a class with new code.

    Use this to change a method you have looked at with `expand_method`.
    `new_code` is the COMPLETE new method (decorators, signature, docstring and
    body); pass it without the class indentation, exactly as `expand_method` shows
    it. It may use a different name. To change just a few lines inside a method,
    use `replace_text` instead. To add a method that doesn't exist yet, use
    `add_method`.

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
        cls_name: Name of the class that owns the method.
        method_name: Name of the method to replace.
        new_code: The complete new method (exactly one function definition).
        occurrence: Needed if the name is defined more than once in the class,
            e.g. a property getter and its setter (1 = first).
    """

    def edit(text: str) -> tuple[str, str]:
        _, visitor = _analyse(text)
        cls = _pick(visitor.get_classes(), "Class", cls_name, None)
        rec = _pick(
            _methods_of(visitor, cls),
            "Method",
            method_name,
            occurrence,
            f" in class `{cls_name}`",
        )
        _parse_new(new_code, _FUNCTIONS, "method definition", single=True)
        new_text, first, last = _replace_span(
            text, rec["start_line"], rec["end_line"], new_code
        )
        return new_text, (
            f"Replaced method `{cls_name}.{method_name}` (was lines "
            f"{rec['start_line']}-{rec['end_line']}, now lines {first}-{last})"
        )

    return _edit_file(ctx, file_name, edit)


def replace_class(
    ctx: RunContext[DatabaseDeps],
    file_name: str,
    cls_name: str,
    new_code: str,
    occurrence: int | None = None,
) -> str:
    """Replace an existing class (all of its methods included) with new code.

    Use this only to rewrite a whole class that you have looked at with
    `expand_class`. `new_code` is the COMPLETE new class and replaces the old one
    entirely. To change one method use `replace_method`; to add a method use
    `add_method`.

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
        cls_name: Name of the class to replace.
        new_code: The complete new class (exactly one class definition).
        occurrence: Only needed if the name is defined more than once (1 = first).
    """

    def edit(text: str) -> tuple[str, str]:
        _, visitor = _analyse(text)
        rec = _pick(visitor.get_classes(), "Class", cls_name, occurrence)
        _parse_new(
            new_code,
            _CLASSES,
            "class definition",
            single=True,
            hint=" Use replace_function for functions.",
        )
        new_text, first, last = _replace_span(
            text, rec["start_line"], rec["end_line"], new_code
        )
        return new_text, (
            f"Replaced class `{cls_name}` (was lines {rec['start_line']}-"
            f"{rec['end_line']}, now lines {first}-{last})"
        )

    return _edit_file(ctx, file_name, edit)


def replace_text(
    ctx: RunContext[DatabaseDeps],
    file_name: str,
    old_text: str,
    new_text: str,
) -> str:
    """Replace a piece of text in a file with other text.

    The general-purpose edit, for everything that is not a whole function, method
    or class: changing an import, a constant, the `if __name__ == "__main__":`
    block, or a few lines inside a function. `old_text` must match the file
    EXACTLY (including indentation) and must appear exactly once; include enough
    surrounding lines to make it unique. Use `new_text` = "" to remove the text.

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
        old_text: The exact text to replace (must occur exactly once).
        new_text: What to put there instead.
    """

    def edit(text: str) -> tuple[str, str]:
        old = _normalize_newlines(old_text)
        new = _normalize_newlines(new_text)
        if not old:
            raise _EditError("`old_text` must not be empty.")
        count = text.count(old)
        if count == 0:
            raise _EditError(
                "`old_text` was not found. It must match the file exactly, "
                "including indentation and line breaks."
            )
        if count > 1:
            found = [m.start() for m in re.finditer(re.escape(old), text)]
            lines = ", ".join(str(text.count("\n", 0, i) + 1) for i in found[:10])
            raise _EditError(
                f"`old_text` was found {count} times (starting on lines {lines}). "
                "Include more surrounding code so it matches only once."
            )
        i = text.index(old)
        line = text.count("\n", 0, i) + 1
        summary = (
            f"Replaced the text at line {line}"
            if old != new
            else "The replacement is identical to the existing text,"
        )
        return text[:i] + new + text[i + len(old) :], summary

    return _edit_file(ctx, file_name, edit)


# ----------------------------------------------------------------------
# Writing tools: add something new
# ----------------------------------------------------------------------
def add_function(
    ctx: RunContext[DatabaseDeps],
    file_name: str,
    new_code: str,
    after: str | None = None,
) -> str:
    """Add a NEW top-level function to a file.

    `new_code` is one or more complete function definitions (decorators,
    signature, docstring and body). By default they go at the end of the file
    (above an `if __name__ == "__main__":` block, if there is one); give `after`
    to put them right after an existing top-level function or class instead.
    Refuses if a function with that name already exists: use `replace_function`
    to change an existing one. To add a method to a class use `add_method`; for
    imports use `add_imports`.

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
        new_code: One or more complete function definitions.
        after: Optional name of an existing top-level function or class to place
            the new code after.
    """

    def edit(text: str) -> tuple[str, str]:
        tree, _ = _analyse(text)
        nodes = _parse_new(
            new_code,
            _FUNCTIONS,
            "function definition",
            hint=" Use add_class for classes.",
        )
        existing = {n.name: n for n in tree.body if isinstance(n, _FUNCTIONS)}
        for node in nodes:
            if node.name in existing:
                old = existing[node.name]
                raise _EditError(
                    f"Function `{node.name}` already exists (lines {old.lineno}-"
                    f"{old.end_lineno}). Use replace_function to change it."
                )
        new_text, first, last = _insert_top_level(text, tree, new_code, after)
        names = ", ".join(f"`{n.name}`" for n in nodes)
        return new_text, f"Added function {names} (lines {first}-{last})"

    return _edit_file(ctx, file_name, edit)


def add_method(
    ctx: RunContext[DatabaseDeps],
    file_name: str,
    cls_name: str,
    new_code: str,
    after_method: str | None = None,
) -> str:
    """Add a NEW method to an existing class.

    `new_code` is one or more complete method definitions (decorators, signature,
    docstring and body). Write them without the class indentation; the right
    indentation is added for you. By default they go at the end of the class; give
    `after_method` to put them right after an existing method. Refuses if the
    method already exists (use `replace_method` to change it), except for a
    decorated method such as a property setter.

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
        cls_name: Name of the class to add the method to.
        new_code: One or more complete method definitions.
        after_method: Optional name of an existing method to place the new code after.
    """

    def edit(text: str) -> tuple[str, str]:
        tree, visitor = _analyse(text)
        cls = _pick(visitor.get_classes(), "Class", cls_name, None)
        nodes = _parse_new(
            new_code,
            _FUNCTIONS,
            "method definition",
            hint=" Use add_class / add_function for classes and functions.",
        )
        methods = _methods_of(visitor, cls)
        for node in nodes:
            clash = [m for m in methods if m["name"] == node.name]
            if clash and not node.decorator_list:
                raise _EditError(
                    f"Method `{node.name}` already exists in class `{cls_name}` "
                    f"(lines {clash[0]['start_line']}-{clash[0]['end_line']}). "
                    "Use replace_method to change it."
                )

        class_node = _class_node(tree, cls)
        first_stmt = class_node.body[0]
        if first_stmt.lineno == class_node.lineno:
            raise _EditError(
                f"Class `{cls_name}` is written on a single line. "
                "Use replace_class to rewrite it."
            )
        lines = _split(text)
        indent = _indent_of(lines[first_stmt.lineno - 1])

        if after_method is None:
            anchor = cls["end_line"]
        else:
            ends = [m["end_line"] for m in methods if m["name"] == after_method]
            if not ends:
                known = _quoted([m["name"] for m in methods])
                hint = f" Methods of this class: {known}." if known else ""
                raise _EditError(
                    f"Method `{after_method}` was not found in class "
                    f"`{cls_name}`.{hint}"
                )
            anchor = ends[-1]

        block = [""] + _block(new_code, indent)
        lines[anchor:anchor] = block
        names = ", ".join(f"`{n.name}`" for n in nodes)
        return _join(lines), (
            f"Added method {names} to class `{cls_name}` "
            f"(lines {anchor + 2}-{anchor + len(block)})"
        )

    return _edit_file(ctx, file_name, edit)


def add_class(
    ctx: RunContext[DatabaseDeps],
    file_name: str,
    new_code: str,
    after: str | None = None,
) -> str:
    """Add a NEW top-level class to a file.

    `new_code` is one or more complete class definitions. By default they go at
    the end of the file (above an `if __name__ == "__main__":` block, if there is
    one); give `after` to put them right after an existing top-level function or
    class instead. Refuses if a class with that name already exists: use
    `replace_class` to change an existing one.

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
        new_code: One or more complete class definitions.
        after: Optional name of an existing top-level function or class to place
            the new code after.
    """

    def edit(text: str) -> tuple[str, str]:
        tree, _ = _analyse(text)
        nodes = _parse_new(
            new_code,
            _CLASSES,
            "class definition",
            hint=" Use add_function for functions.",
        )
        existing = {n.name: n for n in tree.body if isinstance(n, _CLASSES)}
        for node in nodes:
            if node.name in existing:
                old = existing[node.name]
                raise _EditError(
                    f"Class `{node.name}` already exists (lines {old.lineno}-"
                    f"{old.end_lineno}). Use replace_class to change it."
                )
        new_text, first, last = _insert_top_level(text, tree, new_code, after)
        names = ", ".join(f"`{n.name}`" for n in nodes)
        return new_text, f"Added class {names} (lines {first}-{last})"

    return _edit_file(ctx, file_name, edit)


def add_imports(
    ctx: RunContext[DatabaseDeps],
    file_name: str,
    imports: str,
) -> str:
    """Add import statements to a file.

    `imports` holds one or more complete `import ...` / `from ... import ...`
    statements. They are placed after the file's existing imports (or after the
    module docstring if there are none), and any statement that is already in the
    file is skipped. To change or remove an existing import use `replace_text`.

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
        imports: The import statements to add, one per line.
    """

    def edit(text: str) -> tuple[str, str]:
        tree, _ = _analyse(text)
        source_lines = _block(imports, "")
        if not source_lines:
            raise _EditError("`imports` is empty.")
        source = "\n".join(source_lines)
        try:
            nodes = ast.parse(source).body
        except SyntaxError as e:
            raise _EditError(
                f"`imports` is not valid Python (line {e.lineno}: {e.msg})."
            ) from None
        for node in nodes:
            if not isinstance(node, _IMPORTS):
                raise _EditError(
                    "`imports` must contain only import statements, but it "
                    f"contains {_describe(node)}."
                )
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                raise _EditError(
                    "`from __future__` imports must be the first statement of a "
                    "file; add them with `replace_text` instead."
                )

        seen = {ast.dump(n) for n in tree.body if isinstance(n, _IMPORTS)}
        fresh = []
        for node in nodes:
            if ast.dump(node) not in seen:
                seen.add(ast.dump(node))
                fresh.append(ast.get_source_segment(source, node))
        if not fresh:
            return text, "All of these imports are already present;"

        body = tree.body
        has_docstring = (
            bool(body)
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        )
        i = 1 if has_docstring else 0
        last_import = None
        while i < len(body) and isinstance(body[i], _IMPORTS):
            last_import = body[i]
            i += 1

        lines = _split(text)
        new_lines = "\n".join(fresh).split("\n")
        if last_import is not None:
            at, block = last_import.end_lineno, new_lines
        elif has_docstring:
            at, block = body[0].end_lineno, [""] + new_lines
        else:
            at, block = 0, new_lines + ([""] if lines else [])
        lines[at:at] = block
        first = at + (1 if block[0] == "" else 0) + 1
        return _join(lines), (
            f"Added {len(fresh)} import statement(s) (lines "
            f"{first}-{first + len(new_lines) - 1})"
        )

    return _edit_file(ctx, file_name, edit)


# ----------------------------------------------------------------------
# Writing tools: remove something
# ----------------------------------------------------------------------
def delete_definition(
    ctx: RunContext[DatabaseDeps],
    file_name: str,
    name: str,
    cls_name: str | None = None,
    occurrence: int | None = None,
) -> str:
    """Delete a function, a class or a method from a file.

    Without `cls_name`, `name` is a top-level function or a class. With `cls_name`,
    `name` is a method of that class. The code is removed completely (decorators
    included) and the surrounding blank lines are tidied. Refuses if the result
    would not be valid Python, e.g. deleting the only method of a class that has
    no other body.

    Args:
        file_name: Path to the .py file (absolute, or relative to the cwd).
        name: Name of the function, class or method to delete.
        cls_name: Set this when deleting a method: the class that owns it.
        occurrence: Only needed if the name is defined more than once (1 = first).
    """

    def edit(text: str) -> tuple[str, str]:
        _, visitor = _analyse(text)
        if cls_name is not None:
            cls = _pick(visitor.get_classes(), "Class", cls_name, None)
            rec = _pick(
                _methods_of(visitor, cls),
                "Method",
                name,
                occurrence,
                f" in class `{cls_name}`",
            )
            label = f"method `{cls_name}.{name}`"
        else:
            records = [{**r, "kind": "function"} for r in visitor.get_functions()]
            records += [{**r, "kind": "class"} for r in visitor.get_classes()]
            rec = _pick(records, "Function or class", name, occurrence)
            label = f"{rec['kind']} `{name}`"

        lines = _split(text)
        start, end = rec["start_line"], rec["end_line"]
        first = start - 1
        after = end
        while after < len(lines) and not lines[after].strip():
            after += 1
        indent = _indent_of(lines[first])

        if after < len(lines) and len(_indent_of(lines[after])) >= len(indent):
            # a sibling follows: remove the blank lines after the deleted code,
            # so the blank lines before it keep separating the neighbours
            del lines[first:after]
        else:
            # it was the last thing in its block: remove the blank lines before it
            # (and everything after it if it was the end of the file)
            while first > 0 and not lines[first - 1].strip():
                first -= 1
            del lines[first : after if after >= len(lines) else end]
        return _join(lines), f"Deleted {label} (was lines {start}-{end})"

    return _edit_file(ctx, file_name, edit)


# What the agent is given. The Veto layer should ask the developer before any of
# WRITE_TOOLS runs; READ_TOOLS only need the "may I read this?" permission.
READ_TOOLS = (
    get_file_imports,
    get_file_top_level_chunk,
    get_file_classes,
    get_class_methods_signature,
    get_file_functions,
    expand_function,
    expand_class,
    expand_method,
)
WRITE_TOOLS = (
    replace_function,
    replace_method,
    replace_class,
    replace_text,
    add_function,
    add_method,
    add_class,
    add_imports,
    delete_definition,
)
