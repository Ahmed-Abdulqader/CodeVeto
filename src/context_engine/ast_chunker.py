"""Turns Python source into the chunks CodeVeto's tools work with.

Only what the agent actually sees is extracted:

* module-level imports            -> code + line range
* other module-level statements   -> code + line range
* classes / functions / methods   -> code + line range + signature + docstring

Every chunk is a whole-line span, so the line range stored with a chunk is
exactly what `write_code_chunk` needs to replace it later.
"""

import ast
import copy
import hashlib
import os


def normalize_path(file_path: str) -> str:
    """The one place that decides how a path is stored in / looked up in the index."""
    return os.path.realpath(file_path)


def content_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def get_file_data(file_path: str, source: str) -> dict:
    """Data for the `files` table."""
    return {
        "file_path": normalize_path(file_path),
        "content_hash": content_hash(source),
    }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _first_line(node: ast.AST) -> int:
    """First line of a definition, *including* its decorators.

    `node.lineno` of a decorated function/class is the `def`/`class` line, so
    using it alone would silently leave the decorators out of the chunk.
    """
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)])


def _dedent(lines: list[str]) -> str:
    """Strip the first line's indentation from every line.

    Lets a method read like a top-level function. Unlike `textwrap.dedent`,
    lines that are indented less than the first one (e.g. the inside of a
    multi-line string) are left untouched instead of cancelling the dedent.
    """
    first = lines[0]
    prefix = first[: len(first) - len(first.lstrip())]
    return "\n".join(
        line[len(prefix):] if line.startswith(prefix) else line for line in lines
    )


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    """Decorators + the `def` / `class` line, with the body (and docstring) removed.

    Built with `ast.unparse` on a body-less copy so async, defaults, annotations,
    keyword-only args, base classes, metaclass keywords and generics all come out
    right without hand-assembling the header.
    """
    clone = copy.copy(node)
    clone.body = [ast.Pass()]
    return ast.unparse(clone).rsplit("\n", 1)[0]


class ASTVisitor(ast.NodeVisitor):
    def __init__(self, source: str):
        self.source = source
        self._lines = source.split("\n")
        self.imports = []
        self.classes = []
        self.functions = []
        self.methods = []
        # Scope stack: ("class", class_record) or ("function", None)
        self._scope_stack = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _definition_span(self, node: ast.AST) -> tuple[int, int, str]:
        """(start_line, end_line, code) of a class/function, decorators included."""
        start, end = _first_line(node), node.end_lineno
        return start, end, _dedent(self._lines[start - 1 : end])

    # ------------------------------------------------------------------
    # Getters: one per table
    # ------------------------------------------------------------------
    def get_imports(self) -> list[dict]:
        """`imports` table: module-level imports (start_line, end_line, code)."""
        return self.imports

    def get_classes(self) -> list[dict]:
        """`classes` table: name, signature, docstring, start_line, end_line, code."""
        return self.classes

    def get_functions(self) -> list[dict]:
        """`functions` table: top-level functions only."""
        return self.functions

    def get_methods(self) -> list[dict]:
        """`methods` table. class_name + class_start_line identify the owning class."""
        return self.methods

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------
    def visit_Module(self, node: ast.Module):
        # Only module-level imports are kept. An import inside a function is
        # already part of that function's code, and one inside a top-level
        # `try:` / `if TYPE_CHECKING:` is part of that top-level chunk.
        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                self.imports.append(
                    {
                        "start_line": stmt.lineno,
                        "end_line": stmt.end_lineno,
                        "code": ast.get_source_segment(self.source, stmt)
                        or ast.unparse(stmt),
                    }
                )
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef):
        start, end, code = self._definition_span(node)
        class_record = {
            "name": node.name,
            "signature": _signature(node),
            "docstring": ast.get_docstring(node),
            "start_line": start,
            "end_line": end,
            "code": code,
        }
        self.classes.append(class_record)

        self._scope_stack.append(("class", class_record))
        self.generic_visit(node)
        self._scope_stack.pop()

    # ------------------------------------------------------------------
    # Functions / methods
    # ------------------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function(node)

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        scope_kind, scope_record = (
            self._scope_stack[-1] if self._scope_stack else (None, None)
        )

        if scope_kind is None:
            # top-level function
            self.functions.append(self._record_function(node))
        elif scope_kind == "class":
            # direct method of a class
            method = self._record_function(node)
            method["class_name"] = scope_record["name"]
            method["class_start_line"] = scope_record["start_line"]
            self.methods.append(method)
        # scope_kind == "function": nested function -> not recorded as its own
        # row (it is part of the enclosing function's code)

        self._scope_stack.append(("function", None))
        self.generic_visit(node)
        self._scope_stack.pop()

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
        start, end, code = self._definition_span(node)
        return {
            "name": node.name,
            "signature": _signature(node),
            "docstring": ast.get_docstring(node),
            "start_line": start,
            "end_line": end,
            "code": code,
        }


class TopLevelCodeVisitor(ast.NodeVisitor):
    """Module-level statements that are not an import, a function or a class."""

    _SKIPPED = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Import,
        ast.ImportFrom,
    )

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.chunks: list[dict] = []

    def visit_Module(self, node: ast.Module) -> None:
        """Iterates over top-level statements in the file body."""
        for stmt in node.body:
            # Functions/classes and imports are stored in their own tables.
            if isinstance(stmt, self._SKIPPED):
                continue

            self.chunks.append(
                {
                    "start_line": stmt.lineno,
                    "end_line": stmt.end_lineno,
                    "code": ast.get_source_segment(self.source_code, stmt)
                    or ast.unparse(stmt),
                }
            )

    def get_chunks(self) -> list[dict]:
        """`top_level_chunks` table: start_line, end_line, code."""
        return self.chunks
