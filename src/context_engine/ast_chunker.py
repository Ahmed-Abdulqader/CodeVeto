import ast
import hashlib
import os
import textwrap

def get_file_data(file_path: str, source: str) -> dict:
    """Data for the `files` table."""
    return {
        "file_path": os.path.abspath(file_path),
        "file_name": os.path.basename(file_path),
        "content_hash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "last_modified": os.path.getmtime(file_path),
    }


class ASTVisitor(ast.NodeVisitor):
    def __init__(self, source: str):
        self.source = source
        self.classes = []
        self.class_inheritance = []
        self.functions = []
        self.methods = []
        self.imports = []
        # Scope stack: ("class", class_record) or ("function", None)
        self._scope_stack = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_code(self, node: ast.AST) -> str | None:
        """Return the node's source text, dedented so nested defs (e.g. methods) read cleanly."""
        segment = ast.get_source_segment(self.source, node, padded=True)
        return textwrap.dedent(segment) if segment is not None else None

    # ------------------------------------------------------------------
    # Getters: one per table
    # ------------------------------------------------------------------
    def get_classes(self) -> list[dict]:
        """`classes` table: name, docstring, start_line, end_line, decorators, code."""
        return self.classes

    def get_class_inheritance(self) -> list[dict]:
        """
        `class_inheritance` table.
        class_name + class_start_line identify the child class (used to find class_id on insert).
        parent_class_id is None here; it is resolved in SQL after insertion.
        """
        return self.class_inheritance

    def get_functions(self) -> list[dict]:
        """`functions` table: top-level functions only (includes code)."""
        return self.functions

    def get_methods(self) -> list[dict]:
        """`methods` table (includes code). class_name + class_start_line identify the owning class."""
        return self.methods

    def get_imports(self) -> list[dict]:
        """`imports` table (target_file_id is None until resolved)."""
        return self.imports

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append({
                "import_type": "import",
                "module_name": alias.name,
                "imported_name": None,
                "alias": alias.asname,
                "target_file_id": None,
                "line_number": node.lineno,
            })

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # Keep the dots for relative imports: `from ..pkg import x` -> "..pkg"
        module_name = "." * node.level + (node.module or "")
        for alias in node.names:
            self.imports.append({
                "import_type": "import_from",
                "module_name": module_name,
                "imported_name": alias.name,
                "alias": alias.asname,
                "target_file_id": None,
                "line_number": node.lineno,
            })

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef):
        class_record = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "decorators": [ast.unparse(d) for d in node.decorator_list],
            "code": self._get_code(node),
        }
        self.classes.append(class_record)

        for base in node.bases:
            self.class_inheritance.append({
                "class_name": node.name,
                "class_start_line": node.lineno,
                "parent_class_name": ast.unparse(base),
                "parent_class_id": None,
            })

        self._scope_stack.append(("class", class_record))
        self.generic_visit(node)
        self._scope_stack.pop()

    # ------------------------------------------------------------------
    # Functions / methods
    # ------------------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function(node, is_async=True)

    def _process_function(self, node: ast.AST, is_async: bool):
        scope_kind, scope_record = self._scope_stack[-1] if self._scope_stack else (None, None)

        if scope_kind is None:
            # top-level function
            self.functions.append(self._record_function(node, is_async))
        elif scope_kind == "class":
            # direct method of a class
            method = self._record_function(node, is_async)
            method["class_name"] = scope_record["name"]
            method["class_start_line"] = scope_record["start_line"]
            self.methods.append(method)
        # scope_kind == "function": nested function -> not recorded as its own row
        # (its calls are still counted in the parent via ast.walk)

        self._scope_stack.append(("function", None))
        self.generic_visit(node)
        self._scope_stack.pop()

    def get_call_name(self, node_func: ast.AST) -> str:
        if isinstance(node_func, ast.Name):
            return node_func.id
        elif isinstance(node_func, ast.Attribute):
            return node_func.attr

        try:
            return ast.unparse(node_func)
        except AttributeError:
            return "Unknown"

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> dict:
        def _extract_arg_info(arg_node: ast.arg, prefix: str = "") -> dict:
            return {
                "name": f"{prefix}{arg_node.arg}",
                "type": ast.unparse(arg_node.annotation) if arg_node.annotation else None
            }

        args = []
        args.extend([_extract_arg_info(arg) for arg in node.args.posonlyargs])
        args.extend([_extract_arg_info(arg) for arg in node.args.args])

        if node.args.vararg:
            args.append(_extract_arg_info(node.args.vararg, prefix="*"))

        args.extend([_extract_arg_info(arg) for arg in node.args.kwonlyargs])

        if node.args.kwarg:
            args.append(_extract_arg_info(node.args.kwarg, prefix="**"))

        calls = [
            self.get_call_name(child.func)
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
        ]

        return {
            "type": "async_function" if is_async else "function",
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "returns": ast.unparse(node.returns) if node.returns else None,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "start_column": node.col_offset,
            "end_column": node.end_col_offset,
            "args": args,
            "calls": calls,
            "decorators": [ast.unparse(d) for d in node.decorator_list],
            "code": self._get_code(node),
        }

class TopLevelCodeVisitor(ast.NodeVisitor):
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.chunks: list[dict] = []

    def visit_Module(self, node: ast.Module) -> None:
        """Iterates over top-level statements in the file body."""
        for stmt in node.body:
            # 1. Skip functions, async functions, and classes
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue

            # 2. Skip imports (since you already extract them elsewhere)
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                continue

            # 3. Extract source text and metadata
            segment = ast.get_source_segment(self.source_code, stmt)

            self.chunks.append({
                "category": self._categorize_statement(stmt),
                "node_type": type(stmt).__name__,
                "start_line": stmt.lineno,
                "end_line": stmt.end_lineno,
                "code": segment
            })

    def _categorize_statement(self, stmt: ast.stmt) -> str:
        """Categorizes top-level code chunks."""
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            return "constant_or_global"
        elif isinstance(stmt, ast.If):
            if self._is_main_block(stmt):
                return "main_guard"
            return "top_level_conditional"
        elif isinstance(stmt, ast.Expr):
            return "top_level_expression"
        elif isinstance(stmt, (ast.Try, ast.With)):
            return "top_level_setup"
        return "other_top_level_code"

    def _is_main_block(self, node: ast.If) -> bool:
        """Checks if the statement is `if __name__ == '__main__':`."""
        test = node.test
        if isinstance(test, ast.Compare):
            left_is_name = isinstance(test.left, ast.Name) and test.left.id == "__name__"
            right_is_main = any(
                isinstance(comp, ast.Constant) and comp.value == "__main__"
                for comp in test.comparators
            )
            return left_is_name and right_is_main
        return False
    def get_chunks(self) -> list[dict]:
        """`top_level_chunks` table: category, node_type, start_line, end_line, code."""
        return self.chunks

    # Prevent visiting children inside functions or classes
    def visit_FunctionDef(self, node): pass
    def visit_AsyncFunctionDef(self, node): pass
    def visit_ClassDef(self, node): pass


if __name__=="__main__":
    with open("test_chunker.py", "rb") as f:
        code = f.read()
    code_str = code.decode("utf-8")
    tree = ast.parse(code)

    extractor = ASTVisitor()
    extractor.visit(tree)
    for importer in extractor.imports:
        print(importer)
    for import_from in extractor.import_from:
        print(import_from)

    for record in extractor.records:
        print(record)

    tree = ast.parse(code_str)
    visitor = TopLevelCodeVisitor(code_str)
    visitor.visit(tree)

    # 3. Access chunks (list of dicts)
    chunks = visitor.chunks
    print(chunks)