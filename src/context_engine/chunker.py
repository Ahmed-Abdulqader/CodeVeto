import os
import uuid
from typing import Any

import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
import tree_sitter_rust as tsrust
from tree_sitter import Language, Node, Parser

from models.context_engine_models import Chunk, FileMetadata

# ---------------------------------------------------------------------------
# Language setup
# ---------------------------------------------------------------------------

LANGUAGES = {
    "python": Language(tspython.language()),
    "javascript": Language(tsjavascript.language()),
    "rust": Language(tsrust.language()),
}

EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".rs": "rust",
}

# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class Chunks:
    """
    Parses a single source file (Python / Rust / JS) with tree-sitter and
    extracts a list of Chunk objects (functions, methods, classes, etc.).

    Usage:
        chunks = Chunks("example.py")

        for chunk in chunks:
            print(chunk.chunk_type, chunk.signature)
            print(chunk.docstring)

        chunks.chunk.docstring      # convenience: first chunk's docstring
        chunks[0].content           # index access
        chunks.to_list()            # list[dict] matching the requested schema
    """

    # node type -> default chunk_type (before parent-based refinement)
    _NODE_TYPES = {
        "python": {
            "function_definition": "function",
            "class_definition": "class",
        },
        "javascript": {
            "function_declaration": "function",
            "generator_function_declaration": "function",
            "method_definition": "method",
            "class_declaration": "class",
        },
        "rust": {
            "function_item": "function",
            "struct_item": "struct",
            "enum_item": "enum",
            "trait_item": "trait",
            "impl_item": "impl",
        },
    }

    def __init__(self, file_path: str, language: str | None = None):
        self.file_path = file_path
        self.language = language or self._detect_language(file_path)
        if self.language not in LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}")

        self.source_bytes = self._read_file(file_path)
        self.parser = Parser(LANGUAGES[self.language])
        self.tree = self.parser.parse(self.source_bytes)
        self.chunks: list[Chunk] = self._extract_chunks()

    # -- setup helpers ------------------------------------------------

    @staticmethod
    def _detect_language(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        lang = EXTENSION_MAP.get(ext)
        if not lang:
            raise ValueError(f"Cannot detect language for extension: {ext}")
        return lang

    @staticmethod
    def _read_file(file_path: str) -> bytes:
        with open(file_path, "rb") as f:
            return f.read()

    def _build_file_metadata(self) -> FileMetadata:
        return FileMetadata(
            name=os.path.basename(self.file_path),
            path=os.path.abspath(self.file_path),
            modified_ts=os.path.getmtime(self.file_path),
        )

    def _node_text(self, node: Node) -> str:
        return self.source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )

    # -- tree walking ---------------------------------------------------

    def _extract_chunks(self) -> list[Chunk]:
        file_metadata = self._build_file_metadata()
        results: list[Chunk] = []

        def walk(node: Node, parent: Node | None):
            chunk_type = self._classify_node(node, parent)
            if chunk_type:
                results.append(self._build_chunk(node, chunk_type, file_metadata))
            for child in node.children:
                walk(child, node)

        walk(self.tree.root_node, None)
        return results

    def _classify_node(self, node: Node, parent: Node | None) -> str | None:
        t = node.type

        if self.language == "python":
            if t == "function_definition":
                if (
                    parent is not None
                    and parent.type == "block"
                    and parent.parent is not None
                    and parent.parent.type == "class_definition"
                ):
                    return "method"
                return "function"
            return self._NODE_TYPES["python"].get(t)

        if self.language == "javascript":
            if t == "variable_declarator":
                value = node.child_by_field_name("value")
                if value is not None and value.type in (
                    "arrow_function",
                    "function",
                    "function_expression",
                ):
                    return "function"
                return None
            return self._NODE_TYPES["javascript"].get(t)

        if self.language == "rust":
            if t == "function_item":
                if (
                    parent is not None
                    and parent.type == "declaration_list"
                    and parent.parent is not None
                    and parent.parent.type == "impl_item"
                ):
                    return "method"
                return "function"
            return self._NODE_TYPES["rust"].get(t)

        return None

    def _build_chunk(
        self, node: Node, chunk_type: str, file_metadata: FileMetadata
    ) -> Chunk:
        return Chunk(
            id=str(uuid.uuid4()),
            language=self.language,
            chunk_type=chunk_type,
            docstring=self._extract_docstring(node),
            signature=self._extract_signature(node),
            content=self._node_text(node),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            file_metadata=file_metadata,
        )

    # -- docstring extraction -------------------------------------------

    def _extract_docstring(self, node: Node) -> str | None:
        if self.language == "python":
            return self._python_docstring(node)
        if self.language == "javascript":
            return self._js_docstring(node)
        if self.language == "rust":
            return self._rust_docstring(node)
        return None

    def _python_docstring(self, node: Node) -> str | None:
        body = node.child_by_field_name("body")
        if not body or body.child_count == 0:
            return None
        first_stmt = body.children[0]
        if first_stmt.type == "expression_statement" and first_stmt.child_count > 0:
            expr = first_stmt.children[0]
            if expr.type == "string":
                return self._strip_python_string_quotes(self._node_text(expr))
        return None

    @staticmethod
    def _strip_python_string_quotes(text: str) -> str:
        for q in ('"""', "'''"):
            if text.startswith(q) and text.endswith(q):
                return text[3:-3].strip()
        for q in ('"', "'"):
            if text.startswith(q) and text.endswith(q):
                return text[1:-1].strip()
        return text.strip()

    def _js_docstring(self, node: Node) -> str | None:
        target = node.parent if node.type == "variable_declarator" else node
        prev = target.prev_sibling if target else None
        if prev and prev.type == "comment":
            text = self._node_text(prev)
            if text.startswith("/**"):
                return self._clean_jsdoc(text)
        return None

    @staticmethod
    def _clean_jsdoc(text: str) -> str:
        text = text.strip()
        if text.startswith("/**"):
            text = text[3:]
        if text.endswith("*/"):
            text = text[:-2]
        lines = [ln.strip().lstrip("*").strip() for ln in text.splitlines()]
        return "\n".join(line for line in lines if line).strip()

    def _rust_docstring(self, node: Node) -> str | None:
        collected = []
        prev = node.prev_sibling
        while prev and prev.type in ("line_comment", "block_comment"):
            text = self._node_text(prev)
            if text.startswith("///") or text.startswith("/**"):
                collected.insert(0, text)
                prev = prev.prev_sibling
            else:
                break
        if not collected:
            return None
        lines = []
        for c in collected:
            if c.startswith("///"):
                lines.append(c[3:].strip())
            else:
                lines.append(self._clean_jsdoc(c))
        return "\n".join(lines).strip() or None

    # -- signature extraction --------------------------------------------

    def _extract_signature(self, node: Node) -> str | None:
        if self.language == "python":
            return self._header_before(
                node, node.child_by_field_name("body"), strip_colon=True
            )
        if self.language == "javascript":
            if node.type == "variable_declarator":
                value = node.child_by_field_name("value")
                body = value.child_by_field_name("body") if value else None
            else:
                body = node.child_by_field_name("body")
            return self._header_before(node, body)
        if self.language == "rust":
            body = node.child_by_field_name("body")
            return self._header_before(node, body, strip_brace=True)
        return None

    def _header_before(
        self,
        node: Node,
        body: Node | None,
        strip_colon: bool = False,
        strip_brace: bool = False,
    ) -> str:
        text = self._node_text(node)
        if body is not None:
            header = text[: body.start_byte - node.start_byte]
        else:
            header = text.splitlines()[0]
        header = header.rstrip()
        if strip_colon and header.endswith(":"):
            header = header[:-1]
        if strip_brace and header.endswith("{"):
            header = header[:-1]
        return " ".join(header.split())

    # -- convenience access -----------------------------------------------

    def to_list(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.chunks]

    def __iter__(self):
        return iter(self.chunks)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]

    @property
    def chunk(self) -> Chunk | None:
        """First chunk — convenience for chunks.chunk.docstring style access."""
        return self.chunks[0] if self.chunks else None


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    path = Path(__file__).resolve().parent
    this_file = path / "chunker.py"
    chunks = Chunks(this_file)

    for chunk in chunks:
        line_range = f"{chunk.start_line}-{chunk.end_line}"
        print(f"[{chunk.chunk_type}] {chunk.signature}  (lines {line_range})")
        if chunk.docstring:
            print("  doc:", chunk.docstring.splitlines()[0])

    # quick single-chunk access
    print(chunks.chunk.docstring)
    print(chunks.chunk.signature)

    # full schema as list[dict], ready to serialize/index
    import json

    print(json.dumps(chunks.to_list()[0], indent=2, default=str))
