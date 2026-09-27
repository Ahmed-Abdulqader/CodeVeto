"""
graph_builder.py

Builds a language-agnostic `SymbolGraph` by walking a tree-sitter AST once,
using a declarative `LanguageGraphSpec` (see `graph.language_spec`) to
decide what each node means. This replaces the old graph_builder.py, whose
~600 lines were mostly three copies of the same traversal shape
(`_traverse_python`, `_traverse_javascript`, `_traverse_rust`) that only
differed in tree-sitter node-type names. Adding a fourth language now means
adding one `LanguageGraphSpec`, not one ~150-line method.

Builds on the same tree-sitter parse chunker.py already produces — pass in
the `Chunks` instance's `.tree.root_node` and `.source_bytes` (see
`indexer.py`) rather than re-parsing the file.
"""

from __future__ import annotations

import os

from context_engine.graph.language_spec import LANGUAGE_SPECS
from models.context_engine_models import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    SymbolGraph,
)


class GraphBuilder:
    """Builds a `SymbolGraph` from a tree-sitter AST, aligned with the
    language support declared in `graph.language_spec.LANGUAGE_SPECS`
    (which mirrors chunker.py's `EXTENSION_MAP`)."""

    def __init__(self) -> None:
        self.graph = SymbolGraph()

    def extract_graph(
        self,
        file_path: str,
        ast_root,
        source_code: bytes,
        language: str | None = None,
    ) -> SymbolGraph:
        """Extract nodes/edges from `ast_root` and merge them into the
        builder's running `SymbolGraph` (safe to call once per file across
        many files to build a whole-repo graph)."""
        language = language or _detect_language(file_path)
        spec = LANGUAGE_SPECS.get(language)
        if spec is None:
            raise ValueError(f"Unsupported language: {language}")

        file_node_id = _make_node_id(file_path, NodeType.FILE, file_path)
        self.graph.add_node(
            GraphNode(
                id=file_node_id,
                type=NodeType.FILE,
                name=os.path.basename(file_path),
                file_path=file_path,
                start_line=ast_root.start_point[0] + 1,
                end_line=ast_root.end_point[0] + 1,
            )
        )

        self._walk(
            ast_root,
            file_path=file_path,
            file_node_id=file_node_id,
            source=source_code,
            spec=spec,
            parent_container=None,
            parent_func=None,
        )
        return self.graph

    # -- the one generic walk, replacing three duplicated traversals -------

    def _walk(
        self,
        node,
        *,
        file_path: str,
        file_node_id: str,
        source: bytes,
        spec,
        parent_container: str | None,
        parent_func: str | None,
    ) -> None:
        next_container = parent_container
        next_func = parent_func

        if spec.is_class(node):
            class_id = self._add_class(node, file_path, file_node_id, source, spec)
            next_container, next_func = class_id, None

        elif spec.is_impl(node):
            next_container = (
                self._enter_impl(node, file_path, source, spec) or next_container
            )

        elif spec.is_function(node):
            next_func = self._add_function(
                node, file_path, file_node_id, source, spec, parent_container
            )

        elif spec.is_call(node) and parent_func:
            self._add_call(node, file_path, source, spec, parent_func)

        elif spec.is_import(node):
            self._add_import(node, file_path, file_node_id, source, spec)

        for child in node.children:
            self._walk(
                child,
                file_path=file_path,
                file_node_id=file_node_id,
                source=source,
                spec=spec,
                parent_container=next_container,
                parent_func=next_func,
            )

    # -- one small builder method per node kind, shared by every language --

    def _add_class(self, node, file_path, file_node_id, source, spec) -> str:
        name = spec.class_name(node, source) or "unknown_class"
        class_id = _make_node_id(file_path, NodeType.CLASS, name, node.start_point[0])
        self.graph.add_node(
            GraphNode(
                id=class_id,
                type=NodeType.CLASS,
                name=name,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            )
        )
        self.graph.add_edge(
            GraphEdge(
                source_id=file_node_id, target_id=class_id, type=EdgeType.CONTAINS
            )
        )
        if spec.extends_names is not None:
            for base_name in spec.extends_names(node, source):
                base_id = _make_node_id(file_path, NodeType.CLASS, base_name, 0)
                self.graph.add_node(
                    GraphNode(
                        id=base_id,
                        type=NodeType.CLASS,
                        name=base_name,
                        file_path=file_path,
                    )
                )
                self.graph.add_edge(
                    GraphEdge(
                        source_id=class_id, target_id=base_id, type=EdgeType.EXTENDS
                    )
                )
        return class_id

    def _enter_impl(self, node, file_path, source, spec) -> str | None:
        if spec.impl_target_name is None:
            return None
        target = spec.impl_target_name(node, source)
        if not target:
            return None
        impl_id = _make_node_id(file_path, NodeType.CLASS, target, 0)
        self.graph.add_node(
            GraphNode(id=impl_id, type=NodeType.CLASS, name=target, file_path=file_path)
        )
        return impl_id

    def _add_function(
        self, node, file_path, file_node_id, source, spec, parent_container
    ) -> str:
        name = spec.function_name(node, source) or "unknown_func"
        is_method = spec.method_predicate(node, parent_container)
        node_type = NodeType.METHOD if is_method else NodeType.FUNCTION
        func_id = _make_node_id(file_path, node_type, name, node.start_point[0])
        self.graph.add_node(
            GraphNode(
                id=func_id,
                type=node_type,
                name=name,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            )
        )
        self.graph.add_edge(
            GraphEdge(
                source_id=parent_container if is_method else file_node_id,
                target_id=func_id,
                type=EdgeType.CONTAINS,
            )
        )
        return func_id

    def _add_call(self, node, file_path, source, spec, parent_func) -> None:
        name = spec.call_name(node, source)
        if not name:
            return
        target_id = _make_node_id(file_path, NodeType.FUNCTION, name, 0)
        self.graph.add_node(
            GraphNode(
                id=target_id, type=NodeType.FUNCTION, name=name, file_path=file_path
            )
        )
        self.graph.add_edge(
            GraphEdge(source_id=parent_func, target_id=target_id, type=EdgeType.CALLS)
        )

    def _add_import(self, node, file_path, file_node_id, source, spec) -> None:
        name = spec.import_name(node, source)
        if not name:
            return
        mod_id = _make_node_id(file_path, NodeType.MODULE, name, node.start_point[0])
        self.graph.add_node(
            GraphNode(id=mod_id, type=NodeType.MODULE, name=name, file_path=file_path)
        )
        self.graph.add_edge(
            GraphEdge(source_id=file_node_id, target_id=mod_id, type=EdgeType.IMPORTS)
        )


def _make_node_id(file_path: str, node_type: NodeType, name: str, line: int = 0) -> str:
    return f"{file_path}::{node_type.value}::{name}::{line}"


def _detect_language(file_path: str) -> str:
    from context_engine.chunker import EXTENSION_MAP

    ext = os.path.splitext(file_path)[1].lower()
    language = EXTENSION_MAP.get(ext)
    if not language:
        raise ValueError(f"Unsupported language for extension: {ext}")
    return language
