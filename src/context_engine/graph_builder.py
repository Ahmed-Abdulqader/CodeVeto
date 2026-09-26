"""
graph_builder.py

Analyzes AST structures across source code files to identify symbols
and structural dependencies, producing an in-memory SymbolGraph.
Decoupled from database storage logic.

Builds upon Tree-Sitter AST parsing established in chunker.py.
"""

import os

# Import the Pydantic models from your models directory
from models.context_engine_models import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    SymbolGraph,
)

# ==========================================
# Graph Builder Logic
# ==========================================


class GraphBuilder:
    """
    Builds a SymbolGraph from Tree-Sitter ASTs.
    Aligned with the language support defined in chunker.py.
    """

    # Exact same mapping as chunker.py to ensure consistency
    EXTENSION_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".rs": "rust",
    }

    def __init__(self):
        self.graph = SymbolGraph()

    def extract_graph(
        self,
        file_path: str,
        ast_root,
        source_code: bytes,
        language: str | None = None,
    ) -> SymbolGraph:
        """
        Extracts nodes and edges from the given AST and adds them to the internal graph.

        :param file_path: Relative path to the source file.
        :param ast_root: The root node of the Tree-Sitter AST (from chunker.py).
        :param source_code: The raw source code bytes.
        :param language: Optional language override. If None, detected from file_path.
        :return: The updated SymbolGraph.
        """
        # 1. Create File Node
        file_node_id = self._make_node_id(file_path, NodeType.FILE, file_path)
        file_node = GraphNode(
            id=file_node_id,
            type=NodeType.FILE,
            name=os.path.basename(file_path),
            file_path=file_path,
            start_line=ast_root.start_point[0] + 1,
            end_line=ast_root.end_point[0] + 1,
        )
        self.graph.add_node(file_node)

        # 2. Determine language (mirroring chunker.py's logic)
        if language is None:
            ext = os.path.splitext(file_path)[1].lower()
            language = self.EXTENSION_MAP.get(ext)
            if not language:
                raise ValueError(f"Unsupported language for extension: {ext}")

        # 3. Route to the correct language extractor
        if language == "python":
            self._traverse_python(
                ast_root, file_path, file_node_id, source_code, None, None
            )
        elif language == "javascript":
            self._traverse_javascript(
                ast_root, file_path, file_node_id, source_code, None, None
            )
        elif language == "rust":
            self._traverse_rust(
                ast_root, file_path, file_node_id, source_code, None, None
            )
        else:
            raise ValueError(f"Unsupported language: {language}")

        return self.graph

    def _make_node_id(
        self, file_path: str, node_type: NodeType, name: str, line: int = 0
    ) -> str:
        return f"{file_path}::{node_type.value}::{name}::{line}"

    def _extract_text(self, node, source_code: bytes) -> str:
        return source_code[node.start_byte : node.end_byte].decode(
            "utf-8", errors="ignore"
        )

    # ------------------------------------------
    # Python Extraction
    # ------------------------------------------
    def _traverse_python(
        self, node, file_path, file_node_id, source_code, parent_class, parent_func
    ):
        next_parent_class = parent_class
        next_parent_func = parent_func

        if node.type == "class_definition":
            class_name = self._get_python_class_name(node, source_code)
            class_id = self._make_node_id(
                file_path, NodeType.CLASS, class_name, node.start_point[0]
            )
            self.graph.add_node(
                GraphNode(
                    id=class_id,
                    type=NodeType.CLASS,
                    name=class_name,
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

            for child in node.children:
                if child.type == "argument_list":
                    for arg in child.children:
                        if arg.type == "identifier":
                            parent_name = self._extract_text(arg, source_code)
                            parent_id = self._make_node_id(
                                file_path, NodeType.CLASS, parent_name, 0
                            )
                            self.graph.add_node(
                                GraphNode(
                                    id=parent_id,
                                    type=NodeType.CLASS,
                                    name=parent_name,
                                    file_path=file_path,
                                )
                            )
                            self.graph.add_edge(
                                GraphEdge(
                                    source_id=class_id,
                                    target_id=parent_id,
                                    type=EdgeType.EXTENDS,
                                )
                            )
            next_parent_class = class_id
            next_parent_func = None

        elif node.type == "function_definition":
            func_name = self._get_python_func_name(node, source_code)
            is_method = parent_class is not None
            node_type = NodeType.METHOD if is_method else NodeType.FUNCTION
            func_id = self._make_node_id(
                file_path, node_type, func_name, node.start_point[0]
            )
            self.graph.add_node(
                GraphNode(
                    id=func_id,
                    type=node_type,
                    name=func_name,
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                )
            )
            self.graph.add_edge(
                GraphEdge(
                    source_id=parent_class if is_method else file_node_id,
                    target_id=func_id,
                    type=EdgeType.CONTAINS,
                )
            )
            next_parent_func = func_id

        elif node.type == "call" and parent_func:
            func_name = self._get_python_call_name(node, source_code)
            if func_name:
                target_id = self._make_node_id(
                    file_path, NodeType.FUNCTION, func_name, 0
                )
                self.graph.add_node(
                    GraphNode(
                        id=target_id,
                        type=NodeType.FUNCTION,
                        name=func_name,
                        file_path=file_path,
                    )
                )
                self.graph.add_edge(
                    GraphEdge(
                        source_id=parent_func, target_id=target_id, type=EdgeType.CALLS
                    )
                )

        elif node.type in ("import_statement", "import_from_statement"):
            module_name = self._get_python_import_name(node, source_code)
            if module_name:
                mod_id = self._make_node_id(
                    file_path, NodeType.MODULE, module_name, node.start_point[0]
                )
                self.graph.add_node(
                    GraphNode(
                        id=mod_id,
                        type=NodeType.MODULE,
                        name=module_name,
                        file_path=file_path,
                    )
                )
                self.graph.add_edge(
                    GraphEdge(
                        source_id=file_node_id, target_id=mod_id, type=EdgeType.IMPORTS
                    )
                )

        for child in node.children:
            self._traverse_python(
                child,
                file_path,
                file_node_id,
                source_code,
                next_parent_class,
                next_parent_func,
            )

    def _get_python_class_name(self, node, source_code):
        for child in node.children:
            if child.type == "identifier":
                return self._extract_text(child, source_code)
        return "unknown_class"

    def _get_python_func_name(self, node, source_code):
        for child in node.children:
            if child.type == "identifier":
                return self._extract_text(child, source_code)
        return "unknown_func"

    def _get_python_call_name(self, node, source_code):
        for child in node.children:
            if child.type in ("identifier", "attribute"):
                return self._extract_text(child, source_code)
        return None

    def _get_python_import_name(self, node, source_code):
        for child in node.children:
            if child.type in ("dotted_name", "relative_import"):
                return self._extract_text(child, source_code)
        return None

    # ------------------------------------------
    # JavaScript Extraction
    # ------------------------------------------
    def _traverse_javascript(
        self, node, file_path, file_node_id, source_code, parent_class, parent_func
    ):
        next_parent_class = parent_class
        next_parent_func = parent_func

        if node.type == "class_declaration":
            class_name = self._get_js_class_name(node, source_code)
            class_id = self._make_node_id(
                file_path, NodeType.CLASS, class_name, node.start_point[0]
            )
            self.graph.add_node(
                GraphNode(
                    id=class_id,
                    type=NodeType.CLASS,
                    name=class_name,
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

            for child in node.children:
                if child.type == "class_heritage":
                    for sub_child in child.children:
                        if sub_child.type == "identifier":
                            parent_name = self._extract_text(sub_child, source_code)
                            parent_id = self._make_node_id(
                                file_path, NodeType.CLASS, parent_name, 0
                            )
                            self.graph.add_node(
                                GraphNode(
                                    id=parent_id,
                                    type=NodeType.CLASS,
                                    name=parent_name,
                                    file_path=file_path,
                                )
                            )
                            self.graph.add_edge(
                                GraphEdge(
                                    source_id=class_id,
                                    target_id=parent_id,
                                    type=EdgeType.EXTENDS,
                                )
                            )
            next_parent_class = class_id
            next_parent_func = None

        elif node.type in (
            "function_declaration",
            "method_definition",
            "arrow_function",
        ):
            func_name = self._get_js_func_name(node, source_code)
            is_method = parent_class is not None and node.type == "method_definition"
            node_type = NodeType.METHOD if is_method else NodeType.FUNCTION
            func_id = self._make_node_id(
                file_path, node_type, func_name, node.start_point[0]
            )
            self.graph.add_node(
                GraphNode(
                    id=func_id,
                    type=node_type,
                    name=func_name,
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                )
            )
            self.graph.add_edge(
                GraphEdge(
                    source_id=parent_class if is_method else file_node_id,
                    target_id=func_id,
                    type=EdgeType.CONTAINS,
                )
            )
            next_parent_func = func_id

        elif node.type == "call_expression" and parent_func:
            func_name = self._get_js_call_name(node, source_code)
            if func_name:
                target_id = self._make_node_id(
                    file_path, NodeType.FUNCTION, func_name, 0
                )
                self.graph.add_node(
                    GraphNode(
                        id=target_id,
                        type=NodeType.FUNCTION,
                        name=func_name,
                        file_path=file_path,
                    )
                )
                self.graph.add_edge(
                    GraphEdge(
                        source_id=parent_func, target_id=target_id, type=EdgeType.CALLS
                    )
                )

        elif node.type == "import_statement":
            module_name = self._get_js_import_name(node, source_code)
            if module_name:
                mod_id = self._make_node_id(
                    file_path, NodeType.MODULE, module_name, node.start_point[0]
                )
                self.graph.add_node(
                    GraphNode(
                        id=mod_id,
                        type=NodeType.MODULE,
                        name=module_name,
                        file_path=file_path,
                    )
                )
                self.graph.add_edge(
                    GraphEdge(
                        source_id=file_node_id, target_id=mod_id, type=EdgeType.IMPORTS
                    )
                )

        for child in node.children:
            self._traverse_javascript(
                child,
                file_path,
                file_node_id,
                source_code,
                next_parent_class,
                next_parent_func,
            )

    def _get_js_class_name(self, node, source_code):
        for child in node.children:
            if child.type == "identifier":
                return self._extract_text(child, source_code)
        return "unknown_class"

    def _get_js_func_name(self, node, source_code):
        for child in node.children:
            if child.type in ("identifier", "property_identifier"):
                return self._extract_text(child, source_code)
        return "unknown_func"

    def _get_js_call_name(self, node, source_code):
        for child in node.children:
            if child.type in ("identifier", "member_expression"):
                return self._extract_text(child, source_code)
        return None

    def _get_js_import_name(self, node, source_code):
        for child in node.children:
            if child.type == "string":
                return self._extract_text(child, source_code).strip("'\"")
        return None

    # ------------------------------------------
    # Rust Extraction
    # ------------------------------------------
    def _traverse_rust(
        self, node, file_path, file_node_id, source_code, parent_struct, parent_func
    ):
        next_parent_struct = parent_struct
        next_parent_func = parent_func

        if node.type in ("struct_item", "enum_item"):
            class_name = self._get_rust_struct_name(node, source_code)
            class_id = self._make_node_id(
                file_path, NodeType.CLASS, class_name, node.start_point[0]
            )
            self.graph.add_node(
                GraphNode(
                    id=class_id,
                    type=NodeType.CLASS,
                    name=class_name,
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
            next_parent_struct = class_id

        elif node.type == "function_item":
            func_name = self._get_rust_func_name(node, source_code)
            is_method = parent_struct is not None
            node_type = NodeType.METHOD if is_method else NodeType.FUNCTION
            func_id = self._make_node_id(
                file_path, node_type, func_name, node.start_point[0]
            )
            self.graph.add_node(
                GraphNode(
                    id=func_id,
                    type=node_type,
                    name=func_name,
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                )
            )
            self.graph.add_edge(
                GraphEdge(
                    source_id=parent_struct if is_method else file_node_id,
                    target_id=func_id,
                    type=EdgeType.CONTAINS,
                )
            )
            next_parent_func = func_id

        elif node.type == "call_expression" and parent_func:
            func_name = self._get_rust_call_name(node, source_code)
            if func_name:
                target_id = self._make_node_id(
                    file_path, NodeType.FUNCTION, func_name, 0
                )
                self.graph.add_node(
                    GraphNode(
                        id=target_id,
                        type=NodeType.FUNCTION,
                        name=func_name,
                        file_path=file_path,
                    )
                )
                self.graph.add_edge(
                    GraphEdge(
                        source_id=parent_func, target_id=target_id, type=EdgeType.CALLS
                    )
                )

        elif node.type in ("use_declaration", "mod_declaration"):
            module_name = self._get_rust_import_name(node, source_code)
            if module_name:
                mod_id = self._make_node_id(
                    file_path, NodeType.MODULE, module_name, node.start_point[0]
                )
                self.graph.add_node(
                    GraphNode(
                        id=mod_id,
                        type=NodeType.MODULE,
                        name=module_name,
                        file_path=file_path,
                    )
                )
                self.graph.add_edge(
                    GraphEdge(
                        source_id=file_node_id, target_id=mod_id, type=EdgeType.IMPORTS
                    )
                )

        elif node.type == "impl_item":
            struct_name = self._get_rust_impl_target(node, source_code)
            if struct_name:
                struct_id = self._make_node_id(
                    file_path, NodeType.CLASS, struct_name, 0
                )
                self.graph.add_node(
                    GraphNode(
                        id=struct_id,
                        type=NodeType.CLASS,
                        name=struct_name,
                        file_path=file_path,
                    )
                )
                next_parent_struct = struct_id

        for child in node.children:
            self._traverse_rust(
                child,
                file_path,
                file_node_id,
                source_code,
                next_parent_struct,
                next_parent_func,
            )

    def _get_rust_struct_name(self, node, source_code):
        for child in node.children:
            if child.type == "type_identifier":
                return self._extract_text(child, source_code)
        return "unknown_struct"

    def _get_rust_func_name(self, node, source_code):
        for child in node.children:
            if child.type == "identifier":
                return self._extract_text(child, source_code)
        return "unknown_func"

    def _get_rust_call_name(self, node, source_code):
        for child in node.children:
            if child.type in ("identifier", "field_expression"):
                return self._extract_text(child, source_code)
        return None

    def _get_rust_import_name(self, node, source_code):
        for child in node.children:
            if child.type in ("scoped_identifier", "identifier"):
                return self._extract_text(child, source_code)
        return None

    def _get_rust_impl_target(self, node, source_code):
        for child in node.children:
            if child.type == "type_identifier":
                return self._extract_text(child, source_code)
        return None


# ==========================================
# Manual Testing Block
# ==========================================
if __name__ == "__main__":
    """
    Manual Testing Block
    --------------------
    To run this test, ensure you have tree-sitter and the language bindings installed.
    
    pip install tree-sitter tree-sitter-python tree-sitter-javascript pydantic
    """
    import sys

    try:
        import tree_sitter_javascript as tsjavascript
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
    except ImportError as e:
        print(f"❌ Missing dependencies for manual test: {e}")
        print(
            "Please run: pip install tree-sitter "
            "tree-sitter-python tree-sitter-javascript"
        )
        sys.exit(1)

    def print_graph_summary(graph: SymbolGraph):
        print("\n" + "=" * 70)
        print("GRAPH EXTRACTION RESULTS")
        print("=" * 70)

        # Print Nodes (Cleaned up to hide the long IDs)
        print(f"\n📦 Nodes ({len(graph.nodes)}):")
        for node in graph.nodes:
            line_info = (
                f" (Lines {node.start_line}-{node.end_line})" if node.start_line else ""
            )
            print(f"  [{node.type.value:<8}] {node.name}{line_info}")

        # Print Edges (Resolved to actual Node Names and Types)
        print(f"\n🔗 Edges ({len(graph.edges)}):")
        for edge in graph.edges:
            # Look up the actual nodes from the graph using their IDs
            src_node = graph.get_node(edge.source_id)
            tgt_node = graph.get_node(edge.target_id)

            src_name = src_node.name if src_node else "Unknown"
            tgt_name = tgt_node.name if tgt_node else "Unknown"

            src_type = src_node.type.value if src_node else "?"
            tgt_type = tgt_node.type.value if tgt_node else "?"

            # Format: [Type] Name --(Edge)--> [Type] Name
            print(
                f"  [{src_type}] {src_name:<20} --({edge.type.value})--> "
                f"[{tgt_type}] {tgt_name}"
            )

        print("=" * 70 + "\n")

    # ==========================================
    # 1. Test Python Extraction
    # ==========================================
    print("🐍 Testing Python AST Extraction...")
    PY_LANGUAGE = Language(tspython.language())
    py_parser = Parser(PY_LANGUAGE)

    python_code = b"""
import os
from typing import List

class BaseClass:
    pass

class MyClass(BaseClass):
    def my_method(self):
        print("Hello")
        
def my_function():
    my_method()
    os.path.join("a", "b")
"""
    py_tree = py_parser.parse(python_code)
    py_builder = GraphBuilder()
    py_builder.extract_graph("src/test_script.py", py_tree.root_node, python_code)
    print_graph_summary(py_builder.graph)

    # ==========================================
    # 2. Test JavaScript Extraction (.mjs to match chunker.py support)
    # ==========================================
    print("🟨 Testing JavaScript AST Extraction...")
    JS_LANGUAGE = Language(tsjavascript.language())
    js_parser = Parser(JS_LANGUAGE)

    js_code = b"""
import { something } from 'my-module';

class ParentClass {}

class ChildClass extends ParentClass {
    constructor() {
        super();
    }
    
    myMethod() {
        console.log("test");
    }
}

function myFunc() {
    myMethod();
}
"""
    js_tree = js_parser.parse(js_code)
    js_builder = GraphBuilder()
    # Using .mjs to explicitly test the extension map alignment with chunker.py
    js_builder.extract_graph("src/test_script.mjs", js_tree.root_node, js_code)
    print_graph_summary(js_builder.graph)

    print("✅ Manual testing complete!")
