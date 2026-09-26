"""
test_graph_builder.py

Pytest suite for the Context Engine Graph Builder.
Tests the extraction of nodes and edges from ASTs across Python, JavaScript,
and Rust, and verifies architectural constraints (memory-only, no DB deps).
"""

import inspect
import os
import sys

import pytest
import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
import tree_sitter_rust as tsrust
from pydantic import BaseModel
from tree_sitter import Language, Parser

from context_engine.graph_builder import GraphBuilder
from models.context_engine_models import SymbolGraph

# ==========================================
# Path Setup
# ==========================================
# Ensure the 'src' directory is in the Python path.
# Adjust if your project structure differs (e.g., CodeVeto/CodeVeto/src)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
if os.path.exists(src_path):
    sys.path.insert(0, src_path)
else:
    # Fallback for nested structures like CodeVeto/CodeVeto/src
    sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../CodeVeto/src")))
    sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..")))


# ==========================================
# Helpers & Fixtures
# ==========================================


def get_edge_names(graph: SymbolGraph):
    """Helper to convert edges from IDs to human-readable names."""
    return [
        (
            graph.get_node(e.source_id).name,
            e.type.value,
            graph.get_node(e.target_id).name,
        )
        for e in graph.edges
        if graph.get_node(e.source_id) and graph.get_node(e.target_id)
    ]


@pytest.fixture
def py_parser():
    """Provides a Tree-Sitter parser for Python."""
    PY_LANGUAGE = Language(tspython.language())
    return Parser(PY_LANGUAGE)


@pytest.fixture
def js_parser():
    """Provides a Tree-Sitter parser for JavaScript."""
    JS_LANGUAGE = Language(tsjavascript.language())
    return Parser(JS_LANGUAGE)


@pytest.fixture
def rs_parser():
    """Provides a Tree-Sitter parser for Rust."""
    RS_LANGUAGE = Language(tsrust.language())
    return Parser(RS_LANGUAGE)


# ==========================================
# Python Extraction Tests
# ==========================================


class TestPythonGraphExtraction:
    def test_file_containment(self, py_parser):
        """Scenario: Extract file containment edges (CONTAINS) for Python."""
        code = b"""
class MyClass:
    def my_method(self):
        pass

def my_function():
    pass
"""
        tree = py_parser.parse(code)
        graph = GraphBuilder().extract_graph("test.py", tree.root_node, code)
        edges = get_edge_names(graph)

        assert ("test.py", "CONTAINS", "MyClass") in edges
        assert ("test.py", "CONTAINS", "my_function") in edges
        assert ("MyClass", "CONTAINS", "my_method") in edges

    def test_imports(self, py_parser):
        """Scenario: Extract import relationships (IMPORTS) for Python."""
        code = b"""
import os
from typing import List
"""
        tree = py_parser.parse(code)
        graph = GraphBuilder().extract_graph("test.py", tree.root_node, code)
        edges = get_edge_names(graph)

        assert ("test.py", "IMPORTS", "os") in edges
        assert ("test.py", "IMPORTS", "typing") in edges

    def test_inheritance(self, py_parser):
        """Scenario: Extract class inheritance (EXTENDS) for Python."""
        code = b"""
class BaseClass:
    pass

class ChildClass(BaseClass):
    pass
"""
        tree = py_parser.parse(code)
        graph = GraphBuilder().extract_graph("test.py", tree.root_node, code)
        edges = get_edge_names(graph)

        assert ("ChildClass", "EXTENDS", "BaseClass") in edges

    def test_function_calls(self, py_parser):
        """Verify CALLS edges are generated for function invocations."""
        code = b"""
def func_a():
    func_b()

def func_b():
    pass
"""
        tree = py_parser.parse(code)
        graph = GraphBuilder().extract_graph("test.py", tree.root_node, code)
        edges = get_edge_names(graph)

        assert ("func_a", "CALLS", "func_b") in edges


# ==========================================
# JavaScript Extraction Tests
# ==========================================


class TestJavaScriptGraphExtraction:
    def test_file_containment_and_inheritance(self, js_parser):
        """Scenario: Extract CONTAINS and EXTENDS for JavaScript."""
        code = b"""
class Parent {}
class Child extends Parent {
    myMethod() {}
}
function myFunc() {}
"""
        tree = js_parser.parse(code)
        # Using .js to match the exact condition in graph_builder.py
        graph = GraphBuilder().extract_graph("test.js", tree.root_node, code)
        edges = get_edge_names(graph)

        assert ("test.js", "CONTAINS", "Parent") in edges
        assert ("test.js", "CONTAINS", "Child") in edges
        assert ("test.js", "CONTAINS", "myFunc") in edges
        assert ("Child", "CONTAINS", "myMethod") in edges
        assert ("Child", "EXTENDS", "Parent") in edges

    def test_imports(self, js_parser):
        """Scenario: Extract IMPORTS for JavaScript."""
        code = b"""
import { foo } from 'my-module';
import bar from "another-module";
"""
        tree = js_parser.parse(code)
        graph = GraphBuilder().extract_graph("test.js", tree.root_node, code)
        edges = get_edge_names(graph)

        assert ("test.js", "IMPORTS", "my-module") in edges
        assert ("test.js", "IMPORTS", "another-module") in edges

    def test_function_calls(self, js_parser):
        """Verify CALLS edges for JavaScript."""
        code = b"""
function funcA() {
    funcB();
}
function funcB() {}
"""
        tree = js_parser.parse(code)
        graph = GraphBuilder().extract_graph("test.js", tree.root_node, code)
        edges = get_edge_names(graph)

        assert ("funcA", "CALLS", "funcB") in edges


# ==========================================
# Rust Extraction Tests
# ==========================================


class TestRustGraphExtraction:
    def test_file_containment_and_methods(self, rs_parser):
        """Scenario: Extract CONTAINS for Rust structs, impls, and functions."""
        code = b"""
struct MyStruct {
    x: i32,
}

impl MyStruct {
    fn my_method(&self) {}
}

fn my_function() {}
"""
        tree = rs_parser.parse(code)
        graph = GraphBuilder().extract_graph("test.rs", tree.root_node, code)
        edges = get_edge_names(graph)

        assert ("test.rs", "CONTAINS", "MyStruct") in edges
        assert ("test.rs", "CONTAINS", "my_function") in edges
        # Methods inside impl blocks should be contained by the struct
        assert ("MyStruct", "CONTAINS", "my_method") in edges

    def test_imports(self, rs_parser):
        """Scenario: Extract IMPORTS for Rust."""
        code = b"""
use std::collections::HashMap;
mod my_module;
"""
        tree = rs_parser.parse(code)
        graph = GraphBuilder().extract_graph("test.rs", tree.root_node, code)
        edges = get_edge_names(graph)

        # Verify that use/mod declarations are captured as IMPORTS
        import_edges = [tgt for src, edge_type, tgt in edges if edge_type == "IMPORTS"]
        assert len(import_edges) >= 1


# ==========================================
# Architecture & Constraints Tests
# ==========================================


class TestArchitectureConstraints:
    def test_memory_only_no_db_imports(self):
        """Scenario: Output memory-only data structure independent of DB."""
        from context_engine import graph_builder
        from models import context_engine_models

        db_keywords = ["sqlite3", "sqlalchemy", "neo4j", "psycopg2", "pymongo", "motor"]

        gb_source = inspect.getsource(graph_builder)
        for keyword in db_keywords:
            assert keyword not in gb_source, (
                f"Found forbidden DB import '{keyword}' in graph_builder.py"
            )

        models_source = inspect.getsource(context_engine_models)
        for keyword in db_keywords:
            assert keyword not in models_source, (
                f"Found forbidden DB import '{keyword}' in models.py"
            )

    def test_symbol_graph_is_pydantic_model(self):
        """Verify that SymbolGraph is a pure Pydantic BaseModel."""
        assert issubclass(SymbolGraph, BaseModel)

    def test_deduplication_logic(self, py_parser):
        """Verify adding the same node/edge twice doesn't create duplicates."""
        code = b"""
def my_func():
    pass
"""
        tree = py_parser.parse(code)
        builder = GraphBuilder()

        # First pass
        graph = builder.extract_graph("test.py", tree.root_node, code)
        initial_nodes = len(graph.nodes)
        initial_edges = len(graph.edges)

        # Second pass on the exact same builder and file
        builder.extract_graph("test.py", tree.root_node, code)

        # Counts should remain exactly the same due to internal indexing
        assert len(graph.nodes) == initial_nodes
        assert len(graph.edges) == initial_edges
