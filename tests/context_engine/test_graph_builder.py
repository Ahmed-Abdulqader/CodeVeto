from __future__ import annotations

from context_engine.chunker import Chunks
from context_engine.graph.graph_builder import GraphBuilder
from models.context_engine_models import EdgeType, NodeType


def _build_graph(tmp_path, filename, content):
    path = tmp_path / filename
    path.write_text(content)
    parsed = Chunks(str(path))
    graph = GraphBuilder().extract_graph(
        str(path), parsed.tree.root_node, parsed.source_bytes, parsed.language
    )
    return graph


def _names_of_type(graph, node_type):
    return {n.name for n in graph.nodes if n.type == node_type}


def _edges_of_type(graph, edge_type):
    return {(e.source_id, e.target_id) for e in graph.edges if e.type == edge_type}


class TestPythonGraph:
    def test_class_and_methods(self, tmp_path):
        graph = _build_graph(
            tmp_path,
            "m.py",
            "class Animal:\n"
            "    def speak(self):\n"
            "        return noise()\n"
            "\n"
            "def noise():\n"
            "    return 'generic sound'\n",
        )
        assert "Animal" in _names_of_type(graph, NodeType.CLASS)
        # speak() is nested in a class body -> METHOD, not FUNCTION
        assert "speak" in _names_of_type(graph, NodeType.METHOD)
        assert "noise" in _names_of_type(graph, NodeType.FUNCTION)

    def test_extends_edge_for_multiple_inheritance(self, tmp_path):
        graph = _build_graph(
            tmp_path,
            "m.py",
            "class Base1:\n    pass\n\n"
            "class Base2:\n    pass\n\n"
            "class Child(Base1, Base2):\n    pass\n",
        )
        child_id = next(n.id for n in graph.nodes if n.name == "Child")
        extends_targets = {
            graph.get_node(tgt).name
            for src, tgt in _edges_of_type(graph, EdgeType.EXTENDS)
            if src == child_id
        }
        assert extends_targets == {"Base1", "Base2"}

    def test_calls_edge(self, tmp_path):
        graph = _build_graph(
            tmp_path,
            "m.py",
            "def outer():\n    return inner()\n\ndef inner():\n    return 1\n",
        )
        call_targets = {
            graph.get_node(tgt).name for _, tgt in _edges_of_type(graph, EdgeType.CALLS)
        }
        assert "inner" in call_targets

    def test_import_edge(self, tmp_path):
        graph = _build_graph(tmp_path, "m.py", "import os\n")
        assert "os" in _names_of_type(graph, NodeType.MODULE)


class TestJavaScriptGraph:
    def test_method_definition_is_method_but_bare_function_is_not(self, tmp_path):
        graph = _build_graph(
            tmp_path,
            "m.js",
            "class Widget {\n"
            "    render() {\n"
            "        return 1;\n"
            "    }\n"
            "}\n"
            "\n"
            "function helper() {\n"
            "    return 2;\n"
            "}\n",
        )
        assert "render" in _names_of_type(graph, NodeType.METHOD)
        assert "helper" in _names_of_type(graph, NodeType.FUNCTION)
        # a plain function must never be classified as a method, even though
        # the generic walk tracks "currently inside a class" the same way
        # python does -- this is exactly the per-language override
        # method_predicate exists for.
        assert "helper" not in _names_of_type(graph, NodeType.METHOD)

    def test_class_heritage_extends_edge(self, tmp_path):
        graph = _build_graph(
            tmp_path, "m.js", "class Base {}\nclass Child extends Base {}\n"
        )
        child_id = next(n.id for n in graph.nodes if n.name == "Child")
        targets = {
            graph.get_node(tgt).name
            for src, tgt in _edges_of_type(graph, EdgeType.EXTENDS)
            if src == child_id
        }
        assert targets == {"Base"}


class TestRustGraph:
    def test_impl_block_methods_attach_to_struct(self, tmp_path):
        graph = _build_graph(
            tmp_path,
            "m.rs",
            "struct Point { x: i32, y: i32 }\n\n"
            "impl Point {\n    fn norm(&self) -> i32 { self.x }\n}\n",
        )
        assert "Point" in _names_of_type(graph, NodeType.CLASS)
        norm_node = next(n for n in graph.nodes if n.name == "norm")
        assert norm_node.type == NodeType.METHOD
        # CONTAINS edge from the impl's struct node to the method
        contains_sources = {
            src
            for src, tgt in _edges_of_type(graph, EdgeType.CONTAINS)
            if tgt == norm_node.id
        }
        struct_id = next(n.id for n in graph.nodes if n.name == "Point")
        assert struct_id in contains_sources

    def test_free_function_is_not_a_method(self, tmp_path):
        graph = _build_graph(
            tmp_path, "m.rs", "fn add(a: i32, b: i32) -> i32 { a + b }\n"
        )
        node = next(n for n in graph.nodes if n.name == "add")
        assert node.type == NodeType.FUNCTION
