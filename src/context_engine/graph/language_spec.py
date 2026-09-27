"""
Declarative per-language node classification for the symbol graph.

Why this file exists: the old graph_builder.py had one ~150-line
`_traverse_<language>` method per language, each repeating the same
"build a node, add a CONTAINS edge, recurse" shape with only the tree-sitter
node-type names differing. That's what pushed it past 600 lines and made it
hard to add a fourth language without copy-pasting a fifth traversal method.

Here, the *shape* of the walk lives once in `graph_builder.GraphBuilder`.
What differs per language — which node types are classes, which are calls,
how to read a name back out of a node — lives here as data, in a
`LanguageGraphSpec`. Adding a language means adding one spec, not one
method.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from tree_sitter import Node

NameExtractor = Callable[[Node, bytes], str | None]
NamesExtractor = Callable[[Node, bytes], list[str]]
MethodPredicate = Callable[[Node, str | None], bool]


# ---------------------------------------------------------------------------
# Small extractor-building helpers, shared across every language below
# ---------------------------------------------------------------------------


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def _first_child_of_type(node: Node, types: tuple[str, ...]) -> Node | None:
    for child in node.children:
        if child.type in types:
            return child
    return None


def name_from_child(
    types: tuple[str, ...], default: str | None = None
) -> NameExtractor:
    """Most node names in every grammar here are 'the text of the first
    child whose type is X' (an `identifier`, a `type_identifier`, ...).
    This covers that common case so each language only has to say *which*
    child types to look for."""

    def extractor(node: Node, source: bytes) -> str | None:
        child = _first_child_of_type(node, types)
        return _text(child, source) if child is not None else default

    return extractor


def _default_is_method(_node: Node, parent_container: str | None) -> bool:
    """Default rule: a function/def is a method if we're currently walking
    inside a class/struct/impl body. Python and Rust both use this as-is;
    JavaScript overrides it (see below) because a plain `function` or arrow
    function that happens to be lexically inside a class body is still not
    a method there — only an explicit `method_definition` node is."""
    return parent_container is not None


# ---------------------------------------------------------------------------
# The spec itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LanguageGraphSpec:
    """Everything graph_builder needs to know about one tree-sitter grammar."""

    class_types: frozenset[str]
    function_types: frozenset[str]
    call_types: frozenset[str]
    import_types: frozenset[str]

    class_name: NameExtractor
    function_name: NameExtractor
    call_name: NameExtractor
    import_name: NameExtractor

    # Optional: base classes/interfaces to record as EXTENDS edges.
    extends_names: NamesExtractor | None = None

    # Optional: node types that open a class context without being a class
    # node themselves — Rust's `impl Foo { ... }` blocks. None for
    # languages that don't need it.
    impl_types: frozenset[str] = field(default_factory=frozenset)
    impl_target_name: NameExtractor | None = None

    method_predicate: MethodPredicate = _default_is_method

    def is_class(self, node: Node) -> bool:
        return node.type in self.class_types

    def is_function(self, node: Node) -> bool:
        return node.type in self.function_types

    def is_call(self, node: Node) -> bool:
        return node.type in self.call_types

    def is_import(self, node: Node) -> bool:
        return node.type in self.import_types

    def is_impl(self, node: Node) -> bool:
        return node.type in self.impl_types


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def _python_extends_names(node: Node, source: bytes) -> list[str]:
    names = []
    arglist = _first_child_of_type(node, ("argument_list",))
    if arglist is not None:
        for arg in arglist.children:
            if arg.type == "identifier":
                names.append(_text(arg, source))
    return names


PYTHON_SPEC = LanguageGraphSpec(
    class_types=frozenset({"class_definition"}),
    function_types=frozenset({"function_definition"}),
    call_types=frozenset({"call"}),
    import_types=frozenset({"import_statement", "import_from_statement"}),
    class_name=name_from_child(("identifier",), "unknown_class"),
    function_name=name_from_child(("identifier",), "unknown_func"),
    call_name=name_from_child(("identifier", "attribute"), None),
    import_name=name_from_child(("dotted_name", "relative_import"), None),
    extends_names=_python_extends_names,
)


# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------


def _js_extends_names(node: Node, source: bytes) -> list[str]:
    names = []
    heritage = _first_child_of_type(node, ("class_heritage",))
    if heritage is not None:
        for child in heritage.children:
            if child.type == "identifier":
                names.append(_text(child, source))
    return names


def _js_import_name(node: Node, source: bytes) -> str | None:
    child = _first_child_of_type(node, ("string",))
    return _text(child, source).strip("'\"") if child is not None else None


JAVASCRIPT_SPEC = LanguageGraphSpec(
    class_types=frozenset({"class_declaration"}),
    function_types=frozenset(
        {"function_declaration", "method_definition", "arrow_function"}
    ),
    call_types=frozenset({"call_expression"}),
    import_types=frozenset({"import_statement"}),
    class_name=name_from_child(("identifier",), "unknown_class"),
    function_name=name_from_child(
        ("identifier", "property_identifier"), "unknown_func"
    ),
    call_name=name_from_child(("identifier", "member_expression"), None),
    import_name=_js_import_name,
    extends_names=_js_extends_names,
    method_predicate=lambda node, parent: (
        parent is not None and node.type == "method_definition"
    ),
)


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------

RUST_SPEC = LanguageGraphSpec(
    class_types=frozenset({"struct_item", "enum_item"}),
    function_types=frozenset({"function_item"}),
    call_types=frozenset({"call_expression"}),
    import_types=frozenset({"use_declaration", "mod_declaration"}),
    class_name=name_from_child(("type_identifier",), "unknown_struct"),
    function_name=name_from_child(("identifier",), "unknown_func"),
    call_name=name_from_child(("identifier", "field_expression"), None),
    import_name=name_from_child(("scoped_identifier", "identifier"), None),
    impl_types=frozenset({"impl_item"}),
    # NOTE: for `impl Trait for Foo`, this grabs the first `type_identifier`
    # (`Trait`), not `Foo` — same limitation the original hand-written Rust
    # traversal had. Fine for plain `impl Foo { ... }` blocks, which is the
    # common case; a real fix needs to special-case the `for` clause.
    impl_target_name=name_from_child(("type_identifier",), None),
)


LANGUAGE_SPECS: dict[str, LanguageGraphSpec] = {
    "python": PYTHON_SPEC,
    "javascript": JAVASCRIPT_SPEC,
    "rust": RUST_SPEC,
}
