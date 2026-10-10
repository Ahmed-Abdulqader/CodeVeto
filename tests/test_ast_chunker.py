import ast
import os

from context_engine.ast_chunker import (
    ASTVisitor,
    TopLevelCodeVisitor,
    content_hash,
    get_file_data,
    normalize_path,
)


def visit(source: str) -> tuple[ASTVisitor, TopLevelCodeVisitor]:
    tree = ast.parse(source)
    visitor = ASTVisitor(source)
    visitor.visit(tree)
    top = TopLevelCodeVisitor(source)
    top.visit(tree)
    return visitor, top


def by_name(records: list[dict], name: str) -> list[dict]:
    return [r for r in records if r["name"] == name]


# ---------------------------------------------------------------- file data
def test_file_data_only_holds_path_and_hash(tmp_path):
    path = tmp_path / "a.py"
    path.write_text("x = 1\n")
    data = get_file_data(str(path), "x = 1\n")
    assert set(data) == {"file_path", "content_hash"}
    assert data["content_hash"] == content_hash("x = 1\n")


def test_content_hash_changes_with_content():
    assert content_hash("a") != content_hash("b")
    assert content_hash("a") == content_hash("a")


def test_normalize_path_handles_relative_paths_and_symlinks(tmp_path, monkeypatch):
    real = tmp_path / "real.py"
    real.write_text("")
    link = tmp_path / "link.py"
    os.symlink(real, link)
    monkeypatch.chdir(tmp_path)
    assert normalize_path("real.py") == normalize_path(str(real))
    assert normalize_path(str(link)) == normalize_path(str(real))


# ------------------------------------------------------------------ imports
def test_imports_are_module_level_statements_only(sample_source):
    visitor, _ = visit(sample_source)
    imports = visitor.get_imports()
    assert [i["code"] for i in imports] == [
        "import os",
        "import sys as system",
        "from collections import (\n    OrderedDict,\n    defaultdict,\n)",
        "from . import sibling",
    ]
    assert [(i["start_line"], i["end_line"]) for i in imports] == [
        (2, 2),
        (3, 3),
        (4, 7),
        (8, 8),
    ]
    # `import json` (inside try) and `import socket` (inside a function) belong
    # to other chunks, not to the import list
    assert all("json" not in i["code"] and "socket" not in i["code"] for i in imports)


def test_import_records_only_have_what_the_agent_uses(sample_source):
    visitor, _ = visit(sample_source)
    assert all(
        set(i) == {"start_line", "end_line", "code"} for i in visitor.get_imports()
    )


# ---------------------------------------------------------- top-level chunks
def test_top_level_chunks_exclude_imports_functions_and_classes(sample_source):
    _, top = visit(sample_source)
    chunks = top.get_chunks()
    assert [c["code"] for c in chunks] == [
        '"""Sample module."""',
        "MAX_RETRIES = 3",
        "try:\n    import json\nexcept ImportError:\n    json = None",
        'if __name__ == "__main__":\n    helper(1)',
    ]
    assert [(c["start_line"], c["end_line"]) for c in chunks] == [
        (1, 1),
        (10, 10),
        (12, 15),
        (66, 67),
    ]
    assert all(set(c) == {"start_line", "end_line", "code"} for c in chunks)


# ---------------------------------------------------------------- functions
def test_only_top_level_functions_are_functions(sample_source):
    visitor, _ = visit(sample_source)
    assert [f["name"] for f in visitor.get_functions()] == ["helper", "fetch"]


def test_function_record(sample_source):
    visitor, _ = visit(sample_source)
    (fetch,) = by_name(visitor.get_functions(), "fetch")
    assert set(fetch) == {
        "name",
        "signature",
        "docstring",
        "start_line",
        "end_line",
        "code",
    }
    # decorators are part of the chunk, and of its line range
    assert (fetch["start_line"], fetch["end_line"]) == (22, 33)
    assert fetch["code"].startswith("@cache\nasync def fetch(")
    assert fetch["code"].endswith("    return inner()")
    assert fetch["signature"] == (
        "@cache\nasync def fetch(url: str, *, retries: int=MAX_RETRIES) -> str:"
    )
    assert fetch["docstring"] == "Fetch a URL.\n\nRetries on failure."


def test_function_without_docstring_has_none(sample_source):
    visitor, _ = visit(sample_source)
    (helper,) = by_name(visitor.get_functions(), "helper")
    assert helper["docstring"] is None
    assert helper["signature"] == "def helper(a, b=2):"
    assert (helper["start_line"], helper["end_line"]) == (18, 19)


def test_nested_functions_are_not_separate_chunks(sample_source):
    visitor, _ = visit(sample_source)
    names = [f["name"] for f in visitor.get_functions()] + [
        m["name"] for m in visitor.get_methods()
    ]
    assert "inner" not in names


# ------------------------------------------------------------------ classes
def test_class_record(sample_source):
    visitor, _ = visit(sample_source)
    (account,) = by_name(visitor.get_classes(), "Account")
    assert account["signature"] == "@dataclass\nclass Account(Base, metaclass=Meta):"
    assert account["docstring"] == "A bank account."
    assert (account["start_line"], account["end_line"]) == (40, 63)
    assert account["code"].startswith("@dataclass\nclass Account")
    assert account["code"].endswith('await fetch("x")')


def test_class_without_bases_or_docstring(sample_source):
    visitor, _ = visit(sample_source)
    (tiny,) = by_name(visitor.get_classes(), "Tiny")
    assert tiny["signature"] == "class Tiny:"
    assert tiny["docstring"] is None
    assert tiny["code"] == "class Tiny:\n    pass"


# ------------------------------------------------------------------ methods
def test_methods_know_their_class(sample_source):
    visitor, _ = visit(sample_source)
    methods = visitor.get_methods()
    assert [m["name"] for m in methods] == [
        "deposit",
        "create",
        "value",
        "value",
        "sync",
    ]
    assert {(m["class_name"], m["class_start_line"]) for m in methods} == {
        ("Account", 40)
    }


def test_method_code_is_dedented_and_keeps_decorators(sample_source):
    visitor, _ = visit(sample_source)
    deposit = by_name(visitor.get_methods(), "deposit")[0]
    assert deposit["code"] == (
        "def deposit(self, amount: int) -> None:\n"
        '    """Add money."""\n'
        "    self.balance += amount"
    )
    assert (deposit["start_line"], deposit["end_line"]) == (46, 48)

    create = by_name(visitor.get_methods(), "create")[0]
    assert create["code"].startswith("@staticmethod\ndef create()")
    assert create["start_line"] == 50
    assert create["signature"].startswith("@staticmethod\ndef create() ->")


def test_property_getter_and_setter_are_both_kept(sample_source):
    visitor, _ = visit(sample_source)
    getter, setter = by_name(visitor.get_methods(), "value")
    assert getter["signature"] == "@property\ndef value(self):"
    assert setter["signature"] == "@value.setter\ndef value(self, v):"


def test_async_method_signature(sample_source):
    visitor, _ = visit(sample_source)
    (sync,) = by_name(visitor.get_methods(), "sync")
    assert sync["signature"] == "async def sync(self):"


def test_dedent_leaves_lines_less_indented_than_the_def_alone():
    source = (
        'class A:\n    def f(self):\n        return """\nunindented line\n        """\n'
    )
    visitor, _ = visit(source)
    (f,) = visitor.get_methods()
    assert f["code"] == 'def f(self):\n    return """\nunindented line\n    """'


def test_nested_class_is_recorded_with_its_own_methods():
    source = "class Outer:\n    class Inner:\n        def m(self): ...\n"
    visitor, _ = visit(source)
    assert [c["name"] for c in visitor.get_classes()] == ["Outer", "Inner"]
    (m,) = visitor.get_methods()
    assert (m["class_name"], m["class_start_line"]) == ("Inner", 2)


def test_empty_source_gives_no_chunks():
    visitor, top = visit("")
    assert (
        visitor.get_imports() == visitor.get_classes() == visitor.get_functions() == []
    )
    assert visitor.get_methods() == top.get_chunks() == []
