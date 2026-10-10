import pytest

from context_engine import chunk_query
from context_engine.chunk_query import (
    count_tokens,
    get_class_code,
    get_class_methods,
    get_classes,
    get_function_code,
    get_functions,
    get_imports,
    get_method_code,
    get_top_level_chunks,
)
from context_engine.chunks_saver import save_file


def lines_of(source: str, first: int, last: int) -> str:
    """Source lines first..last (1-based, inclusive)."""
    return "\n".join(source.split("\n")[first - 1 : last])


def force_outline(monkeypatch):
    monkeypatch.setattr(
        chunk_query, "count_tokens", lambda t: 1 if "    ..." in t else 1000
    )


def force_full(monkeypatch):
    monkeypatch.setattr(
        chunk_query, "count_tokens", lambda t: 1000 if "    ..." in t else 1
    )


@pytest.fixture
def path(sample_file) -> str:
    return str(sample_file)


# ----------------------------------------------------------- imports / chunks
def test_imports_are_only_the_code(indexed, path):
    assert get_imports(indexed, path) == (
        "```python\n"
        "import os\n"
        "import sys as system\n"
        "from collections import (\n    OrderedDict,\n    defaultdict,\n)\n"
        "from . import sibling\n"
        "```"
    )


def test_top_level_chunks_are_only_the_code(indexed, path):
    assert get_top_level_chunks(indexed, path) == (
        "```python\n"
        '"""Sample module."""\n\n'
        "MAX_RETRIES = 3\n\n"
        "try:\n    import json\nexcept ImportError:\n    json = None\n\n"
        'if __name__ == "__main__":\n    helper(1)\n'
        "```"
    )


def test_file_without_imports_or_top_level_code(conn, tmp_path):
    empty = tmp_path / "only_def.py"
    empty.write_text("def f():\n    pass\n")
    save_file(conn, str(empty))
    assert get_imports(conn, str(empty)) == "`only_def.py` has no module-level imports."
    assert (
        get_top_level_chunks(conn, str(empty))
        == "`only_def.py` has no other module-level code."
    )
    assert get_classes(conn, str(empty)) == "`only_def.py` has no classes."


def test_file_without_functions(conn, tmp_path):
    p = tmp_path / "consts.py"
    p.write_text("X = 1\n")
    save_file(conn, str(p))
    assert get_functions(conn, str(p)) == "`consts.py` has no top-level functions."


# -------------------------------------------------------------------- classes
def test_classes_outline_for_big_class_and_full_source_for_tiny_one(indexed, path):
    # (uses the ~4 chars/token estimate; for `Tiny` the two forms cost the same
    # and a tie goes to the full source, which says more for the same price)
    out = get_classes(indexed, path)
    assert out.startswith("## Classes in `sample.py`")

    tiny, account = out.split("### class ")[1:]
    assert tiny.startswith(
        "`Tiny` (lines 36-37) — full source\n```python\nclass Tiny:\n    pass\n```"
    )
    # the big class is shown as an outline: header + docstring + method names, no bodies
    assert account.startswith(
        "`Account` (lines 40-63)\n"
        "```python\n"
        "@dataclass\n"
        "class Account(Base, metaclass=Meta):"
    )
    assert '"""A bank account."""' in account
    assert "Methods: `deposit`, `create`, `value`, `sync`" in account
    assert "self.balance" not in account


def test_class_outline_lists_each_method_name_once(indexed, path):
    # `value` is defined twice (property getter + setter)
    assert get_classes(indexed, path).count("`value`") == 1


def test_classes_can_be_forced_to_full_source(indexed, path, monkeypatch):
    force_full(monkeypatch)
    out = get_classes(indexed, path)
    assert out.count("— full source") == 2
    assert "self.balance += amount" in out


def test_classes_can_be_forced_to_outline(indexed, path, monkeypatch):
    force_outline(monkeypatch)
    out = get_classes(indexed, path)
    assert "— full source" not in out
    assert "class Tiny:\n    ...\n```" in out


# ------------------------------------------------------------------ functions
def test_functions_outline(indexed, path, monkeypatch):
    force_outline(monkeypatch)
    out = get_functions(indexed, path)
    assert out.startswith("## Functions in `sample.py`")
    assert (
        "### function `helper` (lines 18-19)\n"
        "```python\n"
        "def helper(a, b=2):\n"
        "    ...\n"
        "```" in out
    )
    assert "### function `fetch` (lines 22-33)" in out
    assert "@cache\nasync def fetch(" in out
    assert '    """Fetch a URL.\n\n    Retries on failure."""\n    ...' in out
    assert "import socket" not in out  # the body is left out


def test_functions_full_source_when_cheaper(indexed, path, monkeypatch):
    force_full(monkeypatch)
    out = get_functions(indexed, path)
    assert "### function `helper` (lines 18-19) — full source" in out
    assert "return a + b" in out


# -------------------------------------------------------------------- methods
def test_class_methods_signatures(indexed, path, monkeypatch):
    force_outline(monkeypatch)
    out = get_class_methods(indexed, path, "Account")
    assert out.startswith("## Methods of `Account` (lines 40-63)")
    assert (
        "### method `Account.deposit` (lines 46-48)\n```python\n"
        'def deposit(self, amount: int) -> None:\n    """Add money."""\n    ...\n```'
    ) in out
    assert (
        "### method `Account.sync` (lines 62-63)\n"
        "```python\n"
        "async def sync(self):\n"
        "    ...\n"
        "```" in out
    )
    assert out.count("### method ") == 5
    assert "self.balance" not in out


def test_class_without_methods(indexed, path):
    assert (
        get_class_methods(indexed, path, "Tiny")
        == "Class `Tiny` (lines 36-37) has no methods."
    )


# --------------------------------------------------------------------- expand
def test_function_code(indexed, path, sample_source):
    assert get_function_code(indexed, path, "fetch") == (
        "### function `fetch` (lines 22-33)\n```python\n"
        + lines_of(sample_source, 22, 33)
        + "\n```"
    )


def test_class_code(indexed, path, sample_source):
    assert get_class_code(indexed, path, "Account") == (
        "### class `Account` (lines 40-63)\n```python\n"
        + lines_of(sample_source, 40, 63)
        + "\n```"
    )


def test_method_code_is_dedented(indexed, path):
    assert get_method_code(indexed, path, "Account", "deposit") == (
        "### method `Account.deposit` (lines 46-48)\n"
        "```python\n"
        "def deposit(self, amount: int) -> None:\n"
        '    """Add money."""\n'
        "    self.balance += amount\n"
        "```"
    )


def test_method_with_getter_and_setter_returns_both(indexed, path):
    out = get_method_code(indexed, path, "Account", "value")
    assert out.count("### method `Account.value`") == 2
    assert "@property\ndef value(self):" in out
    assert "@value.setter\ndef value(self, v):" in out


def test_functions_sharing_a_name_are_all_returned(conn, tmp_path):
    p = tmp_path / "overloads.py"
    p.write_text(
        "@overload\n"
        "def f(x: int): ...\n\n"
        "@overload\n"
        "def f(x: str): ...\n\n"
        "def f(x): return x\n"
    )
    save_file(conn, str(p))
    out = get_function_code(conn, str(p), "f")
    assert out.count("### function `f`") == 3


# --------------------------------------------------------------------- errors
@pytest.mark.parametrize(
    "call",
    [
        lambda c, p: get_imports(c, p),
        lambda c, p: get_top_level_chunks(c, p),
        lambda c, p: get_classes(c, p),
        lambda c, p: get_functions(c, p),
        lambda c, p: get_class_methods(c, p, "Account"),
        lambda c, p: get_function_code(c, p, "helper"),
        lambda c, p: get_class_code(c, p, "Account"),
        lambda c, p: get_method_code(c, p, "Account", "deposit"),
    ],
)
def test_unindexed_file_is_an_error(conn, tmp_path, call):
    missing = str(tmp_path / "nope.py")
    assert (
        call(conn, missing) == f"**Error:** File `{missing}` was not found in the index"
    )


def test_unknown_function_lists_the_available_ones(indexed, path):
    assert get_function_code(indexed, path, "nope") == (
        "**Error:** Function `nope` was not found in `sample.py`. "
        "Functions in this file: `helper`, `fetch`."
    )


def test_unknown_class_lists_the_available_ones(indexed, path):
    expected = (
        "**Error:** Class `Nope` was not found in `sample.py`. "
        "Classes in this file: `Tiny`, `Account`."
    )
    assert get_class_code(indexed, path, "Nope") == expected
    assert get_class_methods(indexed, path, "Nope") == expected
    assert get_method_code(indexed, path, "Nope", "deposit") == expected


def test_unknown_method_lists_the_available_ones(indexed, path):
    assert get_method_code(indexed, path, "Account", "nope") == (
        "**Error:** Method `nope` was not found in class `Account`. "
        "Methods of this class: `deposit`, `create`, `value`, `sync`."
    )
    assert "This class has no methods." in get_method_code(indexed, path, "Tiny", "x")


def test_file_with_no_functions_or_classes_says_so(conn, tmp_path):
    p = tmp_path / "x.py"
    p.write_text("X = 1\n")
    save_file(conn, str(p))
    assert "This file has no top-level functions." in get_function_code(
        conn, str(p), "f"
    )
    assert "This file has no classes." in get_class_code(conn, str(p), "C")


def test_long_name_lists_are_capped():
    names = [f"n{i}" for i in range(35)]
    out = chunk_query._names(names)
    assert out.endswith("(and 5 more)")
    assert "`n29`" in out and "`n30`" not in out


# ------------------------------------------------------------- output format
@pytest.mark.parametrize(
    "name", ["get_classes", "get_functions", "get_imports", "get_top_level_chunks"]
)
def test_nothing_is_json(indexed, path, name):
    out = getattr(chunk_query, name)(indexed, path)
    assert not out.lstrip().startswith(("{", "["))
    assert "```python" in out


def test_relative_paths_find_the_indexed_file(indexed, sample_file, monkeypatch):
    monkeypatch.chdir(sample_file.parent)
    assert get_imports(indexed, "sample.py").startswith("```python\nimport os")


def test_fence_grows_when_code_contains_backticks():
    code = 'x = "```"'
    fenced = chunk_query._fence(code)
    assert fenced == f"````python\n{code}\n````"
    assert chunk_query._fence("x = 1") == "```python\nx = 1\n```"


def test_docstring_with_backticks_does_not_break_the_block(conn, tmp_path, monkeypatch):
    p = tmp_path / "doc.py"
    p.write_text('def f():\n    """Use ```code``` here."""\n    return 1\n')
    save_file(conn, str(p))
    force_outline(monkeypatch)
    out = get_functions(conn, str(p))
    assert "````python\ndef f():" in out
    assert out.rstrip().endswith("````")


# -------------------------------------------------------------- token counting
def test_count_tokens_falls_back_to_a_character_estimate():
    assert count_tokens("a" * 40) == 11
    assert count_tokens("") == 1


def test_count_tokens_uses_the_tokenizer_when_available(monkeypatch):
    class FakeEncoder:
        def encode(self, text):
            return text.split()

    monkeypatch.setattr(chunk_query, "_encoder", lambda: FakeEncoder())
    assert count_tokens("one two three") == 3
