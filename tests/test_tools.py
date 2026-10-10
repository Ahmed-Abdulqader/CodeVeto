import ast
import re
import stat
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agents import (
    tools,
)

# ----------------------------------------------------------------------
# Sample sources
# ----------------------------------------------------------------------
SAMPLE = '''\
"""Module docstring."""
import os
import sys


def alpha(x):
    return x + 1


def beta(y):
    return y * 2


class Foo:
    """A class."""

    def method_one(self):
        return 1

    def method_two(self):
        return 2


if __name__ == "__main__":
    alpha(1)
'''

DECORATED = """\
def decorator(f):
    return f


@decorator
def decorated():
    return 1


def after():
    return 0
"""

DUPLICATES = """\
def dup():
    return 1


def dup():
    return 2
"""

PROPERTY = """\
class P:
    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
"""


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------
@pytest.fixture
def db():
    return object()


@pytest.fixture
def ctx(db):
    """tools.py only ever touches `ctx.deps.db`."""
    return SimpleNamespace(deps=SimpleNamespace(db=db))


@pytest.fixture(autouse=True)
def save_file_mock(monkeypatch):
    """Replace the SQLite indexer and make error strings easy to recognise."""
    mock = MagicMock(name="save_file")
    monkeypatch.setattr(tools, "save_file", mock)
    monkeypatch.setattr(tools, "_error", lambda msg: f"ERROR: {msg}")
    return mock


@pytest.fixture
def cq(monkeypatch):
    mock = MagicMock(name="chunk_query")
    monkeypatch.setattr(tools, "chunk_query", mock)
    return mock


@pytest.fixture
def make_py(tmp_path):
    def _make(content=SAMPLE, name="sample.py"):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8", newline="")  # no newline translation
        return path

    return _make


def read(path) -> str:
    return path.read_bytes().decode("utf-8")


def top_level_names(path) -> list[str]:
    body = ast.parse(read(path)).body
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    return [n.name for n in body if isinstance(n, defs)]


def method_names(path, cls_name) -> list[str]:
    tree = ast.parse(read(path))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    return [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]


def assert_rejected(result, path, original, save_mock, *fragments):
    """The edit was refused: error shown, file untouched, index not refreshed."""
    assert result.startswith("ERROR:"), result
    assert "Nothing was written" in result, result
    for fragment in fragments:
        assert fragment in result, result
    assert read(path) == original
    save_mock.assert_not_called()


# ----------------------------------------------------------------------
# _file_path_wrapper
# ----------------------------------------------------------------------
class TestFilePathWrapper:
    def test_existing_py_file_is_resolved(self, make_py):
        p = make_py()
        assert tools._file_path_wrapper(str(p)) == p.resolve()

    def test_surrounding_whitespace_is_stripped(self, make_py):
        p = make_py()
        assert tools._file_path_wrapper(f"  {p}\n") == p.resolve()

    def test_uppercase_extension_is_accepted(self, make_py):
        p = make_py(name="UPPER.PY")
        assert tools._file_path_wrapper(str(p)) == p.resolve()

    def test_missing_file(self, tmp_path):
        assert tools._file_path_wrapper(str(tmp_path / "nope.py")) is None

    def test_directory_named_like_a_py_file(self, tmp_path):
        (tmp_path / "pkg.py").mkdir()
        assert tools._file_path_wrapper(str(tmp_path / "pkg.py")) is None

    def test_other_extension(self, make_py):
        p = make_py(name="notes.txt")
        assert tools._file_path_wrapper(str(p)) is None

    def test_tilde_is_expanded(self, make_py, tmp_path, monkeypatch):
        make_py(name="home_file.py")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        found = tools._file_path_wrapper("~/home_file.py")
        assert found == (tmp_path / "home_file.py").resolve()

    def test_relative_path_uses_cwd(self, make_py, tmp_path, monkeypatch):
        make_py(name="rel.py")
        monkeypatch.chdir(tmp_path)
        assert tools._file_path_wrapper("rel.py") == (tmp_path / "rel.py").resolve()


# ----------------------------------------------------------------------
# _indexed_file
# ----------------------------------------------------------------------
class TestIndexedFile:
    def test_success_indexes_and_returns_resolved_path(
        self, ctx, db, make_py, save_file_mock
    ):
        p = make_py()
        assert tools._indexed_file(ctx, str(p)) == (str(p.resolve()), None)
        save_file_mock.assert_called_once_with(db, str(p.resolve()))

    def test_bad_path_is_reported_without_indexing(self, ctx, tmp_path, save_file_mock):
        path, err = tools._indexed_file(ctx, str(tmp_path / "nope.py"))
        assert path is None
        assert "is not an existing Python (.py) file" in err
        save_file_mock.assert_not_called()

    @pytest.mark.parametrize(
        "exc",
        [
            OSError("disk on fire"),
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        ],
        ids=["OSError", "UnicodeDecodeError"],
    )
    def test_read_failures(self, exc, ctx, make_py, save_file_mock):
        save_file_mock.side_effect = exc
        path, err = tools._indexed_file(ctx, str(make_py()))
        assert path is None
        assert "could not be read" in err

    def test_syntax_error_reports_line_and_reason(self, ctx, make_py, save_file_mock):
        save_file_mock.side_effect = SyntaxError(
            "invalid syntax", ("f.py", 7, 1, "x =")
        )
        path, err = tools._indexed_file(ctx, str(make_py()))
        assert path is None
        assert "cannot be parsed (line 7): invalid syntax" in err
        assert "Fix it before using the code tools." in err

    def test_value_error_has_no_line_number(self, ctx, make_py, save_file_mock):
        save_file_mock.side_effect = ValueError("source contains null bytes")
        path, err = tools._indexed_file(ctx, str(make_py()))
        assert path is None
        assert "cannot be parsed: source contains null bytes" in err
        assert "(line" not in err


# ----------------------------------------------------------------------
# Reading tools
# ----------------------------------------------------------------------
READ_CASES = [
    pytest.param(tools.get_file_imports, (), "get_imports", id="get_file_imports"),
    pytest.param(
        tools.get_file_top_level_chunk, (), "get_top_level_chunks", id="top_level_chunk"
    ),
    pytest.param(tools.get_file_classes, (), "get_classes", id="get_file_classes"),
    pytest.param(
        tools.get_class_methods_signature,
        ("Foo",),
        "get_class_methods",
        id="get_class_methods_signature",
    ),
    pytest.param(
        tools.get_file_functions, (), "get_functions", id="get_file_functions"
    ),
    pytest.param(
        tools.expand_function, ("alpha",), "get_function_code", id="expand_function"
    ),
    pytest.param(tools.expand_class, ("Foo",), "get_class_code", id="expand_class"),
    pytest.param(
        tools.expand_method,
        ("Foo", "method_one"),
        "get_method_code",
        id="expand_method",
    ),
]


class TestReadingTools:
    @pytest.mark.parametrize(("tool", "extra", "query"), READ_CASES)
    def test_reindexes_then_delegates_to_chunk_query(
        self, tool, extra, query, ctx, db, make_py, cq, save_file_mock
    ):
        p = make_py()
        getattr(cq, query).return_value = "RESULT"

        assert tool(ctx, str(p), *extra) == "RESULT"

        save_file_mock.assert_called_once_with(db, str(p.resolve()))
        getattr(cq, query).assert_called_once_with(db, str(p.resolve()), *extra)

    @pytest.mark.parametrize(("tool", "extra", "query"), READ_CASES)
    def test_bad_path_returns_error_and_skips_everything(
        self, tool, extra, query, ctx, tmp_path, cq, save_file_mock
    ):
        result = tool(ctx, str(tmp_path / "missing.py"), *extra)
        assert result.startswith("ERROR:")
        assert "is not an existing Python (.py) file" in result
        assert not getattr(cq, query).called
        save_file_mock.assert_not_called()

    @pytest.mark.parametrize(("tool", "extra", "query"), READ_CASES)
    def test_unparseable_file_returns_error_and_skips_query(
        self, tool, extra, query, ctx, make_py, cq, save_file_mock
    ):
        save_file_mock.side_effect = SyntaxError("bad", ("f.py", 3, 1, "x"))
        result = tool(ctx, str(make_py()), *extra)
        assert result.startswith("ERROR:")
        assert "cannot be parsed (line 3)" in result
        assert not getattr(cq, query).called


# ----------------------------------------------------------------------
# Small text helpers
# ----------------------------------------------------------------------
class TestTextHelpers:
    def test_normalize_newlines(self):
        assert tools._normalize_newlines("a\r\nb\rc\nd") == "a\nb\nc\nd"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("", []),
            ("a\nb\n", ["a", "b"]),
            ("a\nb", ["a", "b"]),
            ("a\n\n", ["a", ""]),
            ("\n", [""]),
        ],
    )
    def test_split(self, text, expected):
        assert tools._split(text) == expected

    def test_join(self):
        assert tools._join([]) == ""
        assert tools._join(["a", "b"]) == "a\nb\n"

    def test_split_and_join_round_trip(self):
        assert tools._join(tools._split(SAMPLE)) == SAMPLE

    @pytest.mark.parametrize(
        ("line", "expected"),
        [("x", ""), ("    x", "    "), ("\tx", "\t"), (" \t x", " \t "), ("", "")],
    )
    def test_indent_of(self, line, expected):
        assert tools._indent_of(line) == expected

    def test_quoted_deduplicates_and_keeps_order(self):
        assert tools._quoted(["b", "a", "b"]) == "`b`, `a`"

    def test_quoted_truncates_with_count(self):
        assert tools._quoted(["a", "b", "c"], limit=2) == "`a`, `b` (and 1 more)"

    def test_quoted_empty(self):
        assert tools._quoted([]) == ""

    def test_block_strips_blank_edges_dedents_and_indents(self):
        code = "\n\n    def f():\n        pass\n\n\n"
        assert tools._block(code, "  ") == ["  def f():", "      pass"]

    def test_block_handles_crlf(self):
        assert tools._block("def f():\r\n    pass\r\n", "") == ["def f():", "    pass"]

    @pytest.mark.parametrize("code", ["", "\n\n", "   \n  \n"])
    def test_block_of_nothing_is_empty(self, code):
        assert tools._block(code, "    ") == []

    def test_block_keeps_blank_lines_inside(self):
        assert tools._block("a\n\nb", "  ") == ["  a", "", "  b"]


# ----------------------------------------------------------------------
# AST helpers
# ----------------------------------------------------------------------
class TestAstHelpers:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("def f(): pass", "a function (`f`)"),
            ("async def g(): pass", "a function (`g`)"),
            ("class C: pass", "a class (`C`)"),
            ("import os", "an import statement"),
            ("from os import path", "an import statement"),
            ("x = 1", "other code (`Assign`)"),
        ],
    )
    def test_describe(self, code, expected):
        assert tools._describe(ast.parse(code).body[0]) == expected

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ('if __name__ == "__main__":\n    pass', True),
            ("if __name__ == '__main__':\n    pass", True),
            ('if __name__ == "other":\n    pass', False),
            ('if name == "__main__":\n    pass', False),
            ('if __name__ != "__main__":\n    pass', True),  # only compares left/right
            ("x = 1", False),
        ],
    )
    def test_is_main_guard(self, code, expected):
        # `!=` still has one comparator equal to "__main__", which the helper accepts
        assert tools._is_main_guard(ast.parse(code).body[0]) is expected

    def test_parse_new_returns_nodes(self):
        nodes = tools._parse_new(
            "def a(): pass\ndef b(): pass", tools._FUNCTIONS, "function definition"
        )
        assert [n.name for n in nodes] == ["a", "b"]

    def test_parse_new_accepts_indented_code(self):
        nodes = tools._parse_new(
            "    def a(self):\n        pass", tools._FUNCTIONS, "x"
        )
        assert nodes[0].name == "a"

    def test_parse_new_empty(self):
        with pytest.raises(tools._EditError, match="`new_code` is empty"):
            tools._parse_new("  \n", tools._FUNCTIONS, "function definition")

    def test_parse_new_syntax_error_has_line(self):
        with pytest.raises(tools._EditError, match=r"not valid Python \(line 1"):
            tools._parse_new("def f(:", tools._FUNCTIONS, "function definition")

    def test_parse_new_wrong_kind_includes_hint(self):
        with pytest.raises(tools._EditError) as exc:
            tools._parse_new(
                "class C: pass", tools._FUNCTIONS, "function definition", hint=" Use X."
            )
        message = str(exc.value)
        assert "only function definitions" in message
        assert "a class (`C`)" in message
        assert message.endswith("Use X.")

    def test_parse_new_single_rejects_two(self):
        with pytest.raises(
            tools._EditError, match="exactly one function definition, found 2"
        ):
            tools._parse_new(
                "def a(): pass\ndef b(): pass",
                tools._FUNCTIONS,
                "function definition",
                single=True,
            )

    def test_analyse_returns_tree_and_visitor(self):
        tree, visitor = tools._analyse(SAMPLE)
        assert isinstance(tree, ast.Module)
        assert [f["name"] for f in visitor.get_functions()] == ["alpha", "beta"]
        assert [c["name"] for c in visitor.get_classes()] == ["Foo"]

    def test_analyse_syntax_error(self):
        with pytest.raises(tools._EditError) as exc:
            tools._analyse("def f(:\n    pass\n")
        assert "cannot be parsed (line 1)" in str(exc.value)
        assert "Fix it first with `replace_text`" in str(exc.value)

    def test_class_node_not_found(self):
        tree = ast.parse(SAMPLE)
        with pytest.raises(tools._EditError, match="could not be located"):
            tools._class_node(tree, {"name": "Foo", "start_line": 1, "end_line": 1})

    def test_class_node_found_with_decorator(self):
        source = "@dec\nclass C:\n    x = 1\n"
        tree, visitor = tools._analyse(source)
        node = tools._class_node(tree, visitor.get_classes()[0])
        assert node.name == "C"


class TestPick:
    RECORDS = [
        {"name": "a", "start_line": 1, "end_line": 2},
        {"name": "b", "start_line": 4, "end_line": 5},
        {"name": "a", "start_line": 7, "end_line": 8},
    ]

    def test_unique_match(self):
        assert tools._pick(self.RECORDS, "Function", "b", None)["start_line"] == 4

    def test_not_found_lists_available(self):
        with pytest.raises(tools._EditError) as exc:
            tools._pick(self.RECORDS, "Function", "zzz", None)
        assert "Function `zzz` was not found." in str(exc.value)
        assert "Available: `a`, `b`." in str(exc.value)

    def test_not_found_with_no_records_has_no_hint(self):
        with pytest.raises(tools._EditError) as exc:
            tools._pick([], "Class", "X", None, " in here")
        assert str(exc.value) == "Class `X` was not found in here."

    def test_ambiguous_requires_occurrence(self):
        with pytest.raises(tools._EditError) as exc:
            tools._pick(self.RECORDS, "Function", "a", None)
        message = str(exc.value)
        assert "defined 2 times" in message
        assert "1: lines 1-2, 2: lines 7-8" in message
        assert "Pass `occurrence`" in message

    def test_occurrence_selects(self):
        assert tools._pick(self.RECORDS, "Function", "a", 1)["start_line"] == 1
        assert tools._pick(self.RECORDS, "Function", "a", 2)["start_line"] == 7

    @pytest.mark.parametrize("occurrence", [0, 3, -1])
    def test_occurrence_out_of_range(self, occurrence):
        with pytest.raises(
            tools._EditError, match="occurrence must be between 1 and 2"
        ):
            tools._pick(self.RECORDS, "Function", "a", occurrence)

    def test_occurrence_on_unique_name_still_validated(self):
        with pytest.raises(tools._EditError, match="between 1 and 1"):
            tools._pick(self.RECORDS, "Function", "b", 2)


class TestTopLevelAnchor:
    def test_default_is_end_of_file_above_main_guard(self):
        tree = ast.parse(SAMPLE)
        foo_end = next(n for n in tree.body if isinstance(n, ast.ClassDef)).end_lineno
        assert tools._top_level_anchor(tree, None) == foo_end

    def test_default_without_main_guard(self):
        tree = ast.parse("def a():\n    pass\n")
        assert tools._top_level_anchor(tree, None) == 2

    def test_empty_file_anchor_is_zero(self):
        assert tools._top_level_anchor(ast.parse(""), None) == 0

    def test_after_named_definition(self):
        tree = ast.parse(SAMPLE)
        assert tools._top_level_anchor(tree, "alpha") == 7

    def test_after_unknown_name(self):
        with pytest.raises(tools._EditError) as exc:
            tools._top_level_anchor(ast.parse(SAMPLE), "nope")
        assert "no top-level function or class named `nope`" in str(exc.value)
        assert "`alpha`, `beta`, `Foo`" in str(exc.value)


# ----------------------------------------------------------------------
# _atomic_write
# ----------------------------------------------------------------------
class TestAtomicWrite:
    def test_writes_content_and_leaves_no_temp_files(self, tmp_path):
        target = tmp_path / "t.py"
        target.write_text("old")
        tools._atomic_write(target, "new\n")
        assert target.read_text() == "new\n"
        assert [p.name for p in tmp_path.iterdir()] == ["t.py"]

    def test_does_not_translate_newlines(self, tmp_path):
        target = tmp_path / "t.py"
        target.write_text("old")
        tools._atomic_write(target, "a\r\nb\n")
        assert target.read_bytes() == b"a\r\nb\n"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    def test_preserves_permissions(self, tmp_path):
        target = tmp_path / "script.py"
        target.write_text("old")
        target.chmod(0o755)
        tools._atomic_write(target, "new")
        assert stat.S_IMODE(target.stat().st_mode) == 0o755

    def test_failure_cleans_up_and_keeps_original(self, tmp_path, monkeypatch):
        target = tmp_path / "t.py"
        target.write_text("original")

        def boom(*_args, **_kwargs):
            raise OSError("rename failed")

        with monkeypatch.context() as m:
            m.setattr(tools.os, "replace", boom)
            with pytest.raises(OSError, match="rename failed"):
                tools._atomic_write(target, "new")

        assert target.read_text() == "original"
        assert [p.name for p in tmp_path.iterdir()] == ["t.py"]


# ----------------------------------------------------------------------
# _edit_file (the engine behind every write tool)
# ----------------------------------------------------------------------
class TestEditFile:
    def test_rejects_non_python_target(self, ctx, tmp_path, save_file_mock):
        target = tmp_path / "notes.txt"
        target.write_text("hello")
        called = MagicMock()
        result = tools._edit_file(ctx, str(target), called)
        assert "is not an existing Python (.py) file" in result
        called.assert_not_called()
        save_file_mock.assert_not_called()

    def test_unreadable_file(self, ctx, make_py, save_file_mock):
        p = make_py()
        p.write_bytes(b"\xff\xfe\x00bad")
        called = MagicMock()
        result = tools._edit_file(ctx, str(p), called)
        assert result.startswith("ERROR:")
        assert "could not be read" in result
        called.assert_not_called()
        save_file_mock.assert_not_called()

    def test_edit_receives_text_with_lf_newlines(self, ctx, make_py):
        p = make_py("a = 1\r\nb = 2\r\n")
        seen = []

        def edit(text):
            seen.append(text)
            return text, "Nothing to do"

        tools._edit_file(ctx, str(p), edit)
        assert seen == ["a = 1\nb = 2\n"]

    def test_edit_error_is_reported_and_nothing_written(
        self, ctx, make_py, save_file_mock
    ):
        p = make_py()

        def edit(_text):
            raise tools._EditError("nope, not doing that.")

        result = tools._edit_file(ctx, str(p), edit)
        assert_rejected(result, p, SAMPLE, save_file_mock, "nope, not doing that.")

    def test_unchanged_result_skips_write_and_reindex(
        self, ctx, make_py, save_file_mock
    ):
        p = make_py()
        result = tools._edit_file(ctx, str(p), lambda text: (text, "Same."))
        assert result == "Same. `sample.py` was left unchanged."
        save_file_mock.assert_not_called()

    def test_result_that_does_not_parse_is_refused(self, ctx, make_py, save_file_mock):
        p = make_py()
        result = tools._edit_file(ctx, str(p), lambda text: ("def f(:\n", "Did it."))
        assert_rejected(
            result,
            p,
            SAMPLE,
            save_file_mock,
            "would leave",
            "with a syntax error at line 1",
        )

    def test_success_writes_and_reindexes_once(self, ctx, db, make_py, save_file_mock):
        p = make_py()
        result = tools._edit_file(
            ctx, str(p), lambda text: (text + "X = 1\n", "Added X")
        )
        assert read(p) == SAMPLE + "X = 1\n"
        assert result == "Added X in `sample.py`. The file was re-indexed."
        save_file_mock.assert_called_once_with(db, str(p.resolve()))

    def test_crlf_files_stay_crlf(self, ctx, make_py):
        crlf = SAMPLE.replace("\n", "\r\n")
        p = make_py(crlf)
        tools.replace_text(ctx, str(p), "import sys", "import json")
        data = p.read_bytes()
        assert b"import json" in data
        assert b"\r\n" in data
        assert re.search(rb"(?<!\r)\n", data) is None

    def test_lf_files_stay_lf(self, ctx, make_py):
        p = make_py()
        tools.replace_text(ctx, str(p), "import sys", "import json")
        assert b"\r" not in p.read_bytes()


# ----------------------------------------------------------------------
# replace_function
# ----------------------------------------------------------------------
class TestReplaceFunction:
    def test_replaces_only_the_target(self, ctx, make_py):
        p = make_py()
        result = tools.replace_function(
            ctx, str(p), "alpha", "def alpha(x):\n    return x + 100\n"
        )
        text = read(p)
        assert "return x + 100" in text
        assert "return x + 1\n" not in text
        assert top_level_names(p) == ["alpha", "beta", "Foo"]
        assert "def beta(y):\n    return y * 2" in text
        assert 'if __name__ == "__main__":' in text
        assert result.startswith(
            "Replaced function `alpha` (was lines 6-7, now lines 6-7)"
        )
        assert result.endswith("in `sample.py`. The file was re-indexed.")

    def test_reports_new_line_range(self, ctx, make_py):
        p = make_py()
        result = tools.replace_function(
            ctx, str(p), "alpha", "def alpha(x):\n    y = x\n    return y + 1\n"
        )
        assert "(was lines 6-7, now lines 6-8)" in result

    def test_reindexes_after_success(self, ctx, db, make_py, save_file_mock):
        p = make_py()
        tools.replace_function(ctx, str(p), "alpha", "def alpha(x):\n    return x\n")
        save_file_mock.assert_called_once_with(db, str(p.resolve()))

    def test_new_code_may_have_a_different_name(self, ctx, make_py):
        p = make_py()
        tools.replace_function(ctx, str(p), "alpha", "def renamed(x):\n    return x\n")
        assert top_level_names(p) == ["renamed", "beta", "Foo"]

    def test_decorators_are_replaced_with_the_function(self, ctx, make_py):
        p = make_py(DECORATED)
        tools.replace_function(
            ctx, str(p), "decorated", "@decorator\ndef decorated():\n    return 2\n"
        )
        text = read(p)
        assert text.count("@decorator") == 1
        assert "return 2" in text
        assert "return 1" not in text
        assert "def after():" in text

    def test_occurrence_selects_among_duplicates(self, ctx, make_py):
        p = make_py(DUPLICATES)
        tools.replace_function(
            ctx, str(p), "dup", "def dup():\n    return 99\n", occurrence=2
        )
        assert read(p) == DUPLICATES.replace("return 2", "return 99")

    def test_duplicates_without_occurrence_are_refused(
        self, ctx, make_py, save_file_mock
    ):
        p = make_py(DUPLICATES)
        result = tools.replace_function(ctx, str(p), "dup", "def dup():\n    pass\n")
        assert_rejected(
            result,
            p,
            DUPLICATES,
            save_file_mock,
            "defined 2 times",
            "Pass `occurrence`",
        )

    @pytest.mark.parametrize(
        ("kwargs", "fragments"),
        [
            pytest.param(
                {"function_name": "nope", "new_code": "def nope(): pass"},
                ["Function `nope` was not found", "Available: `alpha`, `beta`"],
                id="unknown-function",
            ),
            pytest.param(
                {"function_name": "alpha", "new_code": ""},
                ["`new_code` is empty."],
                id="empty",
            ),
            pytest.param(
                {"function_name": "alpha", "new_code": "def alpha(:"},
                ["`new_code` is not valid Python"],
                id="syntax-error",
            ),
            pytest.param(
                {"function_name": "alpha", "new_code": "class alpha:\n    pass"},
                [
                    "must contain only function definitions",
                    "a class (`alpha`)",
                    "Use replace_class",
                ],
                id="class-instead-of-function",
            ),
            pytest.param(
                {"function_name": "alpha", "new_code": "def a(): pass\ndef b(): pass"},
                ["exactly one function definition, found 2"],
                id="two-functions",
            ),
            pytest.param(
                {
                    "function_name": "alpha",
                    "new_code": "def alpha(): pass",
                    "occurrence": 2,
                },
                ["occurrence must be between 1 and 1"],
                id="occurrence-out-of-range",
            ),
        ],
    )
    def test_refusals(self, kwargs, fragments, ctx, make_py, save_file_mock):
        p = make_py()
        result = tools.replace_function(ctx, str(p), **kwargs)
        assert_rejected(result, p, SAMPLE, save_file_mock, *fragments)


# ----------------------------------------------------------------------
# replace_method
# ----------------------------------------------------------------------
class TestReplaceMethod:
    def test_indentation_is_added_automatically(self, ctx, make_py):
        p = make_py()
        result = tools.replace_method(
            ctx, str(p), "Foo", "method_one", "def method_one(self):\n    return 100\n"
        )
        text = read(p)
        assert "    def method_one(self):\n        return 100\n" in text
        assert method_names(p, "Foo") == ["method_one", "method_two"]
        assert result.startswith("Replaced method `Foo.method_one`")

    def test_already_indented_code_is_accepted(self, ctx, make_py):
        p = make_py()
        tools.replace_method(
            ctx,
            str(p),
            "Foo",
            "method_one",
            "    def method_one(self):\n        return 100\n",
        )
        assert "    def method_one(self):\n        return 100\n" in read(p)
        assert method_names(p, "Foo") == ["method_one", "method_two"]

    def test_other_methods_are_untouched(self, ctx, make_py):
        p = make_py()
        tools.replace_method(
            ctx, str(p), "Foo", "method_one", "def method_one(self):\n    return 100\n"
        )
        assert "    def method_two(self):\n        return 2\n" in read(p)

    def test_property_setter_needs_occurrence(self, ctx, make_py, save_file_mock):
        p = make_py(PROPERTY)
        new = "@x.setter\ndef x(self, value):\n    self._x = value * 2\n"

        result = tools.replace_method(ctx, str(p), "P", "x", new)
        assert_rejected(
            result, p, PROPERTY, save_file_mock, "defined 2 times", "Pass `occurrence`"
        )

        tools.replace_method(ctx, str(p), "P", "x", new, occurrence=2)
        text = read(p)
        assert "value * 2" in text
        assert text.count("@x.setter") == 1
        assert text.count("@property") == 1

    @pytest.mark.parametrize(
        ("kwargs", "fragments"),
        [
            pytest.param(
                {
                    "cls_name": "Nope",
                    "method_name": "method_one",
                    "new_code": "def f(self): pass",
                },
                ["Class `Nope` was not found", "Available: `Foo`"],
                id="unknown-class",
            ),
            pytest.param(
                {
                    "cls_name": "Foo",
                    "method_name": "nope",
                    "new_code": "def f(self): pass",
                },
                [
                    "Method `nope` was not found in class `Foo`",
                    "Available: `method_one`, `method_two`",
                ],
                id="unknown-method",
            ),
            pytest.param(
                {
                    "cls_name": "Foo",
                    "method_name": "method_one",
                    "new_code": "class X: pass",
                },
                ["must contain only method definitions"],
                id="class-instead-of-method",
            ),
            pytest.param(
                {
                    "cls_name": "Foo",
                    "method_name": "method_one",
                    "new_code": "def a(self): pass\ndef b(self): pass",
                },
                ["exactly one method definition, found 2"],
                id="two-methods",
            ),
            pytest.param(
                {"cls_name": "Foo", "method_name": "method_one", "new_code": " \n"},
                ["`new_code` is empty."],
                id="empty",
            ),
        ],
    )
    def test_refusals(self, kwargs, fragments, ctx, make_py, save_file_mock):
        p = make_py()
        result = tools.replace_method(ctx, str(p), **kwargs)
        assert_rejected(result, p, SAMPLE, save_file_mock, *fragments)


# ----------------------------------------------------------------------
# replace_class
# ----------------------------------------------------------------------
class TestReplaceClass:
    def test_replaces_the_whole_class(self, ctx, make_py):
        p = make_py()
        result = tools.replace_class(ctx, str(p), "Foo", "class Foo:\n    pass\n")
        text = read(p)
        assert "method_one" not in text
        assert "class Foo:\n    pass\n" in text
        assert 'if __name__ == "__main__":' in text
        assert top_level_names(p) == ["alpha", "beta", "Foo"]
        assert result.startswith("Replaced class `Foo`")

    def test_new_class_may_have_a_different_name(self, ctx, make_py):
        p = make_py()
        tools.replace_class(ctx, str(p), "Foo", "class Bar:\n    pass\n")
        assert top_level_names(p) == ["alpha", "beta", "Bar"]

    @pytest.mark.parametrize(
        ("kwargs", "fragments"),
        [
            pytest.param(
                {"cls_name": "Nope", "new_code": "class Nope: pass"},
                ["Class `Nope` was not found", "Available: `Foo`"],
                id="unknown-class",
            ),
            pytest.param(
                {"cls_name": "Foo", "new_code": "def Foo(): pass"},
                ["must contain only class definitions", "Use replace_function"],
                id="function-instead-of-class",
            ),
            pytest.param(
                {"cls_name": "Foo", "new_code": "class A: pass\nclass B: pass"},
                ["exactly one class definition, found 2"],
                id="two-classes",
            ),
        ],
    )
    def test_refusals(self, kwargs, fragments, ctx, make_py, save_file_mock):
        p = make_py()
        result = tools.replace_class(ctx, str(p), **kwargs)
        assert_rejected(result, p, SAMPLE, save_file_mock, *fragments)


# ----------------------------------------------------------------------
# replace_text
# ----------------------------------------------------------------------
class TestReplaceText:
    def test_replaces_unique_text_and_reports_line(self, ctx, make_py):
        p = make_py()
        result = tools.replace_text(ctx, str(p), "def alpha(x):", "def alpha(x, y=0):")
        assert "def alpha(x, y=0):" in read(p)
        assert result.startswith("Replaced the text at line 6")

    def test_multiline_replacement(self, ctx, make_py):
        p = make_py()
        tools.replace_text(
            ctx,
            str(p),
            "def alpha(x):\n    return x + 1",
            "def alpha(x):\n    return x - 1",
        )
        assert "return x - 1" in read(p)

    def test_empty_new_text_deletes(self, ctx, make_py):
        p = make_py()
        tools.replace_text(ctx, str(p), "import sys\n", "")
        assert "import sys" not in read(p)
        assert "import os" in read(p)

    def test_crlf_in_arguments_is_normalised(self, ctx, make_py):
        p = make_py()
        tools.replace_text(
            ctx,
            str(p),
            "def alpha(x):\r\n    return x + 1",
            "def alpha(x):\r\n    return 0",
        )
        text = read(p)
        assert "return 0" in text
        assert "\r" not in text

    def test_identical_replacement_leaves_file_unchanged(
        self, ctx, make_py, save_file_mock
    ):
        p = make_py()
        result = tools.replace_text(ctx, str(p), "import os", "import os")
        assert "identical to the existing text" in result
        assert "was left unchanged" in result
        assert read(p) == SAMPLE
        save_file_mock.assert_not_called()

    def test_can_repair_a_file_that_does_not_parse(self, ctx, make_py):
        p = make_py("def f(:\n    pass\n")
        result = tools.replace_text(ctx, str(p), "def f(:", "def f():")
        assert read(p) == "def f():\n    pass\n"
        assert "re-indexed" in result

    @pytest.mark.parametrize(
        ("old_text", "new_text", "fragments"),
        [
            pytest.param("", "x", ["`old_text` must not be empty."], id="empty-old"),
            pytest.param(
                "does not exist", "x", ["`old_text` was not found"], id="missing"
            ),
            pytest.param(
                "return",
                "yield",
                ["was found 4 times", "starting on lines 7, 11, 18, 21"],
                id="ambiguous",
            ),
            pytest.param(
                "return x + 1",
                "return (x +",
                ["would leave", "with a syntax error"],
                id="breaks-syntax",
            ),
        ],
    )
    def test_refusals(
        self, old_text, new_text, fragments, ctx, make_py, save_file_mock
    ):
        p = make_py()
        result = tools.replace_text(ctx, str(p), old_text, new_text)
        assert_rejected(result, p, SAMPLE, save_file_mock, *fragments)


# ----------------------------------------------------------------------
# add_function
# ----------------------------------------------------------------------
class TestAddFunction:
    def test_default_goes_above_main_guard_with_pep8_spacing(self, ctx, make_py):
        p = make_py()
        result = tools.add_function(ctx, str(p), "def gamma():\n    return 3\n")
        text = read(p)
        assert (
            "        return 2\n\n\ndef gamma():\n    return 3\n\n\nif __name__" in text
        )
        assert top_level_names(p) == ["alpha", "beta", "Foo", "gamma"]
        assert result.startswith("Added function `gamma` (lines 24-25)")

    def test_appends_to_end_without_main_guard(self, ctx, make_py):
        p = make_py("def a():\n    pass\n")
        tools.add_function(ctx, str(p), "def b():\n    pass\n")
        assert read(p) == "def a():\n    pass\n\n\ndef b():\n    pass\n"

    def test_after_places_it_right_after_the_named_definition(self, ctx, make_py):
        p = make_py()
        tools.add_function(ctx, str(p), "def gamma():\n    return 3\n", after="alpha")
        assert top_level_names(p) == ["alpha", "gamma", "beta", "Foo"]
        ast.parse(read(p))

    def test_after_a_class(self, ctx, make_py):
        p = make_py()
        tools.add_function(ctx, str(p), "def gamma():\n    return 3\n", after="Foo")
        assert top_level_names(p) == ["alpha", "beta", "Foo", "gamma"]

    def test_several_functions_at_once(self, ctx, make_py):
        p = make_py()
        result = tools.add_function(
            ctx, str(p), "def g1():\n    pass\n\n\ndef g2():\n    pass\n"
        )
        assert "Added function `g1`, `g2`" in result
        assert top_level_names(p) == ["alpha", "beta", "Foo", "g1", "g2"]

    def test_indented_new_code_is_dedented(self, ctx, make_py):
        p = make_py()
        tools.add_function(ctx, str(p), "    def gamma():\n        return 3\n")
        assert "\ndef gamma():\n    return 3\n" in read(p)

    def test_into_empty_file(self, ctx, make_py):
        p = make_py("")
        result = tools.add_function(ctx, str(p), "def f():\n    pass\n")
        assert read(p) == "def f():\n    pass\n"
        assert "(lines 1-2)" in result

    def test_into_file_with_only_a_comment(self, ctx, make_py):
        p = make_py("# just a comment\n")
        tools.add_function(ctx, str(p), "def f():\n    pass\n")
        assert read(p) == "def f():\n    pass\n\n\n# just a comment\n"

    @pytest.mark.parametrize(
        ("kwargs", "fragments"),
        [
            pytest.param(
                {"new_code": "def alpha():\n    pass\n"},
                ["Function `alpha` already exists", "Use replace_function"],
                id="already-exists",
            ),
            pytest.param(
                {"new_code": "class C:\n    pass\n"},
                ["must contain only function definitions", "Use add_class"],
                id="class-instead-of-function",
            ),
            pytest.param(
                {"new_code": "x = 1"},
                ["other code (`Assign`)"],
                id="plain-statement",
            ),
            pytest.param(
                {"new_code": "def g(): pass", "after": "nope"},
                ["no top-level function or class named `nope`"],
                id="unknown-after",
            ),
            pytest.param({"new_code": ""}, ["`new_code` is empty."], id="empty"),
            pytest.param(
                {"new_code": "def g(:"}, ["not valid Python"], id="syntax-error"
            ),
        ],
    )
    def test_refusals(self, kwargs, fragments, ctx, make_py, save_file_mock):
        p = make_py()
        result = tools.add_function(ctx, str(p), **kwargs)
        assert_rejected(result, p, SAMPLE, save_file_mock, *fragments)


# ----------------------------------------------------------------------
# add_method
# ----------------------------------------------------------------------
class TestAddMethod:
    def test_default_goes_at_end_of_class(self, ctx, make_py):
        p = make_py()
        result = tools.add_method(
            ctx, str(p), "Foo", "def method_new(self):\n    return 3\n"
        )
        assert method_names(p, "Foo") == ["method_one", "method_two", "method_new"]
        assert "    def method_new(self):\n        return 3\n" in read(p)
        assert result.startswith("Added method `method_new` to class `Foo`")

    def test_after_method(self, ctx, make_py):
        p = make_py()
        tools.add_method(
            ctx,
            str(p),
            "Foo",
            "def method_new(self):\n    return 3\n",
            after_method="method_one",
        )
        assert method_names(p, "Foo") == ["method_one", "method_new", "method_two"]
        ast.parse(read(p))

    def test_several_methods_at_once(self, ctx, make_py):
        p = make_py()
        result = tools.add_method(
            ctx, str(p), "Foo", "def a(self):\n    pass\n\ndef b(self):\n    pass\n"
        )
        assert "`a`, `b`" in result
        assert method_names(p, "Foo") == ["method_one", "method_two", "a", "b"]

    def test_decorated_duplicate_is_allowed_for_property_setters(self, ctx, make_py):
        p = make_py(
            "class P:\n    @property\n    def x(self):\n        return self._x\n"
        )
        tools.add_method(
            ctx, str(p), "P", "@x.setter\ndef x(self, v):\n    self._x = v\n"
        )
        assert method_names(p, "P") == ["x", "x"]

    def test_follows_the_classes_own_indentation(self, ctx, make_py):
        p = make_py("class Two:\n  def a(self):\n    return 1\n")
        tools.add_method(ctx, str(p), "Two", "def b(self):\n    return 2\n")
        assert method_names(p, "Two") == ["a", "b"]
        assert "\n  def b(self):\n" in read(p)

    def test_file_is_reindexed(self, ctx, db, make_py, save_file_mock):
        p = make_py()
        tools.add_method(ctx, str(p), "Foo", "def m(self):\n    pass\n")
        save_file_mock.assert_called_once_with(db, str(p.resolve()))

    @pytest.mark.parametrize(
        ("source", "kwargs", "fragments"),
        [
            pytest.param(
                SAMPLE,
                {"cls_name": "Nope", "new_code": "def m(self): pass"},
                ["Class `Nope` was not found", "Available: `Foo`"],
                id="unknown-class",
            ),
            pytest.param(
                SAMPLE,
                {"cls_name": "Foo", "new_code": "def method_one(self): pass"},
                [
                    "Method `method_one` already exists in class `Foo`",
                    "Use replace_method",
                ],
                id="already-exists",
            ),
            pytest.param(
                SAMPLE,
                {"cls_name": "Foo", "new_code": "class X: pass"},
                [
                    "must contain only method definitions",
                    "Use add_class / add_function",
                ],
                id="class-instead-of-method",
            ),
            pytest.param(
                SAMPLE,
                {
                    "cls_name": "Foo",
                    "new_code": "def m(self): pass",
                    "after_method": "nope",
                },
                [
                    "Method `nope` was not found in class `Foo`",
                    "Methods of this class: `method_one`, `method_two`",
                ],
                id="unknown-after-method",
            ),
            pytest.param(
                "class One: pass\n",
                {"cls_name": "One", "new_code": "def m(self): pass"},
                ["written on a single line", "Use replace_class"],
                id="one-line-class",
            ),
        ],
    )
    def test_refusals(self, source, kwargs, fragments, ctx, make_py, save_file_mock):
        p = make_py(source)
        result = tools.add_method(ctx, str(p), **kwargs)
        assert_rejected(result, p, source, save_file_mock, *fragments)


# ----------------------------------------------------------------------
# add_class
# ----------------------------------------------------------------------
class TestAddClass:
    def test_default_goes_above_main_guard(self, ctx, make_py):
        p = make_py()
        result = tools.add_class(ctx, str(p), "class Bar:\n    pass\n")
        text = read(p)
        assert "        return 2\n\n\nclass Bar:\n    pass\n\n\nif __name__" in text
        assert top_level_names(p) == ["alpha", "beta", "Foo", "Bar"]
        assert result.startswith("Added class `Bar`")

    def test_after_a_function(self, ctx, make_py):
        p = make_py()
        tools.add_class(ctx, str(p), "class Bar:\n    pass\n", after="alpha")
        assert top_level_names(p) == ["alpha", "Bar", "beta", "Foo"]

    @pytest.mark.parametrize(
        ("kwargs", "fragments"),
        [
            pytest.param(
                {"new_code": "class Foo:\n    pass\n"},
                ["Class `Foo` already exists", "Use replace_class"],
                id="already-exists",
            ),
            pytest.param(
                {"new_code": "def f(): pass"},
                ["must contain only class definitions", "Use add_function"],
                id="function-instead-of-class",
            ),
            pytest.param(
                {"new_code": "class B: pass", "after": "nope"},
                ["no top-level function or class named `nope`"],
                id="unknown-after",
            ),
        ],
    )
    def test_refusals(self, kwargs, fragments, ctx, make_py, save_file_mock):
        p = make_py()
        result = tools.add_class(ctx, str(p), **kwargs)
        assert_rejected(result, p, SAMPLE, save_file_mock, *fragments)


# ----------------------------------------------------------------------
# add_imports
# ----------------------------------------------------------------------
class TestAddImports:
    def test_added_after_existing_imports(self, ctx, make_py):
        p = make_py()
        result = tools.add_imports(ctx, str(p), "import json")
        lines = read(p).splitlines()
        assert lines[:4] == [
            '"""Module docstring."""',
            "import os",
            "import sys",
            "import json",
        ]
        assert result.startswith("Added 1 import statement(s) (lines 4-4)")

    def test_multiple_statements(self, ctx, make_py):
        p = make_py()
        result = tools.add_imports(ctx, str(p), "import json\nfrom pathlib import Path")
        lines = read(p).splitlines()
        assert lines[3:5] == ["import json", "from pathlib import Path"]
        assert "Added 2 import statement(s) (lines 4-5)" in result

    def test_already_present_import_is_a_no_op(self, ctx, make_py, save_file_mock):
        p = make_py()
        result = tools.add_imports(ctx, str(p), "import os")
        assert "already present" in result
        assert "was left unchanged" in result
        assert read(p) == SAMPLE
        save_file_mock.assert_not_called()

    def test_only_new_statements_are_added(self, ctx, make_py):
        p = make_py()
        result = tools.add_imports(ctx, str(p), "import os\nimport json")
        assert read(p).count("import os") == 1
        assert "import json" in read(p)
        assert "Added 1 import statement(s)" in result

    def test_duplicates_within_the_input_are_collapsed(self, ctx, make_py):
        p = make_py()
        result = tools.add_imports(ctx, str(p), "import json\nimport json")
        assert read(p).count("import json") == 1
        assert "Added 1 import statement(s)" in result

    def test_after_docstring_when_no_imports(self, ctx, make_py):
        p = make_py('"""Doc."""\n\nx = 1\n')
        tools.add_imports(ctx, str(p), "import os")
        assert read(p) == '"""Doc."""\n\nimport os\n\nx = 1\n'

    def test_top_of_file_when_no_docstring_or_imports(self, ctx, make_py):
        p = make_py("x = 1\n")
        tools.add_imports(ctx, str(p), "import os")
        assert read(p) == "import os\n\nx = 1\n"

    def test_into_empty_file(self, ctx, make_py):
        p = make_py("")
        tools.add_imports(ctx, str(p), "import os")
        assert read(p) == "import os\n"

    @pytest.mark.parametrize(
        ("imports", "fragments"),
        [
            pytest.param("", ["`imports` is empty."], id="empty"),
            pytest.param("   \n\n", ["`imports` is empty."], id="blank"),
            pytest.param(
                "import (", ["`imports` is not valid Python"], id="syntax-error"
            ),
            pytest.param(
                "def f(): pass",
                ["must contain only import statements", "a function (`f`)"],
                id="function",
            ),
            pytest.param(
                "import os\nx = 1",
                ["must contain only import statements", "other code (`Assign`)"],
                id="mixed",
            ),
            pytest.param(
                "from __future__ import annotations",
                ["__future__", "replace_text"],
                id="future-import",
            ),
        ],
    )
    def test_refusals(self, imports, fragments, ctx, make_py, save_file_mock):
        p = make_py()
        result = tools.add_imports(ctx, str(p), imports)
        assert_rejected(result, p, SAMPLE, save_file_mock, *fragments)


# ----------------------------------------------------------------------
# delete_definition
# ----------------------------------------------------------------------
class TestDeleteDefinition:
    def test_delete_function_keeps_neighbour_spacing(self, ctx, make_py):
        p = make_py()
        result = tools.delete_definition(ctx, str(p), "alpha")
        text = read(p)
        assert top_level_names(p) == ["beta", "Foo"]
        assert "import sys\n\n\ndef beta(y):" in text
        assert result.startswith("Deleted function `alpha` (was lines 6-7)")

    def test_delete_class_before_main_guard(self, ctx, make_py):
        p = make_py()
        result = tools.delete_definition(ctx, str(p), "Foo")
        assert top_level_names(p) == ["alpha", "beta"]
        assert "    return y * 2\n\n\nif __name__" in read(p)
        assert result.startswith("Deleted class `Foo`")

    def test_delete_last_definition_removes_trailing_blank_lines(self, ctx, make_py):
        p = make_py("def a():\n    pass\n\n\ndef b():\n    pass\n")
        tools.delete_definition(ctx, str(p), "b")
        assert read(p) == "def a():\n    pass\n"

    def test_delete_method(self, ctx, make_py):
        p = make_py()
        result = tools.delete_definition(ctx, str(p), "method_one", cls_name="Foo")
        assert method_names(p, "Foo") == ["method_two"]
        assert '    """A class."""\n\n    def method_two(self):' in read(p)
        assert result.startswith("Deleted method `Foo.method_one`")

    def test_delete_last_method_of_a_class(self, ctx, make_py):
        p = make_py()
        tools.delete_definition(ctx, str(p), "method_two", cls_name="Foo")
        assert method_names(p, "Foo") == ["method_one"]
        ast.parse(read(p))
        assert 'if __name__ == "__main__":' in read(p)

    def test_decorators_are_deleted_too(self, ctx, make_py):
        p = make_py(DECORATED)
        tools.delete_definition(ctx, str(p), "decorated")
        text = read(p)
        assert "@decorator" not in text
        assert top_level_names(p) == ["decorator", "after"]

    def test_property_setter_with_occurrence(self, ctx, make_py):
        p = make_py(PROPERTY)
        tools.delete_definition(ctx, str(p), "x", cls_name="P", occurrence=2)
        text = read(p)
        assert "@x.setter" not in text
        assert "@property" in text

    def test_duplicate_names_need_occurrence(self, ctx, make_py, save_file_mock):
        p = make_py(DUPLICATES)
        result = tools.delete_definition(ctx, str(p), "dup")
        assert_rejected(result, p, DUPLICATES, save_file_mock, "defined 2 times")

        tools.delete_definition(ctx, str(p), "dup", occurrence=1)
        assert read(p) == "def dup():\n    return 2\n"

    def test_deleting_only_method_of_bodyless_class_is_refused(
        self, ctx, make_py, save_file_mock
    ):
        source = "class Solo:\n    def only(self):\n        pass\n"
        p = make_py(source)
        result = tools.delete_definition(ctx, str(p), "only", cls_name="Solo")
        assert_rejected(
            result, p, source, save_file_mock, "would leave", "syntax error"
        )

    @pytest.mark.parametrize(
        ("kwargs", "fragments"),
        [
            pytest.param(
                {"name": "nope"},
                ["Function or class `nope` was not found", "`alpha`, `beta`, `Foo`"],
                id="unknown-top-level",
            ),
            pytest.param(
                {"name": "method_one", "cls_name": "Nope"},
                ["Class `Nope` was not found"],
                id="unknown-class",
            ),
            pytest.param(
                {"name": "zzz", "cls_name": "Foo"},
                ["Method `zzz` was not found in class `Foo`"],
                id="unknown-method",
            ),
            pytest.param(
                {"name": "method_one"},
                ["Function or class `method_one` was not found"],
                id="method-without-cls_name",
            ),
        ],
    )
    def test_refusals(self, kwargs, fragments, ctx, make_py, save_file_mock):
        p = make_py()
        result = tools.delete_definition(ctx, str(p), **kwargs)
        assert_rejected(result, p, SAMPLE, save_file_mock, *fragments)


# ----------------------------------------------------------------------
# Behaviour shared by every write tool
# ----------------------------------------------------------------------
AST_WRITE_CALLS = [
    pytest.param(
        lambda c, f: tools.replace_function(c, f, "alpha", "def alpha(): pass"),
        id="replace_function",
    ),
    pytest.param(
        lambda c, f: tools.replace_method(
            c, f, "Foo", "method_one", "def method_one(self): pass"
        ),
        id="replace_method",
    ),
    pytest.param(
        lambda c, f: tools.replace_class(c, f, "Foo", "class Foo: pass"),
        id="replace_class",
    ),
    pytest.param(
        lambda c, f: tools.add_function(c, f, "def new(): pass"), id="add_function"
    ),
    pytest.param(
        lambda c, f: tools.add_method(c, f, "Foo", "def new(self): pass"),
        id="add_method",
    ),
    pytest.param(lambda c, f: tools.add_class(c, f, "class New: pass"), id="add_class"),
    pytest.param(lambda c, f: tools.add_imports(c, f, "import json"), id="add_imports"),
    pytest.param(
        lambda c, f: tools.delete_definition(c, f, "alpha"), id="delete_definition"
    ),
]
ALL_WRITE_CALLS = [
    *AST_WRITE_CALLS,
    pytest.param(
        lambda c, f: tools.replace_text(c, f, "import os", "import json"),
        id="replace_text",
    ),
]


class TestEveryWriteTool:
    @pytest.mark.parametrize("call", ALL_WRITE_CALLS)
    def test_rejects_non_python_target(self, call, ctx, tmp_path, save_file_mock):
        target = tmp_path / "notes.txt"
        target.write_text("hello")
        result = call(ctx, str(target))
        assert result.startswith("ERROR:")
        assert "is not an existing Python (.py) file" in result
        assert target.read_text() == "hello"
        save_file_mock.assert_not_called()

    @pytest.mark.parametrize("call", ALL_WRITE_CALLS)
    def test_rejects_missing_file(self, call, ctx, tmp_path, save_file_mock):
        result = call(ctx, str(tmp_path / "missing.py"))
        assert "is not an existing Python (.py) file" in result
        save_file_mock.assert_not_called()

    @pytest.mark.parametrize("call", AST_WRITE_CALLS)
    def test_file_that_does_not_parse_points_to_replace_text(
        self, call, ctx, make_py, save_file_mock
    ):
        broken = "def f(:\n    pass\n"
        p = make_py(broken)
        result = call(ctx, str(p))
        assert_rejected(
            result,
            p,
            broken,
            save_file_mock,
            "cannot be parsed",
            "Fix it first with `replace_text`",
        )

    @pytest.mark.parametrize("call", ALL_WRITE_CALLS)
    def test_success_always_leaves_a_parseable_file_and_reindexes(
        self, call, ctx, db, make_py, save_file_mock
    ):
        p = make_py()
        result = call(ctx, str(p))
        ast.parse(read(p))
        assert not result.startswith("ERROR:"), result
        assert "The file was re-indexed." in result
        save_file_mock.assert_called_once_with(db, str(p.resolve()))


# ----------------------------------------------------------------------
# Tool registries
# ----------------------------------------------------------------------
class TestRegistries:
    def test_read_tools(self):
        assert {t.__name__ for t in tools.READ_TOOLS} == {
            "get_file_imports",
            "get_file_top_level_chunk",
            "get_file_classes",
            "get_class_methods_signature",
            "get_file_functions",
            "expand_function",
            "expand_class",
            "expand_method",
        }

    def test_write_tools(self):
        assert {t.__name__ for t in tools.WRITE_TOOLS} == {
            "replace_function",
            "replace_method",
            "replace_class",
            "replace_text",
            "add_function",
            "add_method",
            "add_class",
            "add_imports",
            "delete_definition",
        }

    def test_read_and_write_tools_do_not_overlap(self):
        assert not set(tools.READ_TOOLS) & set(tools.WRITE_TOOLS)

    @pytest.mark.parametrize(
        "tool", [*tools.READ_TOOLS, *tools.WRITE_TOOLS], ids=lambda t: t.__name__
    )
    def test_every_tool_is_documented_for_the_agent(self, tool):
        # pydantic_ai builds the tool description from the docstring
        assert tool.__doc__ and tool.__doc__.strip()
        assert "file_name" in tool.__doc__
