import inspect
import os
import stat

import pytest

from agents import tools
from agents.tools import (
    expand_class,
    expand_function,
    expand_method,
    get_class_methods_signature,
    get_file_classes,
    get_file_functions,
    get_file_imports,
    get_file_top_level_chunk,
    write_code_chunk,
)
from context_engine import chunk_query
from context_engine.ast_chunker import content_hash
from context_engine.chunks_saver import get_indexed_hash, save_file

READ_TOOLS = [
    get_file_imports,
    get_file_top_level_chunk,
    get_file_classes,
    get_class_methods_signature,
    get_file_functions,
    expand_function,
    expand_class,
    expand_method,
]
ALL_TOOLS = [*READ_TOOLS, write_code_chunk]


@pytest.fixture
def path(sample_file) -> str:
    return str(sample_file)


def read_text(p) -> str:
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


# ----------------------------------------------------------- tool definitions
@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.__name__)
def test_tool_is_described_for_the_model(tool):
    params = list(inspect.signature(tool).parameters)
    assert params[0] == "ctx"
    doc = inspect.getdoc(tool)
    assert doc and len(doc) > 40
    for name in params[1:]:
        assert f"{name}:" in doc, f"{tool.__name__}: parameter {name} is not documented"


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.__name__)
def test_tool_returns_str(tool):
    assert inspect.signature(tool).return_annotation in (str, "str")


# ------------------------------------------------------------------- reading
def test_read_tools_return_markdown(ctx, path):
    calls = [
        get_file_imports(ctx, path),
        get_file_top_level_chunk(ctx, path),
        get_file_classes(ctx, path),
        get_class_methods_signature(ctx, path, "Account"),
        get_file_functions(ctx, path),
        expand_function(ctx, path, "helper"),
        expand_class(ctx, path, "Tiny"),
        expand_method(ctx, path, "Account", "deposit"),
    ]
    for out in calls:
        assert isinstance(out, str)
        assert "```python" in out
        assert not out.lstrip().startswith(("{", "["))


def test_tools_return_what_chunk_query_returns(ctx, path):
    db = ctx.deps.db
    assert get_file_imports(ctx, path) == chunk_query.get_imports(db, path)
    assert get_file_top_level_chunk(ctx, path) == chunk_query.get_top_level_chunks(db, path)
    assert get_file_classes(ctx, path) == chunk_query.get_classes(db, path)
    assert get_file_functions(ctx, path) == chunk_query.get_functions(db, path)
    assert expand_function(ctx, path, "fetch") == chunk_query.get_function_code(db, path, "fetch")
    assert expand_class(ctx, path, "Account") == chunk_query.get_class_code(db, path, "Account")
    assert expand_method(ctx, path, "Account", "sync") == chunk_query.get_method_code(
        db, path, "Account", "sync"
    )
    assert get_class_methods_signature(ctx, path, "Account") == chunk_query.get_class_methods(
        db, path, "Account"
    )


def test_get_file_imports_is_only_code(ctx, path):
    assert get_file_imports(ctx, path).startswith("```python\nimport os\nimport sys as system")


def test_first_call_indexes_the_file(ctx, path):
    assert get_indexed_hash(ctx.deps.db, path) is None
    get_file_functions(ctx, path)
    assert get_indexed_hash(ctx.deps.db, path) is not None


def test_edits_made_outside_the_agent_are_picked_up(ctx, sample_file, path):
    assert "added_later" not in get_file_functions(ctx, path)
    sample_file.write_text(read_text(sample_file) + "\n\ndef added_later():\n    pass\n")
    assert "added_later" in get_file_functions(ctx, path)


def test_relative_path(ctx, sample_file, monkeypatch):
    monkeypatch.chdir(sample_file.parent)
    assert get_file_imports(ctx, "sample.py").startswith("```python\nimport os")
    assert get_file_imports(ctx, "  sample.py  ").startswith("```python\nimport os")


def test_unknown_names_are_errors_not_exceptions(ctx, path):
    assert expand_function(ctx, path, "nope").startswith("**Error:** Function `nope`")
    assert expand_class(ctx, path, "Nope").startswith("**Error:** Class `Nope`")
    assert expand_method(ctx, path, "Account", "nope").startswith("**Error:** Method `nope`")
    assert get_class_methods_signature(ctx, path, "Nope").startswith("**Error:** Class `Nope`")


@pytest.mark.parametrize("tool", READ_TOOLS[:5], ids=lambda t: t.__name__)
def test_bad_paths_are_errors(ctx, tmp_path, tool):
    args = ("Account",) if tool is get_class_methods_signature else ()

    missing = str(tmp_path / "missing.py")
    assert tool(ctx, missing, *args) == f"**Error:** `{missing}` is not an existing Python (.py) file."

    text = tmp_path / "notes.txt"
    text.write_text("hello")
    assert tool(ctx, str(text), *args).startswith("**Error:**")
    assert tool(ctx, str(tmp_path), *args).startswith("**Error:**")  # a directory


def test_file_that_does_not_parse_is_reported(ctx, tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("def f(:\n    pass\n")
    out = get_file_functions(ctx, str(broken))
    assert out.startswith("**Error:**")
    assert "cannot be parsed (line 1)" in out


def test_file_that_stops_parsing_never_serves_stale_chunks(ctx, sample_file, path):
    get_file_functions(ctx, path)
    sample_file.write_text("def f(:\n")
    assert get_file_functions(ctx, path).startswith("**Error:**")


def test_unreadable_encoding_is_reported(ctx, tmp_path):
    p = tmp_path / "latin.py"
    p.write_bytes(b"x = '\xe9'\n")
    assert get_file_imports(ctx, str(p)).startswith("**Error:** `")
    assert "could not be read" in get_file_imports(ctx, str(p))


# ------------------------------------------------------------ write_code_chunk
def prime(ctx, path):
    """Read the file through a tool first, as the agent has to."""
    get_file_functions(ctx, path)


def test_replace_a_function(ctx, sample_file, path, sample_source):
    prime(ctx, path)
    out = write_code_chunk(ctx, path, 18, 19, "def helper(a, b=2):\n    return a * b")

    expected = sample_source.replace("    return a + b", "    return a * b")
    assert read_text(sample_file) == expected
    assert out.startswith("Replaced lines 18-19 with 2 line(s) (now lines 18-19) in `sample.py`.")


def test_write_reindexes_the_file(ctx, path):
    prime(ctx, path)
    write_code_chunk(ctx, path, 18, 19, "def helper2(a):\n    return a\n\n\ndef extra():\n    pass")
    assert "def helper2(a):" in expand_function(ctx, path, "helper2")
    assert expand_function(ctx, path, "helper").startswith("**Error:**")
    # the lines of everything after the edit moved down by 4
    assert "(lines 26-37)" in expand_function(ctx, path, "fetch")


def test_replacing_a_method_matches_the_indentation_of_the_original(ctx, sample_file, path):
    prime(ctx, path)
    # exactly the way expand_method shows it: no indentation
    new = 'def deposit(self, amount: int) -> None:\n    """Add money."""\n    self.balance += amount * 2'
    write_code_chunk(ctx, path, 46, 48, new)

    assert '    def deposit(self, amount: int) -> None:\n        """Add money."""\n        self.balance += amount * 2\n' in read_text(sample_file)
    assert "amount * 2" in expand_method(ctx, path, "Account", "deposit")


def test_already_indented_code_is_not_indented_twice(ctx, sample_file, path):
    prime(ctx, path)
    new = "    def deposit(self, amount):\n        self.balance += amount"
    write_code_chunk(ctx, path, 46, 48, new)
    assert "\n    def deposit(self, amount):\n        self.balance += amount\n" in read_text(sample_file)


def test_insert_before_a_line_is_verbatim(ctx, sample_file, path, sample_source):
    prime(ctx, path)
    out = write_code_chunk(ctx, path, 18, 17, "def added():\n    return 0\n\n\n")
    assert read_text(sample_file) == sample_source.replace(
        "def helper(a, b=2):", "def added():\n    return 0\n\n\ndef helper(a, b=2):"
    )
    assert out.startswith("Inserted 4 line(s) before line 18")


def test_insert_at_the_end(ctx, sample_file, path, sample_source):
    prime(ctx, path)
    n_lines = sample_source.count("\n")
    write_code_chunk(ctx, path, n_lines + 1, n_lines, "\n\ndef last():\n    pass\n")
    assert read_text(sample_file) == sample_source + "\n\ndef last():\n    pass\n"


def test_append_to_a_file_without_a_trailing_newline(ctx, tmp_path):
    p = tmp_path / "nonl.py"
    p.write_text("def f():\n    pass")  # no trailing newline
    prime(ctx, str(p))
    write_code_chunk(ctx, str(p), 3, 2, "x = 1")
    assert read_text(p) == "def f():\n    pass\nx = 1\n"


def test_replace_the_last_line_of_a_file_without_a_trailing_newline(ctx, tmp_path):
    p = tmp_path / "nonl.py"
    p.write_text("x = 1\ny = 2")
    prime(ctx, str(p))
    write_code_chunk(ctx, str(p), 2, 2, "y = 3")
    assert read_text(p) == "x = 1\ny = 3\n"


def test_insert_into_an_empty_file(ctx, tmp_path):
    p = tmp_path / "empty.py"
    p.write_text("")
    prime(ctx, str(p))
    write_code_chunk(ctx, str(p), 1, 0, "x = 1\n")
    assert read_text(p) == "x = 1\n"


def test_empty_new_code_deletes_the_lines(ctx, sample_file, path, sample_source):
    prime(ctx, path)
    out = write_code_chunk(ctx, path, 18, 21, "")
    assert read_text(sample_file) == sample_source.replace("def helper(a, b=2):\n    return a + b\n\n\n", "")
    assert out.startswith("Deleted lines 18-21")


def test_edit_part_of_a_line_with_columns(ctx, sample_file, path, sample_source):
    prime(ctx, path)
    line = sample_source.split("\n")[17]  # "def helper(a, b=2):"
    col = line.index("2")
    out = write_code_chunk(ctx, path, 18, 18, "3", start_column=col, end_column=col + 1)
    assert read_text(sample_file) == sample_source.replace("b=2", "b=3")
    assert out.startswith("Replaced the selected text on lines 18-18")


def test_columns_can_span_several_lines(ctx, sample_file, path, sample_source):
    prime(ctx, path)
    # from `a, b=2` on line 18 through `a + b` on line 19
    start = sample_source.split("\n")[17].index("a, b=2")
    end = sample_source.split("\n")[18].index("a + b") + len("a + b")
    write_code_chunk(ctx, path, 18, 19, "x):\n    return x", start_column=start, end_column=end)
    assert "def helper(x):\n    return x\n" in read_text(sample_file)


def test_only_start_column_runs_to_the_end_of_the_line(ctx, sample_file, path, sample_source):
    prime(ctx, path)
    col = sample_source.split("\n")[17].index("b=2")
    write_code_chunk(ctx, path, 18, 18, "b=5):", start_column=col)
    assert "def helper(a, b=5):\n" in read_text(sample_file)


def test_two_edits_in_a_row_need_no_re_read(ctx, sample_file, path):
    prime(ctx, path)
    write_code_chunk(ctx, path, 10, 10, "MAX_RETRIES = 4")
    write_code_chunk(ctx, path, 10, 10, "MAX_RETRIES = 5")
    assert "MAX_RETRIES = 5\n" in read_text(sample_file)


# ---- write_code_chunk: refusals (the file must stay untouched) ---------------
def test_edit_that_breaks_the_syntax_is_refused(ctx, sample_file, path):
    prime(ctx, path)
    before = sample_file.read_bytes()
    out = write_code_chunk(ctx, path, 18, 19, "def helper(a, b=2:\n    return a")
    assert out.startswith("**Error:** This edit would leave")
    assert "syntax error" in out and "Nothing was written" in out
    assert sample_file.read_bytes() == before


def test_edit_that_breaks_the_indentation_is_refused(ctx, sample_file, path):
    prime(ctx, path)
    before = sample_file.read_bytes()
    out = write_code_chunk(ctx, path, 19, 19, "    return a +\n  return b", start_column=0)
    assert out.startswith("**Error:**")
    assert sample_file.read_bytes() == before


def test_file_that_was_never_read_is_refused(ctx, sample_file, path):
    before = sample_file.read_bytes()
    out = write_code_chunk(ctx, path, 18, 19, "x = 1")
    assert out.startswith("**Error:**") and "has not been read yet" in out
    assert sample_file.read_bytes() == before


def test_file_changed_since_it_was_read_is_refused(ctx, sample_file, path):
    prime(ctx, path)
    sample_file.write_text("# someone else edited this\n" + read_text(sample_file))
    before = sample_file.read_bytes()
    out = write_code_chunk(ctx, path, 18, 19, "x = 1")
    assert out.startswith("**Error:**") and "changed on disk" in out
    assert sample_file.read_bytes() == before

    # after reading it again the new line numbers work
    prime(ctx, path)
    assert write_code_chunk(ctx, path, 1, 1, "# replaced").startswith("Replaced lines 1-1")


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (0, 1, "start_line must be between 1 and 68"),
        (69, 69, "start_line must be between 1 and 68"),
        (10, 9999, "end_line must be between 9 and 67"),
        (10, 8, "end_line must be between 9 and 67"),
    ],
)
def test_out_of_range_lines_are_refused(ctx, sample_file, path, start, end, message):
    prime(ctx, path)
    before = sample_file.read_bytes()
    out = write_code_chunk(ctx, path, start, end, "x = 1")
    assert out.startswith("**Error:**") and message in out
    assert sample_file.read_bytes() == before


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start_column": 999}, "start_column must be between 0 and"),
        ({"start_column": -1}, "start_column must be between 0 and"),
        ({"end_column": 999}, "end_column must be between 0 and"),
        ({"start_column": 15, "end_column": 3}, "before its start"),
    ],
)
def test_bad_columns_are_refused(ctx, sample_file, path, kwargs, message):
    prime(ctx, path)
    before = sample_file.read_bytes()
    out = write_code_chunk(ctx, path, 18, 18, "x", **kwargs)
    assert out.startswith("**Error:**") and message in out
    assert sample_file.read_bytes() == before


def test_columns_cannot_be_combined_with_an_insert(ctx, path):
    prime(ctx, path)
    out = write_code_chunk(ctx, path, 18, 17, "x", start_column=0)
    assert out.startswith("**Error:**") and "end_line >= start_line" in out


def test_non_python_files_are_refused(ctx, tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello\n")
    out = write_code_chunk(ctx, str(p), 1, 1, "x")
    assert out.startswith("**Error:**")
    assert p.read_text() == "hello\n"


# ---- write_code_chunk: file hygiene -----------------------------------------
def test_crlf_line_endings_are_preserved(ctx, tmp_path):
    p = tmp_path / "crlf.py"
    p.write_bytes(b"def f():\r\n    return 1\r\n\r\nX = 1\r\n")
    prime(ctx, str(p))
    write_code_chunk(ctx, str(p), 1, 2, "def f():\n    return 2")
    assert p.read_bytes() == b"def f():\r\n    return 2\r\n\r\nX = 1\r\n"


def test_no_temp_files_are_left_behind(ctx, sample_file, path):
    prime(ctx, path)
    write_code_chunk(ctx, path, 10, 10, "MAX_RETRIES = 4")
    leftovers = [n for n in os.listdir(sample_file.parent) if n.endswith(".tmp") or n.startswith(".")]
    assert leftovers == []


def test_file_permissions_are_preserved(ctx, sample_file, path):
    sample_file.chmod(0o755)
    prime(ctx, path)
    write_code_chunk(ctx, path, 10, 10, "MAX_RETRIES = 4")
    assert stat.S_IMODE(sample_file.stat().st_mode) == 0o755


def test_writing_through_a_symlink_edits_the_real_file(ctx, sample_file, tmp_path):
    link = tmp_path / "link.py"
    os.symlink(sample_file, link)
    prime(ctx, str(link))
    write_code_chunk(ctx, str(link), 10, 10, "MAX_RETRIES = 9")
    assert link.is_symlink()  # the link itself was not replaced by a regular file
    assert "MAX_RETRIES = 9" in read_text(sample_file)


def test_index_hash_matches_the_file_after_a_write(ctx, sample_file, path):
    prime(ctx, path)
    write_code_chunk(ctx, path, 10, 10, "MAX_RETRIES = 4")
    assert get_indexed_hash(ctx.deps.db, path) == content_hash(read_text(sample_file))
    assert save_file(ctx.deps.db, path) is False  # already in sync


# ------------------------------------------------------------------- _splice
def test_splice_counts_inserted_lines():
    text = "a\nb\nc\n"
    assert tools._splice(text, 2, 2, "x\ny", None, None) == ("a\nx\ny\nc\n", 2)
    assert tools._splice(text, 2, 1, "x\n", None, None) == ("a\nx\nb\nc\n", 1)
    assert tools._splice(text, 1, 3, "", None, None) == ("", 0)
