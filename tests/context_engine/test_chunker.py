"""
Tests for context_engine.chunker.Chunks.

Assumptions this suite makes about models/context_engine_models.py, based only
on how chunker.py uses it (I don't have that file's actual content):
  - Chunk exposes attributes: id, language, chunk_type, docstring, signature,
    content, start_line, end_line, file_metadata, and a .to_dict() method.
  - FileMetadata exposes attributes: name, path, modified_ts.
If any of those names differ in your actual implementation, adjust the
attribute accesses below accordingly.

Also assumes tree-sitter-rust's grammar distinguishes a trait method
*declaration* (no body -> function_signature_item) from a function
*definition* (has body -> function_item). If your installed grammar version
differs, the "only one match for fn area" assertion may need loosening.
"""

import os
import uuid

import pytest

from context_engine.chunker import Chunks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_source(tmp_path, filename, content):
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


def get_chunk(chunks, contains):
    """Find the single chunk whose signature contains `contains`."""
    matches = [c for c in chunks if c.signature and contains in c.signature]
    assert len(matches) == 1, (
        f"Expected exactly one chunk with signature containing {contains!r}, "
        f"found {len(matches)}. All signatures: {[c.signature for c in chunks]}"
    )
    return matches[0]


# ---------------------------------------------------------------------------
# Fixture sources
# ---------------------------------------------------------------------------

PYTHON_SOURCE = '''"""Module docstring, not attached to any chunk."""


def top_level_function(a: int, b: int = 2) -> int:
    """Add two numbers together."""
    return a + b


def _no_docstring_function():
    return 42


class Greeter:
    """A friendly greeter class."""

    def greet(self, name: str) -> str:
        """Return a greeting for name."""
        return f"Hello, {name}!"

    def _private_helper(self):
        return None


def outer():
    def inner():
        return 1
    return inner()
'''

JS_SOURCE = """/**
 * Adds two numbers.
 * @param {number} a
 * @param {number} b
 */
function add(a, b) {
  return a + b;
}

/**
 * Multiplies two numbers.
 */
const multiply = (a, b) => {
  return a * b;
};

const notAFunction = 42;

class Calculator {
  constructor() {
    this.value = 0;
  }

  add(n) {
    this.value += n;
    return this.value;
  }
}

function* idGenerator() {
  let i = 0;
  while (true) {
    yield i++;
  }
}
"""

RUST_SOURCE = """/// Adds two numbers together.
fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// A simple point in 2D space.
struct Point {
    x: f64,
    y: f64,
}

/// Represents a color.
enum Color {
    Red,
    Green,
    Blue,
}

/// A trait for shapes.
trait Shape {
    fn area(&self) -> f64;
}

struct Circle {
    radius: f64,
}

impl Circle {
    /// Creates a new circle.
    fn new(radius: f64) -> Self {
        Circle { radius }
    }

    /// Computes the area of the circle.
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
}
"""


@pytest.fixture
def python_chunks(tmp_path):
    path = write_source(tmp_path, "sample.py", PYTHON_SOURCE)
    return Chunks(path)


@pytest.fixture
def js_chunks(tmp_path):
    path = write_source(tmp_path, "sample.js", JS_SOURCE)
    return Chunks(path)


@pytest.fixture
def rust_chunks(tmp_path):
    path = write_source(tmp_path, "sample.rs", RUST_SOURCE)
    return Chunks(path)


# ---------------------------------------------------------------------------
# Language detection / construction
# ---------------------------------------------------------------------------


class TestLanguageDetection:
    @pytest.mark.parametrize(
        "ext,expected",
        [
            (".py", "python"),
            (".js", "javascript"),
            (".jsx", "javascript"),
            (".mjs", "javascript"),
            (".rs", "rust"),
        ],
    )
    def test_detect_language_from_extension(self, tmp_path, ext, expected):
        path = write_source(tmp_path, f"file{ext}", "")
        chunks = Chunks(path)
        assert chunks.language == expected
        assert len(chunks) == 0

    def test_unsupported_extension_raises(self, tmp_path):
        path = write_source(tmp_path, "file.txt", "irrelevant")
        with pytest.raises(ValueError, match="Cannot detect language"):
            Chunks(path)

    def test_explicit_language_overrides_extension(self, tmp_path):
        path = write_source(tmp_path, "script.txt", "def f():\n    pass\n")
        chunks = Chunks(path, language="python")
        assert chunks.language == "python"
        assert len(chunks) == 1

    def test_unsupported_language_raises(self, tmp_path):
        path = write_source(tmp_path, "file.py", "")
        with pytest.raises(ValueError, match="Unsupported language"):
            Chunks(path, language="cobol")

    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.py"
        with pytest.raises(FileNotFoundError):
            Chunks(str(missing))


# ---------------------------------------------------------------------------
# Python chunking
# ---------------------------------------------------------------------------


class TestPythonChunking:
    def test_top_level_function(self, python_chunks):
        fn = get_chunk(python_chunks, "top_level_function")
        assert fn.chunk_type == "function"
        assert fn.language == "python"
        assert fn.signature == "def top_level_function(a: int, b: int = 2) -> int"
        assert fn.docstring == "Add two numbers together."

    def test_function_without_docstring(self, python_chunks):
        fn = get_chunk(python_chunks, "_no_docstring_function")
        assert fn.docstring is None

    def test_class_classified_and_docstring_extracted(self, python_chunks):
        cls = get_chunk(python_chunks, "class Greeter")
        assert cls.chunk_type == "class"
        assert cls.docstring == "A friendly greeter class."

    def test_method_inside_class_classified_as_method(self, python_chunks):
        method = get_chunk(python_chunks, "def greet")
        assert method.chunk_type == "method"
        assert method.signature == "def greet(self, name: str) -> str"
        assert method.docstring == "Return a greeting for name."

    def test_private_method_without_docstring(self, python_chunks):
        method = get_chunk(python_chunks, "_private_helper")
        assert method.chunk_type == "method"
        assert method.docstring is None

    def test_nested_function_is_function_not_method(self, python_chunks):
        inner = get_chunk(python_chunks, "def inner")
        assert inner.chunk_type == "function"


# ---------------------------------------------------------------------------
# JavaScript chunking
# ---------------------------------------------------------------------------


class TestJavaScriptChunking:
    def test_function_declaration_with_jsdoc(self, js_chunks):
        fn = get_chunk(js_chunks, "function add")
        assert fn.chunk_type == "function"
        assert fn.signature == "function add(a, b)"
        assert fn.docstring == (
            "Adds two numbers.\n@param {number} a\n@param {number} b"
        )

    def test_arrow_function_assigned_to_const(self, js_chunks):
        fn = get_chunk(js_chunks, "multiply")
        assert fn.chunk_type == "function"
        assert fn.signature == "multiply = (a, b) =>"
        assert fn.docstring == "Multiplies two numbers."

    def test_non_function_variable_declarator_skipped(self, js_chunks):
        assert not any(c.signature and "notAFunction" in c.signature for c in js_chunks)

    def test_class_declaration(self, js_chunks):
        cls = get_chunk(js_chunks, "class Calculator")
        assert cls.chunk_type == "class"

    def test_constructor_is_method(self, js_chunks):
        ctor = get_chunk(js_chunks, "constructor()")
        assert ctor.chunk_type == "method"

    def test_regular_method_is_method(self, js_chunks):
        method = get_chunk(js_chunks, "add(n)")
        assert method.chunk_type == "method"

    def test_generator_function_declaration(self, js_chunks):
        gen = get_chunk(js_chunks, "idGenerator")
        assert gen.chunk_type == "function"
        assert "idGenerator" in gen.signature


# ---------------------------------------------------------------------------
# Rust chunking
# ---------------------------------------------------------------------------


class TestRustChunking:
    def test_function_item(self, rust_chunks):
        fn = get_chunk(rust_chunks, "fn add")
        assert fn.chunk_type == "function"
        assert fn.signature == "fn add(a: i32, b: i32) -> i32"
        assert fn.docstring == "Adds two numbers together."

    def test_struct_item(self, rust_chunks):
        struct = get_chunk(rust_chunks, "struct Point")
        assert struct.chunk_type == "struct"
        assert struct.docstring == "A simple point in 2D space."

    def test_enum_item(self, rust_chunks):
        enum = get_chunk(rust_chunks, "enum Color")
        assert enum.chunk_type == "enum"
        assert enum.docstring == "Represents a color."

    def test_trait_item(self, rust_chunks):
        trait = get_chunk(rust_chunks, "trait Shape")
        assert trait.chunk_type == "trait"
        assert trait.docstring == "A trait for shapes."

    def test_impl_block_no_leading_doc_comment(self, rust_chunks):
        impl = get_chunk(rust_chunks, "impl Circle")
        assert impl.chunk_type == "impl"
        assert impl.docstring is None

    def test_method_inside_impl_classified_as_method(self, rust_chunks):
        new_fn = get_chunk(rust_chunks, "fn new")
        assert new_fn.chunk_type == "method"
        assert new_fn.signature == "fn new(radius: f64) -> Self"
        assert new_fn.docstring == "Creates a new circle."

    def test_second_method_inside_impl(self, rust_chunks):
        area_fn = get_chunk(rust_chunks, "fn area(&self) -> f64")
        assert area_fn.chunk_type == "method"
        assert area_fn.docstring == "Computes the area of the circle."


# ---------------------------------------------------------------------------
# Container behavior (__iter__, __len__, __getitem__, .chunk, to_list)
# ---------------------------------------------------------------------------


class TestContainerBehavior:
    def test_len_and_iter_agree(self, python_chunks):
        assert len(python_chunks) > 0
        assert len(list(python_chunks)) == len(python_chunks)

    def test_getitem_matches_iteration_order(self, python_chunks):
        as_list = list(python_chunks)
        assert python_chunks[0] is as_list[0]

    def test_chunk_property_returns_first(self, python_chunks):
        assert python_chunks.chunk is python_chunks[0]

    def test_chunk_property_none_when_empty(self, tmp_path):
        path = write_source(tmp_path, "empty.py", "# just a comment\nx = 1\n")
        chunks = Chunks(path)
        assert len(chunks) == 0
        assert chunks.chunk is None
        assert list(chunks) == []
        assert chunks.to_list() == []

    def test_to_list_shape(self, python_chunks):
        as_list = python_chunks.to_list()
        assert isinstance(as_list, list)
        assert len(as_list) == len(python_chunks)
        assert all(isinstance(item, dict) for item in as_list)


# ---------------------------------------------------------------------------
# Chunk metadata correctness (id, file metadata, content/line consistency)
# ---------------------------------------------------------------------------


class TestChunkMetadata:
    def test_chunk_id_is_valid_uuid(self, python_chunks):
        for c in python_chunks:
            uuid.UUID(c.id)  # raises ValueError if malformed

    def test_file_metadata_fields(self, tmp_path):
        path = write_source(tmp_path, "meta_check.py", "def f():\n    pass\n")
        chunks = Chunks(path)
        meta = chunks[0].file_metadata
        assert meta.name == "meta_check.py"
        assert meta.path == os.path.abspath(path)
        assert isinstance(meta.modified_ts, (int, float))

    def test_content_is_verbatim_source_slice(self, python_chunks):
        for c in python_chunks:
            assert c.content in PYTHON_SOURCE

    def test_line_span_matches_content_line_count(self, python_chunks):
        for c in python_chunks:
            spanned_lines = c.end_line - c.start_line + 1
            assert spanned_lines == c.content.count("\n") + 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
