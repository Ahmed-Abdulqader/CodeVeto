from pydantic import BaseModel, ConfigDict


# ======================================================================
# What gets stored
#
# The index only keeps what the agent's tools actually show:
#   - the code itself (imports, top-level code, classes, functions, methods)
#   - the line range of every chunk (needed to edit a chunk later)
#   - for classes / functions / methods: a one-line-ish `signature`
#     (decorators + `def ...` / `class ...` header) and the docstring,
#     which is what the "outline" views are built from.
#
# Nothing else (columns, call lists, argument lists, import aliases,
# parent-class ids, ...) is saved.
# ======================================================================
class Files(BaseModel):
    file_path: str
    content_hash: str


class Chunk(BaseModel):
    """A span of source code: where it is and what it says."""

    model_config = ConfigDict(extra="ignore")

    start_line: int
    end_line: int
    code: str

    def to_db_params(self, file_id: int, **extra) -> dict:
        return {**self.model_dump(), "file_id": file_id, **extra}


class Imports(Chunk):
    """One module-level `import ...` / `from ... import ...` statement."""


class TopLevelChunks(Chunk):
    """Any other module-level statement (constants, `if __name__...`, ...)."""


class Definition(Chunk):
    """A class, function or method."""

    name: str
    signature: str  # decorators + header, e.g. "@cache\nasync def f(x: int) -> str"
    docstring: str | None = None


class Classes(Definition):
    pass


class FunctionsAndMethods(Definition):
    pass


class Methods(FunctionsAndMethods):
    # class_name + class_start_line identify the owning class (-> class_id on insert)
    class_name: str
    class_start_line: int


# ======================================================================
# Query side (SELECT -> validated). Imports, TopLevelChunks and
# FunctionsAndMethods are reused as-is; only classes need their DB id,
# because the methods of a class are looked up through it.
# ======================================================================
class ClassOut(Classes):
    id: int
