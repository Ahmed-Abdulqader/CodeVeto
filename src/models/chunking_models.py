import json
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _loads(v):
    """DB stores lists as JSON text; accept both text and real lists."""
    if v is None:
        return []
    if isinstance(v, (str, bytes)):
        return json.loads(v)
    return v


class Arg(BaseModel):
    name: str
    type: Optional[str] = None


JsonStrList = Annotated[list[str], BeforeValidator(_loads)]
JsonArgList = Annotated[list[Arg], BeforeValidator(_loads)]


# ======================================================================
# INPUT models (parser output -> validated -> INSERT)
# ======================================================================
class Files(BaseModel):
    file_path: str
    file_name: str
    content_hash: str
    last_modified: float


class Classes(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    docstring: Optional[str] = None
    start_line: int
    end_line: int
    decorators: JsonStrList = Field(default_factory=list)
    code: str

    def to_db_params(self, file_id: int) -> dict:
        data = self.model_dump()
        data["decorators"] = json.dumps(data["decorators"])
        data["file_id"] = file_id
        return data


class ClassInheritance(BaseModel):
    # class_name + class_start_line identify the child class (-> class_id on insert)
    class_name: str
    class_start_line: int
    parent_class_name: str
    parent_class_id: Optional[int] = None


class FunctionsAndMethods(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["function", "async_function"]
    name: str
    docstring: Optional[str] = None
    returns: Optional[str] = None
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    args: JsonArgList = Field(default_factory=list)
    calls: JsonStrList = Field(default_factory=list)
    decorators: JsonStrList = Field(default_factory=list)
    code: str

    def to_db_params(self, file_id: int, class_id: Optional[int] = None) -> dict:
        data = self.model_dump()
        data["args"] = json.dumps(data["args"])
        data["calls"] = json.dumps(data["calls"])
        data["decorators"] = json.dumps(data["decorators"])
        data["file_id"] = file_id
        data["class_id"] = class_id
        return data


class Methods(FunctionsAndMethods):
    # class_name + class_start_line identify the owning class (-> class_id on insert)
    class_name: str
    class_start_line: int


class Imports(BaseModel):
    model_config = ConfigDict(extra="ignore")

    import_type: Literal["import", "import_from"]
    module_name: str
    imported_name: Optional[str] = None
    alias: Optional[str] = None
    target_file_id: Optional[int] = None
    line_number: int

    def to_db_params(self, file_id: int) -> dict:
        data = self.model_dump()
        data["source_file_id"] = file_id
        return data


class TopLevelChunks(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: str
    node_type: str
    start_line: int
    end_line: int
    code: Optional[str] = None

    def to_db_params(self, file_id: int) -> dict:
        data = self.model_dump()
        data["file_id"] = file_id
        return data


# ======================================================================
# OUTPUT models (SELECT -> validated -> use in your code)
# Same fields as the input models + the DB ids.
# ======================================================================
class FileOut(Files):
    id: int
    parsed_at: Optional[str] = None


class ClassOut(Classes):
    id: int



class ParentRef(BaseModel):
    parent_class_name: str
    parent_class_id: Optional[int] = None  # None = external / unresolved


class ClassInheritanceOut(BaseModel):
    class_id: int
    parent_class_name: str
    parent_class_id: Optional[int] = None


class ClassWithParents(ClassOut):
    """name, docstring, start_line, end_line, decorators + parents (name + id)."""
    parents: list[ParentRef] = Field(default_factory=list)


class FunctionOut(FunctionsAndMethods):
    id: int


class MethodOut(FunctionsAndMethods):
    id: int
    file_id: int
    class_id: int


class ImportOut(Imports):
    id: int
    source_file_id: int


class TopLevelChunkOut(TopLevelChunks):
    id: int
    file_id: int


# class Signature(BaseModel):
#     file_id: int
#     functions: [FunctionOut]
