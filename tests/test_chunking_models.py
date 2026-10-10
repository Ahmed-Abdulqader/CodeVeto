import pytest
from pydantic import ValidationError

from models import chunking_models
from models.chunking_models import (
    ClassOut,
    Classes,
    FunctionsAndMethods,
    Imports,
    Methods,
    TopLevelChunks,
)

DEFINITION = {
    "name": "f",
    "signature": "def f():",
    "start_line": 1,
    "end_line": 2,
    "code": "def f():\n    pass",
}


def test_docstring_is_optional():
    assert FunctionsAndMethods.model_validate(DEFINITION).docstring is None


def test_unknown_fields_are_ignored():
    # the old parser output (columns, calls, args, ...) must not break validation
    chunk = TopLevelChunks.model_validate(
        {"start_line": 1, "end_line": 1, "code": "x = 1", "category": "constant", "node_type": "Assign"}
    )
    assert not hasattr(chunk, "category")
    assert set(chunk.model_dump()) == {"start_line", "end_line", "code"}


@pytest.mark.parametrize("model", [Imports, TopLevelChunks])
def test_code_chunks_require_a_line_range_and_code(model):
    with pytest.raises(ValidationError):
        model.model_validate({"start_line": 1, "end_line": 1})


def test_to_db_params_adds_file_id_and_extras():
    params = FunctionsAndMethods.model_validate(DEFINITION).to_db_params(7)
    assert params["file_id"] == 7
    assert params["name"] == "f"

    method = Methods.model_validate({**DEFINITION, "class_name": "C", "class_start_line": 1})
    assert method.to_db_params(7, class_id=3)["class_id"] == 3


def test_methods_carry_the_owning_class():
    with pytest.raises(ValidationError):
        Methods.model_validate(DEFINITION)  # no class_name / class_start_line


def test_class_out_needs_its_db_id():
    cls = {**DEFINITION, "name": "C", "signature": "class C:"}
    assert Classes.model_validate(cls).name == "C"
    with pytest.raises(ValidationError):
        ClassOut.model_validate(cls)
    assert ClassOut.model_validate({**cls, "id": 4}).id == 4


@pytest.mark.parametrize(
    "removed",
    [
        "Arg",
        "ClassInheritance",
        "ClassInheritanceOut",
        "ParentRef",
        "ClassWithParents",
        "FileOut",
        "FunctionOut",
        "MethodOut",
        "ImportOut",
        "TopLevelChunkOut",
        "JsonStrList",
        "JsonArgList",
    ],
)
def test_unused_models_are_gone(removed):
    assert not hasattr(chunking_models, removed)
