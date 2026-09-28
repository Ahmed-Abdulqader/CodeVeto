import pytest
from pydantic import ValidationError

from agents.models import FunctionInput


def test_valid_function_input():
    data = FunctionInput(
        language="python",
        function_name="calculate_hash",
        purpose="Calculate the SHA-256 hash of input data.",
        context="Used by the security utility module.",
        requirements=[
            "Use hashlib.",
            "Return the hexadecimal digest.",
        ],
    )

    assert data.language == "python"
    assert data.function_name == "calculate_hash"
    assert len(data.requirements) == 2


def test_missing_function_name():
    with pytest.raises(ValidationError):
        FunctionInput(
            language="python",
            function_name="",
            purpose="Calculate a hash.",
            context="Security module.",
            requirements=["Use hashlib."],
        )


def test_missing_purpose():
    with pytest.raises(ValidationError):
        FunctionInput(
            language="python",
            function_name="calculate_hash",
            purpose="",
            context="Security module.",
            requirements=["Use hashlib."],
        )


def test_missing_context():
    with pytest.raises(ValidationError):
        FunctionInput(
            language="python",
            function_name="calculate_hash",
            purpose="Calculate a hash.",
            context="",
            requirements=["Use hashlib."],
        )


def test_empty_requirements():
    with pytest.raises(ValidationError):
        FunctionInput(
            language="python",
            function_name="calculate_hash",
            purpose="Calculate a hash.",
            context="Security module.",
            requirements=[],
        )


def test_invalid_language():
    with pytest.raises(ValidationError):
        FunctionInput(
            language="pythonnn",
            function_name="calculate_hash",
            purpose="Calculate a hash.",
            context="Security module.",
            requirements=["Use hashlib."],
        )


def test_empty_requirement_item():
    with pytest.raises(ValidationError):
        FunctionInput(
            language="python",
            function_name="calculate_hash",
            purpose="Calculate a hash.",
            context="Security module.",
            requirements=["Use hashlib.", ""],
        )


def test_whitespace_is_stripped():
    data = FunctionInput(
        language="python",
        function_name="  calculate_hash  ",
        purpose="  Calculate a hash.  ",
        context="  Security module.  ",
        requirements=["  Use hashlib.  "],
    )

    assert data.function_name == "calculate_hash"
    assert data.purpose == "Calculate a hash."
    assert data.context == "Security module."
    assert data.requirements == ["Use hashlib."]