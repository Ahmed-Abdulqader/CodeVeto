import pytest

from agents.agent import FunctionGenerationError, generate_function
from agents.models import FunctionInput


@pytest.mark.anyio
async def test_generate_function():
    function_data = FunctionInput(
        language="python",
        function_name="validate_email",
        purpose="Validate an email address",
        context="Used by the authentication module",
        requirements=[
            "Return True for valid emails",
            "Return False for invalid emails",
            "Do not use external libraries",
        ],
    )

    result = await generate_function(function_data)

    assert isinstance(result, str)
    assert result.strip()