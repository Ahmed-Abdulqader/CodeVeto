import asyncio
import os

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .models import FunctionInput

load_dotenv()


class FunctionGenerationError(Exception):
    """Raised when the AI Agent fails to generate a function."""


def create_agent() -> Agent:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise FunctionGenerationError("OPENROUTER_API_KEY is not configured.")

    model = OpenAIChatModel(
        "nvidia/nemotron-3.5-lightning:free",
        provider=OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        ),
    )

    return Agent(
        model,
        system_prompt=(
            "You are a function-generation assistant. "
            "Generate only the requested function. "
            "Do not access files, execute commands, or modify the system. "
            "Follow the provided language, context, and requirements."
        ),
    )


agent = create_agent()


async def generate_function(function_data: FunctionInput) -> str:
    prompt = f"""
Generate the requested function.

Target language:
{function_data.language}

Function name:
{function_data.function_name}

Purpose:
{function_data.purpose}

Context:
{function_data.context}

Requirements:
{chr(10).join(f"- {requirement}" for requirement in function_data.requirements)}

Return only the generated function code.
Do not include explanations or Markdown code fences.
"""

    try:
        result = await agent.run(prompt)
    except Exception as exc:
        raise FunctionGenerationError(f"Failed to generate function: {exc}") from exc

    return result.output


async def main():
    function_data = FunctionInput(
        language="python",
        function_name="validate_email",
        purpose="Validate whether an email address is valid",
        context="Used by the authentication module",
        requirements=[
            "Return true for valid emails",
            "Return false for invalid emails",
            "Do not use external libraries",
        ],
    )

    print("Generating function...\n")

    result = await generate_function(function_data)

    print("Generated function:")
    print("-" * 50)
    print(result)
    print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
