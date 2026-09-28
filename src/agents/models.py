from typing import Literal

from pydantic import BaseModel, Field, field_validator

SupportedLanguage = Literal["python"]


class FunctionInput(BaseModel):
    """
    Structured input received by the AI Agent for function generation.

    The data must already be prepared by the upstream component.
    The Agent is responsible only for processing validated input.
    """

    language: SupportedLanguage = Field(
        description="Target programming language."
    )

    function_name: str = Field(
        min_length=1,
        description="Name of the function to generate."
    )

    purpose: str = Field(
        min_length=1,
        description="Purpose of the function."
    )

    context: str = Field(
        min_length=1,
        description="Context in which the function will be used."
    )

    requirements: list[str] = Field(
        min_length=1,
        description="Requirements the generated function must satisfy."
    )

    @field_validator("function_name", "purpose", "context")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("value cannot be empty")

        return value

    @field_validator("requirements")
    @classmethod
    def validate_requirements(cls, value: list[str]) -> list[str]:
        cleaned = []

        for requirement in value:
            requirement = requirement.strip()

            if not requirement:
                raise ValueError(
                    "requirements cannot contain empty values"
                )

            cleaned.append(requirement)

        if not cleaned:
            raise ValueError(
                "at least one requirement is required"
            )

        return cleaned