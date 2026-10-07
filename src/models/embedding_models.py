from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"


def humanize_code_name(name: str) -> str:
    """
    Converts symbols like:
        createUser
        create_user
        UserFactory
        user.save_account

    Into more searchable text like:
        create user
        create user
        user factory
        user save account
    """
    name = name.strip()
    if not name:
        return name

    # Replace separators with spaces.
    name = re.sub(r"[._\-/\\]+", " ", name)

    # Split camelCase / PascalCase.
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", name)

    # Collapse spaces.
    name = re.sub(r"\s+", " ", name).strip()

    return name


class CodeItemIn(BaseModel):
    """
    Input model used when inserting/updating an item into the vector DB.

    source_id is the ID from your original database table.
    It is only unique inside its own entity_type.
    """

    entity_type: EntityType
    source_id: int = Field(
        ...,
        description=(
            "ID from the original functions/classes/methods table. "
            "Unique only within its entity_type."
        ),
    )
    name: str = Field(..., min_length=1)
    extra_text: Optional[str] = Field(
        default=None,
        description=(
            "Optional signature, docstring, namespace, or context. "
            "This can improve semantic search."
        ),
    )

    def embedding_text(self) -> str:
        """
        Text that will be embedded.
        You can include extra_text if you want richer semantic matching.
        """
        base = humanize_code_name(self.name)

        if self.extra_text:
            extra = self.extra_text.strip()
            if extra:
                return f"{base} {extra}"

        return base


class CodeItemOut(BaseModel):
    """
    Output model representing a stored row in the SQLite vector index.
    """

    row_id: int = Field(..., description="Internal SQLite row id.")
    entity_type: EntityType
    source_id: int
    name: str


class SearchResult(BaseModel):
    """
    Search hit.

    Use entity_type + source_id to query your original database.
    """

    entity_type: EntityType
    source_id: int
    name: str
    score: float = Field(..., description="Cosine similarity. Higher is closer.")
    row_id: Optional[int] = Field(default=None, description="Internal SQLite row id.")


class SearchQuery(BaseModel):
    """
    Search request model.
    """

    query: str = Field(..., min_length=1)
    entity_types: Optional[List[EntityType]] = Field(
        default=None,
        description="Optional filter. Example: only functions, only classes, etc.",
    )
    limit: int = Field(default=10, ge=1)
    min_score: float = Field(default=-1.0, ge=-1.0, le=1.0)


class SearchResponse(BaseModel):
    """
    Optional response wrapper if you want a clean API payload.
    """

    query: str
    count: int
    results: List[SearchResult]