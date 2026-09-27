"""
Shared data models for the CodeVeto context engine.

This is the single source of truth for every shape that crosses a module
boundary: chunker -> storage, graph builder -> storage, storage -> search,
search/memory -> tools. Keeping them in one place is what lets chunk_store,
graph_store, search, and tools agree on field names without importing each
other's internals.

The `Chunk`, `FileMetadata`, `EmbeddedChunk`, `NodeType`, `EdgeType`,
`GraphNode`, `GraphEdge`, and `SymbolGraph` shapes are unchanged from the
original `main` / `feature/graph-builder` branches on purpose, so nothing
upstream of this file (chunker.py's callers, existing scripts) has to
change. Everything below the "New for the context engine" marker is new.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

# ---------------------------------------------------------------------------
# Files & chunks
# ---------------------------------------------------------------------------


class FileMetadata(BaseModel):
    name: str
    path: str
    modified_ts: float


class Chunk(BaseModel):
    id: str
    language: str
    chunk_type: str
    docstring: str | None
    signature: str | None
    content: str
    start_line: int
    end_line: int
    file_metadata: FileMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "docstring": self.docstring,
            "signature": self.signature,
            "content": self.content,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "file_metadata": self.file_metadata,
        }


class EmbeddedChunk(BaseModel):
    """A `Chunk` paired with its resulting dense vector embedding."""

    chunk: Chunk
    embedding: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {"chunk": self.chunk.to_dict(), "embedding": self.embedding}


# ---------------------------------------------------------------------------
# Symbol graph
# ---------------------------------------------------------------------------


class NodeType(StrEnum):
    FILE = "File"
    CLASS = "Class"
    FUNCTION = "Function"
    METHOD = "Method"
    MODULE = "Module"


class EdgeType(StrEnum):
    CONTAINS = "CONTAINS"
    IMPORTS = "IMPORTS"
    EXTENDS = "EXTENDS"
    CALLS = "CALLS"


class GraphNode(BaseModel):
    id: str
    type: NodeType
    name: str
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GraphNode) and self.id == other.id


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    type: EdgeType
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.source_id, self.target_id, self.type))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GraphEdge)
            and self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.type == other.type
        )


class SymbolGraph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)

    _node_index: dict[str, GraphNode] = PrivateAttr(default_factory=dict)
    _edge_index: dict[str, GraphEdge] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        for node in self.nodes:
            self._node_index[node.id] = node
        for edge in self.edges:
            self._edge_index[self._edge_key(edge)] = edge

    @staticmethod
    def _edge_key(edge: GraphEdge) -> str:
        return f"{edge.source_id}::{edge.target_id}::{edge.type.value}"

    def add_node(self, node: GraphNode) -> GraphNode:
        if node.id not in self._node_index:
            self.nodes.append(node)
            self._node_index[node.id] = node
        return self._node_index[node.id]

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        key = self._edge_key(edge)
        if key not in self._edge_index:
            self.edges.append(edge)
            self._edge_index[key] = edge
        return self._edge_index[key]

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._node_index.get(node_id)


# ---------------------------------------------------------------------------
# New for the context engine: search results + session memory
# ---------------------------------------------------------------------------


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class SearchHit(BaseModel):
    """One ranked result from `search_context`. Deliberately does not carry
    the full chunk body — that costs tokens the developer didn't ask to
    spend yet. Call `expand_chunk(chunk_id)` for the full content."""

    chunk_id: str
    file_path: str
    chunk_type: str
    signature: str | None
    preview: str
    score: float
    matched_by: list[str] = Field(default_factory=list)


class Session(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("sess"))
    label: str | None = None
    created_ts: float = Field(default_factory=time.time)


class SessionEvent(BaseModel):
    id: int | None = None
    session_id: str
    event_type: str
    data: dict[str, Any] = Field(default_factory=dict)
    created_ts: float = Field(default_factory=time.time)


class Decision(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("dec"))
    session_id: str
    decision: str
    rationale: str | None = None
    related_chunk_ids: list[str] = Field(default_factory=list)
    created_ts: float = Field(default_factory=time.time)


class CodeArea(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("area"))
    session_id: str
    file_path: str
    chunk_id: str | None = None
    note: str | None = None
    created_ts: float = Field(default_factory=time.time)


class ContextBrief(BaseModel):
    """The condensed, budget-capped memory handed to the agent instead of
    raw conversation history."""

    session_id: str
    text: str
    decision_count: int
    code_area_count: int
    event_count: int
