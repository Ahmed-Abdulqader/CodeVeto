from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


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
        d = {
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
        return d


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

    # Custom hashing/equality based on ID
    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, GraphNode):
            return self.id == other.id
        return False


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    type: EdgeType
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Custom hashing/equality based on source, target, and type
    def __hash__(self):
        return hash((self.source_id, self.target_id, self.type))

    def __eq__(self, other):
        if isinstance(other, GraphEdge):
            return (
                self.source_id == other.source_id
                and self.target_id == other.target_id
                and self.type == other.type
            )
        return False


class SymbolGraph(BaseModel):
    # Allow arbitrary types if needed for future extensions
    model_config = ConfigDict(arbitrary_types_allowed=True)

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)

    # Private attributes for internal indexing (not serialized/validated as fields)
    _node_index: dict[str, GraphNode] = PrivateAttr(default_factory=dict)
    # Using Dict with string key instead of Set to avoid Pydantic unhashable type errors
    _edge_index: dict[str, GraphEdge] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Rebuilds internal indices after the model is initialized with data."""
        for node in self.nodes:
            self._node_index[node.id] = node
        for edge in self.edges:
            edge_key = f"{edge.source_id}::{edge.target_id}::{edge.type.value}"
            self._edge_index[edge_key] = edge

    def add_node(self, node: GraphNode) -> GraphNode:
        """Adds a node to the graph, preventing duplicates."""
        if node.id not in self._node_index:
            self.nodes.append(node)
            self._node_index[node.id] = node
        return self._node_index[node.id]

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        """Adds an edge to the graph, preventing duplicates."""
        edge_key = f"{edge.source_id}::{edge.target_id}::{edge.type.value}"
        if edge_key not in self._edge_index:
            self.edges.append(edge)
            self._edge_index[edge_key] = edge
        return self._edge_index[edge_key]

    def get_node(self, node_id: str) -> GraphNode | None:
        """Retrieves a node by its ID."""
        return self._node_index.get(node_id)
