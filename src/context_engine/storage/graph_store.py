"""
graph_store.py

Persists a `SymbolGraph` (nodes + edges) and answers the traversal
questions `expand_chunk` needs: "what calls this?", "what does this call?",
"what does this file contain?". The graph itself is built in-memory by
`graph.graph_builder.GraphBuilder`; this module is purely storage.
"""

from __future__ import annotations

import json

from context_engine.storage.database import Database
from models.context_engine_models import EdgeType, GraphNode, NodeType, SymbolGraph


class GraphStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def save_graph(self, graph: SymbolGraph) -> None:
        if graph.nodes:
            self.db.executemany(
                """
                INSERT OR REPLACE INTO graph_nodes
                    (id, type, name, file_path, start_line, end_line, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        n.id,
                        n.type.value,
                        n.name,
                        n.file_path,
                        n.start_line,
                        n.end_line,
                        json.dumps(n.metadata),
                    )
                    for n in graph.nodes
                ],
            )
        if graph.edges:
            self.db.executemany(
                """
                INSERT OR REPLACE INTO graph_edges
                    (source_id, target_id, type, metadata)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (e.source_id, e.target_id, e.type.value, json.dumps(e.metadata))
                    for e in graph.edges
                ],
            )

    def delete_file_nodes(self, file_path: str) -> None:
        # Edges referencing this file's nodes are cleaned up as a second
        # pass since graph_edges has no FK (an edge's endpoints can point at
        # nodes declared in other files, e.g. a call to an imported symbol).
        rows = self.db.query(
            "SELECT id FROM graph_nodes WHERE file_path = ?", (file_path,)
        )
        node_ids = [r["id"] for r in rows]
        self.db.execute("DELETE FROM graph_nodes WHERE file_path = ?", (file_path,))
        for node_id in node_ids:
            self.db.execute(
                "DELETE FROM graph_edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id),
            )

    def get_node(self, node_id: str) -> GraphNode | None:
        row = self.db.query_one("SELECT * FROM graph_nodes WHERE id = ?", (node_id,))
        return _row_to_node(row) if row else None

    def nodes_for_file(self, file_path: str) -> list[GraphNode]:
        rows = self.db.query(
            "SELECT * FROM graph_nodes WHERE file_path = ?", (file_path,)
        )
        return [_row_to_node(r) for r in rows]

    def callers_of(self, node_id: str) -> list[GraphNode]:
        """Nodes with a CALLS edge pointing at `node_id`."""
        rows = self.db.query(
            """
            SELECT gn.* FROM graph_edges ge
            JOIN graph_nodes gn ON gn.id = ge.source_id
            WHERE ge.target_id = ? AND ge.type = ?
            """,
            (node_id, EdgeType.CALLS.value),
        )
        return [_row_to_node(r) for r in rows]

    def callees_of(self, node_id: str) -> list[GraphNode]:
        """Nodes `node_id` has a CALLS edge pointing at."""
        rows = self.db.query(
            """
            SELECT gn.* FROM graph_edges ge
            JOIN graph_nodes gn ON gn.id = ge.target_id
            WHERE ge.source_id = ? AND ge.type = ?
            """,
            (node_id, EdgeType.CALLS.value),
        )
        return [_row_to_node(r) for r in rows]

    def node_for_chunk_span(self, file_path: str, start_line: int) -> GraphNode | None:
        """Find the FUNCTION/METHOD/CLASS node whose span starts on the same
        line as a stored chunk, so `expand_chunk` can look up its
        callers/callees. Chunk IDs and graph node IDs aren't the same
        namespace (chunks come from chunker.py, nodes from graph_builder.py,
        built from two independent walks of the same tree), so this matches
        by (file, start_line) instead."""
        row = self.db.query_one(
            """
            SELECT * FROM graph_nodes
            WHERE file_path = ? AND start_line = ?
              AND type IN (?, ?, ?)
            LIMIT 1
            """,
            (
                file_path,
                start_line,
                NodeType.FUNCTION.value,
                NodeType.METHOD.value,
                NodeType.CLASS.value,
            ),
        )
        return _row_to_node(row) if row else None


def _row_to_node(row) -> GraphNode:
    return GraphNode(
        id=row["id"],
        type=NodeType(row["type"]),
        name=row["name"],
        file_path=row["file_path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
    )
