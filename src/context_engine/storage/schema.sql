-- CodeVeto context engine schema.
--
-- One SQLite file per project (default: <project_root>/.codeveto/context.db).
-- SQLite (stdlib `sqlite3`, no download) backs everything: chunk/file
-- metadata, the symbol graph, the lexical search index, and session
-- memory. No vector database, no separate service to run.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------
-- Files & chunks
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    language TEXT NOT NULL,
    modified_ts REAL NOT NULL,
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL REFERENCES files (path) ON DELETE CASCADE,
    language TEXT NOT NULL,
    chunk_type TEXT NOT NULL,
    signature TEXT,
    docstring TEXT,
    content TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks (file_path);

-- ---------------------------------------------------------------------
-- Lexical (BM25) search index — a plain inverted index, built and queried
-- in pure Python (see search/bm25.py). No FTS5 dependency, so this works
-- the same on every SQLite build regardless of compile-time extensions.
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS bm25_postings (
    term TEXT NOT NULL,
    -- No FK to chunks(id): cleanup is done explicitly by
    -- LexicalIndex.remove_chunk() (called from indexer.remove_file()
    -- before the chunk row itself is deleted), which also lets
    -- LexicalIndex be indexed and tested against arbitrary chunk_ids
    -- without a matching chunks row.
    chunk_id TEXT NOT NULL,
    term_freq INTEGER NOT NULL,
    PRIMARY KEY (term, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_postings_term ON bm25_postings (term);

CREATE TABLE IF NOT EXISTS bm25_doc_stats (
    chunk_id TEXT PRIMARY KEY,
    doc_length INTEGER NOT NULL
);

-- ---------------------------------------------------------------------
-- Optional dense vectors. Empty and unused unless the developer configures
-- an embedding provider (see search/embedding.py) — the engine works fully
-- on the lexical index alone with this table empty.
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS embeddings (
    -- No FK to chunks(id), for the same reason as bm25_postings above:
    -- VectorStore.remove() is called explicitly by indexer.remove_file(),
    -- and this keeps VectorStore testable in isolation.
    chunk_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    vector TEXT NOT NULL -- JSON-encoded list[float]
);

-- ---------------------------------------------------------------------
-- Symbol graph
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS graph_nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    file_path TEXT,
    start_line INTEGER,
    end_line INTEGER,
    metadata TEXT -- JSON-encoded dict
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_file ON graph_nodes (file_path);

CREATE TABLE IF NOT EXISTS graph_edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    type TEXT NOT NULL,
    metadata TEXT, -- JSON-encoded dict
    PRIMARY KEY (source_id, target_id, type)
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges (source_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges (target_id);

-- ---------------------------------------------------------------------
-- Session memory. This is what lets the agent pick a conversation back up
-- without being handed raw chat history: session_event() logs what
-- happened, record_decision()/record_code_area() log what mattered, and
-- get_context_brief() condenses all of it into a small budget-capped brief.
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    label TEXT,
    created_ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS session_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    data TEXT, -- JSON-encoded dict
    created_ts REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_session ON session_events (session_id, created_ts);

CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    decision TEXT NOT NULL,
    rationale TEXT,
    related_chunk_ids TEXT, -- JSON-encoded list[str]
    created_ts REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_session ON decisions (session_id, created_ts);

CREATE TABLE IF NOT EXISTS code_areas (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    chunk_id TEXT,
    note TEXT,
    created_ts REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_areas_session ON code_areas (session_id, created_ts);
