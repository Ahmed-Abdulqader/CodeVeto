"""
engine.py

The single entry point. Construct one `ContextEngine` per project, call
`.index()` once, then hand `.tools` to whatever agent framework CodeVeto
ends up using — every tool it needs (`search_context`, `expand_chunk`,
`list_files`, `session_event`, `record_decision`, `record_code_area`,
`get_context_brief`) is a bound method on `.tools`, and `.tool_specs` is
the framework-agnostic JSON-schema version of the same list.

    engine = ContextEngine(EngineConfig(project_root=Path(".")))
    engine.index()
    session = engine.new_session(label="fix-login-bug")
    engine.tools.search_context("password hashing", session_id=session.id)

No agent loop, no LLM calls, no `write_file`/`run_code` live here — this
class only indexes and answers questions about the code that's already on
disk.
"""

from __future__ import annotations

from context_engine.config import EngineConfig
from context_engine.indexer import RepoIndexer
from context_engine.logging_config import setup_logging
from context_engine.memory.session_manager import SessionManager
from context_engine.search.hybrid import ContextSearch
from context_engine.search.lexical import LexicalIndex
from context_engine.storage.chunk_store import ChunkStore
from context_engine.storage.database import Database
from context_engine.storage.graph_store import GraphStore
from context_engine.storage.session_store import SessionStore
from context_engine.storage.vector_store import VectorStore
from context_engine.tools.context_tools import ContextTools
from context_engine.tools.schemas import TOOL_SPECS
from models.context_engine_models import Session


class ContextEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        setup_logging(config.log_dir)

        self.db = Database(config.db_path)
        self.chunk_store = ChunkStore(self.db)
        self.graph_store = GraphStore(self.db)
        self.session_store = SessionStore(self.db)
        self.lexical_index = LexicalIndex(self.db)
        self.vector_store = VectorStore(self.db)

        self.embedding_provider = config.build_embedding_provider()

        self.indexer = RepoIndexer(
            project_root=config.project_root,
            chunk_store=self.chunk_store,
            graph_store=self.graph_store,
            lexical_index=self.lexical_index,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            ignore_patterns=config.ignore_patterns,
            max_file_size_bytes=config.max_file_size_bytes,
        )
        self.search = ContextSearch(
            chunk_store=self.chunk_store,
            lexical_index=self.lexical_index,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
        )
        self.session_manager = SessionManager(
            self.session_store, brief_budget_chars=config.brief_budget_chars
        )
        self.tools = ContextTools(
            chunk_store=self.chunk_store,
            graph_store=self.graph_store,
            search=self.search,
            session_manager=self.session_manager,
        )

    # -- indexing ------------------------------------------------------

    def index(self, force: bool = False) -> dict[str, int]:
        """Index (or re-index) the whole project. Safe to call again later
        — unchanged files are skipped unless `force=True`."""
        return self.indexer.index_repository(force=force)

    def index_file(self, path: str) -> int:
        """Index just one file — the case that matters most in practice:
        the developer let the agent edit one named file, and the engine
        needs to catch up on just that file, cheaply."""
        return self.indexer.index_file(path)

    # -- sessions --------------------------------------------------------

    def new_session(self, label: str | None = None) -> Session:
        return self.session_manager.start_session(label)

    # -- tool metadata for wiring into an agent framework -------------------

    @property
    def tool_specs(self) -> list[dict]:
        return TOOL_SPECS

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> ContextEngine:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
