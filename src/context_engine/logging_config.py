"""
logging_config.py

Two logs, both under `<project_root>/.codeveto/logs/` by default:

- `context_engine.log` — normal rotating application log (warnings,
  errors, provider fallbacks) for debugging the engine itself.
- `audit.log` — one JSON object per line, one line per tool call
  (`search_context`, `expand_chunk`, `record_decision`, ...). This is the
  "what did the AI actually look at and decide" trail — the log-saving
  system CodeVeto's design asked for, and a natural fit for a tool that
  markets itself on developers staying in control.

The audit log intentionally never stores full chunk content or full file
contents — only paths, chunk ids, counts, and truncated argument summaries
— so it stays small and doesn't become a second copy of the codebase sitting
in a log file.
"""

from __future__ import annotations

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_MAX_LOG_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3
_MAX_FIELD_CHARS = 300

_configured = False


def setup_logging(log_dir: str | Path, level: int = logging.INFO) -> None:
    """Idempotent: safe to call every time `ContextEngine` is constructed."""
    global _configured
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("context_engine")
    if _configured:
        return
    root.setLevel(level)

    file_handler = RotatingFileHandler(
        log_dir / "context_engine.log",
        maxBytes=_MAX_LOG_BYTES,
        backupCount=_BACKUP_COUNT,
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(file_handler)

    audit_logger = logging.getLogger("context_engine.audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False
    audit_handler = RotatingFileHandler(
        log_dir / "audit.log", maxBytes=_MAX_LOG_BYTES, backupCount=_BACKUP_COUNT
    )
    audit_handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(audit_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_FIELD_CHARS:
        return value[:_MAX_FIELD_CHARS] + "...(truncated)"
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v) for v in value[:20]]
    return value


class AuditLogger:
    """Records one JSON line per tool call. Used by `tools.context_tools`
    so every `search_context` / `expand_chunk` / `record_decision` /
    `record_code_area` / `session_event` call leaves a trace, regardless of
    which agent framework ends up calling these tools."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("context_engine.audit")

    def log_call(
        self,
        tool: str,
        session_id: str | None,
        args: dict[str, Any],
        ok: bool,
        result_summary: Any = None,
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        record = {
            "ts": time.time(),
            "tool": tool,
            "session_id": session_id,
            "args": _truncate(args),
            "ok": ok,
            "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
        }
        if result_summary is not None:
            record["result"] = _truncate(result_summary)
        if error is not None:
            record["error"] = _truncate(error)
        self._logger.info(json.dumps(record))
