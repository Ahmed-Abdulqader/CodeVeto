"""
ignore.py

Small, dependency-free "should this path be indexed" filter. Not a full
.gitignore parser — matches directory names and glob patterns against path
components, which covers the common cases (`.git`, `node_modules`,
`__pycache__`, virtualenvs, build output) without pulling in a
`gitignore`-parsing library for it.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath

DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    "target",  # Rust/Cargo build output
    ".codeveto",  # the engine's own db/log directory
    "*.lock",
    "*.min.js",
)


def should_ignore(
    path: Path, root: Path, patterns: tuple[str, ...] = DEFAULT_IGNORE_PATTERNS
) -> bool:
    """`path` and `root` are both filesystem paths; `path` is checked
    relative to `root` so patterns match on directory/file names rather
    than absolute paths (which would never match a plain `"node_modules"`
    pattern)."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    rel_posix = PurePosixPath(relative.as_posix())
    parts = rel_posix.parts
    for pattern in patterns:
        if any(fnmatch.fnmatch(part, pattern) for part in parts):
            return True
        if fnmatch.fnmatch(rel_posix.as_posix(), pattern):
            return True
    return False
