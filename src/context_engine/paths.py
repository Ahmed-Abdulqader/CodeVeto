"""
paths.py

Where CodeVeto's global, cross-project state lives: `~/.codeveto/`. Today
that's the local embedding model and its tokenizer — downloaded once,
shared by every project, so switching between two CodeVeto projects never
re-downloads a ~160MB file. This is also where future global configuration
(other embedding models, user preferences, ...) belongs, so the developer
sets things up once instead of re-entering them per project.

Per-project state (the SQLite index, logs) stays under
`<project_root>/.codeveto/` — see `config.EngineConfig` — because a
project's index is specific to that project's code and would collide with
a second project's index if it lived here instead.
"""

from __future__ import annotations

import os
from pathlib import Path

# Overridable so tests (and anyone who wants a non-default location) don't
# have to touch the real user home directory.
GLOBAL_CONFIG_DIR_ENV = "CODEVETO_HOME"

# The one local model this ships support for today. Keyed by directory
# name so a second model can be added later without disturbing this one.
JINA_CODE_MODEL_DIR_NAME = "jina-embeddings-v2-base-code"
JINA_CODE_MODEL_FILENAME = "model_quantized.onnx"
JINA_CODE_TOKENIZER_FILENAME = "tokenizer.json"


def global_config_dir() -> Path:
    """`~/.codeveto` by default; override with the `CODEVETO_HOME`
    environment variable."""
    override = os.environ.get(GLOBAL_CONFIG_DIR_ENV)
    return Path(override) if override else Path.home() / ".codeveto"


def models_dir() -> Path:
    return global_config_dir() / "models"


def jina_code_model_dir() -> Path:
    return models_dir() / JINA_CODE_MODEL_DIR_NAME


def jina_code_model_path() -> Path:
    return jina_code_model_dir() / JINA_CODE_MODEL_FILENAME


def jina_code_tokenizer_path() -> Path:
    return jina_code_model_dir() / JINA_CODE_TOKENIZER_FILENAME


def is_jina_code_model_installed() -> bool:
    return jina_code_model_path().exists() and jina_code_tokenizer_path().exists()
