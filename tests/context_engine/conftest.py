from __future__ import annotations

from pathlib import Path

import pytest

from context_engine.config import EngineConfig
from context_engine.engine import ContextEngine
from context_engine.storage.database import Database

PYTHON_SAMPLE = '''\
class Greeter:
    """Greets people."""

    def __init__(self, name):
        self.name = name

    def greet(self):
        """Say hello."""
        return say_hello(self.name)


def say_hello(name):
    return f"Hello, {name}!"
'''

JS_SAMPLE = """\
class Greeter {
    constructor(name) {
        this.name = name;
    }

    greet() {
        return sayHello(this.name);
    }
}

function sayHello(name) {
    return `Hello, ${name}!`;
}
"""

RUST_SAMPLE = """\
struct Greeter {
    name: String,
}

impl Greeter {
    fn greet(&self) -> String {
        say_hello(&self.name)
    }
}

fn say_hello(name: &str) -> String {
    format!("Hello, {}!", name)
}
"""


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    (tmp_path / "greeter.py").write_text(PYTHON_SAMPLE)
    (tmp_path / "greeter.js").write_text(JS_SAMPLE)
    (tmp_path / "greeter.rs").write_text(RUST_SAMPLE)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.js").write_text("function ignored() {}")
    return tmp_path


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "context.db")
    yield database
    database.close()


@pytest.fixture
def engine(sample_repo: Path) -> ContextEngine:
    config = EngineConfig(
        project_root=sample_repo,
        db_path=sample_repo / ".codeveto" / "context.db",
        log_dir=sample_repo / ".codeveto" / "logs",
    )
    eng = ContextEngine(config)
    yield eng
    eng.close()
