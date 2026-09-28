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


@pytest.fixture(autouse=True)
def isolated_codeveto_home(tmp_path_factory, monkeypatch):
    """Every test gets its own empty CODEVETO_HOME, so no test reads or
    writes the real ~/.codeveto, and no test's behavior depends on
    whether the embedding model happens to already be installed on
    whatever machine the suite runs on. `EmbeddingSettings` now defaults
    to `provider="local_onnx"`, which looks here -- without this, the
    engine fixture below would behave differently on a machine that has
    the model installed than on one that doesn't."""
    fake_home = tmp_path_factory.mktemp("codeveto_home")
    monkeypatch.setenv("CODEVETO_HOME", str(fake_home))


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
