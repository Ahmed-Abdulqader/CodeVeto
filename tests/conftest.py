import sqlite3
from types import SimpleNamespace

import pytest

from context_engine import chunk_query
from context_engine.chunks_saver import init_database, save_file
from models.agents_models import DatabaseDeps

# Line numbers matter in the tests, so keep this file's layout stable.
#  1        module docstring
#  2-8      imports (one is multi-line)
#  10       constant
#  12-15    top-level try/except (an import hidden inside it)
#  18-19    helper()
#  22-33    fetch()      decorated, async, has a docstring, a nested def, a local import
#  36-37    Tiny
#  40-63    Account      decorated, bases + metaclass, docstring, 5 methods
#  66-67    __main__ guard
SAMPLE_SOURCE = '''\
"""Sample module."""
import os
import sys as system
from collections import (
    OrderedDict,
    defaultdict,
)
from . import sibling

MAX_RETRIES = 3

try:
    import json
except ImportError:
    json = None


def helper(a, b=2):
    return a + b


@cache
async def fetch(url: str, *, retries: int = MAX_RETRIES) -> str:
    """Fetch a URL.

    Retries on failure.
    """
    import socket

    def inner():
        return socket

    return inner()


class Tiny:
    pass


@dataclass
class Account(Base, metaclass=Meta):
    """A bank account."""

    balance: int = 0

    def deposit(self, amount: int) -> None:
        """Add money."""
        self.balance += amount

    @staticmethod
    def create() -> "Account":
        return Account()

    @property
    def value(self):
        return self.balance

    @value.setter
    def value(self, v):
        self.balance = v

    async def sync(self):
        await fetch("x")


if __name__ == "__main__":
    helper(1)
'''


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    """Never touch tiktoken (it downloads its vocabulary):
    use the ~4 chars/token fallback."""
    monkeypatch.setattr(chunk_query, "_encoder", lambda: None)


@pytest.fixture
def sample_source() -> str:
    return SAMPLE_SOURCE


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "sample.py"
    path.write_bytes(SAMPLE_SOURCE.encode("utf-8"))
    return path


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "index.db"
    init_database(path)
    return path


@pytest.fixture
def conn(db_path):
    connection = sqlite3.connect(db_path)
    yield connection
    connection.close()


@pytest.fixture
def indexed(conn, sample_file):
    """A connection whose index already contains `sample_file`."""
    save_file(conn, str(sample_file))
    return conn


@pytest.fixture
def ctx(conn):
    """Stand-in for pydantic-ai's RunContext: the tools only use `.deps.db`."""
    return SimpleNamespace(deps=DatabaseDeps(db=conn))
