import os
import sqlite3

import pytest

from context_engine.ast_chunker import content_hash, normalize_path
from context_engine.chunks_saver import (
    SCHEMA_VERSION,
    get_indexed_hash,
    init_database,
    save_file,
)


def count(conn, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def columns(conn, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


# ------------------------------------------------------------------- schema
def test_init_database_creates_only_the_tables_in_use(db_path):
    with sqlite3.connect(db_path) as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert tables == {
            "files",
            "imports",
            "top_level_chunks",
            "classes",
            "functions",
            "methods",
        }
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()


def test_tables_only_store_what_the_agent_uses(conn):
    definition = ["name", "signature", "docstring", "start_line", "end_line", "code"]
    assert columns(conn, "files") == ["id", "file_path", "content_hash"]
    assert columns(conn, "imports") == [
        "id",
        "file_id",
        "start_line",
        "end_line",
        "code",
    ]
    assert columns(conn, "top_level_chunks") == [
        "id",
        "file_id",
        "start_line",
        "end_line",
        "code",
    ]
    assert columns(conn, "classes") == ["id", "file_id", *definition]
    assert columns(conn, "functions") == ["id", "file_id", *definition]
    assert columns(conn, "methods") == ["id", "class_id", "file_id", *definition]


def test_init_database_is_idempotent_and_keeps_data(db_path, sample_file):
    with sqlite3.connect(db_path) as conn:
        save_file(conn, str(sample_file))
    conn.close()

    init_database(db_path)

    with sqlite3.connect(db_path) as conn:
        assert count(conn, "files") == 1
    conn.close()


def test_init_database_rebuilds_an_old_layout(tmp_path):
    old = tmp_path / "old.db"
    with sqlite3.connect(old) as conn:
        conn.executescript(
            """
            CREATE TABLE files (id INTEGER PRIMARY KEY, file_path TEXT, file_name TEXT,
                                content_hash TEXT, last_modified REAL);
            CREATE TABLE class_inheritance (id INTEGER PRIMARY KEY, class_id INTEGER);
            INSERT INTO files (file_path, file_name) VALUES ('/x.py', 'x.py');
            """
        )
    conn.close()

    init_database(old)

    with sqlite3.connect(old) as conn:
        assert columns(conn, "files") == ["id", "file_path", "content_hash"]
        assert count(conn, "files") == 0
        names = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "class_inheritance" not in names
    conn.close()


# ---------------------------------------------------------------- save_file
def test_save_file_stores_every_chunk_kind(conn, sample_file):
    assert save_file(conn, str(sample_file)) is True

    assert count(conn, "files") == 1
    assert count(conn, "imports") == 4
    assert count(conn, "top_level_chunks") == 4
    assert count(conn, "classes") == 2  # Tiny, Account
    assert count(conn, "functions") == 2  # helper, fetch
    assert count(conn, "methods") == 5  # all of Account's


def test_save_file_stores_normalized_path_and_hash(conn, sample_file, sample_source):
    save_file(conn, str(sample_file))
    path, digest = conn.execute("SELECT file_path, content_hash FROM files").fetchone()
    assert path == normalize_path(str(sample_file))
    assert digest == content_hash(sample_source)


def test_save_file_stores_code_signature_and_lines(conn, sample_file):
    save_file(conn, str(sample_file))
    row = conn.execute(
        "SELECT signature, docstring, start_line, end_line, code "
        "FROM functions WHERE name = 'fetch'"
    ).fetchone()
    assert row[0].startswith("@cache\nasync def fetch(")
    assert row[1] == "Fetch a URL.\n\nRetries on failure."
    assert (row[2], row[3]) == (22, 33)
    assert row[4].startswith("@cache")


def test_methods_are_linked_to_their_class(conn, sample_file):
    save_file(conn, str(sample_file))
    rows = conn.execute(
        "SELECT c.name, m.name FROM methods m "
        "JOIN classes c ON c.id = m.class_id "
        "ORDER BY m.start_line"
    ).fetchall()
    assert [r[0] for r in rows] == ["Account"] * 5
    assert [r[1] for r in rows] == ["deposit", "create", "value", "value", "sync"]


def test_unchanged_file_is_skipped(conn, sample_file):
    assert save_file(conn, str(sample_file)) is True
    first_ids = conn.execute("SELECT id FROM functions ORDER BY id").fetchall()
    assert save_file(conn, str(sample_file)) is False
    assert conn.execute("SELECT id FROM functions ORDER BY id").fetchall() == first_ids


def test_changed_file_replaces_all_of_its_rows(conn, sample_file):
    save_file(conn, str(sample_file))
    sample_file.write_text("import os\n\ndef only():\n    return 1\n")

    assert save_file(conn, str(sample_file)) is True
    assert count(conn, "files") == 1
    assert count(conn, "imports") == 1
    assert count(conn, "top_level_chunks") == 0
    assert count(conn, "classes") == 0
    assert count(conn, "methods") == 0
    assert [r[0] for r in conn.execute("SELECT name FROM functions")] == ["only"]


def test_other_files_are_not_touched(conn, sample_file, tmp_path):
    other = tmp_path / "other.py"
    other.write_text("def other():\n    pass\n")
    save_file(conn, str(sample_file))
    save_file(conn, str(other))

    other.write_text("def renamed():\n    pass\n")
    save_file(conn, str(other))

    names = {r[0] for r in conn.execute("SELECT name FROM functions")}
    assert names == {"helper", "fetch", "renamed"}


def test_file_that_does_not_parse_keeps_the_old_index(conn, sample_file):
    save_file(conn, str(sample_file))
    sample_file.write_text("def broken(:\n")

    with pytest.raises(SyntaxError):
        save_file(conn, str(sample_file))

    assert count(conn, "functions") == 2  # still the previous version
    assert get_indexed_hash(conn, str(sample_file)) != content_hash("def broken(:\n")


def test_deleting_a_file_row_cascades_to_every_child_table(conn, sample_file):
    save_file(conn, str(sample_file))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("DELETE FROM files")
    conn.commit()
    for table in ("imports", "top_level_chunks", "classes", "functions", "methods"):
        assert count(conn, table) == 0


def test_relative_and_symlinked_paths_hit_the_same_row(
    conn, sample_file, tmp_path, monkeypatch
):
    link = tmp_path / "link.py"
    os.symlink(sample_file, link)
    monkeypatch.chdir(tmp_path)

    assert save_file(conn, "sample.py") is True
    assert save_file(conn, str(link)) is False  # same file, already indexed
    assert count(conn, "files") == 1


def test_get_indexed_hash(conn, sample_file, sample_source):
    assert get_indexed_hash(conn, str(sample_file)) is None
    save_file(conn, str(sample_file))
    assert get_indexed_hash(conn, str(sample_file)) == content_hash(sample_source)


def test_crlf_files_hash_like_lf_files(conn, tmp_path, sample_source):
    crlf = tmp_path / "crlf.py"
    crlf.write_bytes(sample_source.replace("\n", "\r\n").encode())
    save_file(conn, str(crlf))
    assert get_indexed_hash(conn, str(crlf)) == content_hash(sample_source)
