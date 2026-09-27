from __future__ import annotations

from pathlib import Path

from context_engine.ignore import DEFAULT_IGNORE_PATTERNS, should_ignore


class TestShouldIgnore:
    def test_ignores_known_directory_names(self, tmp_path: Path):
        assert should_ignore(tmp_path / "node_modules" / "x.js", tmp_path)
        assert should_ignore(tmp_path / ".git" / "HEAD", tmp_path)
        assert should_ignore(tmp_path / "src" / "__pycache__" / "a.pyc", tmp_path)

    def test_does_not_ignore_ordinary_source_files(self, tmp_path: Path):
        assert not should_ignore(tmp_path / "src" / "app.py", tmp_path)

    def test_glob_pattern_matches_extension(self, tmp_path: Path):
        assert should_ignore(tmp_path / "uv.lock", tmp_path, patterns=("*.lock",))

    def test_custom_patterns_override_default_usage(self, tmp_path: Path):
        assert not should_ignore(
            tmp_path / "node_modules" / "x.js", tmp_path, patterns=("dist",)
        )

    def test_default_patterns_is_a_nonempty_tuple(self):
        assert isinstance(DEFAULT_IGNORE_PATTERNS, tuple)
        assert len(DEFAULT_IGNORE_PATTERNS) > 0
