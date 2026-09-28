from __future__ import annotations

from pathlib import Path

from context_engine import paths


class TestGlobalConfigDir:
    def test_defaults_to_home_dot_codeveto(self, monkeypatch):
        monkeypatch.delenv(paths.GLOBAL_CONFIG_DIR_ENV, raising=False)
        assert paths.global_config_dir() == Path.home() / ".codeveto"

    def test_env_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv(paths.GLOBAL_CONFIG_DIR_ENV, str(tmp_path / "custom"))
        assert paths.global_config_dir() == tmp_path / "custom"


class TestModelPaths:
    def test_paths_are_nested_under_global_config_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv(paths.GLOBAL_CONFIG_DIR_ENV, str(tmp_path))
        assert paths.models_dir() == tmp_path / "models"
        assert (
            paths.jina_code_model_dir()
            == tmp_path / "models" / "jina-embeddings-v2-base-code"
        )
        assert paths.jina_code_model_path().name == "model_quantized.onnx"
        assert paths.jina_code_tokenizer_path().name == "tokenizer.json"

    def test_is_installed_false_when_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv(paths.GLOBAL_CONFIG_DIR_ENV, str(tmp_path))
        assert paths.is_jina_code_model_installed() is False

    def test_is_installed_requires_both_files(self, monkeypatch, tmp_path):
        monkeypatch.setenv(paths.GLOBAL_CONFIG_DIR_ENV, str(tmp_path))
        paths.jina_code_model_dir().mkdir(parents=True)
        paths.jina_code_model_path().write_bytes(b"fake onnx")
        assert paths.is_jina_code_model_installed() is False  # tokenizer still missing

        paths.jina_code_tokenizer_path().write_text("{}")
        assert paths.is_jina_code_model_installed() is True
