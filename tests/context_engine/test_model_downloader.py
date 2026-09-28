from __future__ import annotations

import hashlib
import io
import urllib.error
from unittest.mock import patch

import pytest

from context_engine import model_downloader as md
from context_engine import paths


class _FakeResponse:
    """Minimal stand-in for `http.client.HTTPResponse` as used by
    `model_downloader._download`: a context manager with `.status`,
    `.headers.get(...)`, and chunked `.read(n)`."""

    def __init__(
        self, body: bytes, status: int = 200, content_length: int | None = None
    ):
        self._buf = io.BytesIO(body)
        self.status = status
        self.headers = (
            {"Content-Length": str(content_length)}
            if content_length is not None
            else {}
        )

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.GLOBAL_CONFIG_DIR_ENV, str(tmp_path))


class TestEnsureModelInstalled:
    def test_noop_when_already_installed(self, tmp_path):
        paths.jina_code_model_dir().mkdir(parents=True)
        paths.jina_code_model_path().write_bytes(b"already here")
        paths.jina_code_tokenizer_path().write_text("{}")

        with patch("urllib.request.urlopen") as mock_urlopen:
            md.ensure_model_installed()
            mock_urlopen.assert_not_called()

    def test_downloads_both_files_when_missing(self, monkeypatch):
        model_bytes = b"x" * (md._MIN_MODEL_BYTES + 1000)
        model_hash = hashlib.sha256(model_bytes).hexdigest()
        monkeypatch.setattr(md, "MODEL_SHA256", model_hash)
        tokenizer_bytes = b"{" + b"a" * 2000 + b"}"

        responses = [
            _FakeResponse(model_bytes, content_length=len(model_bytes)),
            _FakeResponse(tokenizer_bytes, content_length=len(tokenizer_bytes)),
        ]
        with patch("urllib.request.urlopen", side_effect=responses):
            md.ensure_model_installed()

        assert paths.jina_code_model_path().read_bytes() == model_bytes
        assert paths.jina_code_tokenizer_path().read_bytes() == tokenizer_bytes
        assert not paths.jina_code_model_path().with_suffix(".onnx.part").exists()

    def test_progress_callback_is_invoked(self, monkeypatch):
        model_bytes = b"x" * (md._MIN_MODEL_BYTES + 1000)
        monkeypatch.setattr(md, "MODEL_SHA256", hashlib.sha256(model_bytes).hexdigest())
        tokenizer_bytes = b"{}" * 1000

        responses = [
            _FakeResponse(model_bytes, content_length=len(model_bytes)),
            _FakeResponse(tokenizer_bytes, content_length=len(tokenizer_bytes)),
        ]
        seen = []
        with patch("urllib.request.urlopen", side_effect=responses):
            md.ensure_model_installed(progress_callback=seen.append)

        assert seen, "progress_callback should have been called at least once"
        assert seen[-1].label in ("embedding model", "tokenizer")
        assert seen[-1].downloaded_bytes > 0


class TestDownloadFailureModes:
    def test_network_error_raises_model_download_error_with_instructions(self):
        with patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("offline")
        ):
            with pytest.raises(md.ModelDownloadError) as excinfo:
                md.ensure_model_installed()
        assert md.MODEL_URL in excinfo.value.manual_instructions

    def test_truncated_download_raises_and_reports_incomplete(self):
        too_small = b"x" * 1000  # far under _MIN_MODEL_BYTES
        with patch(
            "urllib.request.urlopen",
            return_value=_FakeResponse(too_small, content_length=len(too_small)),
        ):
            with pytest.raises(md.ModelDownloadError, match="incomplete"):
                md.ensure_model_installed()

    def test_checksum_mismatch_raises_and_removes_partial_file(self, monkeypatch):
        model_bytes = b"x" * (md._MIN_MODEL_BYTES + 1000)
        monkeypatch.setattr(md, "MODEL_SHA256", "0" * 64)  # deliberately wrong

        with patch(
            "urllib.request.urlopen",
            return_value=_FakeResponse(model_bytes, content_length=len(model_bytes)),
        ):
            with pytest.raises(md.ModelDownloadError, match="checksum"):
                md.ensure_model_installed()

        part_path = paths.jina_code_model_path().with_suffix(".onnx.part")
        assert not part_path.exists()
        assert not paths.jina_code_model_path().exists()

    def test_manual_instructions_name_exact_urls_and_paths(self):
        text = md.manual_instructions()
        assert md.MODEL_URL in text
        assert md.TOKENIZER_URL in text
        assert str(paths.jina_code_model_path()) in text
        assert str(paths.jina_code_tokenizer_path()) in text


class TestResume:
    def test_existing_part_file_sends_range_header(self, monkeypatch):
        paths.jina_code_model_dir().mkdir(parents=True)
        partial = b"x" * 5000
        part_path = paths.jina_code_model_path().with_suffix(".onnx.part")
        part_path.write_bytes(partial)

        rest = b"y" * (md._MIN_MODEL_BYTES + 1000)
        full = partial + rest
        monkeypatch.setattr(md, "MODEL_SHA256", hashlib.sha256(full).hexdigest())

        captured_requests: list = []

        def fake_urlopen(request, timeout=None):
            captured_requests.append(request)
            if len(captured_requests) == 1:
                return _FakeResponse(rest, status=206, content_length=len(rest))
            tokenizer_bytes = b"{}" * 1000
            return _FakeResponse(tokenizer_bytes, content_length=len(tokenizer_bytes))

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            md.ensure_model_installed()

        first_request = captured_requests[0]
        assert first_request.get_header("Range") == f"bytes={len(partial)}-"
        assert paths.jina_code_model_path().read_bytes() == full
