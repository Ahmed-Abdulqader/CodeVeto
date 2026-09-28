"""
model_downloader.py

Downloads jina-embeddings-v2-base-code's INT8-quantized ONNX export and
its tokenizer into `~/.codeveto/models/` the first time they're needed.
Uses only `urllib.request` from the stdlib — no `huggingface_hub`, no
extra dependency just to fetch two files.

Downloads resume on retry (`Range` requests against a `.part` file) rather
than restarting from zero — worth having given these files are large
enough that a dropped connection partway through is a real scenario, not
an edge case, on a constrained connection.

If the download can't complete for any reason (offline, a firewall, a
Hugging Face outage, ...), this never leaves the caller guessing: it
raises `ModelDownloadError`, whose `.manual_instructions` is the exact
URLs and exact file paths to save them to, for `main.py` to print.

Provenance note: the URLs, file paths, and the model's SHA256 below come
from directly checking https://huggingface.co/jinaai/jina-embeddings-v2-base-code
(the official repo) — not from running this downloader myself. I don't
have network access to huggingface.co from where this code is built, so
this hasn't been tested against a live download. Please run it once
somewhere with normal internet access before relying on it.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from context_engine.paths import (
    is_jina_code_model_installed,
    jina_code_model_dir,
    jina_code_model_path,
    jina_code_tokenizer_path,
)

HF_REPO = "jinaai/jina-embeddings-v2-base-code"
_HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"
MODEL_URL = f"{_HF_BASE}/onnx/model_quantized.onnx"
TOKENIZER_URL = f"{_HF_BASE}/tokenizer.json"

# Cross-checked against two independent copies of the same file (the
# official repo's git history and a third-party mirror) at the time this
# was written — LFS SHA256 is content-addressed, so agreement between
# independent copies is a strong (though not certain, if upstream ever
# replaces the file) signal this is correct.
MODEL_SHA256 = "ed45870251c9f0cf656e78aab0d37a23489066df8a222bb1c8caf8a45f2cb16d"

# No confirmed hash for tokenizer.json (small file, low value target for
# corruption anyway) -- just sanity-check it isn't empty/truncated.
_MIN_TOKENIZER_BYTES = 1024
_MIN_MODEL_BYTES = 100 * 1024 * 1024  # real file is ~162MB; a truncated
# download of that will land far below this, a healthy one comfortably above it.

DOWNLOAD_TIMEOUT_SECONDS = 60.0
_CHUNK_SIZE = 1024 * 1024


class ModelDownloadError(RuntimeError):
    """Raised when the model/tokenizer can't be fetched automatically.
    `manual_instructions` is meant to be shown to the developer as-is."""

    def __init__(self, message: str, manual_instructions: str) -> None:
        super().__init__(message)
        self.manual_instructions = manual_instructions


@dataclass
class DownloadProgress:
    label: str
    downloaded_bytes: int
    total_bytes: int | None


ProgressCallback = Callable[[DownloadProgress], None]


def ensure_model_installed(progress_callback: ProgressCallback | None = None) -> None:
    """Downloads the model + tokenizer into `~/.codeveto/models/` if
    either is missing. Idempotent and safe to call on every startup —
    `main.py` is expected to call this unconditionally."""
    if is_jina_code_model_installed():
        return

    jina_code_model_dir().mkdir(parents=True, exist_ok=True)
    _download(
        MODEL_URL,
        jina_code_model_path(),
        label="embedding model",
        min_bytes=_MIN_MODEL_BYTES,
        expected_sha256=MODEL_SHA256,
        progress_callback=progress_callback,
    )
    _download(
        TOKENIZER_URL,
        jina_code_tokenizer_path(),
        label="tokenizer",
        min_bytes=_MIN_TOKENIZER_BYTES,
        expected_sha256=None,
        progress_callback=progress_callback,
    )


def manual_instructions() -> str:
    return (
        "Automatic download didn't go through. To install the embedding "
        "model manually:\n\n"
        f"  1. Open this link in a browser:\n     {MODEL_URL}\n"
        f"     and save the file as:\n     {jina_code_model_path()}\n\n"
        f"  2. Open this link in a browser:\n     {TOKENIZER_URL}\n"
        f"     and save the file as:\n     {jina_code_tokenizer_path()}\n\n"
        "Then run this program again — it checks for both files before "
        "trying to download them, so it won't re-download anything you've "
        "already saved."
    )


def _download(
    url: str,
    dest: Path,
    *,
    label: str,
    min_bytes: int,
    expected_sha256: str | None,
    progress_callback: ProgressCallback | None,
) -> None:
    part_path = dest.with_suffix(dest.suffix + ".part")
    resume_from = part_path.stat().st_size if part_path.exists() else 0

    headers = {"User-Agent": "codeveto-context-engine"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(
            request, timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            resumed = response.status == 206
            content_length = response.headers.get("Content-Length")
            total = (
                int(content_length) + (resume_from if resumed else 0)
                if content_length is not None
                else None
            )
            mode = "ab" if resumed else "wb"
            downloaded = resume_from if resumed else 0
            with open(part_path, mode) as f:
                while True:
                    chunk = response.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback is not None:
                        progress_callback(
                            DownloadProgress(
                                label=label,
                                downloaded_bytes=downloaded,
                                total_bytes=total,
                            )
                        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Deliberately NOT deleting `part_path` here -- a partial download
        # is exactly what lets the *next* attempt resume instead of
        # starting a ~160MB file over from zero.
        raise ModelDownloadError(
            f"Could not download {label} from {url}: {exc}", manual_instructions()
        ) from exc

    final_size = part_path.stat().st_size
    if final_size < min_bytes:
        raise ModelDownloadError(
            f"Downloaded {label} looks incomplete ({final_size} bytes, "
            f"expected at least {min_bytes}). Try again, or install manually.",
            manual_instructions(),
        )

    if expected_sha256 is not None:
        actual = _sha256_of(part_path)
        if actual != expected_sha256:
            part_path.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"Downloaded {label} failed checksum verification "
                f"(expected {expected_sha256}, got {actual}). The partial "
                "file was removed so the next attempt starts fresh.",
                manual_instructions(),
            )

    part_path.replace(dest)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
