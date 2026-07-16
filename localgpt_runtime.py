"""Shared runtime configuration and boundary validation for LocalGPT services."""

from __future__ import annotations

import os
import hmac
from pathlib import Path, PurePath
from typing import Any, Mapping

try:
    from dotenv import load_dotenv
except ImportError:  # Keep boundary helpers usable in minimal test environments.
    load_dotenv = None

if load_dotenv:
    load_dotenv()


class UploadRejected(ValueError):
    """Raised when an uploaded filename cannot be stored safely."""


SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}


def env_path(name: str, default: Path | str) -> Path:
    """Return an absolute, expanded path from an environment setting."""
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def safe_upload_path(upload_dir: Path | str, submitted_name: str) -> Path:
    """Resolve a client filename beneath *upload_dir* or reject it.

    Browsers normally submit a basename, so path components are never useful
    here. Rejecting them (instead of silently stripping them) also gives API
    clients a clear signal that a traversal attempt was not accepted.
    """
    if not submitted_name or submitted_name in {".", ".."}:
        raise UploadRejected("An upload filename is required")

    normalized = submitted_name.replace("\\", "/")
    if "/" in normalized or PurePath(normalized).name != normalized:
        raise UploadRejected("Upload filenames must not contain path components")

    root = Path(upload_dir).expanduser().resolve()
    destination = (root / normalized).resolve()
    if destination.parent != root:
        raise UploadRejected("Upload path escapes the configured upload directory")
    if destination.suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise UploadRejected(
            f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))}"
        )
    return destination


def validate_index_file_paths(file_paths: list[str], upload_dir: Path | str) -> list[str]:
    """Validate internal indexing paths against the shared upload boundary."""
    root = Path(upload_dir).expanduser().resolve()
    validated: list[str] = []
    for candidate in file_paths:
        path = Path(candidate).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise UploadRejected("Indexing paths must be inside the upload directory") from exc
        if not path.is_file():
            raise UploadRejected(f"Uploaded file does not exist: {path.name}")
        if path.suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise UploadRejected(f"Unsupported file type: {path.suffix}")
        validated.append(str(path))
    return validated


def store_upload(stream, destination: Path | str, max_bytes: int | None = None) -> int:
    """Copy an upload in bounded chunks and remove partial files on failure."""
    limit = max_bytes or int(os.environ.get("LOCALGPT_MAX_UPLOAD_BYTES", 50 * 1024 * 1024))
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with path.open("xb") as output:
            while True:
                chunk = stream.read(min(1024 * 1024, limit - written + 1))
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise UploadRejected(f"Upload exceeds the {limit}-byte size limit")
                output.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return written


def normalize_index_options(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the documented indexing API while accepting legacy camelCase."""
    aliases = {
        "embedding_model": ("embedding_model", "embeddingModel"),
        "enrich_model": ("enrich_model", "enrichModel"),
        "overview_model": ("overview_model", "overviewModel", "overview_model_name"),
        "enable_latechunk": ("enable_latechunk", "latechunk"),
        "enable_docling_chunk": ("enable_docling_chunk", "doclingChunk"),
        "chunk_size": ("chunk_size", "chunkSize"),
        "chunk_overlap": ("chunk_overlap", "chunkOverlap"),
        "retrieval_mode": ("retrieval_mode", "retrievalMode"),
        "window_size": ("window_size", "windowSize"),
        "enable_enrich": ("enable_enrich", "enableEnrich"),
        "batch_size_embed": ("batch_size_embed", "batchSizeEmbed"),
        "batch_size_enrich": ("batch_size_enrich", "batchSizeEnrich"),
    }
    normalized = dict(payload)
    for canonical, candidates in aliases.items():
        value = next(
            (payload[key] for key in candidates if payload.get(key) is not None and payload.get(key) != ""),
            None,
        )
        if value is not None:
            normalized[canonical] = value

    normalized.setdefault("enable_docling_chunk", False)
    normalized.setdefault("enable_latechunk", False)
    normalized.setdefault("enable_enrich", True)
    normalized.setdefault("retrieval_mode", "hybrid")
    normalized.setdefault("chunk_size", 512)
    normalized.setdefault("chunk_overlap", 64)
    normalized.setdefault("window_size", 2)
    normalized.setdefault("batch_size_embed", 50)
    normalized.setdefault("batch_size_enrich", 25)

    mode_aliases = {"vector": "dense", "vector_only": "dense", "fts": "lexical", "bm25": "lexical"}
    mode = mode_aliases.get(str(normalized["retrieval_mode"]).lower(), str(normalized["retrieval_mode"]).lower())
    if mode not in {"hybrid", "dense", "lexical"}:
        raise ValueError("retrieval_mode must be hybrid, dense, or lexical")
    normalized["retrieval_mode"] = mode

    normalized["chunk_size"] = int(normalized["chunk_size"])
    normalized["chunk_overlap"] = int(normalized["chunk_overlap"])
    if normalized["chunk_size"] <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= normalized["chunk_overlap"] < normalized["chunk_size"]:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")
    return normalized


def cors_origin(request_origin: str | None) -> str | None:
    """Return an allowed CORS origin, or None when the origin is denied."""
    configured = os.environ.get(
        "LOCALGPT_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    )
    allowed = {origin.strip() for origin in configured.split(",") if origin.strip()}
    if not request_origin:
        return None
    return request_origin if request_origin in allowed else None


def request_is_authorized(authorization: str | None) -> bool:
    """Validate optional bearer authentication using constant-time comparison."""
    expected = os.environ.get("LOCALGPT_API_TOKEN")
    if not expected:
        return True
    if not authorization or not authorization.startswith("Bearer "):
        return False
    return hmac.compare_digest(authorization.removeprefix("Bearer ").strip(), expected)


def trust_remote_code_enabled() -> bool:
    """Require an explicit opt-in before executing model-repository Python code."""
    return os.environ.get("LOCALGPT_TRUST_REMOTE_CODE", "false").lower() in {
        "1",
        "true",
        "yes",
    }
