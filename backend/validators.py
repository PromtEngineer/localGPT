"""
Input validation and Pydantic models for localGPT backend.
Provides consistent validation across all endpoints.
"""
from pydantic import BaseModel, Field, field_validator
from fastapi import UploadFile
import os
from typing import Optional, Dict, Any


# ============================================================================
# PYDANTIC REQUEST MODELS
# ============================================================================

class SessionRequest(BaseModel):
    """Request model for creating a new session."""
    title: str = Field(..., min_length=1, max_length=100, description="Session title")
    model: str = Field(default="llama3.2:latest", description="LLM model to use")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Project Analysis",
                "model": "qwen3:8b"
            }
        }


class MessageRequest(BaseModel):
    """Request model for chat messages."""
    message: str = Field(..., min_length=1, max_length=10000, description="Chat message")

    @field_validator('message')
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is in this document?"
            }
        }


class RenameSessionRequest(BaseModel):
    """Request model for renaming a session."""
    title: str = Field(..., min_length=1, max_length=100, description="New session title")

    @field_validator('title')
    @classmethod
    def sanitize_title(cls, v: str) -> str:
        return v.strip()


class IndexRequest(BaseModel):
    """Request model for creating an index."""
    name: str = Field(..., min_length=1, max_length=100, description="Index name")
    description: Optional[str] = Field(None, max_length=500, description="Index description")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Index metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "My Documents",
                "description": "Research papers and reports",
                "metadata": {}
            }
        }


class IndexBuildRequest(BaseModel):
    """Request model for building an index."""
    background: Optional[bool] = Field(False, description="Run build in background")
    forceReindex: Optional[bool] = Field(False, description="Force rebuild all files")
    enableEnrich: Optional[bool] = Field(True, description="Enable context enrichment")
    chunkSize: Optional[int] = Field(512, ge=128, le=2048, description="Chunk size in tokens")
    chunkOverlap: Optional[int] = Field(64, ge=0, le=256, description="Chunk overlap in tokens")
    retrievalMode: Optional[str] = Field("hybrid", pattern="^(hybrid|vector_only|bm25)$")
    windowSize: Optional[int] = Field(2, ge=0, le=10, description="Context window size")
    latechunk: Optional[bool] = Field(False, description="Enable late chunking")
    doclingChunk: Optional[bool] = Field(False, description="Enable Docling chunking")
    batchSizeEmbed: Optional[int] = Field(50, ge=1, le=200, description="Embedding batch size")
    batchSizeEnrich: Optional[int] = Field(25, ge=1, le=100, description="Enrichment batch size")
    embeddingModel: Optional[str] = Field(None, description="Embedding model name")
    enrichModel: Optional[str] = Field(None, description="Enrichment model name")
    overviewModel: Optional[str] = Field(None, description="Overview model name")
    checkServices: Optional[bool] = Field(True, description="Check service availability")


# ============================================================================
# FILE VALIDATION
# ============================================================================

class FileValidationResult(BaseModel):
    """Result of file validation."""
    valid: bool
    error: Optional[str] = None
    size: int = 0
    filename: str = ""
    mime_type: Optional[str] = None


# Allowed MIME types and extensions
ALLOWED_MIME_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/markdown": "md",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
}

ALLOWED_EXTENSIONS = {"pdf", "txt", "md", "doc", "docx", "xls", "xlsx", "ppt", "pptx"}

MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500MB


def validate_file_upload(file: UploadFile, max_size_bytes: int = MAX_UPLOAD_SIZE_BYTES) -> FileValidationResult:
    """
    Validate a file upload.

    Args:
        file: FastAPI UploadFile object
        max_size_bytes: Maximum file size in bytes (default 500MB)

    Returns:
        FileValidationResult with validation status and error details
    """
    filename = file.filename or ""
    content_type = file.content_type or ""

    # Check filename
    if not filename:
        return FileValidationResult(
            valid=False,
            error="Filename is required",
            filename=filename
        )

    # Check file extension
    file_ext = os.path.splitext(filename)[1].lstrip(".").lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return FileValidationResult(
            valid=False,
            error=f"File type '.{file_ext}' not allowed. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            filename=filename
        )

    # Check MIME type (advisory, not definitive)
    if content_type and content_type not in ALLOWED_MIME_TYPES:
        # Only warn if it's clearly a disallowed type
        if content_type.startswith("application/x-msdownload") or \
           content_type.startswith("application/x-executable") or \
           content_type.startswith("application/x-elf"):
            return FileValidationResult(
                valid=False,
                error=f"File MIME type '{content_type}' not allowed",
                filename=filename
            )

    # Check file size
    # Note: file.size might be None for streaming uploads, so we validate on actual write
    if file.size and file.size > max_size_bytes:
        size_mb = file.size / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)
        return FileValidationResult(
            valid=False,
            error=f"File size {size_mb:.1f}MB exceeds maximum {max_mb:.0f}MB",
            size=file.size,
            filename=filename,
            mime_type=content_type
        )

    # Validation passed
    return FileValidationResult(
        valid=True,
        filename=filename,
        mime_type=content_type,
        size=file.size or 0
    )


def validate_request_size(content_length: Optional[int], max_bytes: int = MAX_UPLOAD_SIZE_BYTES) -> Optional[str]:
    """
    Validate request size from Content-Length header.

    Args:
        content_length: Content-Length header value
        max_bytes: Maximum allowed size

    Returns:
        Error message if invalid, None if valid
    """
    if content_length is None:
        return None

    if content_length > max_bytes:
        max_mb = max_bytes / (1024 * 1024)
        actual_mb = content_length / (1024 * 1024)
        return f"Request size {actual_mb:.1f}MB exceeds maximum {max_mb:.0f}MB"

    return None
