"""Dependency-free token-window overlap helpers."""

from __future__ import annotations


def add_chunk_overlap(
    chunks: list[str], *, overlap_tokens: int, max_tokens: int
) -> list[str]:
    """Add word-token overlap while preserving all input content."""
    if max_tokens <= 0 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be between 0 and max_tokens - 1")
    if overlap_tokens == 0:
        return chunks

    output: list[str] = []
    previous_content: list[str] = []
    for original in chunks:
        remaining = original.split()
        first_window = not output
        while remaining:
            prefix = [] if first_window else previous_content[-overlap_tokens:]
            capacity = max_tokens - len(prefix)
            content = remaining[:capacity]
            remaining = remaining[capacity:]
            output.append(" ".join(prefix + content))
            previous_content = content
            first_window = False
    return output
