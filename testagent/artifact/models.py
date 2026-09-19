"""Artifact data boundaries (v4 §5.1). Strict types, recursive JSONValue.

LoadedArtifact keeps the raw source bytes so "no change" replays can be
byte-identical. Chunks carry source spans (never shared mutable state).
PublicationResult separates semantic state from on-disk state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Union  # JSONValue needs the recursive alias form

__all__ = [
    "ArtifactChunk",
    "ArtifactFormat",
    "ChunkReviewResult",
    "DocumentReviewOutcome",
    "JSONValue",
    "LoadedArtifact",
    "PublicationResult",
    "sha256_bytes",
]

JSONValue = Union[
    str,
    int,
    float,
    bool,
    None,
    list["JSONValue"],
    dict[str, "JSONValue"],
]


class ArtifactFormat(StrEnum):
    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"


class ChunkStatus(StrEnum):
    REVIEWED = "REVIEWED"
    REVIEW_REJECTED = "REVIEW_REJECTED"
    REVIEW_FAILED = "REVIEW_FAILED"
    REVIEW_DISABLED = "REVIEW_DISABLED"


class DocumentStatus(StrEnum):
    REVIEWED = "REVIEWED"
    REVIEW_REJECTED = "REVIEW_REJECTED"
    REVIEW_FAILED = "REVIEW_FAILED"
    REVIEW_DISABLED = "REVIEW_DISABLED"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class LoadedArtifact:
    """Immutable load result; raw bytes retained for byte-exact no-op paths."""

    source_path: str
    source_sha256: str
    raw_bytes: bytes
    format: ArtifactFormat
    # JSON shape
    json_root_is_envelope: bool = False
    envelope_meta: dict[str, JSONValue] = field(default_factory=dict)
    items: list[JSONValue] = field(default_factory=list)  # JSON objects
    # text shape (MD/TXT)
    text_segments: tuple[str, ...] = ()  # logical blocks in source order
    text_separators: tuple[
        str, ...
    ] = ()  # inter-block source separators (len = len(segments)+1 interleaving)
    bom: bytes = b""  # UTF-8 BOM bytes if present


@dataclass(frozen=True)
class ArtifactChunk:
    """One review chunk: source span + read-only title context."""

    index: int
    # JSON chunks: half-open item index range; text chunks: char span.
    item_start: int = -1
    item_end: int = -1
    char_start: int = -1
    char_end: int = -1
    items: tuple[JSONValue, ...] = ()
    source_text: str = ""
    title_context: str = ""


@dataclass
class ChunkReviewResult:
    """Per-chunk outcome + call ledger (v4 §6.2)."""

    chunk_index: int
    status: ChunkStatus = ChunkStatus.REVIEW_FAILED
    original_count: int = 0
    final_count: int = 0
    parse_ok: int = 0
    rounds_executed: int = 0
    reason: str = ""
    adopted_items: list[JSONValue] = field(default_factory=list)
    calls: list[dict[str, JSONValue]] = field(default_factory=list)


@dataclass
class DocumentReviewOutcome:
    """Whole-document review result (no publication responsibility)."""

    status: DocumentStatus = DocumentStatus.REVIEW_FAILED
    artifact: LoadedArtifact | None = None
    source_unavailable: bool = False
    chunk_results: list[ChunkReviewResult] = field(default_factory=list)
    adopted_chunks: int = 0
    used_review: bool = False
    partial: bool = False
    candidate_reviewed_chunks: int = 0
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class PublicationResult:
    """Semantic vs on-disk state are independent fields (v4 §7.2)."""

    run_id: str
    target_path: str
    report_path: str
    artifact_committed: bool = False
    report_committed: bool = False
    backup_path: str | None = None
    artifact_sha256: str | None = None
    report_sha256: str | None = None
    error: str = ""
