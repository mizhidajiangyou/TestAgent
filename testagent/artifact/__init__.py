"""Artifact layer (plan-modular-capabilities-v4 §5): loading, chunking,
serialization. Depends on nothing above data + stdlib (v4 §1.2)."""

from testagent.artifact.chunking import (
    OVERSIZED_ATOMIC,
    ChunkPlan,
    plan_chunks,
    reassemble_text,
    serialize_json,
)
from testagent.artifact.loader import ArtifactLoadError, load_artifact
from testagent.artifact.models import (
    ArtifactChunk,
    ArtifactFormat,
    ChunkReviewResult,
    ChunkStatus,
    DocumentReviewOutcome,
    DocumentStatus,
    LoadedArtifact,
    PublicationResult,
)

__all__ = [
    "OVERSIZED_ATOMIC",
    "ArtifactChunk",
    "ArtifactFormat",
    "ArtifactLoadError",
    "ChunkPlan",
    "ChunkReviewResult",
    "ChunkStatus",
    "DocumentReviewOutcome",
    "DocumentStatus",
    "LoadedArtifact",
    "PublicationResult",
    "load_artifact",
    "plan_chunks",
    "reassemble_text",
    "serialize_json",
]
