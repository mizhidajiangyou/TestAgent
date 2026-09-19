"""Split-mode requirement resolution (v4 §4.1).

``resolve_split_mode`` is the SINGLE entry for mode resolution; ``single``
builds exactly ONE RequirementItem (id=DOC-1, title=source stem,
description=full parsed text) via the injected DocumentParser — chapter
splitting is not invoked. The document body is preserved verbatim (no
trim/截断/auto-fallback). Pure-whitespace bodies are rejected before any
LLM call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from testagent.config.models import RequirementItem
from testagent.parsers.document_parser import DocumentParser
from testagent.parsers.requirement_parser import RequirementParser

__all__ = ["resolve_requirements", "resolve_split_mode"]

SplitMode = Literal["auto", "single"]


def resolve_split_mode(requested: str | None, settings_value: str) -> SplitMode:
    """Effective split mode: explicit CLI > Settings. Invalid values fail
    fast (v4 §3.1 — never silently fall back)."""
    effective = requested if requested is not None else settings_value
    if effective not in ("auto", "single"):
        raise ValueError(f"invalid split mode: {effective!r} (expected auto|single)")
    return effective  # type: ignore[return-value]


def resolve_requirements(
    requirements_path: str,
    mode: SplitMode,
    document_parser: DocumentParser | None = None,
) -> list[RequirementItem]:
    """Parse the requirement document in the resolved mode.

    ``auto`` delegates to the existing RequirementParser chapter path.
    ``single`` yields exactly one unit carrying the FULL parsed text
    (v4 §4.1 contract). Missing file / empty body fail before the LLM.
    """
    path = Path(requirements_path)
    if not path.exists():
        raise FileNotFoundError(f"requirements document not found: {requirements_path}")

    if mode == "auto":
        return RequirementParser().parse(str(path))

    parser = document_parser or DocumentParser()
    text = parser.parse(path)
    if not text or not text.strip():
        raise ValueError(f"document body is empty after parsing: {requirements_path}")
    item = RequirementItem(
        id="DOC-1",
        title=path.stem,
        description=text,
        module="",
        acceptance_criteria=[],
    )
    return [item]
