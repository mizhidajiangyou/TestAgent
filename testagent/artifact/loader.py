"""Strict artifact loading (v4 §5.2).

Only UTF-8 (BOM tolerated) ``.json/.md/.markdown/.txt``. JSON: non-empty
object array, or an object with a single ``test_cases`` array (envelope);
repeated keys, non-object entries, non-finite numbers are rejected — never
silently filtered. Text: lossless segmentation into logical blocks with
source separators retained (no strip of lines, no blank-line removal).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from testagent.artifact.models import ArtifactFormat, JSONValue, LoadedArtifact, sha256_bytes

__all__ = ["ArtifactLoadError", "load_artifact"]

_BOM = b"\xef\xbb\xbf"


class ArtifactLoadError(Exception):
    """Structured load failure (message carries the reason)."""


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    obj: dict[str, object] = {}
    for key, value in pairs:
        if key in obj:
            raise ArtifactLoadError(f"duplicate object key: {key!r}")
        obj[key] = value
    return obj


def _check_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactLoadError(f"non-finite number in artifact: {value!r}")
    if isinstance(value, dict):
        for v in value.values():
            _check_finite(v)
    elif isinstance(value, list):
        for v in value:
            _check_finite(v)


def _load_json(raw: bytes, source_path: str) -> LoadedArtifact:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactLoadError(f"not valid UTF-8: {source_path}: {exc}") from None
    try:
        data = json.loads(text, object_pairs_hook=_reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ArtifactLoadError(f"invalid JSON: {source_path}: {exc}") from None
    _check_finite(data)

    envelope_meta: dict[str, JSONValue] = {}
    items: list[JSONValue]
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        keys = set(data)
        if "test_cases" in keys and isinstance(data["test_cases"], list):
            items = data["test_cases"]
            envelope_meta = {k: v for k, v in data.items() if k != "test_cases"}
        else:
            raise ArtifactLoadError(
                f"JSON root must be an array or a 'test_cases' envelope: {source_path}"
            )
    else:
        raise ArtifactLoadError(f"JSON root must be an array or object: {source_path}")

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ArtifactLoadError(f"item {index} is not an object: {source_path}")

    return LoadedArtifact(
        source_path=source_path,
        source_sha256=sha256_bytes(raw),
        raw_bytes=raw,
        format=ArtifactFormat.JSON,
        json_root_is_envelope=bool(envelope_meta) or (isinstance(data, dict)),
        envelope_meta=envelope_meta,
        items=items,
    )


_MD_BLOCK_SPLIT = re.compile(r"\n(\s*\n)")


def _segment_text(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Lossless segmentation: logical blocks + the separators BETWEEN them.

    Returns (segments, separators) where joining
    ``segments[i] + separators[i]`` reproduces the input byte-for-byte
    (each separator is what FOLLOWS its segment). Blank-line runs
    are separators; everything else (headings, paragraphs, lists, fenced
    code including their inner blank lines) stays inside one segment.
    """
    segments: list[str] = []
    separators: list[str] = []
    cursor = 0
    # Split on blank-line runs but keep fenced code blocks intact.
    fence_positions = [m.start() for m in re.finditer(r"^[ \t]*(```|~~~)", text, re.MULTILINE)]
    in_fence_ranges: list[tuple[int, int]] = []
    for i in range(0, len(fence_positions) - 1, 2):
        open_pos = fence_positions[i]
        close_match = re.search(r"^[ \t]*(```|~~~)[ \t]*$", text[open_pos + 3 :], re.MULTILINE)
        if close_match:
            in_fence_ranges.append((open_pos, open_pos + 3 + close_match.end()))
        else:
            # Unclosed fence: rest of the document is one block (structure
            # error surfaced at load per v4 §5.2).
            in_fence_ranges.append((open_pos, len(text)))

    def _inside_fence(pos: int) -> bool:
        return any(start <= pos < end for start, end in in_fence_ranges)

    pattern = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")
    last_end = 0
    while True:
        m = pattern.search(text, cursor)
        if not m or _inside_fence(m.start()):
            if m is None:
                break
            cursor = m.end()
            continue
        if m.start() < last_end:
            cursor = m.end()
            continue
        segments.append(text[last_end : m.start()])
        separators.append(text[m.start() : m.end()])
        last_end = m.end()
        cursor = m.end()
    if last_end < len(text):
        segments.append(text[last_end:])
        separators.append("")
    return tuple(segments), tuple(separators)


def _load_text(raw: bytes, source_path: str, fmt: ArtifactFormat) -> LoadedArtifact:
    bom = b""
    body = raw
    if raw.startswith(_BOM):
        bom = _BOM
        body = raw[len(_BOM) :]
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactLoadError(f"not valid UTF-8: {source_path}: {exc}") from None
    segments, separators = _segment_text(text)
    if any(
        text.count("```") % 2 == 1
        and idx == len(segments) - 1
        and "```" in seg
        and seg.count("```") % 2 == 1
        for idx, seg in enumerate(segments)
    ):
        # Unclosed fence overall: explicit structure error (no fake repair).
        raise ArtifactLoadError(f"unclosed code fence in source: {source_path}")
    return LoadedArtifact(
        source_path=source_path,
        source_sha256=sha256_bytes(raw),
        raw_bytes=raw,
        format=fmt,
        text_segments=segments,
        text_separators=separators,
        bom=bom,
    )


def load_artifact(path: str | Path) -> LoadedArtifact:
    """Load and validate a review input artifact (strict, no filtering)."""
    source = Path(path)
    if not source.exists():
        raise ArtifactLoadError(f"artifact not found: {source}")
    if source.is_dir():
        raise ArtifactLoadError(f"artifact path is a directory: {source}")
    suffix = source.suffix.lower()
    fmt_map = {
        ".json": ArtifactFormat.JSON,
        ".md": ArtifactFormat.MARKDOWN,
        ".markdown": ArtifactFormat.MARKDOWN,
        ".txt": ArtifactFormat.TEXT,
    }
    if suffix not in fmt_map:
        raise ArtifactLoadError(
            f"unsupported artifact extension {suffix!r} (supported: .json/.md/.markdown/.txt): {source}"
        )
    raw = source.read_bytes()
    fmt = fmt_map[suffix]
    if fmt is ArtifactFormat.JSON:
        return _load_json(raw, str(source))
    return _load_text(raw, str(source), fmt)
