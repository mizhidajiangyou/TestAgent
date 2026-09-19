"""Chunk assembly + text reassembly (v4 §5.2/§5.4).

Greedy packing in source order under BOTH the logical-entry cap and the
rendered character budget (reference text + title context count too). An
atomic (indivisible) block over budget is marked FAILED with
``oversized_atomic_item`` and zero calls; other chunks continue. Text
reassembly replays separators byte-for-byte and only replaces rewritten
chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from testagent.artifact.models import ArtifactChunk, JSONValue, LoadedArtifact

__all__ = ["ChunkPlan", "plan_chunks", "reassemble_text", "serialize_json"]

OVERSIZED_ATOMIC = "oversized_atomic_item"


@dataclass
class ChunkPlan:
    chunks: list[ArtifactChunk] = field(default_factory=list)
    oversized_atomic: list[int] = field(default_factory=list)  # source indices


def _entry_size(entry: object) -> int:
    import json

    return len(json.dumps(entry, ensure_ascii=False))


def plan_chunks(
    artifact: LoadedArtifact,
    *,
    chunk_size: int,
    char_budget: int,
    system_prompt_chars: int,
    reference_chars: int,
    title_context_chars: int = 0,
) -> ChunkPlan:
    """Greedy source-order packing (v4 §5.4). Both caps apply
    simultaneously; fixed prompt overhead is included in the budget."""
    plan = ChunkPlan()
    fixed = system_prompt_chars + reference_chars + title_context_chars
    if fixed > char_budget:
        # Reference alone busts the budget: caller treats as whole-document
        # FAILED before any call (v4 §5.4); empty plan signals that.
        return plan

    if artifact.format.value == "json":
        index = 0
        chunk_index = 0
        items = artifact.items
        while index < len(items):
            current: list[JSONValue] = []
            size = fixed
            while index < len(items):
                item = items[index]
                item_size = _entry_size(item) + 1  # comma/newline
                if current and size + item_size > char_budget:
                    break
                if len(current) >= chunk_size:
                    break
                if not current and item_size > char_budget - fixed:
                    # Atomic single item over budget: FAILED, no call.
                    plan.oversized_atomic.append(index)
                    index += 1
                    break
                current.append(item)
                size += item_size
                index += 1
            if current:
                plan.chunks.append(
                    ArtifactChunk(
                        index=chunk_index,
                        item_start=index - len(current),
                        item_end=index,
                        items=tuple(current),
                    )
                )
                chunk_index += 1
        return plan

    # Text formats: pack logical blocks.
    index = 0
    chunk_index = 0
    segments = artifact.text_segments
    while index < len(segments):
        current_text: list[str] = []
        size = fixed
        start = index
        while index < len(segments):
            segment = segments[index]
            seg_size = len(segment)
            if current_text and size + seg_size > char_budget:
                break
            if len(current_text) >= chunk_size:
                break
            if not current_text and seg_size > char_budget - fixed:
                plan.oversized_atomic.append(index)
                index += 1
                break
            current_text.append(segment)
            size += seg_size
            index += 1
        if current_text:
            first = current_text[0]
            char_start = sum(len(s) for s in segments[:start]) + sum(
                len(sep) for sep in artifact.text_separators[: start + 1]
            )
            plan.chunks.append(
                ArtifactChunk(
                    index=chunk_index,
                    char_start=char_start,
                    char_end=char_start + sum(len(t) for t in current_text),
                    source_text="\n\n".join(current_text),
                    title_context=first.splitlines()[0][:120] if first else "",
                )
            )
            chunk_index += 1
    return plan


def reassemble_text(
    artifact: LoadedArtifact,
    rewritten: dict[int, str],
) -> str:
    """Byte-faithful reassembly: separators replayed; only rewritten chunk
    indices swap their text (v4 §5.2)."""
    if artifact.format.value == "json":
        raise ValueError("reassemble_text is for text formats")
    segments = list(artifact.text_segments)
    for index, new_text in rewritten.items():
        if index < 0 or index >= len(segments):
            raise ValueError(f"chunk index out of range: {index}")
        segments[index] = new_text
    parts: list[str] = []
    for i, segment in enumerate(segments):
        parts.append(segment)
        parts.append(artifact.text_separators[i])
    return "".join(parts)


def serialize_json(
    artifact: LoadedArtifact,
    rewritten_items: dict[int, object],
) -> bytes:
    """JSON serialization: envelope metadata preserved; items replaced per
    source index. No rewrites replay the SOURCE bytes (v4 §5.2)."""
    if not rewritten_items:
        return artifact.raw_bytes
    items: list[object] = []
    for i, original in enumerate(artifact.items):
        items.append(rewritten_items.get(i, original))
    if artifact.json_root_is_envelope:
        envelope: dict[str, object] = dict(artifact.envelope_meta)
        envelope["test_cases"] = items
        return (json_text(envelope) + "\n").encode("utf-8")
    return (json_text(items) + "\n").encode("utf-8")


def json_text(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)
