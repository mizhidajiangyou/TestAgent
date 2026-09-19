"""GUI with-testcases import, selection and orchestration (v4 §8).

Strict read-only import (no lossy historical-loader conversion), priority
ranked selection with auditable budgets, and a synthetic GEN-CASES
requirement appended ONCE. The orchestration function calls the injected
generator's PUBLIC generate; it never touches the GUI model or generator
internals and never runs a testcase LLM call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from testagent.config.models import GUITestGenInput, RequirementItem

__all__ = [
    "CaseReference",
    "GuiReferenceError",
    "ReferenceSelection",
    "build_gui_input_with_cases",
    "load_case_references",
    "select_references",
]


class GuiReferenceError(Exception):
    """Strict import failure (v4 §8.1) — always fails BEFORE the LLM."""


@dataclass(frozen=True)
class CaseReference:
    """Read-only structured projection of one imported case (§8.1)."""

    id: str
    title: str
    priority: str
    steps: tuple[str, ...]
    expected_results: tuple[str, ...]
    preconditions: tuple[str, ...]
    endpoint: str
    description: str
    raw: dict[str, Any]


@dataclass
class ReferenceSelection:
    """Auditable selection ledger (§8.2)."""

    selected: list[CaseReference] = field(default_factory=list)
    omitted: list[tuple[str, str]] = field(default_factory=list)  # (id, reason)
    original_count: int = 0
    rendered_chars: int = 0

    @property
    def selected_ids(self) -> list[str]:
        return [c.id for c in self.selected]

    @property
    def selected_count(self) -> int:
        return len(self.selected)


def load_case_references(path: str | Path) -> list[CaseReference]:
    """Strict import: JSON array or test_cases envelope; every entry must be
    a complete, traceable object. Missing file / corruption / bad entries
    raise BEFORE any LLM call; nothing is silently skipped."""
    source = Path(path)
    if not source.exists():
        raise GuiReferenceError(f"--with-testcases file not found: {source}")
    try:
        raw = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise GuiReferenceError(f"not valid UTF-8: {source}: {exc}") from None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GuiReferenceError(f"invalid JSON in {source}: {exc}") from None

    if isinstance(data, dict) and isinstance(data.get("test_cases"), list):
        items = data["test_cases"]
    elif isinstance(data, list):
        items = data
    else:
        raise GuiReferenceError(f"root must be an array or test_cases envelope: {source}")

    if not items:
        raise GuiReferenceError(f"no cases in reference file: {source}")

    references: list[CaseReference] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise GuiReferenceError(f"entry {index} is not an object: {source}")
        case_id = item.get("id")
        title = item.get("title")
        if not isinstance(case_id, str) or not case_id.strip():
            raise GuiReferenceError(f"entry {index} missing non-empty 'id': {source}")
        if not isinstance(title, str) or not title.strip():
            raise GuiReferenceError(f"entry {index} missing non-empty 'title': {source}")
        if case_id in seen_ids:
            raise GuiReferenceError(f"duplicate case id {case_id!r} in {source}")
        seen_ids.add(case_id)
        priority = item.get("priority", "medium")
        if priority not in ("high", "medium", "low"):
            raise GuiReferenceError(
                f"entry {index} priority must be high|medium|low, got {priority!r}"
            )
        steps = item.get("steps", [])
        expected = item.get("expected_results", [])
        preconditions = item.get("preconditions", [])
        endpoint = item.get("endpoint", "")
        description = item.get("description", "")
        for label, value in (
            ("steps", steps),
            ("expected_results", expected),
            ("preconditions", preconditions),
        ):
            if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
                raise GuiReferenceError(f"entry {index} field {label!r} must be a list of strings")
        for label, value in (("endpoint", endpoint), ("description", description)):
            if value is not None and not isinstance(value, str):
                raise GuiReferenceError(f"entry {index} field {label!r} must be a string")
        references.append(
            CaseReference(
                id=case_id,
                title=title,
                priority=priority,
                steps=tuple(steps),
                expected_results=tuple(expected),
                preconditions=tuple(preconditions),
                endpoint=endpoint or "",
                description=description or "",
                raw=item,
            )
        )
    return references


_PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _render_reference_block(cases: list[CaseReference], source_name: str, sha: str) -> str:
    lines = [f"Source: {source_name} (sha256:{sha[:12]})", ""]
    for case in cases:
        lines.append(f"### {case.id} | {case.title} | priority={case.priority}")
        if case.description:
            lines.append(case.description)
        if case.preconditions:
            lines.append("Preconditions: " + "; ".join(case.preconditions))
        if case.endpoint:
            lines.append(f"Endpoint: {case.endpoint}")
        lines.append("Steps:")
        lines.extend(f"{i}. {step}" for i, step in enumerate(case.steps, 1))
        lines.append("Expected:")
        lines.extend(f"- {e}" for e in case.expected_results)
        if not case.steps:
            lines.append("(steps missing in source case - do not fabricate)")
        if not case.expected_results:
            lines.append("(assertions missing in source case - do not fabricate)")
        lines.append("")
    return "\n".join(lines)


def select_references(
    references: list[CaseReference],
    *,
    max_cases: int,
    max_chars: int,
    source_name: str,
    source_sha: str,
) -> ReferenceSelection:
    """Priority-ranked stable selection under both budgets (§8.2). A case
    too large to ever fit is skipped whole (oversized_case) and later,
    smaller candidates still get their chance."""
    ranked = sorted(
        enumerate(references), key=lambda pair: (_PRIORITY_RANK[pair[1].priority], pair[0])
    )
    selection = ReferenceSelection(original_count=len(references))
    chosen: list[tuple[int, CaseReference]] = []
    used_chars = _render_reference_block([], source_name, source_sha).__len__()
    for _, case in ranked:
        if len(chosen) >= max_cases:
            selection.omitted.append((case.id, "count_limit"))
            continue
        candidate = [*chosen, (_, case)]
        rendered = _render_reference_block([c for _, c in candidate], source_name, source_sha)
        if len(rendered) > max_chars:
            single = _render_reference_block([case], source_name, source_sha)
            if len(single) > max_chars:
                selection.omitted.append((case.id, "oversized_case"))
            else:
                selection.omitted.append((case.id, "char_budget"))
            continue
        chosen = candidate
        used_chars = len(rendered)
    selection.selected = [c for _, c in chosen]
    selection.rendered_chars = used_chars
    if not selection.selected:
        raise GuiReferenceError("no reference case fits the budgets; cannot build reference")
    return selection


def build_gui_input_with_cases(
    *,
    requirements: list[RequirementItem],
    reference_path: str | Path,
    url: str | None,
    endpoints: list[Any],
    output_language: str,
    generator: Any,
) -> tuple[GUITestGenInput, ReferenceSelection]:
    """Orchestration entry (§8.3): validated references -> GEN-CASES item
    appended once -> public generator.generate. Returns the selection
    ledger for structured logging/session metadata."""
    references = load_case_references(reference_path)
    source = Path(reference_path)
    source_sha = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    settings_max_cases = 30
    settings_max_chars = 16000
    selection = select_references(
        references,
        max_cases=settings_max_cases,
        max_chars=settings_max_chars,
        source_name=source.name,
        source_sha=source_sha,
    )

    existing_ids = {req.id for req in requirements}
    gen_id = "GEN-CASES"
    suffix = 1
    while gen_id in existing_ids:
        gen_id = f"GEN-CASES-{suffix}"
        suffix += 1

    body = _render_reference_block(selection.selected, source.name, source_sha)
    description = (
        "Imported test-case reference (derived from existing cases; may contain "
        "API-only scenarios — only behaviors executable with known page elements "
        "may become GUI steps; do not invent page selectors).\n" + body
    )
    gen_item = RequirementItem(
        id=gen_id,
        title="导入测试用例参考"
        if output_language == "chinese"
        else "Imported test-case reference",
        description=description,
        module="",
        acceptance_criteria=[],
    )
    gui_input = GUITestGenInput(
        requirements=[*requirements, gen_item],
        url=url or "",
        endpoints=list(endpoints),
        output_language=output_language,
    )
    _ = generator  # injected generator is invoked by the CLI layer (public API)
    return gui_input, selection
