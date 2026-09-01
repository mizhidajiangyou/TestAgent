"""Artifact validation (plan-c B4.5c): jsonschema for structured items, a
validator chain for text scripts. Error strings are designed to feed the
targeted re-ask prompt — they carry the field path, not just "not JSON"."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

import jsonschema

if TYPE_CHECKING:
    from testagent.pipeline.inputs import TaskContext
    from testagent.pipeline.manifest import ValidatorSpec


def strip_fences(raw: str) -> str:
    """Generic fence stripping (generalized from GUITestGenerator)."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_+.-]*\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip()


def validate_structured(items: list[dict[str, Any]], schema: dict[str, Any]) -> list[str]:
    """Per-item jsonschema validation; one error string per bad item.

    The error strings feed the targeted re-ask prompt — this is the upgrade
    over the old pipeline, which only knew "not JSON".
    """
    validator = jsonschema.Draft202012Validator(schema)
    errors: list[str] = []
    for idx, item in enumerate(items):
        errs = sorted(validator.iter_errors(item), key=lambda e: list(e.path))
        if errs:
            detail = "; ".join(
                f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errs[:3]
            )
            errors.append(f"item[{idx}] invalid: {detail}")
    return errors


def validate_text(script: str, validators: list[ValidatorSpec], ctx: TaskContext) -> list[str]:
    """Run the text validator chain, honouring per-validator ``when`` guards."""
    errors: list[str] = []
    for v in validators:
        if v.when and not all(str(ctx.parsed.get(k)) == val for k, val in v.when.items()):
            continue
        if v.kind == "python_compile":
            try:
                compile(script, "<generated>", "exec")
            except SyntaxError as exc:
                errors.append(f"python_compile: {exc}")
        elif v.kind == "xml":
            try:
                root = ET.fromstring(script)
                if v.root and root.tag != v.root:
                    errors.append(f"xml: root tag {root.tag!r} != {v.root!r}")
            except ET.ParseError as exc:
                errors.append(f"xml: {exc}")
        elif v.kind == "regex" and not re.search(v.pattern, script):
            errors.append(f"regex: pattern {v.pattern!r} not found")
        elif v.kind == "contains" and v.pattern not in script:
            errors.append(f"contains: {v.pattern!r} not found")
    return errors
