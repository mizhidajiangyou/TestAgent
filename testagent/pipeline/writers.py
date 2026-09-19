"""Artifact writers (plan-c B4.5e): json / csv / markdown / text + meta.

This is a RUNNABLE implementation of the v2 plan's sketched writers (the
``_resolve_extension`` shown there referenced an undefined ``artifact_ctx``
— review #19's "compile-ready vs pseudocode" complaint).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from testagent.pipeline.manifest import Manifest

#: Columns for CSV export of structured test-case artifacts.
CSV_COLUMNS = [
    "id",
    "title",
    "description",
    "endpoint",
    "test_type",
    "priority",
    "preconditions",
    "steps",
    "expected_results",
    "tags",
    # T8/T10 quality contracts (fix-plan §3.2/§3.4)
    "scenario_operation",
    "scenario_scene",
    "scenario_variant",
    "equivalence_class",
    "covers_obligations",
    "binds",
    "executability",
]


def resolve_extension(text_spec: dict[str, Any], ctx: dict[str, Any]) -> str:
    """Resolve the output extension for a text artifact.

    ``text_spec`` shape (manifest.output.formats["text"]):
    ``{"extension": ".py"}`` — fixed;
    ``{"extension": {"format": {"k6": ".js", "jmeter": ".jmx"}}}`` — keyed by
    the parsed ``format`` input's value;
    ``{"extension": {"by_input": "my_input", "map": {...}}}`` — keyed by an
    arbitrary input.
    """
    ext = text_spec.get("extension", ".txt")
    if isinstance(ext, str):
        return ext
    if isinstance(ext, dict):
        keyed = ext.get("format")
        if isinstance(keyed, dict):
            fmt = str(ctx.get("format", ""))
            return str(keyed.get(fmt, ".txt"))
        by_input = ext.get("by_input")
        mapping = ext.get("map")
        if isinstance(by_input, str) and isinstance(mapping, dict):
            return str(mapping.get(str(ctx.get(by_input, "")), ".txt"))
    return ".txt"


def write_artifact(
    manifest: Manifest,
    artifact: list[dict[str, Any]] | str,
    output_path: Path,
    output_format: str,
    ctx: dict[str, Any] | None = None,
    review_meta: dict[str, Any] | None = None,
) -> Path:
    """Write one artifact in the requested format (plan-c B4.5e).

    ``ctx`` carries the parsed inputs (for extension mapping); ``review_meta``
    is serialized next to text artifacts as ``<stem>.meta.json``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = ctx or {}

    #: Package manifests name the structured format ``json_list`` (the artifact
    #: is a JSON array); the writer branch is the same thing.
    output_format = {"json_list": "json"}.get(output_format, output_format)

    if output_format == "json":
        output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    elif output_format == "csv":
        _write_csv(artifact if isinstance(artifact, list) else [], output_path)
    elif output_format == "markdown":
        _write_markdown(artifact if isinstance(artifact, list) else [], output_path, manifest)
    elif output_format == "text":
        text_spec = manifest.output.formats.get("text", {})
        ext = resolve_extension(text_spec, ctx)
        if ext and output_path.suffix != ext:
            output_path = output_path.with_suffix(ext)
        output_path.write_text(str(artifact), encoding="utf-8")
        if review_meta is not None:
            meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
            meta_path.write_text(
                json.dumps(review_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    else:
        raise ValueError(f"unsupported output format {output_format!r}")
    return output_path


#: Columns whose CSV form joins a list, and those serialized as JSON.
_CSV_JOINED = ("preconditions", "steps", "expected_results", "tags", "covers_obligations")
_CSV_JSON = ("binds", "executability")


def csv_row(item: dict[str, Any]) -> dict[str, str]:
    """Flatten one case dict into a CSV row.

    Single holder for the flattening rules and the column list: the legacy
    generator used to own an identical copy, and two holders of one row
    contract is how download and file output silently disagree.
    """
    row: dict[str, str] = {}
    for column in CSV_COLUMNS:
        value = item.get(column, "" if column not in _CSV_JSON else None)
        if column in _CSV_JOINED:
            row[column] = (
                "; ".join(str(v) for v in value) if isinstance(value, list) else str(value)
            )
        elif column in _CSV_JSON:
            row[column] = json.dumps(value or {}, ensure_ascii=False)
        else:
            row[column] = value if isinstance(value, str) else str(value)
    return row


def render_csv_text(items: list[dict[str, Any]]) -> str:
    """CSV document (header + rows, no BOM) for the web download path."""
    import io

    buffer = io.StringIO(newline="")
    # \n, not the csv module's \r\n: the web download used to come from a file
    # read back with universal newlines, so \n IS the HTTP contract here. The
    # file writer below keeps \r\n (legacy CLI artifact bytes).
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for item in items:
        writer.writerow(csv_row(item))
    return buffer.getvalue()


def _write_csv(items: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for item in items:
            writer.writerow(csv_row(item))


def _write_markdown(items: list[dict[str, Any]], path: Path, manifest: Manifest) -> None:
    lines = [f"# {manifest.display_name or manifest.name}", ""]
    for item in items:
        lines.append(f"## {item.get('id', '')} — {item.get('title', '')}")
        lines.append("")
        lines.append(f"- **Endpoint**: {item.get('endpoint', '')}")
        lines.append(f"- **Type**: {item.get('test_type', '')} / {item.get('priority', '')}")
        if item.get("description"):
            lines.append(f"- **Description**: {item['description']}")
        for key in ("preconditions", "steps", "expected_results"):
            values = item.get(key) or []
            if values:
                lines.append(f"- **{key.title()}**:")
                lines.extend(f"  {i}. {v}" for i, v in enumerate(values, 1))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
