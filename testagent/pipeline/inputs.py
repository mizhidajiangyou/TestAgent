"""Input parsing with a parser registry (plan-c B3.2/B4.5a, review P1-7).

``INPUT_PARSERS`` maps each input kind to a parser; ``parse_inputs`` only
dispatches + enforces ``require_any``. Adding a new input kind means
registering one parser class — zero pipeline changes (the v2 hardcoded
swagger/requirements branches inside the generic layer were the leak).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from testagent.parsers.requirement_parser import RequirementParser
from testagent.parsers.swagger_parser import SwaggerParser

if TYPE_CHECKING:
    from testagent.pipeline.manifest import Manifest

logger = logging.getLogger(__name__)


class InputParser(Protocol):
    """One input kind's parser: raw CLI string -> parsed products."""

    def parse(self, value: str, spec: Any) -> dict[str, Any]:
        """Return the context entries this input contributes."""
        ...


class SwaggerInputParser:
    def parse(self, value: str, spec: Any) -> dict[str, Any]:
        endpoints = SwaggerParser().parse(value)
        return {
            "endpoints": endpoints,
            "endpoints_text": SwaggerParser.endpoints_to_text(endpoints),
            "endpoints_signature": _endpoints_signature(endpoints),
        }


class RequirementInputParser:
    def parse(self, value: str, spec: Any) -> dict[str, Any]:
        reqs = RequirementParser().parse(value)
        return {
            "requirements": reqs,
            "requirements_text": RequirementParser.requirements_to_text(reqs),
        }


def _parse_file(parse: str, value: str) -> Any:
    text = Path(value).read_text(encoding="utf-8")
    if parse == "json":
        return json.loads(text)
    if parse == "testcase_history":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("testcase_history file %s is not JSON; treating as []", value)
            return []
        return data if isinstance(data, list) else []
    return text  # parse == "text" or unspecified


class FileInputParser:
    def parse(self, value: str, spec: Any) -> dict[str, Any]:
        return {(spec.context_var or spec.name): _parse_file(spec.parse, value)}


class PlainInputParser:
    def parse(self, value: str, spec: Any) -> dict[str, Any]:
        var = spec.context_var or spec.name
        if spec.kind == "int":
            return {var: int(value)}
        if spec.kind == "bool":
            return {var: value.strip().lower() in ("1", "true", "yes")}
        return {var: value}


#: The registry itself (plan-c B3.2). New kinds register here.
INPUT_PARSERS: dict[str, InputParser] = {
    "swagger": SwaggerInputParser(),
    "requirements": RequirementInputParser(),
    "file": FileInputParser(),
    "text": PlainInputParser(),
    "choice": PlainInputParser(),
    "int": PlainInputParser(),
    "bool": PlainInputParser(),
}


def _format_param(name: str, schema: Any, required: bool) -> str:
    """Format one parameter as ``name(type,req|opt[,enum:a|b])``.

    Byte-parity with ``prompt_builder._format_param`` (B5.1 gate: the perf
    review context renders the signature, so any drift is a fingerprint
    diff). Kept local — the arch gate forbids importing the legacy
    prompt_builder from the pipeline.
    """
    t = schema.get("type", "?") if isinstance(schema, dict) else "?"
    extra = ""
    if isinstance(schema, dict) and schema.get("enum"):
        extra = ",enum:" + "|".join(str(e) for e in schema["enum"])
    return f"{name}({t},{'req' if required else 'opt'}{extra})"


def _endpoints_signature(endpoints: list[Any]) -> str:
    """Compact endpoint signature (arch gate B4.11: the pipeline must not
    import the legacy prompt_builder, so this is a local implementation of
    the same name/type/required/enum format — byte-parity verified by the
    B5.1 fingerprint tests)."""
    lines: list[str] = []
    for ep in endpoints:
        parts: list[str] = []
        for p in getattr(ep, "parameters", None) or []:
            if not isinstance(p, dict):
                continue
            parts.append(
                _format_param(
                    str(p.get("name", "")), p.get("schema") or {}, bool(p.get("required", False))
                )
            )
        line = f"- {ep.method} {ep.path}"
        if parts:
            line += f" params:[{', '.join(sorted(parts))}]"
        body = getattr(ep, "request_body", None) or {}
        if isinstance(body, dict):
            schema = body.get("schema") or {}
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            if isinstance(props, dict) and props:
                req_set = set(schema.get("required", []))
                bparts = [_format_param(k, v or {}, k in req_set) for k, v in props.items()]
                line += f" body:[{', '.join(sorted(bparts))}]"
        lines.append(line)
    return "\n".join(lines)


@dataclass
class TaskContext:
    """Everything a pipeline run needs from its inputs."""

    raw: dict[str, Any] = field(default_factory=dict)  # CLI values as given
    parsed: dict[str, Any] = field(default_factory=dict)  # parsed products
    settings_views: dict[str, Any] = field(default_factory=dict)  # lang/mode/...


def _resolve_settings_value(settings: Any, key: str) -> Any:
    """Resolve a ``from_settings:KEY`` default against the Settings object."""
    # Dotted traversal: "llm.json_mode" -> settings.llm.json_mode.
    obj: Any = settings
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj


def parse_inputs(
    manifest: Manifest,
    raw: dict[str, Any],
    settings: Any = None,
) -> TaskContext:
    """Resolve every input via its kind (plan-c B4.5a) + enforce require_any.

    ``from_settings:KEY`` defaults are resolved when the CLI did not provide
    a value and a Settings object is supplied.
    """
    parsed: dict[str, Any] = {}
    effective: dict[str, Any] = {}
    for spec in manifest.inputs:
        value = raw.get(spec.name)
        if value in (None, ""):
            if (
                isinstance(spec.default, str)
                and spec.default.startswith("from_settings:")
                and settings is not None
            ):
                value = _resolve_settings_value(
                    settings, spec.default.removeprefix("from_settings:")
                )
            elif spec.default is not None:
                value = spec.default
        if value in (None, ""):
            continue
        effective[spec.name] = value
        parser = INPUT_PARSERS.get(spec.kind)
        if parser is None:
            raise ValueError(f"no input parser registered for kind {spec.kind!r}")
        parsed.update(parser.parse(str(value), spec))

    if manifest.require_any and not any(str(k) in effective for k in manifest.require_any):
        raise ValueError(f"task {manifest.name!r} requires at least one of: {manifest.require_any}")

    settings_views: dict[str, Any] = {}
    if settings is not None:
        settings_views["output_language"] = _resolve_settings_value(settings, "output_language")
        settings_views["json_mode"] = _resolve_settings_value(settings, "llm.json_mode")
        settings_views["max_concurrency"] = _resolve_settings_value(settings, "llm.max_concurrency")
    return TaskContext(raw=dict(effective), parsed=parsed, settings_views=settings_views)
