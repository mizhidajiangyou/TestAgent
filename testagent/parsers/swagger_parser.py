"""
Swagger / OpenAPI spec parser.

Supports URL, file path, or pre-loaded dict.
Parses endpoints into standardized ``APIEndpoint`` objects.

Adapted from PerfAI-JMeter-Script-Generator-Analyser swagger_parser.py.
"""

import json
import logging
from pathlib import Path
from typing import Any, cast

import requests
import yaml

from testagent.config.models import APIEndpoint
from testagent.parsers.base import BaseParser, ParseSource

logger = logging.getLogger(__name__)


class SwaggerParser(BaseParser):
    """Parse Swagger/OpenAPI specs into endpoint list."""

    def parse(self, source: ParseSource) -> list[APIEndpoint]:
        """Parse a Swagger/OpenAPI spec.

        Args:
            source: URL, file path, or dict of the spec.

        Returns:
            List of ``APIEndpoint`` objects.
        """
        spec = self._load_spec(source)
        return self._extract_endpoints(spec)

    def _load_spec(self, source: ParseSource) -> dict[str, Any]:
        """Load spec from dict, URL, or file path."""
        if isinstance(source, dict):
            return source

        if source.startswith("http://") or source.startswith("https://"):
            logger.info("Fetching Swagger spec from URL: %s", source)
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "yaml" in content_type or source.endswith((".yaml", ".yml")):
                return cast("dict[str, Any]", yaml.safe_load(response.text))
            return cast("dict[str, Any]", response.json())

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Swagger spec not found: {source}")

        logger.info("Loading Swagger spec from file: %s", source)
        with open(path, encoding="utf-8") as f:
            if path.suffix in (".yaml", ".yml"):
                return cast("dict[str, Any]", yaml.safe_load(f))
            return cast("dict[str, Any]", json.load(f))

    def _extract_endpoints(self, spec: dict[str, Any]) -> list[APIEndpoint]:
        """Extract endpoints from parsed spec."""
        endpoints: list[APIEndpoint] = []
        paths = spec.get("paths", {})
        http_methods = {"get", "post", "put", "patch", "delete", "head", "options"}
        # T3 (fix-plan RC-4): response-schema extraction is an OpenAPI 3.x
        # capability. Swagger 2.0 degrades EXPLICITLY to empty (its response
        # shapes live under ``definitions`` with a different contract) —
        # honest blindness instead of guessed envelope shapes.
        response_schemas_enabled = self._is_openapi3(spec)

        for path, path_item in paths.items():
            path_params = path_item.get("parameters", [])

            for method, operation in path_item.items():
                if method.lower() not in http_methods:
                    continue
                if not isinstance(operation, dict):
                    continue

                op_params = operation.get("parameters", [])
                all_params: dict[str, Any] = {p.get("name"): p for p in path_params}
                all_params.update({p.get("name"): p for p in op_params})

                request_body = self._extract_request_body(operation)
                if request_body is not None:
                    request_body = self._resolve_schema_refs(spec, request_body)

                endpoints.append(
                    APIEndpoint(
                        method=method.upper(),
                        path=path,
                        summary=operation.get("summary", ""),
                        description=operation.get("description", ""),
                        parameters=list(all_params.values()),
                        request_body=request_body,
                        responses=list(operation.get("responses", {}).keys()),
                        tags=operation.get("tags", []),
                        response_schemas=(
                            self._extract_response_schemas(spec, operation)
                            if response_schemas_enabled
                            else {}
                        ),
                    )
                )

        logger.info("Parsed %d endpoints from spec", len(endpoints))
        return endpoints

    @staticmethod
    def _is_openapi3(spec: dict[str, Any]) -> bool:
        """True for OpenAPI 3.x specs; Swagger 2.0 (``swagger: "2.0"``) is not."""
        version = str(spec.get("openapi", ""))
        return version.startswith("3")

    def _extract_response_schemas(
        self, spec: dict[str, Any], operation: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract ``$ref``-resolved response schemas keyed by status code.

        Only codes that actually document a schema get an entry; codes with
        description-only responses (e.g. ``204``) stay absent so renderers can
        distinguish "documented empty" from "documented schema".
        """
        schemas: dict[str, Any] = {}
        responses = operation.get("responses", {})
        if not isinstance(responses, dict):
            return schemas
        for status, resp in responses.items():
            if not isinstance(resp, dict):
                continue
            content = resp.get("content", {})
            if not isinstance(content, dict):
                continue
            for _media_type, media_obj in content.items():
                if not isinstance(media_obj, dict) or "schema" not in media_obj:
                    continue
                schemas[str(status)] = self._resolve_schema_refs(spec, media_obj["schema"])
                break
        return schemas

    def _resolve_schema_refs(self, spec: dict[str, Any], schema: Any) -> Any:
        """Recursively resolve ``$ref`` pointers against the spec (T3).

        Follows ``#/components/schemas/...`` (OpenAPI 3.x) and
        ``#/definitions/...`` (Swagger 2.0) style pointers; nested refs inside
        a resolved schema are resolved as well, with a visiting-set cycle
        guard. Unresolvable refs are left in place (honest, visible) rather
        than silently dropped.
        """
        visited: set[str] = set()
        return self._resolve_node(spec, schema, visited)

    def _resolve_node(self, spec: dict[str, Any], node: Any, visited: set[str]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                if ref in visited:
                    return node  # cycle: keep the pointer, do not recurse
                resolved = self._lookup_ref(spec, ref)
                if resolved is None:
                    return node  # unresolvable: keep the pointer visible
                visited = visited | {ref}
                return self._resolve_node(spec, resolved, visited)
            return {k: self._resolve_node(spec, v, visited) for k, v in node.items()}
        if isinstance(node, list):
            return [self._resolve_node(spec, item, visited) for item in node]
        return node

    @staticmethod
    def _lookup_ref(spec: dict[str, Any], ref: str) -> Any:
        """Follow a JSON pointer like ``#/components/schemas/User``."""
        if not ref.startswith("#/"):
            return None
        current: Any = spec
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    def _extract_request_body(self, operation: dict[str, Any]) -> dict[str, Any] | None:
        """Extract request body schema from operation."""
        body = operation.get("requestBody", {})
        if not body:
            return None
        content = body.get("content", {})
        for media_type, media_obj in content.items():
            schema = media_obj.get("schema", {})
            return {"media_type": media_type, "schema": schema}
        return None

    @staticmethod
    def endpoints_to_text(endpoints: list[APIEndpoint]) -> str:
        """Convert endpoints to a concise text summary for LLM prompts.

        Args:
            endpoints: List of parsed endpoints.

        Returns:
            Plain text representation.
        """
        lines: list[str] = []
        for ep in endpoints:
            line = f"{ep.method} {ep.path}"
            if ep.summary:
                line += f" - {ep.summary}"
            if ep.parameters:
                param_names = [p.get("name", "") for p in ep.parameters]
                line += f" (params: {', '.join(param_names)})"
            if ep.request_body:
                line += f" [body: {ep.request_body['media_type']}]"
            lines.append(line)
        return "\n".join(lines)
