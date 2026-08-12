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

                endpoints.append(
                    APIEndpoint(
                        method=method.upper(),
                        path=path,
                        summary=operation.get("summary", ""),
                        description=operation.get("description", ""),
                        parameters=list(all_params.values()),
                        request_body=self._extract_request_body(operation),
                        responses=list(operation.get("responses", {}).keys()),
                        tags=operation.get("tags", []),
                    )
                )

        logger.info("Parsed %d endpoints from spec", len(endpoints))
        return endpoints

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
