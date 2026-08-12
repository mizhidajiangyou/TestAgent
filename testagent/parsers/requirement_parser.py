"""
Requirement document parser.

Supports plain text (.txt), Markdown (.md), and structured JSON (.json) formats.
Extracts requirement items with ID, title, description, and acceptance criteria.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, cast

from testagent.config.models import RequirementItem, TestPriority
from testagent.parsers.base import BaseParser, ParseSource

logger = logging.getLogger(__name__)


class RequirementParser(BaseParser):
    """Parse requirement documents into structured items."""

    def parse(self, source: ParseSource) -> list[RequirementItem]:
        """Parse a requirement document.

        Args:
            source: File path or dict of requirements.

        Returns:
            List of ``RequirementItem`` objects.
        """
        if isinstance(source, dict):
            return self._parse_dict(source)

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Requirement document not found: {source}")

        logger.info("Parsing requirement document: %s", source)
        suffix = path.suffix.lower()

        if suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data = cast("dict[str, Any]", json.load(f))
            return self._parse_dict(data)
        elif suffix in (".md", ".markdown"):
            with open(path, encoding="utf-8") as f:
                return self._parse_markdown(f.read())
        else:
            with open(path, encoding="utf-8") as f:
                return self._parse_text(f.read())

    def _parse_dict(self, data: dict[str, Any]) -> list[RequirementItem]:
        """Parse from a structured dict."""
        items: list[RequirementItem] = []
        requirements = data.get("requirements", data.get("items", []))

        for idx, req in enumerate(requirements, start=1):
            priority_str = str(req.get("priority", "medium")).lower()
            try:
                priority = TestPriority(priority_str)
            except ValueError:
                priority = TestPriority.MEDIUM

            items.append(
                RequirementItem(
                    id=req.get("id", f"REQ-{idx:03d}"),
                    title=req.get("title", req.get("name", f"Requirement {idx}")),
                    description=req.get("description", ""),
                    module=req.get("module", ""),
                    priority=priority,
                    acceptance_criteria=req.get("acceptance_criteria", []),
                )
            )

        logger.info("Parsed %d requirements from dict", len(items))
        return items

    def _parse_markdown(self, text: str) -> list[RequirementItem]:
        """Parse Markdown requirement document."""
        items: list[RequirementItem] = []
        sections = re.split(r"^#{1,3}\s+", text, flags=re.MULTILINE)

        for idx, section in enumerate(sections[1:], start=1):
            lines = section.strip().split("\n")
            title = lines[0].strip() if lines else f"Requirement {idx}"
            body = "\n".join(lines[1:]).strip()

            acceptance_criteria: list[str] = []
            ac_match = re.search(
                r"(?:验收标准|Acceptance Criteria|AC)[:\s]*\n((?:\s*[-*]\s+.+\n?)+)",
                body,
                re.IGNORECASE,
            )
            if ac_match:
                acceptance_criteria = [
                    line.strip().lstrip("-* ").strip()
                    for line in ac_match.group(1).strip().split("\n")
                    if line.strip()
                ]

            description = body
            if ac_match:
                description = body[: ac_match.start()].strip()

            items.append(
                RequirementItem(
                    id=f"REQ-{idx:03d}",
                    title=title,
                    description=description,
                    acceptance_criteria=acceptance_criteria,
                )
            )

        logger.info("Parsed %d requirements from markdown", len(items))
        return items

    def _parse_text(self, text: str) -> list[RequirementItem]:
        """Parse plain text requirement document."""
        items: list[RequirementItem] = []
        blocks = re.split(r"\n\s*\n", text.strip())

        for idx, block in enumerate(blocks, start=1):
            block = block.strip()
            if not block:
                continue

            lines = block.split("\n")
            title = lines[0].strip().lstrip("#-•* ").strip()
            description = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

            items.append(
                RequirementItem(
                    id=f"REQ-{idx:03d}",
                    title=title,
                    description=description or title,
                )
            )

        logger.info("Parsed %d requirements from text", len(items))
        return items

    @staticmethod
    def requirements_to_text(requirements: list[RequirementItem]) -> str:
        """Convert requirements to text summary for LLM prompts.

        Args:
            requirements: List of parsed requirements.

        Returns:
            Plain text representation.
        """
        lines: list[str] = []
        for req in requirements:
            line = f"[{req.id}] {req.title}"
            if req.module:
                line += f" (module: {req.module})"
            line += f"\n  {req.description}"
            if req.acceptance_criteria:
                line += "\n  Acceptance Criteria:"
                for ac in req.acceptance_criteria:
                    line += f"\n    - {ac}"
            lines.append(line)
        return "\n\n".join(lines)
