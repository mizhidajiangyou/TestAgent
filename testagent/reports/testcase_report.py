"""
Test case report generator.

Generates Markdown and JSON reports from test cases.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from testagent.config.models import ReportMetadata, TestCase, TestCaseReportInput
from testagent.reports.base import BaseReport
from testagent.reports.i18n import priority_label, report_label, type_label

logger = logging.getLogger(__name__)


class TestCaseReport(BaseReport[TestCaseReportInput]):
    """Generate test case reports in Markdown or JSON format."""

    def __init__(self, metadata: ReportMetadata | None = None) -> None:
        self._metadata = metadata or ReportMetadata(title="Test Case Report")

    def generate(self, data: TestCaseReportInput) -> str:
        """Generate test case report.

        Args:
            data: Input payload with test cases and output format.

        Returns:
            Report content as string.
        """
        if data.output_format == "json":
            return self._generate_json(data.test_cases, data.output_language)
        return self._generate_markdown(data.test_cases, data.output_language)

    def save(self, content: str, output_path: Path) -> Path:
        """Save report to file.

        Args:
            content: Report content.
            output_path: Target file path.

        Returns:
            Saved file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Saved test case report to %s", output_path)
        return output_path

    def _generate_markdown(self, test_cases: list[TestCase], output_language: str) -> str:
        """Generate Markdown report.

        Args:
            test_cases: List of test cases.
            output_language: "english" or "chinese".

        Returns:
            Markdown report content.
        """
        lines: list[str] = []

        lines.append(f"# {self._metadata.title}")
        lines.append("")
        lines.append(f"**{report_label('author', output_language)}:** {self._metadata.author}")
        lines.append(
            f"**{report_label('date', output_language)}:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append(f"**{report_label('version', output_language)}:** {self._metadata.version}")
        lines.append(f"**{report_label('total_test_cases', output_language)}:** {len(test_cases)}")
        lines.append("")

        # Summary by type
        type_counts: dict[str, int] = {}
        priority_counts: dict[str, int] = {}
        for tc in test_cases:
            type_counts[tc.test_type.value] = type_counts.get(tc.test_type.value, 0) + 1
            priority_counts[tc.priority.value] = priority_counts.get(tc.priority.value, 0) + 1

        lines.append(f"## {report_label('summary', output_language)}")
        lines.append("")
        lines.append(f"### {report_label('by_type', output_language)}")
        lines.append("")
        lines.append(
            f"| {report_label('type', output_language)} | {report_label('count', output_language)} |"
        )
        lines.append("|------|-------|")
        for t, c in sorted(type_counts.items()):
            lines.append(f"| {type_label(t, output_language)} | {c} |")
        lines.append("")

        lines.append(f"### {report_label('by_priority', output_language)}")
        lines.append("")
        lines.append(
            f"| {report_label('priority', output_language)} | {report_label('count', output_language)} |"
        )
        lines.append("|----------|-------|")
        for p, c in sorted(priority_counts.items()):
            lines.append(f"| {priority_label(p, output_language)} | {c} |")
        lines.append("")

        # Detailed test cases
        lines.append(f"## {report_label('test_cases', output_language)}")
        lines.append("")

        for tc in test_cases:
            lines.append(f"### {tc.id}: {tc.title}")
            lines.append("")
            lines.append(
                f"- **{report_label('endpoint', output_language)}:** `{tc.endpoint.full_path}`"
            )
            lines.append(
                f"- **{report_label('type', output_language)}:** {type_label(tc.test_type.value, output_language)}"
            )
            lines.append(
                f"- **{report_label('priority', output_language)}:** {priority_label(tc.priority.value, output_language)}"
            )
            lines.append(f"- **{report_label('description', output_language)}:** {tc.description}")
            lines.append("")

            if tc.preconditions:
                lines.append(f"**{report_label('preconditions', output_language)}:**")
                for step in tc.preconditions:
                    lines.append(f"1. {step}")
                lines.append("")

            if tc.steps:
                lines.append(f"**{report_label('steps', output_language)}:**")
                for i, step in enumerate(tc.steps, 1):
                    lines.append(f"{i}. {step}")
                lines.append("")

            if tc.expected_results:
                lines.append(f"**{report_label('expected_results', output_language)}:**")
                for result in tc.expected_results:
                    lines.append(f"- {result}")
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _generate_json(self, test_cases: list[TestCase], output_language: str) -> str:
        """Generate JSON report.

        Args:
            test_cases: List of test cases.
            output_language: "english" or "chinese" (affects summary labels).

        Returns:
            JSON report content.
        """
        by_type: dict[str, int] = {}
        by_priority: dict[str, int] = {}
        cases: list[dict[str, Any]] = []

        for tc in test_cases:
            by_type[tc.test_type.value] = by_type.get(tc.test_type.value, 0) + 1
            by_priority[tc.priority.value] = by_priority.get(tc.priority.value, 0) + 1
            cases.append(
                {
                    "id": tc.id,
                    "title": tc.title,
                    "description": tc.description,
                    "endpoint": tc.endpoint.full_path,
                    "test_type": tc.test_type.value,
                    "priority": tc.priority.value,
                    "preconditions": tc.preconditions,
                    "steps": tc.steps,
                    "expected_results": tc.expected_results,
                    "tags": tc.tags,
                }
            )

        data: dict[str, Any] = {
            "metadata": {
                "title": self._metadata.title,
                "author": self._metadata.author,
                "created_at": datetime.now().isoformat(),
                "version": self._metadata.version,
                "total_cases": len(test_cases),
            },
            "summary": {
                "by_type": by_type,
                "by_priority": by_priority,
            },
            "test_cases": cases,
        }

        return json.dumps(data, ensure_ascii=False, indent=2)
