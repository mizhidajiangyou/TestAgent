"""
Performance test report generator.

Generates Markdown reports with performance metrics and analysis.
"""

import logging
from datetime import datetime
from pathlib import Path

from testagent.config.models import PerfReportInput, ReportMetadata
from testagent.reports.base import BaseReport

logger = logging.getLogger(__name__)


class PerformanceReport(BaseReport[PerfReportInput]):
    """Generate performance test reports."""

    def __init__(self, metadata: ReportMetadata | None = None) -> None:
        self._metadata = metadata or ReportMetadata(title="Performance Test Report")

    def generate(self, data: PerfReportInput) -> str:
        """Generate performance test report.

        Args:
            data: Input payload with script path, config, and optional metrics.

        Returns:
            Report content as Markdown string.
        """
        script_path = data.script_path
        config = data.config
        metrics = data.metrics
        analysis = data.analysis

        lines: list[str] = []

        lines.append(f"# {self._metadata.title}")
        lines.append("")
        lines.append(f"**Author:** {self._metadata.author}")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Version:** {self._metadata.version}")
        lines.append("")

        # Test Configuration
        lines.append("## Test Configuration")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Base URL | `{config.base_url}` |")
        lines.append(f"| Virtual Users | {config.virtual_users} |")
        lines.append(f"| Duration | {config.duration_seconds}s |")
        lines.append(f"| Ramp-Up | {config.ramp_up_seconds}s |")
        lines.append(f"| Think Time | {config.think_time_ms}ms |")
        lines.append(f"| Auth Type | {config.auth_type} |")
        lines.append(f"| Script | `{script_path}` |")
        lines.append("")

        # Metrics section (if available)
        if metrics:
            lines.append("## Test Results")
            lines.append("")
            lines.append("### Key Performance Indicators")
            lines.append("")

            summary = metrics.get("summary", {})
            if summary:
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                lines.append(f"| Total Requests | {summary.get('total_requests', 'N/A')} |")
                lines.append(f"| Avg Response Time | {summary.get('avg_ms', 'N/A')}ms |")
                lines.append(f"| P95 Response Time | {summary.get('p95_ms', 'N/A')}ms |")
                lines.append(f"| P99 Response Time | {summary.get('p99_ms', 'N/A')}ms |")
                lines.append(f"| Throughput | {summary.get('throughput_rps', 'N/A')} req/s |")
                lines.append(f"| Error Rate | {summary.get('error_rate_pct', 'N/A')}% |")
                lines.append("")

            # Endpoint breakdown
            endpoints = metrics.get("endpoints", {})
            if endpoints:
                lines.append("### Endpoint Breakdown")
                lines.append("")
                lines.append("| Endpoint | Avg (ms) | P95 (ms) | P99 (ms) | Errors |")
                lines.append("|----------|----------|----------|----------|--------|")
                for ep_name, ep_data in endpoints.items():
                    lines.append(
                        f"| {ep_name} | {ep_data.get('avg_ms', 'N/A')} | "
                        f"{ep_data.get('p95_ms', 'N/A')} | {ep_data.get('p99_ms', 'N/A')} | "
                        f"{ep_data.get('error_count', 0)} |"
                    )
                lines.append("")

        # AI Analysis section (if available)
        if analysis:
            lines.append("## AI Analysis")
            lines.append("")
            verdict = analysis.get("verdict", "unknown")
            lines.append(f"**Verdict:** {verdict.upper()}")
            lines.append("")

            if analysis.get("headline"):
                lines.append(f"**Summary:** {analysis['headline']}")
                lines.append("")

            findings = analysis.get("findings", [])
            if findings:
                lines.append("### Findings")
                lines.append("")
                for finding in findings:
                    ftype = finding.get("type", "info")
                    severity = finding.get("severity", "medium")
                    title = finding.get("title", "")
                    desc = finding.get("description", "")
                    lines.append(f"- **[{ftype.upper()}] [{severity.upper()}]** {title}")
                    if desc:
                        lines.append(f"  {desc}")
                lines.append("")

            next_steps = analysis.get("next_steps", [])
            if next_steps:
                lines.append("### Recommended Next Steps")
                lines.append("")
                for i, step in enumerate(next_steps, 1):
                    lines.append(f"{i}. {step}")
                lines.append("")

        # Usage instructions
        lines.append("## How to Run")
        lines.append("")
        lines.append("### k6")
        lines.append("")
        lines.append("```bash")
        lines.append(f"k6 run {script_path}")
        lines.append("```")
        lines.append("")
        lines.append("### JMeter")
        lines.append("")
        lines.append("```bash")
        lines.append(f"jmeter -n -t {script_path} -l results.jtl")
        lines.append("```")
        lines.append("")

        return "\n".join(lines)

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
        logger.info("Saved performance report to %s", output_path)
        return output_path
