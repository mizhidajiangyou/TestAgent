"""Tests for report generators (language support)."""

from testagent.config.models import (
    APIEndpoint,
    PerformanceConfig,
    PerfReportInput,
    TestCase,
    TestCaseReportInput,
    TestPriority,
    TestType,
)
from testagent.reports.performance_report import PerformanceReport
from testagent.reports.testcase_report import TestCaseReport


def _sample_test_case() -> TestCase:
    return TestCase(
        id="TC-001",
        title="Get users successfully",
        description="Verify GET /users returns 200",
        endpoint=APIEndpoint(method="GET", path="/users"),
        test_type=TestType.FUNCTIONAL,
        priority=TestPriority.HIGH,
        preconditions=["User is authenticated"],
        steps=["Send GET request to /users"],
        expected_results=["Status code is 200"],
        tags=["smoke", "regression"],
    )


class TestTestCaseReportLanguage:
    """Test case report language handling."""

    def test_markdown_english_by_default(self) -> None:
        """Test English headers by default."""
        report = TestCaseReport()
        content = report.generate(
            TestCaseReportInput(test_cases=[_sample_test_case()], output_format="markdown")
        )
        assert "## Summary" in content
        assert "Total Test Cases" in content
        assert "functional" in content

    def test_markdown_chinese(self) -> None:
        """Test Chinese headers and labels."""
        report = TestCaseReport()
        content = report.generate(
            TestCaseReportInput(
                test_cases=[_sample_test_case()],
                output_format="markdown",
                output_language="chinese",
            )
        )
        assert "## 汇总" in content
        assert "测试用例总数" in content
        assert "功能" in content  # functional -> 功能
        assert "高" in content  # high priority -> 高

    def test_json_summary_keyword_not_localized(self) -> None:
        """Test JSON report keeps machine-readable keys."""
        report = TestCaseReport()
        content = report.generate(
            TestCaseReportInput(
                test_cases=[_sample_test_case()],
                output_format="json",
                output_language="chinese",
            )
        )
        assert '"test_type": "functional"' in content


class TestPerformanceReportLanguage:
    """Performance report language handling."""

    def test_markdown_english_by_default(self) -> None:
        """Test English headers by default."""
        report = PerformanceReport()
        content = report.generate(
            PerfReportInput(
                script_path="perf_test.js",
                config=PerformanceConfig(base_url="https://api.example.com"),
            )
        )
        assert "## Test Configuration" in content
        assert "## How to Run" in content

    def test_markdown_chinese(self) -> None:
        """Test Chinese headers."""
        report = PerformanceReport()
        content = report.generate(
            PerfReportInput(
                script_path="perf_test.js",
                config=PerformanceConfig(base_url="https://api.example.com"),
                output_language="chinese",
            )
        )
        assert "## 测试配置" in content
        assert "## 如何运行" in content
