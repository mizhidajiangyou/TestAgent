"""
Data models for parsed API endpoints, test cases, and reports.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class TestPriority(StrEnum):
    """Test case priority levels."""

    __test__ = False

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TestType(StrEnum):
    """Test case types."""

    __test__ = False

    FUNCTIONAL = "functional"
    BOUNDARY = "boundary"
    NEGATIVE = "negative"
    PERFORMANCE = "performance"
    SECURITY = "security"
    INTEGRATION = "integration"


@dataclass
class APIEndpoint:
    """Parsed API endpoint representation."""

    method: str
    path: str
    summary: str = ""
    description: str = ""
    parameters: list[dict[str, Any]] = field(default_factory=list)
    request_body: dict[str, Any] | None = None
    responses: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # --- quality: T3 (fix-plan RC-4; plan-l L-1 field slip) ---
    #: Documented response schemas keyed by status code, ``$ref``-resolved:
    #: ``{"200": {...}, "404": {...}}``. Swagger 2.0 specs degrade to ``{}``
    #: (explicit honest blindness, never guessed shapes). Owner: T3.
    response_schemas: dict[str, Any] = field(default_factory=dict)

    @property
    def full_path(self) -> str:
        """Return method + path string."""
        return f"{self.method} {self.path}"


@dataclass
class TestCase:
    """Generated test case."""

    id: str
    title: str
    description: str
    endpoint: APIEndpoint
    test_type: TestType
    priority: TestPriority
    preconditions: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    expected_results: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class RequirementItem:
    """Parsed requirement item."""

    id: str
    title: str
    description: str
    module: str = ""
    priority: TestPriority = TestPriority.MEDIUM
    acceptance_criteria: list[str] = field(default_factory=list)


@dataclass
class PerformanceConfig:
    """Performance test configuration."""

    base_url: str = "https://api.example.com"
    virtual_users: int = 100
    duration_seconds: int = 300
    ramp_up_seconds: int = 60
    think_time_ms: int = 500
    auth_type: str = "none"


@dataclass
class ReportMetadata:
    """Report metadata."""

    title: str
    created_at: datetime = field(default_factory=datetime.now)
    author: str = "TestAgent"
    version: str = "0.1.0"
    description: str = ""


@dataclass
class TestCaseGenInput:
    """Input payload for test case generation."""

    __test__ = False  # prevent pytest collection of this class

    requirements: list[RequirementItem]
    endpoints: list[APIEndpoint] = field(default_factory=list)
    #: Previously generated test cases (JSON file path or list of TestCase).
    #: When provided, the generator treats them as a baseline and merges
    #: new requirements on top, preserving unchanged historical cases and
    #: only generating net-new cases for the new requirements.
    historical_cases: list[TestCase] = field(default_factory=list)


@dataclass
class PerfGenInput:
    """Input payload for performance script generation."""

    endpoints: list[APIEndpoint]
    config: PerformanceConfig | None = None


@dataclass
class GUITestGenInput:
    """Input payload for GUI (Playwright) test script generation.

    Inspired by stagehand's approach: the LLM analyzes requirements and the
    target URL, then generates Playwright test steps using robust locator
    strategies (``get_by_role`` / ``get_by_label`` / ``get_by_text``).
    """

    __test__ = False  # prevent pytest collection of this class

    requirements: list[RequirementItem]
    url: str | None = None
    endpoints: list[APIEndpoint] = field(default_factory=list)
    output_language: str = "english"


@dataclass
class TestCaseReportInput:
    """Input payload for test case report generation."""

    __test__ = False  # prevent pytest collection of this class

    test_cases: list[TestCase]
    output_format: str = "markdown"
    output_language: str = "english"


@dataclass
class PerfReportInput:
    """Input payload for performance report generation."""

    script_path: str
    config: PerformanceConfig
    metrics: dict[str, Any] | None = None
    analysis: dict[str, Any] | None = None
    output_language: str = "english"
