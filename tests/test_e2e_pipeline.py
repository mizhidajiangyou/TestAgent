"""End-to-end pipeline test using a mock LLM client.

Verifies the minimal chain: Swagger + requirements -> test cases + perf script + reports.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

from testagent.config.models import (
    PerfGenInput,
    PerformanceConfig,
    PerfReportInput,
    ReportMetadata,
    TestCaseGenInput,
    TestCaseReportInput,
)
from testagent.container import Container
from testagent.generators.performance_generator import PerformanceGenerator
from testagent.generators.testcase_generator import TestCaseGenerator

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

MOCK_TESTCASE_RESPONSE = json.dumps(
    [
        {
            "id": "TC-001",
            "title": "List users returns paginated results",
            "description": "Verify GET /users returns 200 with pagination",
            "endpoint": "GET /users",
            "test_type": "functional",
            "priority": "high",
            "preconditions": ["Admin is authenticated"],
            "steps": ["Send GET /users?page=1&limit=20"],
            "expected_results": ["Status 200", "Response contains user list"],
        },
        {
            "id": "TC-002",
            "title": "Create user with valid payload",
            "description": "Verify POST /users creates user",
            "endpoint": "POST /users",
            "test_type": "functional",
            "priority": "high",
            "preconditions": [],
            "steps": ["Send POST /users with valid body"],
            "expected_results": ["Status 201"],
        },
    ]
)

MOCK_K6_RESPONSE = """import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '60s', target: 100 },
    { duration: '240s', target: 100 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  const base = __ENV.BASE_URL || 'https://api.example.com';
  const res = http.get(`${base}/users`);
  check(res, { 'status is 200': (r) => r.status === 200 });
  sleep(0.5);
}
"""


class TestE2EPipeline:
    """Full pipeline test with mocked LLM."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        """Run the minimal chain end-to-end."""
        # Arrange: mock LLM returns testcase JSON for each batch, then k6 script.
        # Two-phase generation:
        #   Phase 1 (requirements): 3 reqs in 1 module batch -> 1 call
        #   Phase 2 (API-specific): 4 endpoints / 2 per batch -> 2 calls
        #   Perf script: 1 call
        #   Total: 4 calls, 6 test cases (3 batches * 2 cases)
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            MOCK_TESTCASE_RESPONSE,  # Phase 1: requirements batch
            MOCK_TESTCASE_RESPONSE,  # Phase 2: API batch 1/2
            MOCK_TESTCASE_RESPONSE,  # Phase 2: API batch 2/2
            MOCK_K6_RESPONSE,  # Performance script
        ]

        container = Container()

        # Step 1: parse inputs
        endpoints = container.swagger_parser.parse(str(EXAMPLES_DIR / "sample_swagger.json"))
        assert len(endpoints) == 4

        requirements = container.requirement_parser.parse(
            str(EXAMPLES_DIR / "sample_requirements.md")
        )
        assert len(requirements) == 3

        # Step 2: generate test cases (two-phase: req batch + 2 api batches)
        tc_generator = TestCaseGenerator(
            llm_client=mock_llm, prompt_builder=container.prompt_builder
        )
        test_cases = tc_generator.generate(
            TestCaseGenInput(endpoints=endpoints, requirements=requirements)
        )
        # 3 batches * 2 cases = 6 cases (re-numbered TC-001..TC-006)
        assert len(test_cases) == 6
        assert test_cases[0].id == "TC-001"
        assert test_cases[-1].id == "TC-006"

        tc_path = tmp_path / "testcases.json"
        tc_generator.save(test_cases, tc_path)
        assert tc_path.exists()

        # Step 3: generate testcase report (markdown)
        report_meta = ReportMetadata(title="用户管理系统测试用例报告")
        container.testcase_report._metadata = report_meta
        tc_report = container.testcase_report.generate(
            TestCaseReportInput(test_cases=test_cases, output_format="markdown")
        )
        tc_report_path = tmp_path / "testcase_report.md"
        container.testcase_report.save(tc_report, tc_report_path)
        assert tc_report_path.exists()
        assert "用户管理系统测试用例报告" in tc_report_path.read_text()

        # Step 4: generate performance script
        perf_generator = PerformanceGenerator(
            llm_client=mock_llm,
            prompt_builder=container.prompt_builder,
            script_format="k6",
        )
        perf_config = PerformanceConfig(base_url="https://api.example.com", virtual_users=100)
        script = perf_generator.generate(PerfGenInput(endpoints=endpoints, config=perf_config))
        script_path = tmp_path / "perf_test.js"
        perf_generator.save(script, script_path)
        assert script_path.exists()
        assert "import http" in script_path.read_text()

        # Step 5: generate performance report template
        perf_report = container.performance_report.generate(
            PerfReportInput(script_path=str(script_path), config=perf_config)
        )
        perf_report_path = tmp_path / "perf_report.md"
        container.performance_report.save(perf_report, perf_report_path)
        assert perf_report_path.exists()
        content = perf_report_path.read_text()
        assert "Test Configuration" in content
        assert "k6 run" in content

        # Verify LLM was called 4 times (1 req batch + 2 api batches + 1 perf script)
        assert mock_llm.chat.call_count == 4
