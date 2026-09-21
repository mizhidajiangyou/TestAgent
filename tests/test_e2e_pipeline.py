"""End-to-end product chain on the SHIPPING code path.

Swagger + requirements -> test cases -> testcase report -> perf script ->
perf report, driven by the task-package chain (``tasks/testcase``,
``tasks/perf`` + ``PipelineExecutor``). It used to drive the legacy
generators, which meant the only "does the whole thing fit together" test was
exercising the code the migration is deleting.
"""

import asyncio
import json
from pathlib import Path

from testagent.config.models import (
    PerformanceConfig,
    PerfReportInput,
    ReportMetadata,
    TestCaseReportInput,
)
from testagent.container import Container
from testagent.engine.llm_client import LLMClient, LLMResponse
from testagent.parsers.requirement_parser import RequirementParser
from testagent.parsers.swagger_parser import SwaggerParser
from testagent.pipeline.executor import PipelineExecutor
from testagent.pipeline.inputs import parse_inputs
from testagent.pipeline.registry import get_registry
from testagent.pipeline.runtime import build_engine_generate_unit, build_generate_unit
from testagent.pipeline.testcase_adapter import dict_to_testcase
from testagent.pipeline.writers import write_artifact
from tests.parity_harness import bare_settings

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
REPO = Path(__file__).parents[1]

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


class _QueuedLLM(LLMClient):
    """Answers from a queue on the async entry point the pipeline uses."""

    intent_capable = False

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.usage = None  # type: ignore[assignment]

    async def achat_with_meta(self, system, user, response_format=None, max_tokens=None):  # type: ignore[override]
        self.calls += 1
        text = self._responses.pop(0) if self._responses else ""
        return LLMResponse(text=text, finish_reason="stop", completion_tokens=50)

    async def achat(self, system, user, response_format=None, max_tokens=None):  # type: ignore[override]
        return (await self.achat_with_meta(system, user)).text


class TestE2EPipeline:
    """Full pipeline test with a scripted LLM (no network)."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        container = Container()
        endpoints = SwaggerParser().parse(str(EXAMPLES_DIR / "sample_swagger.json"))
        assert len(endpoints) == 4
        requirements = RequirementParser().parse(str(EXAMPLES_DIR / "sample_requirements.md"))
        assert len(requirements) == 3

        # Two-phase generation on tasks/testcase: 3 requirement units + 2 API
        # batches (4 endpoints / batch of 2). Batch 2 must carry ITS endpoints
        # (GET/DELETE /users/{id}) or the engine's scope filter drops the items.
        batch2 = json.loads(MOCK_TESTCASE_RESPONSE)
        batch2[0]["endpoint"] = "GET /users/{id}"
        batch2[1]["endpoint"] = "DELETE /users/{id}"

        settings = bare_settings(
            output_dir=str(tmp_path / "output"),
            review_enabled=False,
            llm=bare_settings().llm.model_copy(update={"max_concurrency": 1}),
        )
        case_llm = _QueuedLLM(
            [
                MOCK_TESTCASE_RESPONSE,
                MOCK_TESTCASE_RESPONSE,
                MOCK_TESTCASE_RESPONSE,
                MOCK_TESTCASE_RESPONSE,
                json.dumps(batch2, ensure_ascii=False),
            ]
        )
        task = get_registry(REPO / "tasks").get("testcase")
        executor = PipelineExecutor(
            case_llm,  # type: ignore[arg-type]
            settings,
            generate_unit=build_engine_generate_unit(
                case_llm,  # type: ignore[arg-type]
                output_token_cap=settings.llm.max_output_tokens,
            ),
        )
        ctx = parse_inputs(
            task.manifest,
            {
                "requirements": str(EXAMPLES_DIR / "sample_requirements.md"),
                "swagger": str(EXAMPLES_DIR / "sample_swagger.json"),
            },
            settings,
        )
        result = asyncio.run(executor.arun(task, ctx, session_id="e2e-cases"))
        test_cases = [tc for tc in (dict_to_testcase(item) for item in result.artifact) if tc]
        assert case_llm.calls == 5, "two-phase fan-out changed on the shipping path"
        assert test_cases, "the pipeline produced no usable cases"
        assert test_cases[0].id == "TC-001"
        assert [tc.id for tc in test_cases] == [
            f"TC-{i:03d}" for i in range(1, len(test_cases) + 1)
        ]

        tc_path = tmp_path / "testcases.json"
        write_artifact(task.manifest, result.artifact, tc_path, "json", ctx=ctx.parsed)
        assert tc_path.exists()

        # Step 3: testcase report (markdown) from the same case objects.
        report_meta = ReportMetadata(title="用户管理系统测试用例报告")
        tc_report_service = container.testcase_report()
        tc_report_service._metadata = report_meta
        tc_report = tc_report_service.generate(
            TestCaseReportInput(test_cases=test_cases, output_format="markdown")
        )
        tc_report_path = tmp_path / "testcase_report.md"
        tc_report_service.save(tc_report, tc_report_path)
        assert tc_report_path.exists()
        assert "用户管理系统测试用例报告" in tc_report_path.read_text(encoding="utf-8")

        # Step 4: performance script through tasks/perf.
        perf_llm = _QueuedLLM([MOCK_K6_RESPONSE])
        perf_settings = bare_settings(output_dir=str(tmp_path / "output"), review_enabled=False)
        perf_task = get_registry(REPO / "tasks").get("perf")
        perf_executor = PipelineExecutor(
            perf_llm,  # type: ignore[arg-type]
            perf_settings,
            generate_unit=build_generate_unit(perf_llm),  # type: ignore[arg-type]
        )
        perf_ctx = parse_inputs(
            perf_task.manifest,
            {
                "swagger": str(EXAMPLES_DIR / "sample_swagger.json"),
                "script_format": "k6",
                "base_url": "https://api.example.com",
            },
            perf_settings,
        )
        perf_result = asyncio.run(perf_executor.arun(perf_task, perf_ctx, session_id="e2e-perf"))
        assert perf_llm.calls == 1
        script_path = tmp_path / "perf_test.js"
        write_artifact(
            perf_task.manifest,
            perf_result.artifact,
            script_path,
            "text",
            ctx=perf_ctx.parsed,
        )
        assert script_path.exists()
        assert "import http" in script_path.read_text(encoding="utf-8")

        # Step 5: performance report template.
        perf_config = PerformanceConfig(base_url="https://api.example.com", virtual_users=100)
        perf_report = container.performance_report().generate(
            PerfReportInput(script_path=str(script_path), config=perf_config)
        )
        perf_report_path = tmp_path / "perf_report.md"
        container.performance_report().save(perf_report, perf_report_path)
        assert perf_report_path.exists()
        content = perf_report_path.read_text(encoding="utf-8")
        assert "Test Configuration" in content
        assert "k6 run" in content
