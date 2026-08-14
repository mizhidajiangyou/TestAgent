"""Tests for the FastAPI web app (TestAgent Web GUI)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from testagent.config.models import (
    APIEndpoint,
    TestCase,
    TestCaseGenInput,
)
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.testcase_generator import TestCaseGenerator
from testagent.parsers.requirement_parser import RequirementParser
from testagent.reports.testcase_report import TestCaseReport
from testagent.web.app import create_app

MOCK_LLM_RESPONSE = json.dumps(
    [
        {
            "id": "TC-001",
            "title": "Get users successfully",
            "description": "Verify GET /users returns 200",
            "endpoint": "GET /users",
            "test_type": "functional",
            "priority": "high",
            "preconditions": ["User is authenticated"],
            "steps": ["Send GET request to /users"],
            "expected_results": ["Status code is 200"],
        },
        {
            "id": "TC-002",
            "title": "Create user with valid data",
            "description": "Verify POST /users creates a user",
            "endpoint": "POST /users",
            "test_type": "functional",
            "priority": "high",
            "steps": ["Send POST request with valid body"],
            "expected_results": ["Status code is 201"],
        },
    ]
)

_REQUIREMENTS_MD = (
    "# User Management\n"
    "Users can be created, listed, and deleted.\n\n"
    "Acceptance Criteria:\n"
    "- User can register\n"
    "- User can login\n"
)


def _make_container() -> SimpleNamespace:
    """Build a near-real container: real parser/generator/report, mock LLM."""
    mock_llm = MagicMock()
    mock_llm.chat.return_value = MOCK_LLM_RESPONSE
    mock_llm.usage.summary.return_value = "prompt=10 completion=20 total=30"
    settings = SimpleNamespace(
        output_language="english",
        azure_llm=SimpleNamespace(enabled=False),
        llm=SimpleNamespace(models=["gpt-4o-mini", "gpt-4o"]),
        review_enabled=False,
        review_max_rounds=2,
    )
    return SimpleNamespace(
        testcase_generator=TestCaseGenerator(llm_client=mock_llm, prompt_builder=PromptBuilder()),
        requirement_parser=RequirementParser(),
        # Swagger parser returns real endpoints so generated cases get real
        # endpoint paths (matches the historical baseline for dedup tests).
        swagger_parser=MagicMock(
            parse=MagicMock(
                return_value=[
                    APIEndpoint(method="GET", path="/users", summary="List users"),
                    APIEndpoint(method="POST", path="/users", summary="Create user"),
                ]
            )
        ),
        testcase_report=TestCaseReport(),
        llm_client=mock_llm,
        settings=settings,
    )


@pytest.fixture()
def client() -> TestClient:
    """A TestClient wired to a container with a mocked LLM (no network)."""
    return TestClient(create_app(container=_make_container()))


class TestMetaEndpoints:
    def test_health(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_index_returns_html(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "TestAgent" in r.text

    def test_config_endpoint(self, client: TestClient) -> None:
        r = client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert data["provider"] == "openai"
        assert data["primary_model"] == "gpt-4o-mini"
        assert data["fallback_models"] == ["gpt-4o"]


class TestIframeEmbedding:
    def test_frame_ancestors_default_allows_all(self, client: TestClient) -> None:
        """Default CSP allows any origin to embed via iframe."""
        r = client.get("/")
        csp = r.headers.get("content-security-policy", "")
        assert "frame-ancestors *" in csp

    def test_no_x_frame_options_block(self, client: TestClient) -> None:
        """No X-Frame-Options: DENY header that would block embedding."""
        r = client.get("/")
        # Either absent or not DENY (the middleware strips it).
        xfo = r.headers.get("x-frame-options")
        assert xfo is None or "deny" not in xfo.lower()

    def test_frame_ancestors_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WEB_FRAME_ANCESTORS restricts the allowed embed origins."""
        monkeypatch.setenv("WEB_FRAME_ANCESTORS", "https://app.example.com")
        c = TestClient(create_app(container=_make_container()))
        r = c.get("/")
        csp = r.headers.get("content-security-policy", "")
        assert "frame-ancestors https://app.example.com" in csp
        assert "*" not in csp


class TestGenerate:
    def test_generate_json_returns_cases(self, client: TestClient) -> None:
        r = client.post(
            "/api/generate",
            json={"requirements": _REQUIREMENTS_MD, "output_format": "json"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2
        assert data["output_format"] == "json"
        assert data["test_cases"][0]["title"] == "Get users successfully"
        assert data["download_content"] is None

    def test_generate_empty_requirements_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/api/generate",
            json={"requirements": "   ", "output_format": "json"},
        )
        assert r.status_code == 400

    def test_generate_invalid_format_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/api/generate",
            json={"requirements": _REQUIREMENTS_MD, "output_format": "xml"},
        )
        assert r.status_code == 400

    def test_generate_csv_returns_download(self, client: TestClient) -> None:
        r = client.post(
            "/api/generate",
            json={"requirements": _REQUIREMENTS_MD, "output_format": "csv"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2
        assert data["download_filename"] == "testcases.csv"
        assert data["download_content"] is not None
        assert "title" in data["download_content"]

    def test_generate_markdown_returns_download(self, client: TestClient) -> None:
        r = client.post(
            "/api/generate",
            json={"requirements": _REQUIREMENTS_MD, "output_format": "markdown"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["download_filename"] == "testcases.md"
        assert data["download_content"] is not None

    def test_generate_with_historical_cases(self, client: TestClient) -> None:
        """Historical cases act as a baseline; duplicates are deduped."""
        historical = [
            {
                "id": "TC-OLD-001",
                "title": "Get users successfully",
                "endpoint": "GET /users",
                "test_type": "functional",
                "priority": "high",
            }
        ]
        r = client.post(
            "/api/generate",
            json={
                "requirements": _REQUIREMENTS_MD,
                "swagger_url": "swagger.json",
                "output_format": "json",
                "historical_cases": historical,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["historical_count"] == 1
        # 1 historical + 2 new - 1 duplicate = 2 total
        assert data["count"] == 2

    def test_generate_propagates_generation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A generation failure surfaces as a 500 with the detail."""
        container = _make_container()
        # Make the generator raise by giving it an LLM that always errors.
        container.testcase_generator._llm.chat.side_effect = RuntimeError("boom")
        c = TestClient(create_app(container=container))
        r = c.post(
            "/api/generate",
            json={"requirements": _REQUIREMENTS_MD, "output_format": "json"},
        )
        assert r.status_code == 500
        assert "boom" in r.json()["detail"]


class TestGenerateIntegration:
    """End-to-end-ish: confirm the generator receives parsed requirements."""

    def test_generate_calls_generator_with_parsed_requirements(self) -> None:
        """The web layer parses markdown into RequirementItems before generate."""
        container = _make_container()
        captured: list[TestCaseGenInput] = []
        original_generate = container.testcase_generator.generate

        def spy(data: TestCaseGenInput) -> list[TestCase]:
            captured.append(data)
            return original_generate(data)

        container.testcase_generator.generate = spy  # type: ignore[method-assign]

        c = TestClient(create_app(container=container))
        r = c.post(
            "/api/generate",
            json={"requirements": _REQUIREMENTS_MD, "output_format": "json"},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 2
        # The generator received at least one parsed requirement.
        assert len(captured) == 1
        assert len(captured[0].requirements) >= 1
        assert captured[0].requirements[0].title == "User Management"
