"""Tests for the FastAPI web app (TestAgent Web GUI).

Route behavior only — the HTTP contract of ``/api/generate`` (statuses, body
schema, error texts) is pinned by the recorded cells in
``tests/test_migration_parity_web.py``; these tests cover what the contract
fixtures cannot: config sanitization, CSP headers, and that a failure inside
generation surfaces as an actionable error instead of a silent empty 200.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from testagent.config.settings import LLMSettings
from testagent.web.app import create_app
from tests.parity_harness import bare_settings
from tests.web_fakes import (
    CASES_JSON,
    ScriptedLLM,
    make_container,
)
from tests.web_fakes import (
    REQUIREMENTS as _REQUIREMENTS_MD,
)

_MODELS = "gpt-4o-mini,gpt-4o"


def _make_container(**llm_flags: bool) -> object:
    """Real parsers/report, scripted LLM — the same double the contract cells
    use, so both suites exercise one chain (``models`` only feeds /api/config)."""
    llm = ScriptedLLM(
        [_case_response()] * 12,
        **llm_flags,  # type: ignore[arg-type]
    )
    settings = bare_settings(
        llm=LLMSettings(_env_file=None, model=_MODELS),  # type: ignore[call-arg]
    )
    return make_container(llm, settings)


def _case_response() -> str:
    return CASES_JSON


@pytest.fixture()
def client() -> TestClient:
    """A TestClient wired to a container with a scripted LLM (no network)."""
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
        assert data["test_cases"][0]["title"] == "List users returns 200"
        assert data["download_content"] is None
        # the quality line reaches the response (it used to be legacy-only)
        assert data["test_cases"][0]["executability"].get("grade")

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

    def test_generate_with_historical_cases(self, tmp_path: Path) -> None:
        """Historical cases act as the baseline; a regenerated duplicate keeps
        the historical version and drops the new one (H1 precedence).

        The spec is part of the premise: without endpoints to scope to, a
        generated case lands on the placeholder endpoint and is (correctly) not
        a duplicate of a case bound to GET /users.
        """
        from tests.web_fakes import SPEC

        spec = tmp_path / "spec.json"
        spec.write_text(SPEC, encoding="utf-8")
        client = TestClient(create_app(container=_make_container()))
        historical = [
            {
                "id": "TC-OLD-001",
                "title": "List users returns 200",
                "description": "baseline detail the regenerated copy must not overwrite",
                "endpoint": "GET /users",
                "test_type": "functional",
                "priority": "high",
            }
        ]
        r = client.post(
            "/api/generate",
            json={
                "requirements": _REQUIREMENTS_MD,
                "swagger_url": str(spec),
                "output_format": "json",
                "historical_cases": historical,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["historical_count"] == 1
        # 1 historical + 2 new - 1 duplicate = 2; ids are renumbered after the
        # merge (H1), so precedence is proved by the surviving content.
        assert data["count"] == 2
        assert data["test_cases"][0]["description"].startswith("baseline detail")

    def test_generate_propagates_generation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A total generation failure surfaces as a 500 with actionable guidance.

        Per-requirement LLM failures degrade to an empty unit (so one bad
        requirement does not abort the whole run), but when *every* call fails
        the generator raises a clear error that the web layer returns as 500.
        """
        # Every LLM call raising is the "total failure" case: the pipeline
        # degrades to an empty artifact and the route must say why.
        container = _make_container(fail_call=True)
        c = TestClient(create_app(container=container))
        r = c.post(
            "/api/generate",
            json={"requirements": _REQUIREMENTS_MD, "output_format": "json"},
        )
        assert r.status_code == 500
        assert "0 test cases" in r.json()["detail"]

    def test_generate_model_unavailable_returns_400(self) -> None:
        """A zero-token pre-flight failure surfaces as a 400 with guidance."""
        container = _make_container(fail_verify=True)
        c = TestClient(create_app(container=container))
        r = c.post(
            "/api/generate",
            json={"requirements": _REQUIREMENTS_MD, "output_format": "json"},
        )
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "Model unavailable" in detail
        assert "OPENAI_API_KEY" in detail


class TestGenerateIntegration:
    """End-to-end-ish: confirm the pipeline receives PARSED requirements."""

    def test_generate_parses_markdown_before_generating(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The web layer turns markdown into RequirementItems before the run —
        the claim the legacy spy test made about ``agenerate``, now checked at
        the pipeline boundary (``parse_inputs``)."""
        import testagent.web.app as web_app

        captured: list[object] = []
        original = web_app.parse_inputs

        def spy(manifest: object, raw: dict[str, str], settings: object = None) -> object:
            ctx = original(manifest, raw, settings)
            captured.append(ctx)
            return ctx

        monkeypatch.setattr(web_app, "parse_inputs", spy)
        monkeypatch.chdir(tmp_path)

        c = TestClient(create_app(container=_make_container()))
        r = c.post(
            "/api/generate",
            json={"requirements": _REQUIREMENTS_MD, "output_format": "json"},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 2
        assert len(captured) == 1
        requirements = captured[0].parsed["requirements"]  # type: ignore[attr-defined]
        assert len(requirements) >= 1
        assert requirements[0].title == "User Management"
        # the staged requirement file lives in a temp dir that is gone again
        assert not Path(captured[0].raw["requirements"]).exists()  # type: ignore[attr-defined]


class TestLinksReportDisplay:
    """``links_report`` reaches the browser as the SAME object the run recorded
    (v15 §8.3): no second rate calculation, and no key at all when links off."""

    def test_links_off_response_has_no_report_key(self, client: TestClient) -> None:
        r = client.post("/api/generate", json={"requirements": _REQUIREMENTS_MD})
        assert r.status_code == 200, r.text
        assert "links_report" not in r.json(), "a links-off response grew a contract key"

    def test_links_on_response_shows_the_run_report(self) -> None:
        llm = ScriptedLLM([_case_response()] * 24)
        settings = bare_settings(
            llm=LLMSettings(_env_file=None, model=_MODELS),  # type: ignore[call-arg]
            links_enabled=True,
            links_prose_enabled=True,
        )
        app = create_app(container=make_container(llm, settings))
        # The container's executor needs the injected signature renderer the
        # production container passes — same wiring, so the L0 index is real.
        r = TestClient(app).post(
            "/api/generate", json={"requirements": _REQUIREMENTS_MD, "links": True}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        report = body["links_report"]
        assert {"pair_rate", "grades_by_source_stage", "candidate_paths"} <= set(report)
