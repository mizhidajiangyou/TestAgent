"""Tests for PerformanceGenerator."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from testagent.config.models import APIEndpoint, PerfGenInput
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.performance_generator import PerformanceGenerator

MOCK_K6_SCRIPT = """import http from 'k6/http';
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
  const res = http.get(`${__ENV.BASE_URL}/users`);
  check(res, { 'status is 200': (r) => r.status === 200 });
  sleep(0.5);
}
"""

# A review-refined variant (distinct text so tests can tell them apart).
REVISED_K6_SCRIPT = MOCK_K6_SCRIPT + "\n// reviewed: thresholds tightened\n"

# A minimal but structurally valid JMX document.
MOCK_JMX = """<?xml version="1.0" encoding="UTF-8"?>
<jmeterTestPlan version="1.2">
  <hashTree/>
</jmeterTestPlan>"""


class TestPerformanceGenerator:
    """Test suite for PerformanceGenerator."""

    def setup_method(self) -> None:
        self.mock_llm = MagicMock()
        self.mock_llm.chat.return_value = MOCK_K6_SCRIPT
        self.prompt_builder = PromptBuilder()
        self.generator = PerformanceGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
            script_format="k6",
        )
        self.endpoints = [
            APIEndpoint(method="GET", path="/users", summary="List users"),
        ]

    def _input(self) -> PerfGenInput:
        return PerfGenInput(endpoints=self.endpoints)

    def test_generate_returns_script(self) -> None:
        """Test that generate returns script content."""
        script = self.generator.generate(self._input())
        assert "import http" in script
        assert "k6" in script

    def test_generate_calls_llm(self) -> None:
        """Test that generate calls LLM client."""
        self.generator.generate(self._input())
        self.mock_llm.chat.assert_called_once()

    def test_save_creates_file(self, tmp_path) -> None:
        """Test that save writes script file."""
        script = self.generator.generate(self._input())
        output_path = tmp_path / "test.js"
        result = self.generator.save(script, output_path)

        assert result == output_path
        assert output_path.exists()
        assert "import http" in output_path.read_text()

    def test_strip_markdown_fences(self) -> None:
        """Test that markdown fences are stripped."""
        self.mock_llm.chat.return_value = f"```javascript\n{MOCK_K6_SCRIPT}\n```"
        script = self.generator.generate(self._input())
        assert not script.startswith("```")

    def test_jmeter_validation_missing_xml_decl(self) -> None:
        """Test JMX validation catches missing XML declaration."""
        gen = PerformanceGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
            script_format="jmeter",
        )
        self.mock_llm.chat.return_value = "<jmeterTestPlan></jmeterTestPlan>"
        with pytest.raises(ValueError, match="XML declaration"):
            gen.generate(self._input())

    def test_jmeter_validation_missing_closing_tag(self) -> None:
        """Test JMX validation catches missing closing tag."""
        gen = PerformanceGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
            script_format="jmeter",
        )
        self.mock_llm.chat.return_value = '<?xml version="1.0"?><jmeterTestPlan>'
        with pytest.raises(ValueError, match="closing"):
            gen.generate(self._input())


class TestBuildScriptReviewPrompt:
    """Render verification for script_review_prompt.j2 (plan v2 §4.2).

    Covers all template branches (k6 / jmeter / playwright) with full and
    minimal variables, plus the inline fallback when templates are missing.
    """

    def setup_method(self) -> None:
        self.builder = PromptBuilder()

    def test_k6_branch_full_variables(self) -> None:
        system, user = self.builder.build_script_review_prompt(
            script_kind="k6",
            script=MOCK_K6_SCRIPT,
            context_text="VUs=100 duration=300s",
            output_language="chinese",
        )
        assert "k6" in system
        assert "Simplified Chinese" in system
        assert MOCK_K6_SCRIPT.strip() in user
        assert "VUs=100 duration=300s" in user
        # Output contract: complete script, not a diff.
        assert "COMPLETE" in user

    def test_jmeter_branch(self) -> None:
        system, user = self.builder.build_script_review_prompt(
            script_kind="jmeter",
            script='<?xml version="1.0"?><jmeterTestPlan/>',
            context_text="ThreadGroup: 50 VUs",
        )
        assert "JMX" in system
        assert "jmeterTestPlan" in user
        assert "ThreadGroup: 50 VUs" in user

    def test_playwright_branch_minimal_variables(self) -> None:
        """Minimal-variable render: defaults only, {% if %} pairing intact."""
        system, user = self.builder.build_script_review_prompt(
            script_kind="playwright",
            script="def test_login(): ...",
            context_text="URL: https://example.com",
        )
        assert "Playwright" in system
        assert "def test_login(): ..." in user
        assert "https://example.com" in user

    def test_inline_fallback_when_template_missing(self) -> None:
        builder = PromptBuilder(templates_dir=Path("/nonexistent-templates"))
        system, user = builder.build_script_review_prompt(
            script_kind="k6",
            script=MOCK_K6_SCRIPT,
            context_text="VUs=10",
        )
        assert "k6" in system
        assert MOCK_K6_SCRIPT.strip() in user
        assert "VUs=10" in user
        assert "COMPLETE" in user  # same output contract as the template

    def test_inline_fallback_all_kinds(self) -> None:
        builder = PromptBuilder(templates_dir=Path("/nonexistent-templates"))
        for kind, marker in (
            ("jmeter", "JMX"),
            ("k6", "k6"),
            ("playwright", "Python"),
        ):
            _, user = builder.build_script_review_prompt(
                script_kind=kind, script="SCRIPT-BODY", context_text="CTX"
            )
            assert "SCRIPT-BODY" in user
            assert "CTX" in user
            assert marker in user


class TestPerformanceReview:
    """Review integration for PerformanceGenerator (plan v2 §4.3).

    Review uses the loop's built-in call path (no call_llm injection — R2);
    candidates are validated via _extract_script (+ _validate_jmx for jmeter).
    """

    def setup_method(self) -> None:
        self.mock_llm = MagicMock()
        self.prompt_builder = PromptBuilder()
        self.endpoints = [APIEndpoint(method="GET", path="/users", summary="List users")]

    def _generator(
        self, *, review_enabled: bool, script_format: str = "k6"
    ) -> PerformanceGenerator:
        return PerformanceGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
            script_format=script_format,
            review_enabled=review_enabled,
            review_max_rounds=1,
        )

    def _input(self) -> PerfGenInput:
        return PerfGenInput(endpoints=self.endpoints)

    def test_review_disabled_by_default_single_llm_call(self) -> None:
        self.mock_llm.chat.return_value = MOCK_K6_SCRIPT
        gen = PerformanceGenerator(llm_client=self.mock_llm, prompt_builder=self.prompt_builder)
        gen.generate(self._input())
        self.mock_llm.chat.assert_called_once()

    def test_review_enabled_runs_second_pass(self) -> None:
        self.mock_llm.chat.side_effect = [MOCK_K6_SCRIPT, REVISED_K6_SCRIPT]
        gen = self._generator(review_enabled=True)
        script = gen.generate(self._input())
        assert "reviewed: thresholds tightened" in script
        assert self.mock_llm.chat.call_count == 2  # generate + 1 review round

    def test_review_prompt_carries_config_context(self) -> None:
        """Decision 4 symmetry: review sees load config + endpoints."""
        self.mock_llm.chat.side_effect = [MOCK_K6_SCRIPT, REVISED_K6_SCRIPT]
        gen = self._generator(review_enabled=True)
        gen.generate(self._input())
        review_call = self.mock_llm.chat.call_args_list[1]
        system_prompt, user_prompt = review_call.args
        assert "k6" in system_prompt
        assert "virtual_users" in user_prompt  # load config context
        assert "GET /users" in user_prompt or "/users" in user_prompt
        assert MOCK_K6_SCRIPT.strip() in user_prompt  # the script under review

    def test_review_empty_responses_keep_original(self) -> None:
        # 1 round x 2 attempts (initial + built-in retry), both empty.
        self.mock_llm.chat.side_effect = [MOCK_K6_SCRIPT, "", ""]
        gen = self._generator(review_enabled=True)
        script = gen.generate(self._input())
        # _extract_script strips whitespace; the first-pass script is kept.
        assert script == MOCK_K6_SCRIPT.strip()
        assert self.mock_llm.chat.call_count == 3

    def test_jmeter_review_invalid_candidate_keeps_original(self) -> None:
        """A review answer failing _validate_jmx counts as a failed round."""
        self.mock_llm.chat.side_effect = [MOCK_JMX, "not xml at all", "still not xml"]
        gen = self._generator(review_enabled=True, script_format="jmeter")
        script = gen.generate(self._input())
        assert script == MOCK_JMX.strip()

    def test_jmeter_review_accepts_valid_revised_candidate(self) -> None:
        revised_jmx = MOCK_JMX.replace("<hashTree/>", "<hashTree><ResultCollector/></hashTree>")
        self.mock_llm.chat.side_effect = [MOCK_JMX, revised_jmx]
        gen = self._generator(review_enabled=True, script_format="jmeter")
        script = gen.generate(self._input())
        assert "ResultCollector" in script

    def test_save_writes_meta_when_review_succeeded(self, tmp_path: Path) -> None:
        self.mock_llm.chat.side_effect = [MOCK_K6_SCRIPT, REVISED_K6_SCRIPT]
        gen = self._generator(review_enabled=True)
        script = gen.generate(self._input())
        out = tmp_path / "perf_test.js"
        gen.save(script, out)

        meta = json.loads((tmp_path / "perf_test.meta.json").read_text())
        assert meta == {
            "generator": "performance",
            "script_format": "k6",
            "reviewed": True,
            "rounds_executed": 1,
            "rounds_succeeded": 1,
        }

    def test_save_meta_marks_unreviewed_when_all_rounds_failed(self, tmp_path: Path) -> None:
        self.mock_llm.chat.side_effect = [MOCK_K6_SCRIPT, "", ""]
        gen = self._generator(review_enabled=True)
        script = gen.generate(self._input())
        gen.save(script, tmp_path / "perf_test.js")

        meta = json.loads((tmp_path / "perf_test.meta.json").read_text())
        assert meta["reviewed"] is False
        assert meta["rounds_succeeded"] == 0

    def test_save_no_meta_when_review_did_not_run(self, tmp_path: Path) -> None:
        self.mock_llm.chat.return_value = MOCK_K6_SCRIPT
        gen = self._generator(review_enabled=False)
        script = gen.generate(self._input())
        gen.save(script, tmp_path / "perf_test.js")
        assert not (tmp_path / "perf_test.meta.json").exists()
