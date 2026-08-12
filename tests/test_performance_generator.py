"""Tests for PerformanceGenerator."""

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
