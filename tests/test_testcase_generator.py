"""Tests for TestCaseGenerator."""

import json
from unittest.mock import MagicMock

from testagent.config.models import APIEndpoint, TestCaseGenInput, TestPriority, TestType
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.testcase_generator import MAX_PARSE_RETRIES, TestCaseGenerator

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
            "expected_results": ["Status code is 200", "Response contains user list"],
        },
        {
            "id": "TC-002",
            "title": "Create user with valid data",
            "description": "Verify POST /users creates a user",
            "endpoint": "POST /users",
            "test_type": "functional",
            "priority": "high",
            "preconditions": ["User is authenticated"],
            "steps": ["Send POST request with valid body"],
            "expected_results": ["Status code is 201"],
        },
    ]
)


class TestTestCaseGenerator:
    """Test suite for TestCaseGenerator."""

    def setup_method(self) -> None:
        self.mock_llm = MagicMock()
        self.mock_llm.chat.return_value = MOCK_LLM_RESPONSE
        self.prompt_builder = PromptBuilder()
        self.generator = TestCaseGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
        )
        self.endpoints = [
            APIEndpoint(method="GET", path="/users", summary="List users"),
            APIEndpoint(method="POST", path="/users", summary="Create user"),
        ]

    def _input(self) -> TestCaseGenInput:
        return TestCaseGenInput(endpoints=self.endpoints)

    def test_generate_returns_test_cases(self) -> None:
        """Test that generate returns parsed test cases."""
        test_cases = self.generator.generate(self._input())
        assert len(test_cases) == 2
        assert test_cases[0].id == "TC-001"
        assert test_cases[0].test_type == TestType.FUNCTIONAL
        assert test_cases[0].priority == TestPriority.HIGH

    def test_generate_calls_llm(self) -> None:
        """Test that generate calls LLM client."""
        self.generator.generate(self._input())
        self.mock_llm.chat.assert_called_once()

    def test_save_creates_file(self, tmp_path) -> None:
        """Test that save writes JSON file."""
        test_cases = self.generator.generate(self._input())
        output_path = tmp_path / "testcases.json"
        result = self.generator.save(test_cases, output_path)

        assert result == output_path
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["id"] == "TC-001"

    def test_parse_response_with_markdown_fences(self) -> None:
        """Test parsing response wrapped in markdown code fences."""
        self.mock_llm.chat.return_value = f"```json\n{MOCK_LLM_RESPONSE}\n```"
        test_cases = self.generator.generate(self._input())
        assert len(test_cases) == 2

    def test_parse_response_with_surrounding_prose(self) -> None:
        """Test parsing response with prose around the JSON array."""
        self.mock_llm.chat.return_value = (
            f"Here are the test cases you requested:\n\n{MOCK_LLM_RESPONSE}\n\nHope this helps!"
        )
        test_cases = self.generator.generate(self._input())
        assert len(test_cases) == 2

    def test_parse_response_empty(self, tmp_path, monkeypatch) -> None:
        """Test parsing invalid JSON response triggers retries then gives up."""
        monkeypatch.chdir(tmp_path)
        self.mock_llm.chat.return_value = "not valid json"
        test_cases = self.generator.generate(self._input())
        assert len(test_cases) == 0
        assert self.mock_llm.chat.call_count == MAX_PARSE_RETRIES
