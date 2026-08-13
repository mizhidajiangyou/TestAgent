"""Tests for TestCaseGenerator."""

import csv
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

    def test_save_csv_creates_file(self, tmp_path) -> None:
        """Test that save_csv writes a UTF-8 BOM CSV file."""
        test_cases = self.generator.generate(self._input())
        output_path = tmp_path / "testcases.csv"
        result = self.generator.save_csv(test_cases, output_path)

        assert result == output_path
        assert output_path.exists()

        # UTF-8 BOM for Excel compatibility
        raw = output_path.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")

        with open(output_path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["id"] == "TC-001"
        assert rows[0]["endpoint"] == "GET /users"
        # List fields flattened with "; "
        assert rows[0]["steps"] == "Send GET request to /users"
        assert rows[0]["expected_results"] == "Status code is 200; Response contains user list"

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

    def test_review_disabled_calls_llm_once(self) -> None:
        """Test that review disabled results in a single LLM call."""
        self.generator = TestCaseGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
            review_enabled=False,
        )
        self.generator.generate(self._input())
        self.mock_llm.chat.assert_called_once()

    def test_review_enabled_runs_second_pass(self) -> None:
        """Test that review triggers a second fresh-conversation LLM call."""
        refined = json.dumps(
            [
                {
                    "id": "TC-001",
                    "title": "Get users",
                    "endpoint": "GET /users",
                    "test_type": "functional",
                    "priority": "high",
                    "steps": ["Send GET request to /users"],
                    "expected_results": ["Status code is 200"],
                },
                {
                    "id": "TC-002",
                    "title": "Create user",
                    "endpoint": "POST /users",
                    "test_type": "functional",
                    "priority": "high",
                    "steps": ["Send POST request"],
                    "expected_results": ["Status code is 201"],
                },
                {
                    "id": "TC-003",
                    "title": "Update user",
                    "endpoint": "POST /users",
                    "test_type": "boundary",
                    "priority": "medium",
                    "steps": ["Send request"],
                    "expected_results": ["Status code is 400"],
                },
            ]
        )
        self.mock_llm.chat.side_effect = [MOCK_LLM_RESPONSE, refined]
        generator = TestCaseGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
            review_enabled=True,
        )
        test_cases = generator.generate(self._input())
        assert self.mock_llm.chat.call_count == 2
        assert len(test_cases) == 3
        assert test_cases[2].id == "TC-003"

    def test_review_failure_keeps_original(self) -> None:
        """Test that unparseable review output keeps the original cases."""
        self.mock_llm.chat.side_effect = [
            MOCK_LLM_RESPONSE,
            "invalid json",
            "invalid json",
            "invalid json",
        ]
        generator = TestCaseGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
            review_enabled=True,
        )
        test_cases = generator.generate(self._input())
        assert len(test_cases) == 2
        assert test_cases[0].id == "TC-001"

    def test_output_language_chinese_instructed(self) -> None:
        """Test that chinese output language is passed into prompts."""
        builder = PromptBuilder()
        system_prompt, user_prompt = builder.build_testcase_prompt(
            endpoints_text="GET /users",
            requirements_text="",
            output_language="chinese",
        )
        assert "Simplified Chinese" in system_prompt
        assert "Simplified Chinese" in user_prompt

    def test_build_review_prompt(self) -> None:
        """Test review prompt includes current cases and checklist."""
        builder = PromptBuilder()
        _, user_prompt = builder.build_review_prompt(
            endpoints_text="GET /users",
            requirements_text="REQ-1",
            test_cases_json='[{"id": "TC-001"}]',
            output_language="chinese",
        )
        assert "no prior context" in user_prompt.lower()
        assert '[{"id": "TC-001"}]' in user_prompt
        assert "Simplified Chinese" in user_prompt
