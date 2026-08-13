"""Tests for TestCaseGenerator."""

import csv
import json
from unittest.mock import MagicMock

from testagent.config.models import (
    APIEndpoint,
    RequirementItem,
    TestCaseGenInput,
    TestType,
)
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

_ENDPOINTS = [
    APIEndpoint(method="GET", path="/users", summary="List users"),
    APIEndpoint(method="POST", path="/users", summary="Create user"),
]

_REQUIREMENTS = [
    RequirementItem(
        id="REQ-001",
        title="User Management",
        description="Users can be created, listed, and deleted",
        module="user",
        acceptance_criteria=["User can register", "User can login"],
    ),
]


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

    def _input_endpoints_only(self) -> TestCaseGenInput:
        return TestCaseGenInput(requirements=[], endpoints=_ENDPOINTS)

    def _input_requirements_only(self) -> TestCaseGenInput:
        return TestCaseGenInput(requirements=_REQUIREMENTS, endpoints=[])

    def _input_both(self) -> TestCaseGenInput:
        return TestCaseGenInput(requirements=_REQUIREMENTS, endpoints=_ENDPOINTS)

    def test_generate_from_endpoints_only(self) -> None:
        """Test generation with endpoints only (no requirements)."""
        test_cases = self.generator.generate(self._input_endpoints_only())
        assert len(test_cases) == 2
        assert test_cases[0].id == "TC-001"
        assert test_cases[0].test_type == TestType.FUNCTIONAL

    def test_generate_from_requirements_only(self) -> None:
        """Test generation with requirements only (no endpoints)."""
        test_cases = self.generator.generate(self._input_requirements_only())
        assert len(test_cases) == 2
        # Endpoint should be N/A fallback when no API spec
        assert test_cases[0].endpoint.full_path == "N/A N/A"

    def test_generate_both_phases(self) -> None:
        """Test two-phase generation: requirements + API-specific."""
        test_cases = self.generator.generate(self._input_both())
        # Phase 1 (1 batch req) + Phase 2 (1 batch endpoints) = 2 calls * 2 = 4
        assert len(test_cases) == 4
        assert self.mock_llm.chat.call_count == 2

    def test_generate_calls_llm(self) -> None:
        """Test that generate calls LLM client."""
        self.generator.generate(self._input_endpoints_only())
        self.mock_llm.chat.assert_called_once()

    def test_save_creates_file(self, tmp_path) -> None:
        """Test that save writes JSON file."""
        test_cases = self.generator.generate(self._input_endpoints_only())
        output_path = tmp_path / "testcases.json"
        result = self.generator.save(test_cases, output_path)
        assert result == output_path
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert len(data) == 2

    def test_save_csv_creates_file(self, tmp_path) -> None:
        """Test that save_csv writes a UTF-8 BOM CSV file."""
        test_cases = self.generator.generate(self._input_endpoints_only())
        output_path = tmp_path / "testcases.csv"
        result = self.generator.save_csv(test_cases, output_path)
        assert result == output_path
        assert output_path.exists()
        raw = output_path.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")
        with open(output_path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_parse_response_with_markdown_fences(self) -> None:
        """Test parsing response wrapped in markdown code fences."""
        self.mock_llm.chat.return_value = f"```json\n{MOCK_LLM_RESPONSE}\n```"
        test_cases = self.generator.generate(self._input_endpoints_only())
        assert len(test_cases) == 2

    def test_parse_response_with_surrounding_prose(self) -> None:
        """Test parsing response with prose around the JSON array."""
        self.mock_llm.chat.return_value = (
            f"Here are the test cases:\n\n{MOCK_LLM_RESPONSE}\n\nHope this helps!"
        )
        test_cases = self.generator.generate(self._input_endpoints_only())
        assert len(test_cases) == 2

    def test_parse_response_empty(self, tmp_path, monkeypatch) -> None:
        """Test parsing invalid JSON triggers retries then gives up."""
        monkeypatch.chdir(tmp_path)
        self.mock_llm.chat.return_value = "not valid json"
        test_cases = self.generator.generate(self._input_endpoints_only())
        assert len(test_cases) == 0
        assert self.mock_llm.chat.call_count == MAX_PARSE_RETRIES

    def test_review_disabled_calls_llm_once(self) -> None:
        """Test that review disabled results in a single LLM call (endpoints only)."""
        self.generator = TestCaseGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
            review_enabled=False,
        )
        self.generator.generate(self._input_endpoints_only())
        self.mock_llm.chat.assert_called_once()

    def test_review_enabled_runs_second_pass(self) -> None:
        """Test that review triggers a second fresh-conversation LLM call.

        Uses ``review_max_rounds=1`` so the test only needs one review call
        after the initial generation.
        """
        refined = json.dumps(
            [
                {"id": "TC-001", "title": "Get users", "endpoint": "GET /users",
                 "test_type": "functional", "priority": "high",
                 "steps": ["Send GET"], "expected_results": ["Status 200"]},
                {"id": "TC-002", "title": "Create user", "endpoint": "POST /users",
                 "test_type": "functional", "priority": "high",
                 "steps": ["Send POST"], "expected_results": ["Status 201"]},
                {"id": "TC-003", "title": "Update user", "endpoint": "POST /users",
                 "test_type": "boundary", "priority": "medium",
                 "steps": ["Send request"], "expected_results": ["Status 400"]},
            ]
        )
        self.mock_llm.chat.side_effect = [MOCK_LLM_RESPONSE, refined]
        generator = TestCaseGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
            review_enabled=True,
            review_max_rounds=1,
        )
        test_cases = generator.generate(self._input_endpoints_only())
        assert self.mock_llm.chat.call_count == 2
        assert len(test_cases) == 3
        assert test_cases[2].id == "TC-003"

    def test_review_failure_keeps_original(self) -> None:
        """Test that unparseable review output keeps the original cases.

        With ``review_max_rounds=1`` and 3 parse-retry attempts, the LLM is
        called 4 times total (1 generate + 3 review retries).
        """
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
            review_max_rounds=1,
        )
        test_cases = generator.generate(self._input_endpoints_only())
        assert len(test_cases) == 2

    def test_review_multi_round_alternates_clients(self) -> None:
        """Multi-round review alternates between primary and secondary clients.

        With ``review_max_rounds=2``:
          - Generation: primary (1 call)
          - Round 1: secondary (1 call)
          - Round 2: primary (1 call)
        Total: primary called twice, secondary called once.
        """
        primary_llm = MagicMock()
        primary_llm.chat.return_value = MOCK_LLM_RESPONSE
        secondary_llm = MagicMock()
        refined_round1 = json.dumps(
            [
                {"id": "TC-001", "title": "Get users", "endpoint": "GET /users",
                 "test_type": "functional", "priority": "high",
                 "steps": ["Send GET"], "expected_results": ["Status 200"]},
            ]
        )
        refined_round2 = json.dumps(
            [
                {"id": "TC-001", "title": "Get users (refined)", "endpoint": "GET /users",
                 "test_type": "functional", "priority": "high",
                 "steps": ["Send GET with auth"], "expected_results": ["Status 200"]},
            ]
        )
        secondary_llm.chat.return_value = refined_round1
        # Round 2 uses the primary client again — need a 2nd response
        primary_llm.chat.side_effect = [MOCK_LLM_RESPONSE, refined_round2]

        generator = TestCaseGenerator(
            llm_client=primary_llm,
            prompt_builder=self.prompt_builder,
            review_enabled=True,
            review_llm_client=secondary_llm,
            review_max_rounds=2,
        )
        test_cases = generator.generate(self._input_endpoints_only())
        assert primary_llm.chat.call_count == 2  # generation + round 2
        assert secondary_llm.chat.call_count == 1  # round 1
        assert len(test_cases) == 1
        assert test_cases[0].title == "Get users (refined)"

    def test_review_single_model_logs_warning(self, caplog) -> None:
        """When review client equals primary, a warning is logged."""
        import logging

        with caplog.at_level(logging.WARNING):
            generator = TestCaseGenerator(
                llm_client=self.mock_llm,
                prompt_builder=self.prompt_builder,
                review_enabled=True,
                review_llm_client=None,  # falls back to primary
                review_max_rounds=1,
            )
        assert any(
            "same as the primary client" in rec.message for rec in caplog.records
        )
        # Generator still functions
        self.mock_llm.chat.return_value = MOCK_LLM_RESPONSE
        test_cases = generator.generate(self._input_endpoints_only())
        assert len(test_cases) == 2

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

    def test_build_api_prompt(self) -> None:
        """Test API-specific prompt includes focus areas."""
        builder = PromptBuilder()
        _, user_prompt = builder.build_api_prompt(
            endpoints_text="GET /users",
            requirements_text="REQ-1",
            output_language="chinese",
        )
        assert "boundary" in user_prompt.lower()
        assert "security" in user_prompt.lower()
        assert "integration" in user_prompt.lower()
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

    def test_batch_generation_multiple_endpoint_batches(self) -> None:
        """Test that 4 endpoints are split into 2 batches of 2."""
        endpoints = [
            APIEndpoint(method="GET", path="/users", summary="List"),
            APIEndpoint(method="POST", path="/users", summary="Create"),
            APIEndpoint(method="DELETE", path="/users/{id}", summary="Delete"),
            APIEndpoint(method="POST", path="/users/login", summary="Login"),
        ]
        self.mock_llm.chat.return_value = MOCK_LLM_RESPONSE
        generator = TestCaseGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
        )
        test_cases = generator.generate(TestCaseGenInput(requirements=[], endpoints=endpoints))
        assert len(test_cases) == 4
        assert self.mock_llm.chat.call_count == 2

    def test_module_based_batching(self) -> None:
        """Test that requirements are batched by module."""
        reqs = [
            RequirementItem(id="R1", title="A", description="d", module="auth"),
            RequirementItem(id="R2", title="B", description="d", module="auth"),
            RequirementItem(id="R3", title="C", description="d", module="user"),
            RequirementItem(id="R4", title="D", description="d", module="user"),
            RequirementItem(id="R5", title="E", description="d", module="payment"),
        ]
        self.mock_llm.chat.return_value = MOCK_LLM_RESPONSE
        generator = TestCaseGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
        )
        test_cases = generator.generate(TestCaseGenInput(requirements=reqs, endpoints=[]))
        # 3 modules = 3 batches * 2 cases = 6
        assert len(test_cases) == 6
        assert self.mock_llm.chat.call_count == 3

    def test_targeted_reask(self) -> None:
        """Test that targeted re-ask includes failed output."""
        self.mock_llm.chat.side_effect = ["not json", MOCK_LLM_RESPONSE]
        generator = TestCaseGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
        )
        test_cases = generator.generate(self._input_endpoints_only())
        assert len(test_cases) == 2
        # Second call should include re-ask instructions
        second_call_args = self.mock_llm.chat.call_args_list[1]
        user_prompt_arg = second_call_args.args[1]
        assert "NOT valid JSON" in user_prompt_arg
        assert "not json" in user_prompt_arg

    def test_classify_failure_non_parseable(self) -> None:
        """Garbage with no JSON structure is classified as non_parseable."""
        assert TestCaseGenerator._classify_failure("not json at all") == "non_parseable"
        assert TestCaseGenerator._classify_failure("") == "non_parseable"
        assert TestCaseGenerator._classify_failure("here are the cases") == "non_parseable"

    def test_classify_failure_truncated(self) -> None:
        """Incomplete JSON array/object is classified as truncated."""
        # Array started but never closed
        assert TestCaseGenerator._classify_failure('[{"id": "TC-001"') == "truncated"
        # Object started but never closed
        assert TestCaseGenerator._classify_failure('{"id": "TC-001", "title":') == "truncated"

    def test_reask_truncated_hint(self) -> None:
        """Truncated output triggers a 'generate FEWER cases' re-ask hint."""
        truncated_output = '[{"id": "TC-001", "title": "A"'
        self.mock_llm.chat.side_effect = [truncated_output, MOCK_LLM_RESPONSE]
        generator = TestCaseGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
        )
        generator.generate(self._input_endpoints_only())
        second_call_args = self.mock_llm.chat.call_args_list[1]
        user_prompt_arg = second_call_args.args[1]
        assert "TRUNCATED" in user_prompt_arg
        assert "FEWER" in user_prompt_arg

    def test_salvage_truncated_json(self) -> None:
        """Test that truncated JSON array is salvaged."""
        truncated = (
            "[\n"
            '  {"id": "TC-001", "title": "A", "endpoint": "GET /users",'
            ' "test_type": "functional", "priority": "high",'
            ' "steps": ["s"], "expected_results": ["e"]},\n'
            '  {"id": "TC-002", "title": "B", "endpoint": "POST /users",'
            ' "test_type": "functional", "priority": "high",'
            ' "steps": ["s"], "expected_results": ["e"]},\n'
            '  {"id": "TC-003", "title": "C", "endpoint": "DELETE /users/{id}",'
            ' "test_type": "functional", "priority": "medium",'
            ' "steps": ["send"'
            # Truncated
        )
        salvaged = TestCaseGenerator._salvage_truncated_json(truncated)
        assert salvaged is not None
        assert len(salvaged) == 2

    def test_salvage_returns_none_for_garbage(self) -> None:
        """Test that salvage returns None for non-JSON input."""
        assert TestCaseGenerator._salvage_truncated_json("not json at all") is None
        assert TestCaseGenerator._salvage_truncated_json("") is None
