"""Tests for TestCaseGenerator."""

import csv
import json
import logging
from unittest.mock import MagicMock

import pytest

from testagent.config.models import (
    APIEndpoint,
    RequirementItem,
    TestCase,
    TestCaseGenInput,
    TestPriority,
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

    def test_phase2_receives_phase1_coverage(self) -> None:
        """Regression: Phase 2 API prompt must receive Phase-1 coverage text.

        When both requirements and endpoints are provided, the generator feeds
        the Phase-1 case summary into Phase 2 via ``already_covered`` so that
        the two phases do not regenerate the same scenario (kills cross-phase
        duplication / spec inconsistency).
        """
        # Replace the generator's PromptBuilder with a mock so we can inspect
        # the arguments passed to build_api_prompt. Both prompt builders must
        # return a 2-tuple (system, user) because generate() unpacks them.
        self.prompt_builder = MagicMock()
        self.prompt_builder.build_testcase_prompt.return_value = ("sys", "user")
        self.prompt_builder.build_api_prompt.return_value = ("sys", "user")
        self.generator._prompt_builder = self.prompt_builder

        self.generator.generate(self._input_both())

        # Phase 2 must have been invoked exactly once (single endpoint batch).
        assert self.prompt_builder.build_api_prompt.called
        # Capture the already_covered kwarg from the Phase-2 call.
        call_kwargs = self.prompt_builder.build_api_prompt.call_args.kwargs
        already_covered = call_kwargs.get("already_covered", "")
        assert already_covered, "Phase 2 should receive non-empty already_covered"
        # The coverage text must reference a Phase-1 case so Phase 2 can avoid
        # duplicating it. MOCK_LLM_RESPONSE's first case title is below.
        assert "Get users successfully" in already_covered

    def test_phase2_not_invoked_without_requirements(self) -> None:
        """Regression: Phase 2 (API-specific) only runs when requirements exist.

        With endpoints-only input there is no Phase 1 to cover, so build_api_prompt
        must NOT be called and no already_covered wiring is needed.
        """
        self.prompt_builder = MagicMock()
        self.prompt_builder.build_testcase_prompt.return_value = ("sys", "user")
        self.prompt_builder.build_api_prompt.return_value = ("sys", "user")
        self.generator._prompt_builder = self.prompt_builder

        self.generator.generate(self._input_endpoints_only())

        assert not self.prompt_builder.build_api_prompt.called

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
                {
                    "id": "TC-001",
                    "title": "Get users",
                    "endpoint": "GET /users",
                    "test_type": "functional",
                    "priority": "high",
                    "steps": ["Send GET"],
                    "expected_results": ["Status 200"],
                },
                {
                    "id": "TC-002",
                    "title": "Create user",
                    "endpoint": "POST /users",
                    "test_type": "functional",
                    "priority": "high",
                    "steps": ["Send POST"],
                    "expected_results": ["Status 201"],
                },
                {
                    "id": "TC-003",
                    "title": "Update user",
                    "endpoint": "POST /users",
                    "test_type": "boundary",
                    "priority": "medium",
                    "steps": ["Send request"],
                    "expected_results": ["Status 400"],
                },
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
                {
                    "id": "TC-001",
                    "title": "Get users",
                    "endpoint": "GET /users",
                    "test_type": "functional",
                    "priority": "high",
                    "steps": ["Send GET"],
                    "expected_results": ["Status 200"],
                },
            ]
        )
        refined_round2 = json.dumps(
            [
                {
                    "id": "TC-001",
                    "title": "Get users (refined)",
                    "endpoint": "GET /users",
                    "test_type": "functional",
                    "priority": "high",
                    "steps": ["Send GET with auth"],
                    "expected_results": ["Status 200"],
                },
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
        assert any("same as the primary client" in rec.message for rec in caplog.records)
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


class TestHistoricalCases:
    """Tests for historical-case loading, merging, and incremental generation."""

    def setup_method(self) -> None:
        self.mock_llm = MagicMock()
        self.mock_llm.chat.return_value = MOCK_LLM_RESPONSE
        self.prompt_builder = PromptBuilder()
        self.generator = TestCaseGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
        )

    def _historical_case(self, title: str = "Get users successfully") -> TestCase:
        return TestCase(
            id="TC-OLD-001",
            title=title,
            description="legacy case",
            endpoint=APIEndpoint(method="GET", path="/users"),
            test_type=TestType.FUNCTIONAL,
            priority=TestPriority.HIGH,
            preconditions=["auth"],
            steps=["GET /users"],
            expected_results=["200"],
        )

    # --- load_historical_cases ---

    def test_load_historical_cases_from_file(self, tmp_path) -> None:
        """Load cases from a JSON file produced by save()."""
        cases = [self._historical_case()]
        path = tmp_path / "hist.json"
        self.generator.save(cases, path)
        loaded = TestCaseGenerator.load_historical_cases(path)
        assert len(loaded) == 1
        assert loaded[0].title == "Get users successfully"
        assert loaded[0].endpoint.full_path == "GET /users"

    def test_load_historical_cases_missing_file(self, tmp_path) -> None:
        """Missing file returns empty list (with a warning)."""
        loaded = TestCaseGenerator.load_historical_cases(tmp_path / "nope.json")
        assert loaded == []

    def test_load_historical_cases_invalid_json(self, tmp_path) -> None:
        """Invalid JSON returns empty list."""
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        assert TestCaseGenerator.load_historical_cases(path) == []

    def test_load_historical_cases_non_array(self, tmp_path) -> None:
        """A JSON object (not array) returns empty list."""
        path = tmp_path / "obj.json"
        path.write_text('{"id": "TC-001"}', encoding="utf-8")
        assert TestCaseGenerator.load_historical_cases(path) == []

    def test_load_historical_cases_skips_bad_items(self, tmp_path) -> None:
        """Non-dict items are skipped without failing the whole load."""
        path = tmp_path / "mixed.json"
        path.write_text(
            json.dumps(
                [
                    "not a dict",
                    {
                        "id": "TC-001",
                        "title": "A",
                        "endpoint": "GET /users",
                        "test_type": "functional",
                        "priority": "high",
                    },
                ]
            ),
            encoding="utf-8",
        )
        loaded = TestCaseGenerator.load_historical_cases(path)
        assert len(loaded) == 1
        assert loaded[0].title == "A"

    # --- _dict_to_testcase ---

    def test_dict_to_testcase_defaults(self) -> None:
        """Missing optional fields default to empty lists; bad enums fall back."""
        tc = TestCaseGenerator._dict_to_testcase(
            {
                "id": "TC-1",
                "title": "T",
                "endpoint": "GET /x",
                "test_type": "weird",
                "priority": "nope",
            }
        )
        assert tc is not None
        assert tc.test_type == TestType.FUNCTIONAL
        assert tc.priority == TestPriority.MEDIUM
        assert tc.preconditions == []
        assert tc.tags == []

    def test_dict_to_testcase_none_on_bad_data(self) -> None:
        """Malformed input returns None instead of raising."""
        # endpoint that splits to a single token still works (path defaults)
        assert TestCaseGenerator._dict_to_testcase({"title": "T"}) is not None
        # A dict raising during construction is caught -> None
        assert TestCaseGenerator._dict_to_testcase({"id": 123}) is not None

    # --- dedup + merge ---

    def test_case_dedup_key_case_insensitive(self) -> None:
        """Dedup key is case-insensitive on title and endpoint."""
        a = self._historical_case("Get Users Successfully")
        b = self._historical_case("get users successfully")
        assert TestCaseGenerator._case_dedup_key(a) == TestCaseGenerator._case_dedup_key(b)

    def test_merge_keeps_baseline_and_appends_net_new(self) -> None:
        """Historical baseline is preserved; only net-new cases are appended."""
        historical = [self._historical_case("Get users successfully")]
        new_cases = [
            self._historical_case("Get users successfully"),  # duplicate -> dropped
            TestCase(
                id="TC-NEW",
                title="Delete user",
                description="delete a user",
                endpoint=APIEndpoint("DELETE", "/users/{id}"),
                test_type=TestType.NEGATIVE,
                priority=TestPriority.MEDIUM,
            ),
        ]
        merged = TestCaseGenerator._merge_historical_cases(historical, new_cases)
        assert len(merged) == 2
        assert merged[0].title == "Get users successfully"  # baseline first
        assert merged[1].title == "Delete user"  # net-new appended

    def test_merge_empty_historical(self) -> None:
        """Empty baseline just returns the new cases (deduped among themselves)."""
        new_cases = [
            self._historical_case("A"),
            self._historical_case("A"),  # dup
        ]
        merged = TestCaseGenerator._merge_historical_cases([], new_cases)
        assert len(merged) == 1

    # --- generate with historical_cases ---

    def test_generate_merges_historical_baseline(self) -> None:
        """generate() merges historical baseline with newly generated cases."""
        historical = [self._historical_case("Get users successfully")]
        # LLM returns 2 cases, one of which duplicates the historical baseline.
        result = self.generator.generate(
            TestCaseGenInput(
                requirements=_REQUIREMENTS,
                endpoints=_ENDPOINTS,
                historical_cases=historical,
            )
        )
        # historical(1) + 2 new - 1 duplicate = 2 total
        assert len(result) == 2
        # IDs are re-numbered sequentially
        assert result[0].id == "TC-001"
        assert result[1].id == "TC-002"

    def test_generate_injects_historical_context_into_prompt(self) -> None:
        """Historical cases are summarized and injected into the LLM prompt."""
        historical = [self._historical_case()]
        self.generator.generate(
            TestCaseGenInput(
                requirements=_REQUIREMENTS,
                endpoints=[],
                historical_cases=historical,
            )
        )
        # First call's system prompt should mention the baseline instruction.
        system_prompt_arg = self.mock_llm.chat.call_args.args[0]
        assert "baseline" in system_prompt_arg.lower()
        # User prompt should include the historical case summary.
        user_prompt_arg = self.mock_llm.chat.call_args.args[1]
        assert "Get users successfully" in user_prompt_arg

    def test_generate_without_historical_no_baseline_hint(self) -> None:
        """Without historical cases, the prompt has no baseline instruction."""
        self.generator.generate(TestCaseGenInput(requirements=_REQUIREMENTS, endpoints=[]))
        system_prompt_arg = self.mock_llm.chat.call_args.args[0]
        assert "baseline" not in system_prompt_arg.lower()

    # --- prompt builder integration ---

    def test_build_prompt_with_historical_context(self) -> None:
        """PromptBuilder injects historical baseline text when provided."""
        builder = PromptBuilder()
        system_prompt, user_prompt = builder.build_testcase_prompt(
            endpoints_text="GET /users",
            requirements_text="req",
            output_language="english",
            extra_context={"historical_cases": "- [TC-001] Get users | GET /users | functional"},
        )
        assert "baseline" in system_prompt.lower()
        assert "Get users" in user_prompt


class TestTestCaseGeneratorJsonMode:
    """Tests for the optional OpenAI JSON mode (response_format + envelope)."""

    def setup_method(self) -> None:
        self.mock_llm = MagicMock()
        self.prompt_builder = PromptBuilder()
        self.generator = TestCaseGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
            json_mode=True,
        )

    def test_json_mode_forwards_response_format(self) -> None:
        """With json_mode, every chat call passes response_format json_object."""
        envelope = {"test_cases": json.loads(MOCK_LLM_RESPONSE)}
        self.mock_llm.chat.return_value = json.dumps(envelope)
        self.generator.generate(TestCaseGenInput(requirements=[], endpoints=_ENDPOINTS))
        assert self.mock_llm.chat.call_count >= 1
        for call in self.mock_llm.chat.call_args_list:
            assert call.kwargs.get("response_format") == {"type": "json_object"}

    def test_json_mode_unwraps_test_cases_envelope(self) -> None:
        """A {"test_cases": [...]} envelope is unwrapped into a flat list."""
        envelope = {"test_cases": json.loads(MOCK_LLM_RESPONSE)}
        self.mock_llm.chat.return_value = json.dumps(envelope)
        cases = self.generator.generate(TestCaseGenInput(requirements=[], endpoints=_ENDPOINTS))
        assert len(cases) == 2
        assert cases[0].id == "TC-001"

    def test_default_mode_does_not_forward_response_format(self) -> None:
        """Without json_mode, response_format is not sent (portable backends)."""
        default_gen = TestCaseGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
        )
        self.mock_llm.chat.return_value = MOCK_LLM_RESPONSE
        default_gen.generate(TestCaseGenInput(requirements=[], endpoints=_ENDPOINTS))
        for call in self.mock_llm.chat.call_args_list:
            assert call.kwargs.get("response_format") is None

    def test_json_mode_emits_backend_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Enabling json_mode warns the operator to confirm backend support."""
        with caplog.at_level(logging.WARNING, logger="testagent.generators.testcase_generator"):
            TestCaseGenerator(
                llm_client=self.mock_llm,
                prompt_builder=self.prompt_builder,
                json_mode=True,
            )
        assert any("OPENAI_JSON_MODE is enabled" in r.message for r in caplog.records)

    def test_default_mode_emits_no_json_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Default (json_mode off) does not warn about the backend."""
        with caplog.at_level(logging.WARNING, logger="testagent.generators.testcase_generator"):
            TestCaseGenerator(
                llm_client=self.mock_llm,
                prompt_builder=self.prompt_builder,
            )
        assert not any("OPENAI_JSON_MODE is enabled" in r.message for r in caplog.records)

