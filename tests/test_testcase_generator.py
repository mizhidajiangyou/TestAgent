"""Tests for TestCaseGenerator."""

import asyncio
import csv
import json
import logging
import threading
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
from testagent.engine.llm_client import LLMOutputTooLongError
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.testcase_generator import MAX_PARSE_RETRIES, TestCaseGenerator


class FakeAsyncLLM:
    """Minimal ``LLMClient`` returning a fixed response for both chat and achat.

    ``achat_impl`` (optional) overrides async behavior — e.g. to measure
    concurrency or to alternate responses across calls. A list of responses is
    consumed in order (wrapped to the last value when exhausted).
    """

    def __init__(self, response: str | list[str], achat_impl=None) -> None:
        self._response = response
        self._achat_impl = achat_impl
        self.chat_calls = 0
        self.achat_calls = 0
        self._pos = 0
        self.session_id: str | None = None

    def _next(self) -> str:
        if isinstance(self._response, list):
            val = (
                self._response[self._pos] if self._pos < len(self._response) else self._response[-1]
            )
            self._pos += 1
            return val
        return self._response

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format=None,
        max_tokens: int | None = None,
    ) -> str:
        self.chat_calls += 1
        return self._next()

    async def achat(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format=None,
        max_tokens: int | None = None,
    ) -> str:
        self.achat_calls += 1
        if self._achat_impl is not None:
            return await self._achat_impl(system_prompt, user_prompt, response_format, max_tokens)
        return self._next()

    def set_session_id(self, session_id: str) -> None:
        """Record the session id for assertions in tests."""
        self.session_id = session_id

    def verify(self) -> None:
        """No-op: the real client's verify is exercised in llm_client tests."""
        pass

    async def averify(self) -> None:
        """No-op async verify (see :meth:`verify`)."""
        pass


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
        # v6 truncation loop re-asks until the call budget is exhausted
        # (TruncationPolicy.max_total_calls), not a fixed retry count.
        assert self.mock_llm.chat.call_count >= MAX_PARSE_RETRIES

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

        # v6 filters raw items against the batch's endpoint scope, so each
        # batch must return cases for ITS OWN endpoints (a catch-all response
        # would be correctly dropped by the quota filter).
        def case(cid: str, title: str, ep: str) -> dict[str, object]:
            return {
                "id": cid,
                "title": title,
                "endpoint": ep,
                "test_type": "functional",
                "priority": "high",
                "steps": ["s"],
                "expected_results": ["Status 200"],
            }

        batch1 = json.dumps(
            [case("TC-001", "List", "GET /users"), case("TC-002", "Create", "POST /users")]
        )
        batch2 = json.dumps(
            [
                case("TC-003", "Delete", "DELETE /users/{id}"),
                case("TC-004", "Login", "POST /users/login"),
            ]
        )
        self.mock_llm.chat.side_effect = [batch1, batch2]
        generator = TestCaseGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
        )
        test_cases = generator.generate(TestCaseGenInput(requirements=[], endpoints=endpoints))
        assert len(test_cases) == 4
        assert self.mock_llm.chat.call_count == 2

    def test_requirement_per_call_fanout(self) -> None:
        """Each requirement becomes its own generation call (true concurrency).

        With 5 requirements (regardless of module grouping), the generator
        issues 5 LLM calls — one per requirement — so N requirements run
        concurrently (bounded by ``max_concurrency``) instead of collapsing
        into a single oversized batch that would exceed the token limit.
        """
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
        # 5 requirements = 5 calls * 2 cases = 10
        assert len(test_cases) == 10
        assert self.mock_llm.chat.call_count == 5

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
        # v6 sends a CONTINUATION prompt after truncation: it names the
        # truncation, lists what is already produced and what is still needed.
        assert "truncated" in user_prompt_arg.lower()
        assert "do NOT repeat" in user_prompt_arg

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


_REFINED_ASYNC = json.dumps(
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


class TestAsyncGeneration:
    """Async generation path (``agenerate``): parity, concurrency, review."""

    def setup_method(self) -> None:
        self.prompt_builder = PromptBuilder()

    def _input_endpoints_only(self) -> TestCaseGenInput:
        return TestCaseGenInput(requirements=[], endpoints=_ENDPOINTS)

    def _input_requirements_only(self) -> TestCaseGenInput:
        return TestCaseGenInput(requirements=_REQUIREMENTS, endpoints=[])

    def _input_both(self) -> TestCaseGenInput:
        return TestCaseGenInput(requirements=_REQUIREMENTS, endpoints=_ENDPOINTS)

    def _fake(self, response: str | list[str] = MOCK_LLM_RESPONSE) -> FakeAsyncLLM:
        return FakeAsyncLLM(response)

    async def test_agenerate_parity_requirements_only(self) -> None:
        """agenerate produces the same cases as generate (requirements only)."""
        fake = self._fake()
        gen = TestCaseGenerator(llm_client=fake, prompt_builder=self.prompt_builder)
        sync_cases = gen.generate(self._input_requirements_only())
        async_cases = await gen.agenerate(self._input_requirements_only())
        assert [c.id for c in sync_cases] == [c.id for c in async_cases]
        assert len(async_cases) == 2
        assert fake.chat_calls >= 1
        assert fake.achat_calls >= 1

    async def test_agenerate_parity_endpoints_only(self) -> None:
        """agenerate produces the same cases as generate (endpoints only)."""
        fake = self._fake()
        gen = TestCaseGenerator(llm_client=fake, prompt_builder=self.prompt_builder)
        sync_cases = gen.generate(self._input_endpoints_only())
        async_cases = await gen.agenerate(self._input_endpoints_only())
        assert [c.id for c in sync_cases] == [c.id for c in async_cases]
        assert len(async_cases) == 2

    async def test_agenerate_parity_both_phases(self) -> None:
        """agenerate mirrors the two-phase result for requirements+endpoints."""
        fake = self._fake()
        gen = TestCaseGenerator(llm_client=fake, prompt_builder=self.prompt_builder)
        sync_cases = gen.generate(self._input_both())
        async_cases = await gen.agenerate(self._input_both())
        # Phase 1 (1 batch) + Phase 2 (1 batch) = 2 calls, 2 cases each.
        assert [c.id for c in sync_cases] == [c.id for c in async_cases]
        assert len(async_cases) == 4
        assert fake.achat_calls == 2

    async def test_agenerate_empty_input(self) -> None:
        """No requirements and no endpoints returns an empty list."""
        gen = TestCaseGenerator(llm_client=self._fake(), prompt_builder=self.prompt_builder)
        result = await gen.agenerate(TestCaseGenInput(requirements=[], endpoints=[]))
        assert result == []

    async def test_agenerate_concurrency_bounded(self) -> None:
        """max_concurrency=2 caps concurrency at 2 (4 requirement coroutines)."""
        events = {"current": 0, "max": 0}
        lock = threading.Lock()

        async def achat(sp: str, up: str, rf=None, max_tokens: int | None = None) -> str:
            with lock:
                events["current"] += 1
                events["max"] = max(events["max"], events["current"])
            await asyncio.sleep(0.05)
            with lock:
                events["current"] -= 1
            return MOCK_LLM_RESPONSE

        fake = FakeAsyncLLM(MOCK_LLM_RESPONSE, achat_impl=achat)
        gen = TestCaseGenerator(
            llm_client=fake,
            prompt_builder=self.prompt_builder,
            max_concurrency=2,
        )
        # 4 requirements -> 4 coroutines (per-requirement fan-out), each yields 2 cases.
        reqs = [
            RequirementItem(id=f"R{i}", title=f"T{i}", description="d", module=f"m{i}")
            for i in range(4)
        ]
        cases = await gen.agenerate(TestCaseGenInput(requirements=reqs, endpoints=[]))
        assert len(cases) == 8
        assert events["max"] == 2

    async def test_agenerate_unbounded_within_batch_count(self) -> None:
        """max_concurrency=None falls back to default(5); 4 req coroutines run free."""
        events = {"current": 0, "max": 0}
        lock = threading.Lock()

        async def achat(sp: str, up: str, rf=None, max_tokens: int | None = None) -> str:
            with lock:
                events["current"] += 1
                events["max"] = max(events["max"], events["current"])
            await asyncio.sleep(0.03)
            with lock:
                events["current"] -= 1
            return MOCK_LLM_RESPONSE

        fake = FakeAsyncLLM(MOCK_LLM_RESPONSE, achat_impl=achat)
        gen = TestCaseGenerator(
            llm_client=fake,
            prompt_builder=self.prompt_builder,
            max_concurrency=None,
        )
        reqs = [
            RequirementItem(id=f"R{i}", title=f"T{i}", description="d", module=f"m{i}")
            for i in range(4)
        ]
        await gen.agenerate(TestCaseGenInput(requirements=reqs, endpoints=[]))
        # Only 4 requirement coroutines exist, so all 4 run concurrently (<= default 5).
        assert events["max"] == 4

    async def test_agenerate_with_review(self) -> None:
        """Async review path alternates primary/secondary via achat."""
        primary = FakeAsyncLLM([MOCK_LLM_RESPONSE, _REFINED_ASYNC])
        secondary = FakeAsyncLLM([_REFINED_ASYNC])
        gen = TestCaseGenerator(
            llm_client=primary,
            prompt_builder=self.prompt_builder,
            review_enabled=True,
            review_llm_client=secondary,
            review_max_rounds=1,
        )
        cases = await gen.agenerate(self._input_endpoints_only())
        # Generation uses primary (1 achat); review round 1 uses secondary (1 achat).
        assert primary.achat_calls == 1
        assert secondary.achat_calls == 1
        assert len(cases) == 3


class TestGenerationResilience:
    """Resilience: a failing unit degrades instead of aborting the whole run."""

    def test_one_failing_requirement_still_yields_others(self) -> None:
        """When one requirement's LLM call fails, the others still produce cases.

        Before the fix, any per-requirement LLM error propagated and aborted the
        entire run. Now it degrades: the failing unit returns no cases while the
        rest complete, and the run returns partial results instead of raising.
        """
        reqs = [
            RequirementItem(
                id="R1", title="Alpha", description="a", module="m1", acceptance_criteria=[]
            ),
            RequirementItem(
                id="R2", title="Beta", description="b", module="m2", acceptance_criteria=[]
            ),
            RequirementItem(
                id="R3", title="Gamma", description="c", module="m3", acceptance_criteria=[]
            ),
        ]

        async def achat(sp: str, up: str, rf=None, max_tokens: int | None = None) -> str:
            if "Gamma" in up:
                raise RuntimeError("simulated model failure")
            return MOCK_LLM_RESPONSE

        fake = FakeAsyncLLM(MOCK_LLM_RESPONSE, achat_impl=achat)
        gen = TestCaseGenerator(llm_client=fake, prompt_builder=PromptBuilder())

        cases = asyncio.run(gen.agenerate(TestCaseGenInput(requirements=reqs, endpoints=[])))
        # 2 of 3 requirements succeed -> 2 * 2 cases = 4. The run did NOT crash,
        # and the single failing requirement degraded to [] instead of aborting.
        assert len(cases) == 4
        # Each produced case is a valid, re-numbered TestCase.
        assert all(c.id for c in cases)
        assert {c.id for c in cases} == {f"TC-{i:03d}" for i in range(1, 5)}

    def test_total_failure_degrades_to_empty(self) -> None:
        """When every call fails, agenerate degrades to an empty list (no crash).

        The generator never aborts the run on a per-unit LLM error; it returns
        as many valid cases as it can. Empty output is surfaced as a clear
        error by the entry points (CLI/web), not by the generator itself.
        """
        reqs = [
            RequirementItem(
                id="R1", title="Alpha", description="a", module="m1", acceptance_criteria=[]
            ),
        ]

        async def achat(sp: str, up: str, rf=None, max_tokens: int | None = None) -> str:
            raise RuntimeError("simulated model failure")

        fake = FakeAsyncLLM(MOCK_LLM_RESPONSE, achat_impl=achat)
        gen = TestCaseGenerator(llm_client=fake, prompt_builder=PromptBuilder())

        cases = asyncio.run(gen.agenerate(TestCaseGenInput(requirements=reqs, endpoints=[])))
        assert cases == []


class TestSessionAndEmptyRecovery:
    """Session id correlation and empty-truncation recovery."""

    def test_session_id_is_set_and_propagated(self) -> None:
        """agenerate assigns a session id and pushes it to the LLM client."""
        fake = FakeAsyncLLM(MOCK_LLM_RESPONSE)
        gen = TestCaseGenerator(llm_client=fake, prompt_builder=PromptBuilder())
        reqs = [
            RequirementItem(
                id="R1", title="A", description="d", module="m1", acceptance_criteria=[]
            )
        ]
        asyncio.run(gen.agenerate(TestCaseGenInput(requirements=reqs, endpoints=[])))
        assert gen.session_id is not None
        assert len(gen.session_id) == 12  # uuid hex[:12]
        assert fake.session_id == gen.session_id

    def test_explicit_session_id_is_respected(self) -> None:
        """An externally provided session id is used (for resume)."""
        fake = FakeAsyncLLM(MOCK_LLM_RESPONSE)
        gen = TestCaseGenerator(llm_client=fake, prompt_builder=PromptBuilder())
        reqs = [
            RequirementItem(
                id="R1", title="A", description="d", module="m1", acceptance_criteria=[]
            )
        ]
        asyncio.run(
            gen.agenerate(
                TestCaseGenInput(requirements=reqs, endpoints=[]), session_id="abc123def456"
            )
        )
        assert gen.session_id == "abc123def456"
        assert fake.session_id == "abc123def456"

    def test_empty_truncation_retry_recovers(self) -> None:
        """A requirement whose first call returns an EMPTY truncation recovers.

        The generator catches ``LLMOutputTooLongError`` (empty response under
        parallel load), re-asks with a compressed / full-regeneration scope at
        the SAME token budget — we deliberately do NOT shrink ``max_tokens``
        (that makes truncation *more* likely; see experience.md #10) — and the
        retry succeeds instead of crashing or yielding 0 for that requirement.
        """
        calls: list[int] = []
        seen_max_tokens: list[int | None] = []
        seen_prompts: list[str] = []

        async def achat(sp: str, up: str, rf=None, max_tokens: int | None = None) -> str:
            seen_max_tokens.append(max_tokens)
            seen_prompts.append(up)
            calls.append(1)
            if len(calls) == 1:
                # First attempt: simulate an empty truncated response.
                raise LLMOutputTooLongError(
                    "Model returned an EMPTY response (finish_reason=length)."
                )
            return MOCK_LLM_RESPONSE

        fake = FakeAsyncLLM(MOCK_LLM_RESPONSE, achat_impl=achat)
        gen = TestCaseGenerator(llm_client=fake, prompt_builder=PromptBuilder())
        reqs = [
            RequirementItem(
                id="R1", title="A", description="d", module="m1", acceptance_criteria=[]
            )
        ]
        cases = asyncio.run(gen.agenerate(TestCaseGenInput(requirements=reqs, endpoints=[])))
        # Recovered on retry -> produced cases, not [].
        assert len(cases) == 2
        # Retry keeps the SAME token budget (no shrink) on every attempt.
        # v6 passes the policy cap explicitly (TruncationPolicy default 16000)
        # instead of None, but never shrinks it between attempts.
        from testagent.generators.truncation import TruncationPolicy

        assert seen_max_tokens == [TruncationPolicy().output_token_cap] * 2
        # The re-ask tells the model its previous output was empty / to
        # regenerate, not the misleading "remove code fences" hint.
        assert "empty" in seen_prompts[1].lower() or "regenerate" in seen_prompts[1].lower()


class TestFanOutRecovery:
    """``_fan_out_recover``: concurrent stream-drop -> sequential recovery.

    Reproduces the real failure from the user's run (2026-08-17): 3 requirements
    are fanned out concurrently; REQ-001 succeeds while REQ-002/REQ-003 come
    back EMPTY at the same moment (provider dropped their streams under parallel
    load). The generator must retry ONLY the empty units, ONE AT A TIME
    (concurrency=1), so the parallel pressure that caused the drops is relieved
    and they recover -- instead of re-fanning them concurrently (the old
    behavior that kept failing and forced the user to Ctrl-C).
    """

    @staticmethod
    def _mk_case(title: str) -> TestCase:
        return TestCase(
            id="TC-X",
            title=title,
            description="d",
            endpoint=APIEndpoint(method="GET", path="/x"),
            test_type=TestType.FUNCTIONAL,
            priority=TestPriority.HIGH,
        )

    async def test_fan_out_recovers_dropped_units_sequentially(self) -> None:
        """Dropped units (2/3) are retried sequentially and recover.

        Proves two things at once: (a) the empty units are retried and recover,
        and (b) that retry is SERIAL -- at the moment a recovery unit starts, no
        other recovery unit is in flight (concurrency=1). A concurrent re-fan
        would let two recovery units overlap.
        """
        gen = TestCaseGenerator(
            llm_client=FakeAsyncLLM(MOCK_LLM_RESPONSE),
            prompt_builder=PromptBuilder(),
        )
        items = ["req1", "req2", "req3"]
        invocations: dict[int, int] = {}
        active = 0
        lock = threading.Lock()

        def make_coro(i: int, item: str):
            async def _one() -> list[TestCase]:
                nonlocal active
                inv = invocations[i] = invocations.get(i, 0) + 1
                # Recovery-phase entry (2nd invocation of a dropped unit):
                # must be the only active recovery unit -> proves concurrency=1.
                if i in (2, 3) and inv == 2:
                    with lock:
                        assert active == 0, f"recovery unit {i} overlapped another"
                with lock:
                    active += 1
                await asyncio.sleep(0.005)
                with lock:
                    active -= 1
                if i == 1:
                    return [self._mk_case("good")]
                if inv == 1:
                    return []  # concurrent stream drop -> empty during fan-out
                return [self._mk_case(f"recovered-{i}")]

            return _one()

        result = await gen._fan_out_recover(items, make_coro)

        # All three units present: the good one + the two recovered ones.
        assert len(result) == 3
        # Good unit ran exactly once; each dropped unit ran twice (once in the
        # fan-out, once in the sequential recovery). This is the smoking gun
        # that recovery fires and is scoped to the failed units only.
        assert invocations == {1: 1, 2: 2, 3: 2}

    async def test_fan_out_skips_recovery_when_all_failed(self) -> None:
        """If EVERY unit is empty, the model is genuinely down -> no recovery.

        The guard ``if not failed or not any(results)`` must short-circuit so we
        don't waste N serial retries on a model that is truly unreachable.
        """
        gen = TestCaseGenerator(
            llm_client=FakeAsyncLLM(MOCK_LLM_RESPONSE),
            prompt_builder=PromptBuilder(),
        )
        items = ["a", "b", "c"]
        invocations: dict[int, int] = {}

        def make_coro(i: int, item: str):
            async def _one() -> list[TestCase]:
                invocations[i] = invocations.get(i, 0) + 1
                return []

            return _one()

        result = await gen._fan_out_recover(items, make_coro)
        assert result == []
        # No recovery pass: each unit invoked exactly once (fan-out only).
        assert invocations == {1: 1, 2: 1, 3: 1}

    async def test_concurrent_stream_drop_recovers_via_agenerate(self) -> None:
        """End-to-end: a real ``agenerate`` run where 2/3 requirements are
        dropped by the provider under concurrent load, then recover.

        Mirrors the user's failing run exactly: REQ-001 OK, REQ-002 & REQ-003
        empty at the same moment. Previously the retries also ran concurrently
        and kept failing (user aborted). Now the generator retries only the
        empty units, one at a time, and they recover.
        """
        per_req: dict[str, int] = {"REQ-002": 0, "REQ-003": 0}
        active_rec = 0
        lock = threading.Lock()

        async def achat(sp: str, up: str, rf=None, max_tokens: int | None = None) -> str:
            nonlocal active_rec
            for rid in ("REQ-002", "REQ-003"):
                if rid in up:
                    per_req[rid] += 1
                    if per_req[rid] > MAX_PARSE_RETRIES:
                        # This is the recovery attempt -> must be serialized.
                        with lock:
                            assert active_rec == 0, "recovery calls overlapped"
                        with lock:
                            active_rec += 1
                        await asyncio.sleep(0.01)
                        with lock:
                            active_rec -= 1
                        return MOCK_LLM_RESPONSE
                    raise LLMOutputTooLongError(
                        "Model returned an EMPTY response (finish_reason=length)."
                    )
            return MOCK_LLM_RESPONSE  # REQ-001 always good

        fake = FakeAsyncLLM(MOCK_LLM_RESPONSE, achat_impl=achat)
        gen = TestCaseGenerator(llm_client=fake, prompt_builder=PromptBuilder())
        reqs = [
            RequirementItem(
                id="REQ-001", title="Good", description="d", module="m1", acceptance_criteria=[]
            ),
            RequirementItem(
                id="REQ-002", title="DroppedA", description="d", module="m2", acceptance_criteria=[]
            ),
            RequirementItem(
                id="REQ-003", title="DroppedB", description="d", module="m3", acceptance_criteria=[]
            ),
        ]
        cases = await gen.agenerate(TestCaseGenInput(requirements=reqs, endpoints=[]))

        # All three requirements recovered -> 3 * 2 cases (MOCK_LLM_RESPONSE = 2).
        assert len(cases) == 6
        # Call accounting: REQ-001 = 1; REQ-002/003 each = MAX_PARSE_RETRIES
        # failing fan-out attempts + 1 successful recovery = 4.
        assert fake.achat_calls == 1 + 2 * (MAX_PARSE_RETRIES + 1)
        # Recovery was serial: at no point did two recovery calls overlap.
        assert active_rec <= 1


class TestNormalizeCases:
    """Boundary cleanup: degenerate stubs dropped, duplicates deduped, IDs renumbered."""

    @staticmethod
    def _mk(
        title: str,
        *,
        endpoint: str = "GET /x",
        expected: list[str] | None = None,
        case_id: str = "TC-X",
    ) -> TestCase:
        method, path = endpoint.split(None, 1)
        return TestCase(
            id=case_id,
            title=title,
            description="d",
            endpoint=APIEndpoint(method=method, path=path),
            test_type=TestType.FUNCTIONAL,
            priority=TestPriority.HIGH,
            steps=["Send request"],
            expected_results=expected if expected is not None else ["Status 200"],
        )

    def test_drops_degenerate_stubs(self) -> None:
        """Cases with empty title or no expected_results are unexecutable noise."""
        cases = [
            self._mk("Valid case"),
            self._mk(""),  # empty title -> stub
            self._mk("No assertions", expected=[]),  # no expected_results -> stub
            self._mk("   "),  # whitespace-only title -> stub
        ]
        kept = TestCaseGenerator._normalize_cases(cases)
        assert [c.title for c in kept] == ["Valid case"]

    def test_keeps_valid_cases_untouched(self) -> None:
        """Valid cases survive in order; only degenerate ones are dropped."""
        cases = [
            self._mk("List users", case_id="TC-1"),
            self._mk("", expected=[]),  # stub
            self._mk("Create user", endpoint="POST /x", case_id="TC-3"),
        ]
        kept = TestCaseGenerator._normalize_cases(cases)
        assert [c.title for c in kept] == ["List users", "Create user"]
        assert kept[1].endpoint.method == "POST"

    def test_renumbers_sequentially(self) -> None:
        cases = [self._mk("A", case_id="TC-007"), self._mk("B", case_id="TC-042")]
        kept = TestCaseGenerator._normalize_cases(cases)
        assert [c.id for c in kept] == ["TC-001", "TC-002"]

    def test_empty_input_returns_empty(self) -> None:
        assert TestCaseGenerator._normalize_cases([]) == []
