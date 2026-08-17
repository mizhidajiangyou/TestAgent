"""Tests for the conversational refinement engine (langgraph-inspired).

Covers ConversationSession and ConversationManager: generate/validate/refine/chat
actions, programmatic artifact validation, artifact versioning & lineage,
session lifecycle, and the JSON-extraction helper.
"""

import json
from unittest.mock import MagicMock

import pytest

from testagent.engine.conversation import (
    Artifact,
    ConversationManager,
    ConversationSession,
    _extract_json,
    _strip_code_fences,
)
from testagent.engine.prompt_builder import PromptBuilder

# A valid 2-case JSON array used as the LLM "generate" response.
_VALID_TEST_CASES = [
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
        "title": "Create user",
        "description": "Verify POST /users creates a user",
        "endpoint": "POST /users",
        "test_type": "functional",
        "priority": "high",
        "preconditions": ["User is authenticated"],
        "steps": ["Send POST request with valid body"],
        "expected_results": ["Status code is 201"],
    },
]

_VALID_TEST_CASES_JSON = json.dumps(_VALID_TEST_CASES, ensure_ascii=False)

# A valid (k6) performance script used as the LLM "generate" response.
_VALID_K6_SCRIPT = """import http from 'k6/http';
import { check, sleep } from 'k6';

export default function () {
  const res = http.get(`${__ENV.BASE_URL}/users`);
  check(res, { 'status is 200': (r) => r.status === 200 });
}
"""

# A valid Playwright GUI script used as the LLM "generate" response.
_VALID_GUI_SCRIPT = """import re
from playwright.sync_api import Page, expect


def test_login(page: Page) -> None:
    page.goto("https://example.com/login")
    page.get_by_label("Username").fill("admin")
    page.get_by_label("Password").fill("secret")
    page.get_by_role("button", name="Sign in").click()
    expect(page).to_have_url(re.compile(r"/dashboard"))
"""


class TestExtractJson:
    """Tests for the _extract_json helper."""

    def test_plain_json(self) -> None:
        assert _extract_json('[{"a": 1}]') == [{"a": 1}]

    def test_fenced_json(self) -> None:
        text = '```json\n[{"a": 1}]\n```'
        assert _extract_json(text) == [{"a": 1}]

    def test_json_with_prose(self) -> None:
        text = 'Here are the cases:\n[{"id": 1}]\nDone.'
        assert _extract_json(text) == [{"id": 1}]

    def test_object_extraction(self) -> None:
        text = 'prefix {"passed": true, "count": 5} suffix'
        result = _extract_json(text)
        assert isinstance(result, dict)
        assert result["passed"] is True
        assert result["count"] == 5

    def test_no_json_returns_none(self) -> None:
        assert _extract_json("no json here") is None

    def test_invalid_json_returns_none(self) -> None:
        assert _extract_json("[invalid") is None


class TestStripCodeFences:
    """Tests for the _strip_code_fences helper."""

    def test_strips_python_fence(self) -> None:
        text = "```python\nprint('hi')\n```"
        assert _strip_code_fences(text) == "print('hi')"

    def test_no_fence_unchanged(self) -> None:
        assert _strip_code_fences("plain code") == "plain code"


class TestConversationSessionGenerate:
    """Tests for the generate action."""

    def setup_method(self) -> None:
        self.mock_llm = MagicMock()
        self.prompt_builder = PromptBuilder()
        self.session = ConversationSession(
            session_id="s1",
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
            max_iterations=3,
        )

    def test_generate_test_cases_creates_artifact_v1(self) -> None:
        self.mock_llm.chat.return_value = _VALID_TEST_CASES_JSON
        response = self.session.send("generate test cases")

        artifacts = self.session.get_artifacts()
        assert len(artifacts) == 1
        art = artifacts[0]
        assert art.version == 1
        assert art.parent_id is None
        assert art.type == "test_cases"
        # Response summarizes the artifact and includes validation footer.
        assert "Artifact" in response
        assert "version=1" in response

    def test_generate_uses_context_endpoints_for_validation(self) -> None:
        self.mock_llm.chat.return_value = _VALID_TEST_CASES_JSON
        ctx = {"endpoints": "GET /users\nPOST /users", "requirements": "user mgmt"}
        self.session.send("generate test cases", context=ctx)

        # Validation should pass since endpoints match.
        state = self.session.get_state()
        # No feedback means validation passed.
        assert state.feedback == ""

    def test_generate_flags_endpoint_mismatch(self) -> None:
        self.mock_llm.chat.return_value = _VALID_TEST_CASES_JSON
        ctx = {"endpoints": "GET /products"}  # /users not in spec
        self.session.send("generate test cases", context=ctx)

        state = self.session.get_state()
        assert "GET /users" in state.feedback
        assert "POST /users" in state.feedback

    def test_generate_flags_missing_required_field(self) -> None:
        bad_cases = [{"id": "TC-001", "title": "x"}]  # missing steps/expected_results
        self.mock_llm.chat.return_value = json.dumps(bad_cases)
        self.session.send("generate test cases")

        state = self.session.get_state()
        assert "steps" in state.feedback
        assert "expected_results" in state.feedback

    def test_generate_flags_non_machine_checkable_expected(self) -> None:
        cases = [
            {
                "id": "TC-001",
                "title": "x",
                "steps": ["do something"],
                "expected_results": ["the system should work nicely"],  # not checkable
            }
        ]
        self.mock_llm.chat.return_value = json.dumps(cases)
        self.session.send("generate test cases")

        state = self.session.get_state()
        assert "machine-checkable" in state.feedback

    def test_generate_performance_script(self) -> None:
        self.mock_llm.chat.return_value = _VALID_K6_SCRIPT
        self.session.send("generate performance script")

        art = self.session.get_latest_artifact()
        assert art is not None
        assert art.type == "performance_script"
        assert "import http" in art.content

    def test_generate_gui_script(self) -> None:
        self.mock_llm.chat.return_value = _VALID_GUI_SCRIPT
        self.session.send("generate gui script")

        art = self.session.get_latest_artifact()
        assert art is not None
        assert art.type == "gui_script"
        assert "playwright" in art.content.lower()

    def test_generate_adds_two_messages_to_history(self) -> None:
        self.mock_llm.chat.return_value = _VALID_TEST_CASES_JSON
        self.session.send("generate test cases")

        history = self.session.get_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"


class TestConversationSessionRefine:
    """Tests for the refine action."""

    def setup_method(self) -> None:
        self.mock_llm = MagicMock()
        self.prompt_builder = PromptBuilder()
        self.session = ConversationSession(
            session_id="s1",
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
            max_iterations=2,
        )

    def test_refine_without_artifact_falls_back_to_generate(self) -> None:
        self.mock_llm.chat.return_value = _VALID_TEST_CASES_JSON
        response = self.session.send("refine the test cases")

        artifacts = self.session.get_artifacts()
        assert len(artifacts) == 1
        assert "Artifact" in response

    def test_refine_increments_version_and_links_parent(self) -> None:
        # Seed a v1 artifact via generate.
        self.mock_llm.chat.return_value = _VALID_TEST_CASES_JSON
        self.session.send("generate test cases")
        v1 = self.session.get_latest_artifact()
        assert v1 is not None

        # Refine: LLM returns improved cases.
        refined = json.dumps(_VALID_TEST_CASES * 3, ensure_ascii=False)
        self.mock_llm.chat.return_value = refined
        self.session.send("refine: add more cases")

        v2 = self.session.get_latest_artifact()
        assert v2 is not None
        assert v2.version == 2
        assert v2.parent_id == v1.id

    def test_refine_stops_when_validation_passes(self) -> None:
        self.mock_llm.chat.return_value = _VALID_TEST_CASES_JSON
        self.session.send("generate test cases")
        # v1 is valid, refine returns valid content -> should iterate once.
        self.mock_llm.chat.return_value = _VALID_TEST_CASES_JSON
        self.session.send("refine the cases")
        # Only one refine iteration since validation passes immediately.
        assert self.session.iteration == 1


class TestConversationSessionValidate:
    """Tests for the validate action."""

    def setup_method(self) -> None:
        self.mock_llm = MagicMock()
        self.prompt_builder = PromptBuilder()
        self.session = ConversationSession(
            session_id="s1",
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
        )

    def test_validate_with_no_artifact(self) -> None:
        response = self.session.send("validate")
        assert "No artifact" in response

    def test_validate_returns_programmatic_and_llm_feedback(self) -> None:
        self.mock_llm.chat.return_value = _VALID_TEST_CASES_JSON
        self.session.send("generate test cases")
        # LLM validate response.
        self.mock_llm.chat.return_value = json.dumps(
            {"passed": True, "issues": [], "suggestions": ["add more edge cases"]}
        )
        response = self.session.send("validate the cases")

        assert "PASS" in response or "passed: true" in response
        assert "add more edge cases" in response

    def test_validate_llm_response_not_json_returns_raw(self) -> None:
        self.mock_llm.chat.return_value = _VALID_TEST_CASES_JSON
        self.session.send("generate test cases")
        self.mock_llm.chat.return_value = "The cases look good to me."
        response = self.session.send("validate")
        assert "The cases look good" in response


class TestConversationSessionChat:
    """Tests for the chat action."""

    def setup_method(self) -> None:
        self.mock_llm = MagicMock()
        self.prompt_builder = PromptBuilder()
        self.session = ConversationSession(
            session_id="s1",
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
        )

    def test_chat_returns_llm_response(self) -> None:
        self.mock_llm.chat.return_value = "Sure, I can help with that."
        response = self.session.send("what is testing?")

        assert response == "Sure, I can help with that."
        state = self.session.get_state()
        assert state.status == "idle"

    def test_chat_with_artifacts_summary(self) -> None:
        self.mock_llm.chat.return_value = _VALID_TEST_CASES_JSON
        self.session.send("generate test cases")
        self.mock_llm.chat.return_value = "Here is my answer."
        response = self.session.send("explain the cases")
        assert response == "Here is my answer."


class TestActionInference:
    """Tests for action and artifact-type inference from natural language."""

    def test_infer_refine_keywords(self) -> None:
        assert ConversationSession._infer_action("refine the cases") == "refine"
        assert ConversationSession._infer_action("improve coverage") == "refine"
        assert ConversationSession._infer_action("优化用例") == "refine"

    def test_infer_generate_keywords(self) -> None:
        assert ConversationSession._infer_action("generate tests") == "generate"
        assert ConversationSession._infer_action("创建用例") == "generate"

    def test_infer_validate_keywords(self) -> None:
        assert ConversationSession._infer_action("validate this") == "validate"
        assert ConversationSession._infer_action("检查一下") == "validate"

    def test_infer_chat_default(self) -> None:
        assert ConversationSession._infer_action("hello there") == "chat"

    def test_infer_artifact_type_performance(self) -> None:
        assert (
            ConversationSession._infer_artifact_type("generate perf script") == "performance_script"
        )
        assert ConversationSession._infer_artifact_type("生成性能脚本") == "performance_script"

    def test_infer_artifact_type_gui(self) -> None:
        assert ConversationSession._infer_artifact_type("generate gui test") == "gui_script"
        assert ConversationSession._infer_artifact_type("界面测试") == "gui_script"

    def test_infer_artifact_type_default_test_cases(self) -> None:
        assert ConversationSession._infer_artifact_type("generate tests") == "test_cases"

    def test_explicit_action_in_context_overrides_inference(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat.return_value = _VALID_TEST_CASES_JSON
        session = ConversationSession(
            session_id="s1",
            llm_client=mock_llm,
            prompt_builder=PromptBuilder(),
        )
        # "hello" would infer "chat", but context forces "generate".
        session.send("hello", context={"action": "generate"})
        assert session.get_latest_artifact() is not None


class TestSessionStateAndProperties:
    """Tests for state snapshot, properties and seed artifacts."""

    def test_get_state_snapshot(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat.return_value = _VALID_TEST_CASES_JSON
        session = ConversationSession(
            session_id="abc",
            llm_client=mock_llm,
            prompt_builder=PromptBuilder(),
        )
        session.send("generate test cases")
        state = session.get_state()
        assert state.session_id == "abc"
        assert len(state.messages) == 2
        assert len(state.artifacts) == 1
        assert state.status == "completed"

    def test_session_id_and_iteration_properties(self) -> None:
        session = ConversationSession(
            session_id="xyz",
            llm_client=MagicMock(),
            prompt_builder=PromptBuilder(),
        )
        assert session.session_id == "xyz"
        assert session.iteration == 0

    def test_set_max_iterations(self) -> None:
        session = ConversationSession(
            session_id="s",
            llm_client=MagicMock(),
            prompt_builder=PromptBuilder(),
            max_iterations=3,
        )
        session.set_max_iterations(5)
        # Indirectly verify by triggering a refine loop; generate first.
        session._llm.chat.return_value = _VALID_TEST_CASES_JSON  # type: ignore[attr-defined]
        session.send("generate test cases")
        # Refine with valid content stops at iteration 1 regardless of max.
        session.send("refine")
        assert session.iteration >= 1

    def test_load_seed_artifacts_from_context(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat.return_value = json.dumps({"passed": True, "issues": [], "suggestions": []})
        session = ConversationSession(
            session_id="s",
            llm_client=mock_llm,
            prompt_builder=PromptBuilder(),
        )
        seed = [
            {
                "id": "seed-1",
                "type": "test_cases",
                "content": _VALID_TEST_CASES_JSON,
                "version": 1,
            }
        ]
        session.send("validate", context={"artifacts": seed})
        artifacts = session.get_artifacts()
        assert any(a.id == "seed-1" for a in artifacts)

    def test_get_latest_artifact_filtered_by_type(self) -> None:
        mock_llm = MagicMock()
        mock_llm.chat.return_value = _VALID_TEST_CASES_JSON
        session = ConversationSession(
            session_id="s",
            llm_client=mock_llm,
            prompt_builder=PromptBuilder(),
        )
        # Seed a test_cases and a performance_script artifact.
        session.send(
            "generate test cases",
            context={
                "artifacts": [
                    {
                        "id": "a1",
                        "type": "test_cases",
                        "content": _VALID_TEST_CASES_JSON,
                        "version": 1,
                    }
                ]
            },
        )
        latest_tc = session.get_latest_artifact("test_cases")
        latest_perf = session.get_latest_artifact("performance_script")
        assert latest_tc is not None
        assert latest_perf is None


class TestProgrammaticValidation:
    """Tests for _validate_artifact edge cases."""

    def setup_method(self) -> None:
        self.session = ConversationSession(
            session_id="s",
            llm_client=MagicMock(),
            prompt_builder=PromptBuilder(),
        )

    def test_validate_invalid_json_test_cases(self) -> None:
        art = Artifact(id="x", type="test_cases", content="not json", version=1, created_at="t")
        feedback = self.session._validate_artifact(art)
        assert "not valid JSON" in feedback

    def test_validate_non_array_test_cases(self) -> None:
        art = Artifact(id="x", type="test_cases", content='{"a": 1}', version=1, created_at="t")
        feedback = self.session._validate_artifact(art)
        assert "not a JSON array" in feedback

    def test_validate_empty_test_cases_array(self) -> None:
        art = Artifact(id="x", type="test_cases", content="[]", version=1, created_at="t")
        feedback = self.session._validate_artifact(art)
        assert "empty" in feedback

    def test_validate_jmx_missing_xml_declaration(self) -> None:
        art = Artifact(
            id="x",
            type="performance_script",
            content="<jmeterTestPlan></jmeterTestPlan>",
            version=1,
            created_at="t",
        )
        feedback = self.session._validate_artifact(art)
        assert "XML declaration" in feedback

    def test_validate_jmx_invalid_xml(self) -> None:
        art = Artifact(
            id="x",
            type="performance_script",
            content="<?xml version='1.0'?><jmeterTestPlan><unclosed>",
            version=1,
            created_at="t",
        )
        feedback = self.session._validate_artifact(art)
        assert "parse error" in feedback.lower() or "syntax" in feedback.lower()

    def test_validate_unbalanced_brackets(self) -> None:
        art = Artifact(
            id="x",
            type="performance_script",
            content="function() { return [1, 2; ",  # unbalanced
            version=1,
            created_at="t",
        )
        feedback = self.session._validate_artifact(art)
        assert "bracket" in feedback.lower()

    def test_validate_code_syntax_error(self) -> None:
        art = Artifact(
            id="x",
            type="code",
            content="def f(\n",  # syntax error
            version=1,
            created_at="t",
        )
        feedback = self.session._validate_artifact(art)
        assert "syntax" in feedback.lower()

    def test_validate_unknown_artifact_type(self) -> None:
        art = Artifact(id="x", type="unknown_type", content="x", version=1, created_at="t")
        feedback = self.session._validate_artifact(art)
        assert "Unknown artifact type" in feedback

    def test_validate_empty_script(self) -> None:
        art = Artifact(id="x", type="gui_script", content="   ", version=1, created_at="t")
        feedback = self.session._validate_artifact(art)
        assert "empty" in feedback

    def test_bracket_balance_handles_strings(self) -> None:
        # Brackets inside strings should not count. Include 'import http'
        # so the performance-script keyword check also passes.
        art = Artifact(
            id="x",
            type="performance_script",
            content='import http from "k6"; const s = "( { [ "; export default function() {}',
            version=1,
            created_at="t",
        )
        feedback = self.session._validate_artifact(art)
        assert feedback == ""


class TestConversationManager:
    """Tests for ConversationManager session lifecycle."""

    def setup_method(self) -> None:
        self.mock_llm = MagicMock()
        self.manager = ConversationManager(
            llm_client=self.mock_llm,
            prompt_builder=PromptBuilder(),
            max_iterations=2,
        )

    def test_create_session_auto_id(self) -> None:
        session = self.manager.create_session()
        assert session.session_id
        assert session.session_id in self.manager.list_sessions()

    def test_create_session_explicit_id(self) -> None:
        session = self.manager.create_session("my-session")
        assert session.session_id == "my-session"

    def test_create_session_duplicate_raises(self) -> None:
        self.manager.create_session("dup")
        with pytest.raises(ValueError, match="already exists"):
            self.manager.create_session("dup")

    def test_get_session_unknown_returns_none(self) -> None:
        assert self.manager.get_session("nope") is None

    def test_list_sessions(self) -> None:
        self.manager.create_session("a")
        self.manager.create_session("b")
        assert set(self.manager.list_sessions()) >= {"a", "b"}

    def test_close_session(self) -> None:
        self.manager.create_session("temp")
        self.manager.close_session("temp")
        assert "temp" not in self.manager.list_sessions()

    def test_close_unknown_session_warns_no_raise(self) -> None:
        # Should not raise even if session doesn't exist.
        self.manager.close_session("ghost")

    def test_session_persists_state_across_turns(self) -> None:
        self.mock_llm.chat.return_value = _VALID_TEST_CASES_JSON
        session = self.manager.create_session("persist")
        session.send("generate test cases")
        # Retrieve from manager and send another message.
        retrieved = self.manager.get_session("persist")
        assert retrieved is not None
        assert len(retrieved.get_artifacts()) == 1
        retrieved.send("validate")
        assert len(retrieved.get_history()) == 4  # 2 per turn


class TestConversationGuiRouting:
    """Tests for gui_script routing through build_gui_test_prompt + URL."""

    def setup_method(self) -> None:
        self.mock_llm = MagicMock()
        self.prompt_builder = PromptBuilder()
        self.session = ConversationSession(
            session_id="gui1",
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
            max_iterations=3,
        )

    def test_gui_script_prompt_includes_target_url(self) -> None:
        """gui_script generation routes through build_gui_test_prompt (Playwright)."""
        self.mock_llm.chat.return_value = _VALID_GUI_SCRIPT
        self.session.send("generate gui script")
        call_args = self.mock_llm.chat.call_args
        system_arg = call_args.args[0]
        user_arg = call_args.args[1]
        assert "Playwright" in system_arg
        # Default target URL should be embedded in the generated (user) prompt.
        assert "example.com" in user_arg

    def test_gui_script_respects_gui_url_context(self) -> None:
        """A caller-supplied gui_url is honored by the GUI prompt."""
        self.mock_llm.chat.return_value = _VALID_GUI_SCRIPT
        self.session.send(
            "generate gui script", context={"gui_url": "https://app.example.com"}
        )
        user_arg = self.mock_llm.chat.call_args.args[1]
        assert "app.example.com" in user_arg

    def test_gui_script_validation_flags_syntax_error(self) -> None:
        """gui_script artifacts get a real Python syntax check (new behavior)."""
        self.mock_llm.chat.return_value = "def broken(:\n    pass\n"  # invalid Python
        self.session.send("generate gui script")
        state = self.session.get_state()
        assert "syntax error" in state.feedback.lower()

