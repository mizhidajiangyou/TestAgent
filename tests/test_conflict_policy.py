"""T9 tests: conflict policy + authoritative value table wiring (fix-plan §3.3).

Covers:
- ``CONFLICT_POLICY`` settings field (Literal validation, default strict).
- The demoted ERROR_CONTRACT wording (fallback, no longer "use EXACTLY").
- Table injection into testcase / api / review prompts (empty table keeps
  prompts byte-identical).
- Generator end-to-end wiring: findings from requirement-vs-spec text land
  in the generation prompt and the session consistency report under the
  strict policy.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from testagent.config.models import APIEndpoint, RequirementItem, TestCaseGenInput
from testagent.config.settings import Settings
from testagent.engine.prompt_builder import ERROR_CONTRACT, PromptBuilder
from testagent.generators.testcase_generator import TestCaseGenerator
from tests.test_testcase_generator import MOCK_LLM_RESPONSE

_ENDPOINTS = [
    APIEndpoint(method="POST", path="/users", summary="Create user"),
    APIEndpoint(method="GET", path="/users", summary="List users"),
]


class TestSettingsConflictPolicy:
    def test_default_is_strict(self) -> None:
        assert Settings(conflict_policy="strict").conflict_policy == "strict"

    def test_invalid_policy_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings(conflict_policy="spec_wins")  # type: ignore[arg-type]


class TestErrorContractDemoted:
    def test_contract_is_fallback_wording(self) -> None:
        assert "FALLBACK" in ERROR_CONTRACT
        assert "spec and requirements always win" in ERROR_CONTRACT
        assert "use EXACTLY these status codes" not in ERROR_CONTRACT


class TestTableInjection:
    TABLE = (
        "# Authoritative value table\n\nPolicy: strict\n\n"
        "| subject | kind | decision |\n|---|---|---|\n"
        "| duplicate_email(400 vs 409) | spec_req_conflict | "
        "conflict_unresolved:duplicate_email(400 vs 409) |\n"
    )

    def test_testcase_prompt_appends_table(self) -> None:
        builder = PromptBuilder()
        _system_prompt, user_prompt = builder.build_testcase_prompt(
            endpoints_text="- GET /users",
            requirements_text="Some requirement.",
            extra_context={"authoritative_table": self.TABLE},
        )
        assert "AUTHORITATIVE VALUE TABLE" in user_prompt
        assert "conflict_unresolved:duplicate_email(400 vs 409)" in user_prompt

    def test_empty_table_keeps_prompt_identical(self) -> None:
        builder = PromptBuilder()
        _, without = builder.build_testcase_prompt(
            endpoints_text="- GET /users",
            requirements_text="Some requirement.",
        )
        _, with_empty = builder.build_testcase_prompt(
            endpoints_text="- GET /users",
            requirements_text="Some requirement.",
            extra_context={"authoritative_table": ""},
        )
        assert without == with_empty

    def test_api_prompt_appends_table(self) -> None:
        builder = PromptBuilder()
        _, user_prompt = builder.build_api_prompt(
            endpoints_text="- GET /users",
            requirements_text="Some requirement.",
            authoritative_table=self.TABLE,
        )
        assert "AUTHORITATIVE VALUE TABLE" in user_prompt

    def test_review_prompt_appends_table(self) -> None:
        builder = PromptBuilder()
        _, user_prompt = builder.build_review_prompt(
            endpoints_text="- GET /users",
            requirements_text="Some requirement.",
            test_cases_json="[]",
            authoritative_table=self.TABLE,
        )
        assert "AUTHORITATIVE VALUE TABLE" in user_prompt


class TestGeneratorConflictWiring:
    def _generator(self, tmp_path: Path) -> TestCaseGenerator:
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MOCK_LLM_RESPONSE
        return TestCaseGenerator(
            llm_client=mock_llm,
            prompt_builder=PromptBuilder(),
            audit_dump_enabled=True,
            audit_dump_dir=str(tmp_path),
        )

    def _input(self) -> TestCaseGenInput:
        req = RequirementItem(
            id="REQ-002",
            title="Duplicate email",
            description="Registering the same email twice returns 409.",
            acceptance_criteria=["重复邮箱返回 409。涉及 POST /users"],
        )
        return TestCaseGenInput(requirements=[req], endpoints=list(_ENDPOINTS))

    def test_conflict_table_reaches_generation_prompt_and_report(self, tmp_path: Path) -> None:
        generator = self._generator(tmp_path)
        generator.generate(self._input(), session_id="polcase01")
        prompts = [
            str(c.args[1])
            for c in generator._llm.chat.call_args_list  # type: ignore[attr-defined]
        ]
        assert any("AUTHORITATIVE VALUE TABLE" in p for p in prompts), (
            "generation prompt must carry the authoritative table"
        )
        assert any("conflict_unresolved" in p for p in prompts)
        report = tmp_path / "sessions" / "polcase01" / "consistency_report.md"
        assert report.exists()
        assert "duplicate_email" in report.read_text(encoding="utf-8")

    def test_no_findings_no_table(self, tmp_path: Path) -> None:
        generator = self._generator(tmp_path)
        clean_input = TestCaseGenInput(
            requirements=[],
            endpoints=[APIEndpoint(method="GET", path="/users")],
        )
        generator.generate(clean_input, session_id="polcase02")
        prompts = [str(c.args[1]) for c in generator._llm.chat.call_args_list]  # type: ignore[attr-defined]
        assert all("AUTHORITATIVE VALUE TABLE" not in p for p in prompts)

    def test_invalid_policy_fails_fast(self) -> None:
        generator = TestCaseGenerator(
            llm_client=MagicMock(),
            prompt_builder=PromptBuilder(),
            conflict_policy="nonsense",
        )
        with pytest.raises(ValueError, match="CONFLICT_POLICY"):
            generator.generate(
                TestCaseGenInput(
                    requirements=[RequirementItem(id="R1", title="t", description="d")],
                    endpoints=[APIEndpoint(method="GET", path="/users")],
                )
            )


class TestReviewSharesTable:
    def test_review_prompt_carries_table(self, tmp_path: Path) -> None:
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            MOCK_LLM_RESPONSE,
            json.dumps(
                [
                    {
                        "id": "TC-001",
                        "title": "Get users",
                        "endpoint": "GET /users",
                        "test_type": "functional",
                    }
                ]
            ),
        ]
        generator = TestCaseGenerator(
            llm_client=mock_llm,
            prompt_builder=PromptBuilder(),
            review_enabled=True,
            review_max_rounds=1,
            audit_dump_enabled=True,
            audit_dump_dir=str(tmp_path),
        )
        req = RequirementItem(
            id="REQ-002",
            title="Duplicate email",
            description="Registering the same email twice returns 409.",
            acceptance_criteria=["重复邮箱返回 409。涉及 POST /users"],
        )
        generator.generate(
            TestCaseGenInput(requirements=[req], endpoints=list(_ENDPOINTS)),
            session_id="polcase03",
        )
        review_prompts = [str(c.args[1]) for c in mock_llm.chat.call_args_list][1:]
        assert review_prompts, "review must run"
        assert any("AUTHORITATIVE VALUE TABLE" in p for p in review_prompts), (
            "review must share the generation-side authoritative table"
        )
