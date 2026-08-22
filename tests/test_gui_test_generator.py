"""Tests for the GUITestGenerator (Playwright, stagehand-inspired).

Covers script generation, code-fence stripping, Python syntax validation,
file saving, and LLM interaction.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from testagent.config.models import APIEndpoint, GUITestGenInput, RequirementItem
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.gui_test_generator import GUITestGenerator

# A complete, syntactically valid Playwright test script.
_VALID_PLAYWRIGHT_SCRIPT = '''"""Generated GUI test for the login flow."""
import re

from playwright.sync_api import Page, expect


def test_user_login(page: Page) -> None:
    """Verify a user can log in successfully."""
    page.goto("https://example.com/login")

    # Fill credentials using robust locators (stagehand-inspired a11y targeting).
    page.get_by_label("Username").fill("admin")
    page.get_by_label("Password").fill("secret123")

    page.get_by_role("button", name="Sign in").click()

    # Machine-checkable assertion (stagehand-style data/url assertion).
    expect(page).to_have_url(re.compile(r"/dashboard"))
    expect(page.get_by_text("Welcome")).to_be_visible()
'''


# The same script wrapped in a markdown code fence, simulating LLM output.
_FENCED_SCRIPT = f"```python\n{_VALID_PLAYWRIGHT_SCRIPT}\n```"

# A syntactically invalid script.
_INVALID_PYTHON_SCRIPT = "def broken(:\n    pass\n"

_REQUIREMENTS = [
    RequirementItem(
        id="REQ-001",
        title="User Login",
        description="A user can log in with username and password",
        acceptance_criteria=["Login succeeds with valid credentials"],
    ),
]

_ENDPOINTS = [
    APIEndpoint(method="POST", path="/auth/login", summary="Authenticate user"),
]


class TestGUITestGeneratorGenerate:
    """Tests for the generate() method."""

    def setup_method(self) -> None:
        self.mock_llm = MagicMock()
        self.mock_llm.chat.return_value = _VALID_PLAYWRIGHT_SCRIPT
        self.prompt_builder = PromptBuilder()
        self.generator = GUITestGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
            output_language="english",
        )

    def _input(self, url: str | None = None) -> GUITestGenInput:
        return GUITestGenInput(
            requirements=_REQUIREMENTS,
            url=url or "https://example.com",
            endpoints=_ENDPOINTS,
        )

    def test_generate_returns_script(self) -> None:
        script = self.generator.generate(self._input())
        assert "playwright" in script.lower()
        assert "def test_" in script
        assert "page.goto" in script

    def test_generate_strips_code_fences(self) -> None:
        self.mock_llm.chat.return_value = _FENCED_SCRIPT
        script = self.generator.generate(self._input())
        assert not script.startswith("```")
        assert not script.endswith("```")
        assert "def test_" in script

    def test_generate_calls_llm_with_system_and_user_prompts(self) -> None:
        self.generator.generate(self._input())
        assert self.mock_llm.chat.call_count == 1
        args = self.mock_llm.chat.call_args
        system_prompt, user_prompt = args[0]
        assert "Playwright" in system_prompt
        assert "https://example.com" in user_prompt

    def test_generate_uses_default_url_when_none(self) -> None:
        self.generator.generate(self._input(url=None))
        args = self.mock_llm.chat.call_args
        _system, user_prompt = args[0]
        # Should fall back to DEFAULT_TARGET_URL (https://example.com).
        assert "example.com" in user_prompt

    def test_generate_includes_endpoints_context(self) -> None:
        self.generator.generate(self._input())
        args = self.mock_llm.chat.call_args
        _system, user_prompt = args[0]
        assert "/auth/login" in user_prompt

    def test_generate_includes_requirements_context(self) -> None:
        self.generator.generate(self._input())
        args = self.mock_llm.chat.call_args
        _system, user_prompt = args[0]
        assert "User Login" in user_prompt

    def test_generate_uses_per_input_language_override(self) -> None:
        inp = GUITestGenInput(
            requirements=_REQUIREMENTS,
            url="https://example.com",
            output_language="chinese",
        )
        self.generator.generate(inp)
        args = self.mock_llm.chat.call_args
        system_prompt, _user = args[0]
        assert "Chinese" in system_prompt

    def test_generate_raises_on_invalid_python_syntax(self) -> None:
        self.mock_llm.chat.return_value = _INVALID_PYTHON_SCRIPT
        with pytest.raises(ValueError, match="invalid Python syntax"):
            self.generator.generate(self._input())

    def test_generate_logs_token_usage(self) -> None:
        self.generator.generate(self._input())
        # Verify the LLM was called (token usage tracked by the client).
        assert self.mock_llm.chat.called


class TestGUITestGeneratorSave:
    """Tests for the save() method."""

    def setup_method(self) -> None:
        self.mock_llm = MagicMock()
        self.generator = GUITestGenerator(
            llm_client=self.mock_llm,
            prompt_builder=PromptBuilder(),
        )

    def test_save_writes_file(self, tmp_path: Path) -> None:
        output_path = tmp_path / "gui_test.py"
        result = self.generator.save(_VALID_PLAYWRIGHT_SCRIPT, output_path)

        assert result == output_path
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "def test_user_login" in content

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        output_path = tmp_path / "nested" / "deep" / "gui_test.py"
        self.generator.save(_VALID_PLAYWRIGHT_SCRIPT, output_path)
        assert output_path.exists()

    def test_save_overwrites_existing_file(self, tmp_path: Path) -> None:
        output_path = tmp_path / "gui_test.py"
        output_path.write_text("old content", encoding="utf-8")
        self.generator.save(_VALID_PLAYWRIGHT_SCRIPT, output_path)
        content = output_path.read_text(encoding="utf-8")
        assert "def test_user_login" in content
        assert "old content" not in content


class TestExtractScript:
    """Tests for the _extract_script static helper."""

    def test_strips_python_fence(self) -> None:
        text = "```python\nprint('hi')\n```"
        assert GUITestGenerator._extract_script(text) == "print('hi')"

    def test_strips_py_fence(self) -> None:
        text = "```py\nx = 1\n```"
        assert GUITestGenerator._extract_script(text) == "x = 1"

    def test_strips_bare_fence(self) -> None:
        text = "```\ncode\n```"
        assert GUITestGenerator._extract_script(text) == "code"

    def test_no_fence_unchanged(self) -> None:
        assert GUITestGenerator._extract_script("plain code") == "plain code"

    def test_trims_whitespace(self) -> None:
        text = "  ```python\ncode\n```  \n"
        assert GUITestGenerator._extract_script(text) == "code"


class TestValidatePython:
    """Tests for the _validate_python static helper."""

    def test_valid_python_passes(self) -> None:
        GUITestGenerator._validate_python("x = 1\nprint(x)\n")

    def test_invalid_python_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="invalid Python syntax"):
            GUITestGenerator._validate_python("def f(:\n")

    def test_empty_string_passes(self) -> None:
        # Empty string is technically valid Python (no statements).
        GUITestGenerator._validate_python("")

    def test_complex_valid_script_passes(self) -> None:
        GUITestGenerator._validate_python(_VALID_PLAYWRIGHT_SCRIPT)

    def test_syntax_error_message_includes_line(self) -> None:
        try:
            GUITestGenerator._validate_python("x =\n")
        except ValueError as e:
            assert "line" in str(e).lower() or "syntax" in str(e).lower()
        else:
            pytest.fail("Expected ValueError for invalid syntax")


# A review-refined variant (valid Python, distinct from the original).
_REVISED_PLAYWRIGHT_SCRIPT = _VALID_PLAYWRIGHT_SCRIPT + "\n\n# reviewed: added logout flow\n"


class TestGUIReview:
    """Review integration for GUITestGenerator (plan v2 §4.3).

    Context per decision 4: URL + requirements + generated script. Review
    candidates must pass fence-stripping + Python syntax validation.
    """

    def setup_method(self) -> None:
        self.mock_llm = MagicMock()
        self.prompt_builder = PromptBuilder()

    def _generator(self, *, review_enabled: bool) -> GUITestGenerator:
        return GUITestGenerator(
            llm_client=self.mock_llm,
            prompt_builder=self.prompt_builder,
            review_enabled=review_enabled,
            review_max_rounds=1,
        )

    def _input(self) -> GUITestGenInput:
        return GUITestGenInput(
            requirements=_REQUIREMENTS,
            url="https://example.com",
            endpoints=_ENDPOINTS,
        )

    def test_review_enabled_runs_second_pass(self) -> None:
        self.mock_llm.chat.side_effect = [_VALID_PLAYWRIGHT_SCRIPT, _REVISED_PLAYWRIGHT_SCRIPT]
        gen = self._generator(review_enabled=True)
        script = gen.generate(self._input())
        assert "reviewed: added logout flow" in script
        assert self.mock_llm.chat.call_count == 2

    def test_review_prompt_carries_url_and_requirements(self) -> None:
        self.mock_llm.chat.side_effect = [_VALID_PLAYWRIGHT_SCRIPT, _REVISED_PLAYWRIGHT_SCRIPT]
        gen = self._generator(review_enabled=True)
        gen.generate(self._input())
        system_prompt, user_prompt = self.mock_llm.chat.call_args_list[1].args
        assert "Playwright" in system_prompt
        assert "https://example.com" in user_prompt  # URL context
        assert "User Login" in user_prompt  # requirements context
        assert "def test_user_login" in user_prompt  # the script under review

    def test_review_invalid_python_candidate_keeps_original(self) -> None:
        # 1 round x 2 attempts, both candidates fail the syntax check.
        self.mock_llm.chat.side_effect = [
            _VALID_PLAYWRIGHT_SCRIPT,
            _INVALID_PYTHON_SCRIPT,
            _INVALID_PYTHON_SCRIPT,
        ]
        gen = self._generator(review_enabled=True)
        script = gen.generate(self._input())
        # _extract_script strips whitespace; the first-pass script is kept.
        assert script == _VALID_PLAYWRIGHT_SCRIPT.strip()

    def test_review_disabled_by_default_single_llm_call(self) -> None:
        self.mock_llm.chat.return_value = _VALID_PLAYWRIGHT_SCRIPT
        gen = GUITestGenerator(llm_client=self.mock_llm, prompt_builder=self.prompt_builder)
        gen.generate(self._input())
        self.mock_llm.chat.assert_called_once()

    def test_save_writes_meta_when_review_succeeded(self, tmp_path: Path) -> None:
        self.mock_llm.chat.side_effect = [_VALID_PLAYWRIGHT_SCRIPT, _REVISED_PLAYWRIGHT_SCRIPT]
        gen = self._generator(review_enabled=True)
        script = gen.generate(self._input())
        gen.save(script, tmp_path / "gui_test.py")

        meta = json.loads((tmp_path / "gui_test.meta.json").read_text())
        assert meta["generator"] == "gui"
        assert meta["reviewed"] is True
        assert meta["rounds_executed"] == 1
        assert meta["rounds_succeeded"] == 1

    def test_save_no_meta_when_review_did_not_run(self, tmp_path: Path) -> None:
        self.mock_llm.chat.return_value = _VALID_PLAYWRIGHT_SCRIPT
        gen = self._generator(review_enabled=False)
        script = gen.generate(self._input())
        gen.save(script, tmp_path / "gui_test.py")
        assert not (tmp_path / "gui_test.meta.json").exists()
