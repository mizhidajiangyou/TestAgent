"""
GUI (Playwright) test script generator using LLM.

Generates Playwright Python test scripts for web/GUI testing.

Inspired by stagehand's approach:
- LLM analyzes requirements and the target URL, then generates
  natural-language-driven test steps.
- Uses Playwright's recommended robust locator strategies
  (``get_by_role`` / ``get_by_label`` / ``get_by_placeholder`` /
  ``get_by_text``) instead of fragile CSS/XPath selectors. Stagehand
  similarly relies on the accessibility tree + element IDs for robust
  targeting rather than brittle selectors.
- Uses ``expect()`` for machine-checkable assertions (stagehand uses
  URL/data comparison assertions).
- Emits a clean, maintainable, page-object-style test structure with
  proper waits, fixtures, and cleanup.
"""

import logging
import re
from pathlib import Path

from testagent.config.models import GUITestGenInput
from testagent.engine.llm_client import LLMClient
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.base import BaseGenerator
from testagent.parsers.requirement_parser import RequirementParser
from testagent.parsers.swagger_parser import SwaggerParser

logger = logging.getLogger(__name__)

#: Default target URL used when the caller does not supply one.
DEFAULT_TARGET_URL = "https://example.com"


class GUITestGenerator(BaseGenerator[GUITestGenInput, str]):
    """Generate Playwright Python test scripts for web/GUI testing.

    Inspired by stagehand's approach:
    - LLM analyzes requirements and generates natural-language-driven test steps.
    - Uses Playwright's recommended robust locator strategies (``get_by_role``,
      ``get_by_text``, ``get_by_label``, ``get_by_placeholder``) instead of
      fragile CSS/XPath selectors (stagehand uses the a11y tree + element IDs
      for robust targeting).
    - Uses ``expect()`` for machine-checkable assertions (stagehand uses
      URL/data comparison assertions).
    - Includes proper waits, assertions, and cleanup.
    - Generates clean, maintainable test structure (fixtures + page objects).
    """

    __test__ = False

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder,
        output_language: str = "english",
    ) -> None:
        self._llm = llm_client
        self._prompt_builder = prompt_builder
        self._output_language = output_language

    def generate(self, gen_input: GUITestGenInput) -> str:
        """Generate a complete Playwright Python test script.

        Args:
            gen_input: Input payload with requirements, target URL and optional
                API context for reference.

        Returns:
            Generated script content as a string.
        """
        requirements_text = RequirementParser.requirements_to_text(gen_input.requirements)
        url = gen_input.url or DEFAULT_TARGET_URL
        endpoints_text = (
            SwaggerParser.endpoints_to_text(gen_input.endpoints) if gen_input.endpoints else ""
        )

        # The per-input language wins over the constructor default so callers
        # can override the language on a single generation request.
        output_language = gen_input.output_language or self._output_language

        system_prompt, user_prompt = self._prompt_builder.build_gui_test_prompt(
            url=url,
            requirements_text=requirements_text,
            endpoints_text=endpoints_text,
            output_language=output_language,
        )

        logger.info("Generating Playwright GUI test script for %s ...", url)
        raw_response = self._llm.chat(system_prompt, user_prompt)

        script = self._extract_script(raw_response)
        self._validate_python(script)

        logger.info("Generated GUI test script (%d chars)", len(script))
        return script

    def save(self, output: str, output_path: Path) -> Path:
        """Save the generated script to a file.

        Parent directories are created automatically when missing.

        Args:
            output: Generated script content.
            output_path: Target file path.

        Returns:
            The saved file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
        logger.info("Saved GUI test script to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_script(raw: str) -> str:
        """Strip markdown code fences if present and trim whitespace.

        Handles opening fences such as ```` ```python ````, ```` ```py ````
        and a trailing ```` ``` ```` fence, so the result is pure Python.
        """
        cleaned = raw.strip()
        # Remove an opening fence: ```python / ```py / ```<lang>
        cleaned = re.sub(r"^```[a-zA-Z0-9_+.-]*\s*\n?", "", cleaned)
        # Remove a trailing fence
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def _validate_python(script: str) -> None:
        """Validate basic Python syntax by compiling the generated script.

        Raises:
            ValueError: When the script contains a ``SyntaxError``.
        """
        try:
            compile(script, "<generated>", "exec")
        except SyntaxError as exc:
            raise ValueError(f"Generated GUI test script has invalid Python syntax: {exc}") from exc
