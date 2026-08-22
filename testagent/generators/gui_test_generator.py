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

import json
import logging
import re
from pathlib import Path

from testagent.config.models import GUITestGenInput
from testagent.engine.llm_client import LLMClient
from testagent.engine.prompt_builder import PromptBuilder
from testagent.engine.review import ReviewLoop, ReviewResult
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
        review_enabled: bool = False,
        review_llm_client: LLMClient | None = None,
        review_max_rounds: int = 2,
    ) -> None:
        self._llm = llm_client
        self._prompt_builder = prompt_builder
        self._output_language = output_language
        # Review is opt-in at construction time (plan v2 decision 5): pure
        # code callers never trigger an extra LLM pass by accident; the
        # container wires settings.review_enabled in for CLI/web use.
        self._review_enabled = review_enabled
        self._review_loop = ReviewLoop[str](
            primary_llm=llm_client,
            review_llm=review_llm_client or llm_client,
            prompt_builder=prompt_builder,
            max_rounds=review_max_rounds,
        )
        # Review outcome of the last generate() call (None when review did
        # not run); consumed by save() to persist the .meta.json marker.
        self._last_review: ReviewResult[str] | None = None

    def set_review_enabled(self, enabled: bool) -> None:
        """Runtime override of the review switch (CLI ``--review/--no-review``)."""
        self._review_enabled = enabled

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

        self._last_review = None
        if self._review_enabled and script:
            script, self._last_review = self._review_script(
                script, url, requirements_text, output_language
            )

        self._validate_python(script)

        logger.info("Generated GUI test script (%d chars)", len(script))
        return script

    def save(self, output: str, output_path: Path) -> Path:
        """Save the generated script to a file.

        Parent directories are created automatically when missing. When the
        last generate() ran a review pass, a sibling ``<stem>.meta.json``
        records the review outcome (``reviewed`` flag).

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

        if self._last_review is not None:
            meta_path = output_path.with_suffix(".meta.json")
            meta = {
                "generator": "gui",
                "script_format": "playwright",
                "reviewed": self._last_review.used_review,
                "rounds_executed": self._last_review.rounds_executed,
                "rounds_succeeded": self._last_review.rounds_succeeded,
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Saved review metadata to %s", meta_path)

        return output_path

    def _review_script(
        self,
        script: str,
        url: str,
        requirements_text: str,
        output_language: str,
    ) -> tuple[str, ReviewResult[str]]:
        """Run the shared ReviewLoop over the generated script (no call_llm).

        Review context (plan v2 decision 4): URL + requirements + the script
        itself (passed per round as the artifact). Candidates are validated
        by fence-stripping + Python syntax check; invalid candidates count as
        failed rounds so a bad review answer cannot corrupt a good script.
        """
        context_text = f"Target URL: {url}\n\n## Requirements\n{requirements_text}"

        def build_prompt(current: str, round_idx: int) -> tuple[str, str]:
            return self._prompt_builder.build_script_review_prompt(
                script_kind="playwright",
                script=current,
                context_text=context_text,
                output_language=output_language,
            )

        def parse(raw: str) -> str | None:
            candidate = self._extract_script(raw)
            if not candidate:
                return None
            try:
                self._validate_python(candidate)
            except ValueError:
                return None
            return candidate

        result = self._review_loop.run(
            script,
            build_prompt=build_prompt,
            parse=parse,
            label="playwright-script",
        )
        return result.artifact, result

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
