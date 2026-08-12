"""
Test case generator using LLM.

Generates structured test cases from API endpoints and requirements.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from testagent.config.models import (
    APIEndpoint,
    TestCase,
    TestCaseGenInput,
    TestPriority,
    TestType,
)
from testagent.engine.llm_client import LLMClient
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.base import BaseGenerator
from testagent.parsers.requirement_parser import RequirementParser
from testagent.parsers.swagger_parser import SwaggerParser

logger = logging.getLogger(__name__)

#: Number of attempts when the LLM response cannot be parsed as JSON.
MAX_PARSE_RETRIES = 3


class TestCaseGenerator(BaseGenerator[TestCaseGenInput, list[TestCase]]):
    """Generate test cases from API specs and requirements."""

    __test__ = False

    def __init__(self, llm_client: LLMClient, prompt_builder: PromptBuilder) -> None:
        self._llm = llm_client
        self._prompt_builder = prompt_builder

    def generate(self, data: TestCaseGenInput) -> list[TestCase]:
        """Generate test cases.

        Args:
            data: Input payload containing endpoints and requirements.

        Returns:
            List of generated test cases.
        """
        endpoints = data.endpoints
        requirements = data.requirements

        endpoints_text = SwaggerParser.endpoints_to_text(endpoints)
        requirements_text = (
            RequirementParser.requirements_to_text(requirements)
            if requirements
            else "No specific requirements provided."
        )

        system_prompt, user_prompt = self._prompt_builder.build_testcase_prompt(
            endpoints_text=endpoints_text,
            requirements_text=requirements_text,
        )

        logger.info("Generating test cases via LLM...")

        for attempt in range(1, MAX_PARSE_RETRIES + 1):
            raw_response = self._llm.chat(system_prompt, user_prompt)
            items = self._extract_json(raw_response)

            if items is not None:
                test_cases = self._to_test_cases(items, endpoints)
                logger.info("Generated %d test cases", len(test_cases))
                return test_cases

            logger.error(
                "Attempt %d/%d: LLM response is not valid JSON. Preview: %.300s",
                attempt,
                MAX_PARSE_RETRIES,
                raw_response,
            )
            self._dump_debug_response(raw_response)

        logger.error("Giving up after %d parse attempts", MAX_PARSE_RETRIES)
        return []

    def save(self, output: list[TestCase], output_path: Path) -> Path:
        """Save test cases to JSON file.

        Args:
            output: List of test cases.
            output_path: Target file path.

        Returns:
            Saved file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [self._testcase_to_dict(tc) for tc in output]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Saved %d test cases to %s", len(output), output_path)
        return output_path

    def _extract_json(self, raw: str) -> list[Any] | None:
        """Extract a JSON array from the raw LLM response.

        Tolerates markdown fences and prose surrounding the JSON payload.
        """
        text = raw.strip()

        # Strip markdown code fences if present
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Direct parse
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            pass

        # Locate outermost array/object boundaries within surrounding prose
        for open_ch, close_ch in (("[", "]"), ("{", "}")):
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start != -1 and end > start:
                try:
                    parsed = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
                return parsed if isinstance(parsed, list) else [parsed]

        return None

    @staticmethod
    def _dump_debug_response(raw: str) -> None:
        """Persist the unparseable response for debugging."""
        try:
            debug_path = Path("output/.debug_last_llm_response.txt")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(raw, encoding="utf-8")
            logger.info("Raw response saved to %s for inspection", debug_path)
        except OSError:
            pass

    def _to_test_cases(self, items: list[Any], endpoints: list[APIEndpoint]) -> list[TestCase]:
        """Convert parsed JSON items into TestCase objects."""
        endpoint_map = {ep.full_path: ep for ep in endpoints}
        test_cases: list[TestCase] = []

        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue

            ep_key = item.get("endpoint", "")
            endpoint = endpoint_map.get(ep_key, endpoints[0] if endpoints else None)

            if endpoint is None:
                continue

            try:
                test_type = TestType(item.get("test_type", "functional"))
            except ValueError:
                test_type = TestType.FUNCTIONAL

            try:
                priority = TestPriority(item.get("priority", "medium"))
            except ValueError:
                priority = TestPriority.MEDIUM

            test_cases.append(
                TestCase(
                    id=item.get("id", f"TC-{idx:03d}"),
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    endpoint=endpoint,
                    test_type=test_type,
                    priority=priority,
                    preconditions=item.get("preconditions", []),
                    steps=item.get("steps", []),
                    expected_results=item.get("expected_results", []),
                    tags=item.get("tags", []),
                )
            )

        return test_cases

    @staticmethod
    def _testcase_to_dict(tc: TestCase) -> dict[str, Any]:
        """Convert TestCase to serializable dict."""
        return {
            "id": tc.id,
            "title": tc.title,
            "description": tc.description,
            "endpoint": tc.endpoint.full_path,
            "test_type": tc.test_type.value,
            "priority": tc.priority.value,
            "preconditions": tc.preconditions,
            "steps": tc.steps,
            "expected_results": tc.expected_results,
            "tags": tc.tags,
        }
