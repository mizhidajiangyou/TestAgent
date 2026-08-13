"""
Prompt template builder for LLM interactions.

Uses Jinja2 templates for flexible prompt construction.
"""

import logging
from pathlib import Path
from typing import Any

from jinja2 import BaseLoader, Environment, FileSystemLoader

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"


class PromptBuilder:
    """Build LLM prompts from templates or inline strings."""

    def __init__(self, templates_dir: Path | None = None) -> None:
        self._templates_dir = templates_dir or TEMPLATES_DIR
        if self._templates_dir.exists():
            self._env = Environment(
                loader=FileSystemLoader(str(self._templates_dir)),
                autoescape=False,
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            self._env = Environment(
                loader=BaseLoader(),
                autoescape=False,
            )

    def build_testcase_prompt(
        self,
        endpoints_text: str,
        requirements_text: str,
        output_language: str = "english",
        extra_context: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Build prompts for test case generation.

        Args:
            endpoints_text: API endpoints as plain text.
            requirements_text: Requirements as plain text.
            output_language: Target output language ("english" or "chinese").
            extra_context: Additional template variables.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        system_prompt = (
            "You are a senior QA engineer with 10+ years of experience in software testing. "
            "You generate comprehensive, well-structured test cases based on API specifications "
            "and requirement documents. Output only valid JSON."
        )
        language_instruction = self._language_instruction(output_language)
        if language_instruction:
            system_prompt += " " + language_instruction

        context = {
            "endpoints": endpoints_text,
            "requirements": requirements_text,
            "output_language": output_language,
            **(extra_context or {}),
        }

        try:
            template = self._env.get_template("testcase_prompt.j2")
            user_prompt = template.render(**context)
        except Exception:
            user_prompt = self._build_inline_testcase_prompt(
                endpoints=endpoints_text,
                requirements=requirements_text,
                output_language=output_language,
            )

        return system_prompt, user_prompt

    def build_review_prompt(
        self,
        endpoints_text: str,
        requirements_text: str,
        test_cases_json: str,
        output_language: str = "english",
    ) -> tuple[str, str]:
        """Build prompts for reviewing/refining generated test cases.

        Runs as a fresh conversation with no prior context.

        Args:
            endpoints_text: API endpoints as plain text.
            requirements_text: Requirements as plain text.
            test_cases_json: Existing test cases serialized as JSON.
            output_language: Target output language ("english" or "chinese").

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        system_prompt = (
            "You are a meticulous senior QA reviewer. You detect gaps, inconsistencies and "
            "weak assertions in generated test cases, then produce a complete, improved list. "
            "Output only valid JSON."
        )
        language_instruction = self._language_instruction(output_language)
        if language_instruction:
            system_prompt += " " + language_instruction

        context = {
            "endpoints": endpoints_text,
            "requirements": requirements_text,
            "test_cases_json": test_cases_json,
            "output_language": output_language,
        }

        try:
            template = self._env.get_template("review_prompt.j2")
            user_prompt = template.render(**context)
        except Exception:
            user_prompt = self._build_inline_review_prompt(
                endpoints=endpoints_text,
                requirements=requirements_text,
                test_cases_json=test_cases_json,
                output_language=output_language,
            )

        return system_prompt, user_prompt

    def build_performance_prompt(
        self,
        endpoints_text: str,
        config: dict[str, Any],
        script_format: str = "k6",
        output_language: str = "english",
    ) -> tuple[str, str]:
        """Build prompts for performance script generation.

        Args:
            endpoints_text: API endpoints as plain text.
            config: Performance test configuration.
            script_format: Target format (k6 or jmeter).
            output_language: Target output language ("english" or "chinese").

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        if script_format == "jmeter":
            system_prompt = (
                "You are an expert JMeter performance engineer. "
                "Output only complete, valid JMeter JMX XML. "
                "Never output partial XML, markdown, or explanations."
            )
        else:
            system_prompt = (
                "You are an expert k6 performance engineer. "
                "Output only complete, runnable k6 JavaScript. "
                "Never output markdown fences or explanations."
            )
        language_instruction = self._language_instruction(output_language, code_context=True)
        if language_instruction:
            system_prompt += " " + language_instruction

        context = {
            "endpoints": endpoints_text,
            "config": config,
            "script_format": script_format,
        }

        try:
            template = self._env.get_template("performance_prompt.j2")
            user_prompt = template.render(**context)
        except Exception:
            user_prompt = self._build_inline_performance_prompt(
                endpoints=endpoints_text, config=config, script_format=script_format
            )

        return system_prompt, user_prompt

    @staticmethod
    def _language_instruction(output_language: str, code_context: bool = False) -> str:
        """Return a language instruction snippet for prompts.

        Args:
            output_language: Target language.
            code_context: Whether the output is code (comments/labels only).

        Returns:
            Instruction string, or empty when English.
        """
        if output_language == "chinese":
            if code_context:
                return (
                    "Write all script comments, user-facing labels and summary text in "
                    "Simplified Chinese (keep code keywords and identifiers in English)."
                )
            return (
                "Write all text fields (title, description, preconditions, steps, "
                "expected_results, tags) in Simplified Chinese."
            )
        return ""

    def _build_inline_testcase_prompt(
        self, endpoints: str, requirements: str, **kwargs: Any
    ) -> str:
        """Fallback inline prompt for test case generation."""
        language_hint = self._language_instruction(kwargs.get("output_language", "english"))
        language_section = f"\n{language_hint}" if language_hint else ""
        return f"""Generate comprehensive test cases based on the following API endpoints and requirements.

## API Endpoints
{endpoints}

## Requirements
{requirements}

## Output Format
Return a JSON array of test cases. Each test case must have:
- "id": unique test case ID (e.g., "TC-001")
- "title": concise test case title
- "description": what this test case verifies
- "endpoint": "METHOD /path" (the target endpoint)
- "test_type": one of "functional", "boundary", "negative", "performance", "security", "integration"
- "priority": one of "high", "medium", "low"
- "preconditions": array of setup steps
- "steps": array of test execution steps
- "expected_results": array of expected outcomes
- "tags": array of labels (smoke, regression, auth, crud, boundary, security, performance)

## Quality Checklist (must follow)
1. "endpoint" must EXACTLY match an endpoint above; steps must use the same endpoint.
2. Cover Create/Read/Update/Delete for every resource present in the spec.
3. Every expected_result must be machine-checkable (explicit status code + field value/presence).
4. Test min/max length boundaries, pagination limit maximum, case-sensitivity, empty strings.
5. Add SQL injection / XSS, rate limiting, token expiry & refresh, IDOR, password strength tests.
6. Add P95 response-time baseline and concurrency tests.
7. Include teardown/cleanup and duplicate-submission (idempotency) cases.
8. Cover unicode / emoji / whitespace / special characters.
{language_section}

Cover:
1. Happy path for each endpoint
2. Boundary value tests for parameters
3. Negative tests (invalid input, missing auth, etc.)
4. Integration tests where endpoints depend on each other

Output ONLY the JSON array. No markdown, no explanation."""

    def _build_inline_review_prompt(
        self,
        endpoints: str,
        requirements: str,
        test_cases_json: str,
        output_language: str = "english",
    ) -> str:
        """Fallback inline prompt for review/refinement."""
        language_hint = self._language_instruction(output_language)
        language_section = f"\n{language_hint}" if language_hint else ""
        return f"""Review and improve the test cases below. You have no prior context.

## API Endpoints
{endpoints}

## Requirements
{requirements}

## Existing Test Cases (JSON)
{test_cases_json}

## Review Checklist
1. Every "endpoint" must exactly match an endpoint above; fix mismatches.
2. Ensure Create/Read/Update/Delete are all covered per resource; add missing ones.
3. Rewrite vague expected_results into machine-checkable assertions (status code + field value/presence).
4. Add boundary tests (min/max length, pagination limit, case-sensitivity, empty strings).
5. Add security tests (SQLi/XSS, rate limiting, token expiry/refresh, IDOR, password strength).
6. Add performance tests (P95 baseline, concurrency).
7. Fill "tags" (smoke, regression, auth, crud, boundary, security, performance).
8. Add teardown/cleanup and idempotency (duplicate submission) cases.
9. Add unicode/emoji/whitespace handling cases.
{language_section}

Return the COMPLETE final list of ALL test cases as a single JSON array with the same schema.
Output ONLY the JSON array. No markdown, no explanation."""

    def _build_inline_performance_prompt(
        self, endpoints: str, config: dict[str, Any], script_format: str, **kwargs: Any
    ) -> str:
        """Fallback inline prompt for performance script generation."""
        if script_format == "jmeter":
            return f"""Generate a complete, production-ready JMeter JMX test plan.

## API Endpoints
{endpoints}

## Configuration
- Virtual Users: {config.get("virtual_users", 100)}
- Duration: {config.get("duration_seconds", 300)} seconds
- Ramp-Up: {config.get("ramp_up_seconds", 60)} seconds
- Think Time: {config.get("think_time_ms", 500)} ms
- Base URL: {config.get("base_url", "https://api.example.com")}
- Auth: {config.get("auth_type", "none")}

## Requirements
1. Complete XML from <?xml ...?> to </jmeterTestPlan>
2. ThreadGroup with configured users/ramp-up/duration
3. HTTP Request Defaults, Cookie Manager, Cache Manager
4. HTTP Header Manager with Content-Type: application/json
5. HTTP Samplers for each endpoint with realistic bodies
6. Response Assertions for success status codes
7. Summary Report and Simple Data Writer listeners

Output ONLY raw JMX XML. Start with: <?xml version="1.0" encoding="UTF-8"?>"""
        else:
            return f"""Generate a complete, production-ready k6 JavaScript test script.

## API Endpoints
{endpoints}

## Configuration
- Virtual Users: {config.get("virtual_users", 100)}
- Duration: {config.get("duration_seconds", 300)} seconds
- Ramp-Up: {config.get("ramp_up_seconds", 60)} seconds
- Think Time: {config.get("think_time_ms", 500)} ms
- Base URL: {config.get("base_url", "https://api.example.com")}
- Auth: {config.get("auth_type", "none")}

## Requirements
1. Export default function and options object
2. Configure stages for ramp-up then steady state
3. HTTP requests for each endpoint with proper headers
4. sleep() between requests matching think time
5. check() assertions for HTTP status codes
6. Realistic JSON bodies for POST/PUT requests
7. __ENV variables for BASE_URL and AUTH_TOKEN
8. Thresholds for p(95) < 500ms and error rate < 1%

Output ONLY raw JavaScript. No markdown."""
