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
            self._env = Environment(loader=BaseLoader(), autoescape=False)

    def build_testcase_prompt(
        self,
        endpoints_text: str,
        requirements_text: str,
        output_language: str = "english",
        extra_context: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Build prompts for test case generation.

        Works with or without API endpoints. When endpoints_text is empty,
        the prompt focuses on requirements only.
        """
        system_prompt = (
            "You are a senior QA engineer. Generate comprehensive, well-structured "
            "test cases from requirements and API specifications. Output only valid JSON."
        )
        lang_hint = self._language_instruction(output_language)
        if lang_hint:
            system_prompt += " " + lang_hint

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

    def build_api_prompt(
        self,
        endpoints_text: str,
        requirements_text: str,
        output_language: str = "english",
    ) -> tuple[str, str]:
        """Build prompts for API-specific test case generation.

        This is phase 2: generates boundary, security, integration cases
        that require API-level details (parameters, request body, status codes).
        """
        system_prompt = (
            "You are a senior QA engineer specializing in API testing. "
            "Generate boundary, security, and integration test cases. "
            "Output only valid JSON."
        )
        lang_hint = self._language_instruction(output_language)
        if lang_hint:
            system_prompt += " " + lang_hint

        context = {
            "endpoints": endpoints_text,
            "requirements": requirements_text,
            "output_language": output_language,
        }

        try:
            template = self._env.get_template("api_prompt.j2")
            user_prompt = template.render(**context)
        except Exception:
            user_prompt = self._build_inline_api_prompt(
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
        """
        system_prompt = (
            "You are a meticulous senior QA reviewer. You detect gaps, inconsistencies and "
            "weak assertions in generated test cases, then produce a complete, improved list. "
            "Output only valid JSON."
        )
        lang_hint = self._language_instruction(output_language)
        if lang_hint:
            system_prompt += " " + lang_hint

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
        """Build prompts for performance script generation."""
        if script_format == "jmeter":
            system_prompt = (
                "You are an expert JMeter performance engineer. "
                "Output only complete, valid JMeter JMX XML."
            )
        else:
            system_prompt = (
                "You are an expert k6 performance engineer. "
                "Output only complete, runnable k6 JavaScript."
            )
        lang_hint = self._language_instruction(output_language, code_context=True)
        if lang_hint:
            system_prompt += " " + lang_hint

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
        """Return a language instruction snippet for prompts."""
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
        lang_hint = self._language_instruction(kwargs.get("output_language", "english"))
        lang_section = f"\n{lang_hint}" if lang_hint else ""
        ep_section = f"\n## API Endpoints\n{endpoints}\n" if endpoints else ""
        ep_rule = (
            "(must match an endpoint above)" if endpoints else '(use "N/A" if no API spec)'
        )
        return f"""Generate test cases for the requirements below.

## Requirements
{requirements}
{ep_section}
## Output
Return a JSON array of test cases. Each element:
- "id": "TC-XXX"
- "title": concise title
- "description": what this verifies
- "endpoint": "METHOD /path" {ep_rule}
- "test_type": functional|boundary|negative|security|performance|integration
- "priority": high|medium|low
- "preconditions": ["setup steps"]
- "steps": ["exact execution steps with request details"]
- "expected_results": ["machine-checkable assertions"]
- "tags": ["labels"]

## Rules (must follow)
1. Assertable: every expected_results entry must be verifiable by a machine — explicit HTTP status code + concrete field-level check.
2. Endpoint consistency: when endpoints are given, the endpoint field must match one of them and steps must call it.
3. Tags required: fill tags with relevant labels.

## Guidance (use judgment)
- Prefer fewer, high-value cases over many shallow ones.
- Cover meaningful scenarios (happy path + important boundary/error/security where it matters); do not force every category per requirement.
- For data-creating cases, note cleanup so tests stay re-runnable.
{lang_section}

Output ONLY the JSON array. No markdown, no explanation."""

    def _build_inline_api_prompt(
        self, endpoints: str, requirements: str, **kwargs: Any
    ) -> str:
        """Fallback inline prompt for API-specific generation."""
        lang_hint = self._language_instruction(kwargs.get("output_language", "english"))
        lang_section = f"\n{lang_hint}" if lang_hint else ""
        return f"""Generate API-specific test cases (boundary, security, integration).

## API Endpoints
{endpoints}

## Requirements Context
{requirements}

Focus on: boundary values, security (SQLi/XSS/IDOR/auth), integration (CRUD flow), negative cases.
Each expected_result must be machine-checkable.
{lang_section}

Output ONLY a JSON array. No markdown, no explanation."""

    def _build_inline_review_prompt(
        self,
        endpoints: str,
        requirements: str,
        test_cases_json: str,
        output_language: str = "english",
    ) -> str:
        """Fallback inline prompt for review/refinement."""
        lang_hint = self._language_instruction(output_language)
        lang_section = f"\n{lang_hint}" if lang_hint else ""
        ep_section = f"\n## API Endpoints\n{endpoints}\n" if endpoints else ""
        return f"""Review and improve the test cases below. You have no prior context.

## Business Requirements
{requirements}
{ep_section}
## Existing Test Cases (JSON)
{test_cases_json}

## Review Checklist
1. Fix endpoint mismatches.
2. Ensure CRUD coverage per resource.
3. Rewrite vague expected_results into machine-checkable assertions.
4. Add boundary, security, performance, idempotency, i18n cases.
5. Fill tags.
{lang_section}

Return the COMPLETE final list as a single JSON array.
Output ONLY the JSON array. No markdown, no explanation."""

    def _build_inline_performance_prompt(
        self, endpoints: str, config: dict[str, Any], script_format: str, **kwargs: Any
    ) -> str:
        """Fallback inline prompt for performance script generation."""
        if script_format == "jmeter":
            return f"""Generate a complete JMeter JMX test plan.

## API Endpoints
{endpoints}

## Configuration
- Virtual Users: {config.get("virtual_users", 100)}
- Duration: {config.get("duration_seconds", 300)}s
- Ramp-Up: {config.get("ramp_up_seconds", 60)}s
- Think Time: {config.get("think_time_ms", 500)}ms
- Base URL: {config.get("base_url", "https://api.example.com")}

Output ONLY raw JMX XML starting with <?xml version="1.0"?>"""
        else:
            return f"""Generate a complete k6 JavaScript test script.

## API Endpoints
{endpoints}

## Configuration
- Virtual Users: {config.get("virtual_users", 100)}
- Duration: {config.get("duration_seconds", 300)}s
- Ramp-Up: {config.get("ramp_up_seconds", 60)}s
- Think Time: {config.get("think_time_ms", 500)}ms
- Base URL: {config.get("base_url", "https://api.example.com")}

Output ONLY raw JavaScript. No markdown."""
