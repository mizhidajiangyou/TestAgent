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

    def render_template(self, name: str, **context: Any) -> str | None:
        """Render a named Jinja2 template by name.

        Returns the rendered string, or ``None`` when the template cannot be
        found or rendering fails. Enables callers (e.g. the conversation
        engine) to use optional template files with inline fallbacks.
        """
        try:
            template = self._env.get_template(name)
            return template.render(**context)
        except Exception:
            logger.debug("Template '%s' not available; caller should use inline fallback.", name)
            return None

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

    def build_gui_test_prompt(
        self,
        url: str,
        requirements_text: str,
        endpoints_text: str = "",
        output_language: str = "english",
    ) -> tuple[str, str]:
        """Build prompts for GUI (Playwright) test script generation.

        Inspired by stagehand: the LLM is asked to emit Playwright steps using
        robust locators (``get_by_role`` / ``get_by_label`` / ``get_by_text``)
        and ``expect()`` assertions, rather than fragile CSS/XPath selectors.
        """
        system_prompt = (
            "You are a senior QA automation engineer specializing in Playwright "
            "and pytest. Generate complete, runnable, maintainable GUI test "
            "scripts. Output only valid Python code."
        )
        lang_hint = self._language_instruction(output_language, code_context=True)
        if lang_hint:
            system_prompt += " " + lang_hint

        context = {
            "url": url,
            "requirements": requirements_text,
            "endpoints": endpoints_text,
            "output_language": output_language,
        }

        try:
            template = self._env.get_template("gui_test_prompt.j2")
            user_prompt = template.render(**context)
        except Exception:
            user_prompt = self._build_inline_gui_test_prompt(
                url=url,
                requirements=requirements_text,
                endpoints=endpoints_text,
                output_language=output_language,
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

    def _build_inline_gui_test_prompt(
        self,
        url: str,
        requirements: str,
        endpoints: str,
        output_language: str = "english",
        **kwargs: Any,
    ) -> str:
        """Fallback inline prompt for GUI (Playwright) test script generation."""
        lang_hint = self._language_instruction(output_language, code_context=True)
        lang_section = f"\n## Language\n{lang_hint}\n" if lang_hint else ""
        ep_section = f"\n## API Context (for reference)\n{endpoints}\n" if endpoints else ""
        return f"""Generate a complete Playwright Python test script for web/GUI testing.

## Target URL
{url}

## Requirements
{requirements}
{ep_section}
## Script Requirements
1. Use Playwright's Python sync API (from playwright.sync_api import Page, expect)
2. Use pytest as the test framework
3. Use ROBUST locators (in priority order):
   - page.get_by_role() - FIRST CHOICE for all interactive elements
   - page.get_by_label() - for form inputs with associated labels
   - page.get_by_placeholder() - for inputs without labels
   - page.get_by_text() - for non-interactive text elements
   - page.locator() - LAST RESORT only, with specific CSS selectors
4. Use expect() for ALL assertions (not raw assert)
5. Include proper waits: page.wait_for_load_state("networkidle") after navigation
6. Generate multiple test methods covering: happy path, form validation,
   error handling, navigation/routing, UI state changes
7. Include a setup fixture that navigates to the target URL
8. Add descriptive docstrings to each test method
9. Use pytest markers: @pytest.mark.smoke, @pytest.mark.regression
{lang_section}
## Output
Output ONLY the Python script code. No markdown fences, no explanations.
The script must be syntactically valid Python runnable with:
  pytest test_script.py --browser chromium"""
