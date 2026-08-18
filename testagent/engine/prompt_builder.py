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

#: System-prompt suffix appended when JSON mode is enabled. OpenAI's
#: ``json_object`` response format rejects a bare JSON array, so the payload
#: must be wrapped in a single top-level object keyed ``test_cases``.
JSON_MODE_TEST_CASES_INSTRUCTION = (
    " CRITICAL: You MUST return a JSON OBJECT with exactly one top-level key "
    '"test_cases", whose value is the JSON array of test cases described '
    'above. Example shape: {"test_cases": [ { ...single case object... } ]}. '
    "Do NOT return a bare JSON array."
)

#: Canonical error contract shared by BOTH generation phases and the review
#: pass. Defining it ONCE here (injected into every system prompt) is what
#: prevents Phase 1 and Phase 2 from inventing two different, contradictory
#: status-code / error-code conventions (the "spec inconsistency" defect).
ERROR_CONTRACT = (
    " ERROR CONTRACT — use EXACTLY these status codes and error.code values for "
    "EVERY case (both phases and review must agree): "
    "2xx = 200 OK / 201 Created / 204 No Content. "
    "400 BAD_REQUEST for ALL client-input errors, with error.code: "
    "'VALIDATION_ERROR' + error.details.<field> for body validation (e.g. password "
    "length, email format); 'INVALID_<NAME>' for bad path/query params (INVALID_ID, "
    "INVALID_PAGE, INVALID_LIMIT); 'MALFORMED_JSON' for unparseable body. "
    "Do NOT use 422 — use 400. "
    "401 UNAUTHORIZED = missing / invalid / expired token. "
    "403 FORBIDDEN = authenticated but wrong role (non-admin on an admin endpoint). "
    "404 NOT_FOUND = unknown resource id. "
    "409 DUPLICATE_<FIELD> = unique-constraint violation (DUPLICATE_EMAIL). "
    "415 UNSUPPORTED_MEDIA_TYPE = wrong Content-Type. "
    "429 TOO_MANY_REQUESTS = rate limit / account lockout."
)


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
        json_mode: bool = False,
    ) -> tuple[str, str]:
        """Build prompts for test case generation.

        Works with or without API endpoints. When endpoints_text is empty,
        the prompt focuses on requirements only.

        When ``extra_context["historical_cases"]`` is a non-empty string, it is
        injected into the prompt so the LLM generates only net-new or updated
        cases (avoiding duplicates with the historical baseline).

        When ``json_mode`` is true, the output contract switches to a JSON
        object ``{"test_cases": [...]}`` (required by OpenAI's ``json_object``
        response format) and the wrapper instruction is appended to the system
        prompt; the template / inline fallback both honor the same contract.
        """
        historical_cases = ""
        if extra_context and isinstance(extra_context.get("historical_cases"), str):
            historical_cases = extra_context["historical_cases"]

        system_prompt = (
            "You are a senior QA engineer. Generate comprehensive, well-structured "
            "test cases from requirements and API specifications. Output only valid JSON. "
            "Keep the total output within the model's token limit: prefer a focused set "
            "of high-value cases with concise descriptions and steps over exhaustive "
            "coverage, so the response is never cut off mid-JSON."
        )
        if historical_cases:
            system_prompt += (
                " Historical test cases are provided as a baseline — generate ONLY "
                "net-new or updated cases that are NOT already covered by the baseline."
            )
        lang_hint = self._language_instruction(output_language)
        if lang_hint:
            system_prompt += " " + lang_hint
        system_prompt += ERROR_CONTRACT
        if json_mode:
            system_prompt += JSON_MODE_TEST_CASES_INSTRUCTION

        context = {
            "endpoints": endpoints_text,
            "requirements": requirements_text,
            "output_language": output_language,
            "historical_cases": historical_cases,
            "json_mode": json_mode,
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
                historical_cases=historical_cases,
                json_mode=json_mode,
            )

        return system_prompt, user_prompt

    def build_api_prompt(
        self,
        endpoints_text: str,
        requirements_text: str,
        output_language: str = "english",
        json_mode: bool = False,
        already_covered: str = "",
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
        system_prompt += ERROR_CONTRACT
        if json_mode:
            system_prompt += JSON_MODE_TEST_CASES_INSTRUCTION

        context = {
            "endpoints": endpoints_text,
            "requirements": requirements_text,
            "output_language": output_language,
            "json_mode": json_mode,
            "already_covered": already_covered,
        }

        try:
            template = self._env.get_template("api_prompt.j2")
            user_prompt = template.render(**context)
        except Exception:
            user_prompt = self._build_inline_api_prompt(
                endpoints=endpoints_text,
                requirements=requirements_text,
                output_language=output_language,
                json_mode=json_mode,
                already_covered=already_covered,
            )

        return system_prompt, user_prompt

    def build_review_prompt(
        self,
        endpoints_text: str,
        requirements_text: str,
        test_cases_json: str,
        output_language: str = "english",
        json_mode: bool = False,
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
        system_prompt += ERROR_CONTRACT
        if json_mode:
            system_prompt += JSON_MODE_TEST_CASES_INSTRUCTION

        context = {
            "endpoints": endpoints_text,
            "requirements": requirements_text,
            "test_cases_json": test_cases_json,
            "output_language": output_language,
            "json_mode": json_mode,
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
                json_mode=json_mode,
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
        self, endpoints: str, requirements: str, json_mode: bool = False, **kwargs: Any
    ) -> str:
        """Fallback inline prompt for test case generation."""
        lang_hint = self._language_instruction(kwargs.get("output_language", "english"))
        lang_section = f"\n{lang_hint}" if lang_hint else ""
        ep_section = f"\n## API Endpoints\n{endpoints}\n" if endpoints else ""
        ep_rule = "(must match an endpoint above)" if endpoints else '(use "N/A" if no API spec)'
        historical_cases = kwargs.get("historical_cases", "")
        hist_section = ""
        if historical_cases:
            hist_section = (
                "\n## Historical Test Cases (baseline — do NOT regenerate these)\n"
                "The following cases already exist. Generate ONLY net-new or updated "
                "cases that add coverage NOT already provided below.\n"
                f"{historical_cases}\n"
            )
        if json_mode:
            output_shape = (
                'Return a JSON OBJECT with exactly one top-level key "test_cases" whose '
                "value is the array described below. Do NOT return a bare array."
            )
            output_footer = (
                'Output ONLY the JSON object {"test_cases": [...]}. No markdown, no explanation.'
            )
        else:
            output_shape = "Return a JSON array of test cases. Each element:"
            output_footer = "Output ONLY the JSON array. No markdown, no explanation."
        return f"""Generate test cases for the requirements below.

## Requirements
{requirements}
{ep_section}{hist_section}
## Output
{output_shape}
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
1. Assertable: every expected_results entry MUST contain an explicit HTTP status code + a concrete field-level check (e.g. "Status 201; response.body.email == 'a@x.com'"). Forbidden vague terms: 成功/失败/works/as expected/properly.
2. Endpoint verbatim: when endpoints are given, the endpoint field MUST match one of them exactly (method+path) and steps must call it. Do NOT invent endpoints not in the list.
3. Executable steps: for API cases include the real request (method, path, headers incl. Authorization for protected endpoints, realistic JSON body for POST/PUT). Protected endpoints need a token in preconditions.
4. Cleanup: for create/modify cases, state cleanup (delete created resource) in the last step or description so tests stay re-runnable.
5. Tags required: fill tags with relevant labels from scope (smoke/regression), domain (auth/crud/pagination), type (boundary/security/negative).

## Guidance (use judgment)
- Prefer fewer, high-value cases over many shallow ones.
- Cover meaningful scenarios (happy path + important boundary/error/security where it matters); do not force every category per requirement.
- For data-creating cases, note cleanup so tests stay re-runnable.
- Error contract: 400 for all client-input errors (VALIDATION_ERROR+details.<field> for body, INVALID_<NAME> for params, MALFORMED_JSON); 401 missing/invalid/expired token; 403 wrong role; 404 unknown id; 409 DUPLICATE_<FIELD>; 415 wrong Content-Type. Do NOT use 422.
- Test isolation: use UNIQUE data per case (uuid/timestamp suffix on emails); never reuse fixed emails across cases; clean up only own data; do not assert global counts unless self-contained.
{lang_section}

{output_footer}"""

    def _build_inline_api_prompt(
        self,
        endpoints: str,
        requirements: str,
        json_mode: bool = False,
        already_covered: str = "",
        **kwargs: Any,
    ) -> str:
        """Fallback inline prompt for API-specific generation."""
        lang_hint = self._language_instruction(kwargs.get("output_language", "english"))
        lang_section = f"\n{lang_hint}" if lang_hint else ""
        covered_section = ""
        if already_covered:
            covered_section = f"\n## Already covered by Phase 1 (do NOT regenerate these scenarios)\n{already_covered}\n"
        if json_mode:
            output_footer = (
                'Output ONLY the JSON object {"test_cases": [...]}. No markdown, no explanation.'
            )
        else:
            output_footer = "Output ONLY a JSON array. No markdown, no explanation."
        return f"""Generate API-specific test cases (boundary, security, integration). This is Phase 2 — add ONLY cases not already covered.

## API Endpoints
{endpoints}

## Requirements Context
{requirements}
{covered_section}
Focus on: parameter bounds (page/limit/id extremes), password complexity, security (SQLi/XSS/IDOR/expired token/role), negative (malformed JSON, wrong Content-Type), concurrency/isolation. Skip functional/happy CRUD and uniqueness(409) cases already done in Phase 1.
Each expected_result must be machine-checkable (explicit status code + field assertion; no vague terms like 成功/失败/works).
Error contract: 400 for all client-input errors (VALIDATION_ERROR+details.<field> for body, INVALID_<NAME> for params, MALFORMED_JSON); 401 missing/invalid/expired token; 403 wrong role; 404 unknown id; 409 DUPLICATE_<FIELD>; 415 wrong Content-Type. Do NOT use 422.
Endpoint field must match one of the listed endpoints exactly (method+path); do not invent endpoints. Use UNIQUE data per case (no fixed shared emails); clean up only own data.
{lang_section}

{output_footer}"""

    def _build_inline_review_prompt(
        self,
        endpoints: str,
        requirements: str,
        test_cases_json: str,
        output_language: str = "english",
        json_mode: bool = False,
    ) -> str:
        """Fallback inline prompt for review/refinement."""
        lang_hint = self._language_instruction(output_language)
        lang_section = f"\n{lang_hint}" if lang_hint else ""
        ep_section = f"\n## API Endpoints\n{endpoints}\n" if endpoints else ""
        if json_mode:
            output_footer = (
                'Return the COMPLETE final list as a JSON object {"test_cases": [...]}.\n'
                "Output ONLY that JSON object. No markdown, no explanation."
            )
        else:
            output_footer = (
                "Return the COMPLETE final list as a single JSON array.\n"
                "Output ONLY the JSON array. No markdown, no explanation."
            )
        return f"""Review and improve the test cases below. You have no prior context.

## Business Requirements
{requirements}
{ep_section}
## Existing Test Cases (JSON)
{test_cases_json}

## Review Checklist
1. Fix endpoint mismatches (endpoint must match one listed endpoint exactly; do NOT invent endpoints).
2. Ensure CRUD coverage per resource.
3. Rewrite vague expected_results into machine-checkable assertions (explicit status code + field check; reject 成功/失败/works/as expected).
4. Add boundary, security, performance, idempotency, i18n cases.
5. Fill tags.
6. Preserve existing IDs (do not renumber). State a one-line reason for each change in description.
7. DEDUPLICATE: remove cases that target the SAME endpoint AND the SAME scenario intent, even if titles differ (e.g. "list without token → 401" vs "unauthenticated list → 403" are duplicates of the same scenario). Keep the one with stronger assertions.
8. RECONCILE contradictions: if two cases describe the same (endpoint, scenario) but assert different status codes, force BOTH to the canonical contract — 401 missing/invalid/expired token; 403 wrong role; 404 unknown id; 409 DUPLICATE_<FIELD>; 400 for all client-input errors (VALIDATION_ERROR+details for body, INVALID_<NAME> for params; NO 422).

Return the COMPLETE final list. After the JSON, on a new line output: SUMMARY: added=<N> fixed=<N> removed=<N>
{lang_section}

{output_footer}"""

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
- Auth: {config.get("auth_type", "none")}

Requirements: complete XML; ThreadGroup with ramp-up then steady state; HTTP Request Defaults + Cookie/Cache managers; Header Manager with Content-Type; samplers for EACH endpoint with realistic POST/PUT bodies; Response Assertions for status + a key field; if Auth != none add a login sampler that extracts and reuses a token; Summary Report + Simple Data Writer listeners; teardown that deletes created data.
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
- Auth: {config.get("auth_type", "none")}

Requirements: export options with stages (ramp-up then steady); build URLs from `const BASE_URL = __ENV.BASE_URL || '{config.get("base_url", "https://api.example.com")}'` (never hardcode host); if Auth != none obtain token in setup() and send Authorization header; cover EVERY endpoint with realistic POST/PUT bodies; check() status + a key field per request; thresholds p(95)<500 and error rate<1%; sleep() for pacing matching think time; setup()/teardown() create+cleanup data; no hardcoded secrets.
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
1. Use Playwright's Python sync API; start with: import pytest; from playwright.sync_api import Page, expect, BrowserContext
2. Use pytest as the test framework; type-annotate the page: Page argument
3. Use a fixture that navigates to the Target URL (use page.goto("{url}")), waits with wait_for_load_state("networkidle"), performs login if needed, and yields page
4. Use ROBUST locators (priority order) — NEVER use CSS/XPath as primary:
   - page.get_by_role() - FIRST CHOICE for interactive elements
   - page.get_by_label() - form inputs with associated labels
   - page.get_by_placeholder() - inputs without labels
   - page.get_by_text() - non-interactive text
   - page.locator() - LAST RESORT, specific selector + comment why
5. Use expect() for ALL assertions (never raw assert); assert concrete values
6. Use explicit waits (wait_for_load_state / wait_for_selector / expect(...).to_be_visible()); do NOT use time.sleep or long wait_for_timeout
7. Cover happy path, form validation (assert visible error text), error handling, navigation/routing, UI state changes
8. Each test method needs a docstring + pytest marker (@pytest.mark.smoke, @pytest.mark.regression)
9. Add cleanup (logout / delete created data) so the script is re-runnable
10. Target the Target URL above (do NOT navigate to example.com or any other host)
{lang_section}
## Output
Output ONLY the Python script code. No markdown fences, no explanations.
The script must be syntactically valid Python runnable with:
  pytest test_script.py --browser chromium"""
