"""
Prompt template builder for LLM interactions.

Uses Jinja2 templates for flexible prompt construction.
"""

import logging
import re
from pathlib import Path
from typing import Any

from jinja2 import BaseLoader, Environment, FileSystemLoader

from testagent.config.models import APIEndpoint
from testagent.config.prompt_contract import (
    API_SYSTEM_BASE,
    ERROR_CONTRACT,
    GUI_SYSTEM_BASE,
    JMETER_SYSTEM_BASE,
    JSON_MODE_TEST_CASES_INSTRUCTION,
    K6_SYSTEM_BASE,
    REVIEW_SYSTEM_BASE,
    TESTCASE_HISTORICAL_SUFFIX,
    TESTCASE_SYSTEM_BASE,
    append_authoritative_table,
    compose_system_prompt,
    language_instruction,
)
from testagent.parsers.swagger_parser import endpoints_to_rich_signature

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"

#: The prompt contract fragments moved to ``config/prompt_contract`` so the
#: task-package chain can compose byte-identical prompts without importing this
#: module (architecture gate). They are re-exported above for the frozen legacy
#: import paths (generators, tests) until plan-k B7.3 deletes them.
__all__ = [
    "API_SYSTEM_BASE",
    "ERROR_CONTRACT",
    "GUI_SYSTEM_BASE",
    "JMETER_SYSTEM_BASE",
    "JSON_MODE_TEST_CASES_INSTRUCTION",
    "K6_SYSTEM_BASE",
    "REVIEW_SYSTEM_BASE",
    "TEMPLATES_DIR",
    "TESTCASE_HISTORICAL_SUFFIX",
    "TESTCASE_SYSTEM_BASE",
    "PromptBuilder",
    "endpoints_to_rich_signature",
    "endpoints_to_signature",
    "extract_requirement_summary",
    "language_instruction",
]


#: Task packages live next to the source tree; the composition root passes the
#: configured ``TASKS_DIR`` in.
DEFAULT_TASKS_DIR = "tasks"


class PromptBuilder:
    """Build LLM prompts from templates or inline strings."""

    def __init__(
        self,
        templates_dir: Path | None = None,
        tasks_dir: Path | str = DEFAULT_TASKS_DIR,
    ) -> None:
        self._templates_dir = templates_dir or TEMPLATES_DIR
        self._tasks_dir = Path(tasks_dir)
        self._package_envs: dict[str, Environment] = {}
        if self._templates_dir.exists():
            self._env = Environment(
                loader=FileSystemLoader(str(self._templates_dir)),
                autoescape=False,
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            self._env = Environment(loader=BaseLoader(), autoescape=False)

    def render_package_prompt(
        self, package: str, template_name: str, context: dict[str, Any]
    ) -> str | None:
        """Render a task-package prompt (plan-k B7.2).

        ``tasks/<pkg>/prompts`` is the source of prompt text for the new chain;
        the env is configured exactly like the legacy one (``trim_blocks`` /
        ``lstrip_blocks``), otherwise the bytes would differ by whitespace and
        every prompt fingerprint would move. Returns ``None`` when the package
        template is unavailable, so a partially installed tree still renders
        from ``templates/``.
        """
        env = self._package_envs.get(package)
        if env is None:
            prompts_dir = self._tasks_dir / package / "prompts"
            if not prompts_dir.is_dir():
                return None
            env = Environment(
                loader=FileSystemLoader(str(prompts_dir)),
                autoescape=False,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            self._package_envs[package] = env
        try:
            return env.get_template(template_name).render(**context)
        except Exception:
            logger.debug("Package prompt %s/%s not available", package, template_name)
            return None

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

        system_prompt = compose_system_prompt(
            TESTCASE_SYSTEM_BASE,
            output_language=output_language,
            json_mode=json_mode,
            with_historical_suffix=bool(historical_cases),
        )

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

        table = str((extra_context or {}).get("authoritative_table", "") or "")
        return system_prompt, self._append_authoritative_table(user_prompt, table)

    @staticmethod
    def _append_authoritative_table(user_prompt: str, table: str) -> str:
        """T9: attach the run's authoritative value table (rules live in
        ``config/prompt_contract``, which the task-package chain shares)."""
        return append_authoritative_table(user_prompt, table)

    def build_api_prompt(
        self,
        endpoints_text: str,
        requirements_text: str,
        output_language: str = "english",
        json_mode: bool = False,
        already_covered: str = "",
        authoritative_table: str = "",
    ) -> tuple[str, str]:
        """Build prompts for API-specific test case generation.

        This is phase 2: generates boundary, security, integration cases
        that require API-level details (parameters, request body, status codes).
        """
        system_prompt = compose_system_prompt(
            API_SYSTEM_BASE, output_language=output_language, json_mode=json_mode
        )

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

        return system_prompt, self._append_authoritative_table(user_prompt, authoritative_table)

    def build_slim_continue_context(
        self,
        *,
        endpoints_signature: str,
        requirement_summary: str,
        fingerprint: str,
        label: str,
        pending: dict[str, int],
        max_tokens_budget: int = 2000,
    ) -> str:
        """Slim continuation context after truncation / downgrade (plan A).

        Explicitly EXCLUDES the heavyweight parts of the original prompt:
        the Rules region, worked example and historical cases. When the total
        exceeds the budget, sections are truncated lowest-priority first —
        requirement summary, then endpoint signature — while the "still
        needed" (pending) and "already produced" (fingerprint) sections are
        NEVER truncated (they are what makes the continuation converge).

        ``max_tokens_budget`` is an approximate token budget; it is converted
        to a character budget with a conservative mixed-script coefficient of
        2 chars/token (CJK ≈ 1.5, Latin ≈ 4 — 2 is the safe middle).
        """
        del label  # reserved for diagnostics; the sections below are complete
        pending_text = (
            ", ".join(f"{ep} x{max(0, n)}" for ep, n in pending.items() if n > 0) or "(none)"
        )
        tail = (
            "CONTINUATION: your previous response for this task was TRUNCATED "
            "(or came back empty) before the JSON closed properly.\n\n"
            f"## Already produced (do NOT repeat)\n{fingerprint}\n\n"
            f"## Still needed (endpoint xcount): {pending_text}\n\n"
            "Generate ONLY the still-needed cases, same JSON shape as before. "
            "Keep each case compact. Return a complete, properly closed JSON array."
        )
        char_budget = max(500, max_tokens_budget * 2) - len(tail)

        sig_header = "## Endpoint signatures (compact)\n"
        sum_header = "## Requirement summary\n"
        sig_block = f"{sig_header}{endpoints_signature or '(none)'}\n\n"
        summary_block = f"{sum_header}{requirement_summary or '(unavailable)'}\n\n"

        # Priority truncation: the summary is cut first; the signature only
        # when it alone still exceeds the remaining budget.
        summary_limit = char_budget - len(sig_block) - len(sum_header) - 2
        if summary_limit < len(requirement_summary or ""):
            summary_body = _truncate_marker(requirement_summary or "", max(0, summary_limit))
            summary_block = f"{sum_header}{summary_body}\n\n"
        if len(sig_block) + len(summary_block) > char_budget:
            sig_limit = char_budget - len(summary_block) - len(sig_header) - 2
            sig_body = _truncate_marker(endpoints_signature or "", max(0, sig_limit))
            sig_block = f"{sig_header}{sig_body}\n\n"

        return sig_block + summary_block + tail

    def build_review_prompt(
        self,
        endpoints_text: str,
        requirements_text: str,
        test_cases_json: str,
        output_language: str = "english",
        json_mode: bool = False,
        authoritative_table: str = "",
    ) -> tuple[str, str]:
        """Build prompts for reviewing/refining generated test cases.

        Runs as a fresh conversation with no prior context.
        """
        system_prompt = compose_system_prompt(
            REVIEW_SYSTEM_BASE, output_language=output_language, json_mode=json_mode
        )

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

        return system_prompt, self._append_authoritative_table(user_prompt, authoritative_table)

    def build_script_review_prompt(
        self,
        script_kind: str,
        script: str,
        context_text: str,
        output_language: str = "english",
    ) -> tuple[str, str]:
        """Build prompts for reviewing a generated test SCRIPT (plan v2 §4.2).

        Unlike :meth:`build_review_prompt` (testcase JSON contract), the
        output contract here is the COMPLETE revised script (decision 2:
        diff/patch line alignment is unreliable for generated scripts).

        Args:
            script_kind: ``"k6"`` | ``"jmeter"`` | ``"playwright"`` — selects
                the checklist branch and the system prompt.
            script: Full generated script to review.
            context_text: Rendered review context the script must stay
                consistent with (perf: load config + endpoint signatures;
                GUI: target URL + requirements).
            output_language: Language for comments/labels in the script.
        """
        if script_kind == "jmeter":
            system_prompt = (
                "You are a meticulous senior performance engineer reviewing a JMeter "
                "JMX test plan. Detect parameter mismatches, missing assertions and "
                "structural errors, then produce the complete revised script. "
                "Output only complete, valid JMX XML."
            )
        elif script_kind == "k6":
            system_prompt = (
                "You are a meticulous senior performance engineer reviewing a k6 "
                "JavaScript test script. Detect parameter mismatches, missing "
                "assertions and structural errors, then produce the complete "
                "revised script. Output only complete, runnable k6 JavaScript."
            )
        else:
            system_prompt = (
                "You are a meticulous senior QA automation engineer reviewing a "
                "Playwright (pytest) GUI test script. Detect missing requirement "
                "coverage, fragile locators and weak assertions, then produce the "
                "complete revised script. Output only complete, valid Python code."
            )
        lang_hint = self.language_instruction(output_language, code_context=True)
        if lang_hint:
            system_prompt += " " + lang_hint

        context = {
            "script_kind": script_kind,
            "script": script,
            "context_text": context_text,
        }

        try:
            template = self._env.get_template("script_review_prompt.j2")
            user_prompt = template.render(**context)
        except Exception:
            user_prompt = self._build_inline_script_review_prompt(
                script_kind=script_kind, script=script, context_text=context_text
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
        system_prompt = JMETER_SYSTEM_BASE if script_format == "jmeter" else K6_SYSTEM_BASE
        lang_hint = self.language_instruction(output_language, code_context=True)
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
        system_prompt = GUI_SYSTEM_BASE
        lang_hint = self.language_instruction(output_language, code_context=True)
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
    def language_instruction(output_language: str, code_context: bool = False) -> str:
        """Return a language instruction snippet for prompts.

        Delegates to :func:`testagent.config.prompt_contract.language_instruction`
        — one holder of the wording, shared with the task-package chain.
        """
        return language_instruction(output_language, code_context=code_context)

    def _build_inline_testcase_prompt(
        self, endpoints: str, requirements: str, json_mode: bool = False, **kwargs: Any
    ) -> str:
        """Fallback inline prompt for test case generation."""
        lang_hint = self.language_instruction(kwargs.get("output_language", "english"))
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
        lang_hint = self.language_instruction(kwargs.get("output_language", "english"))
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
        lang_hint = self.language_instruction(output_language)
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

    def _build_inline_script_review_prompt(
        self, script_kind: str, script: str, context_text: str
    ) -> str:
        """Fallback inline prompt for script review (template missing).

        Must stay in sync with ``templates/script_review_prompt.j2`` — the
        output contract (COMPLETE revised script, no diff) is the critical
        part that parse/validate relies on.
        """
        if script_kind == "jmeter":
            checklist = (
                "1. ThreadGroup users/ramp-up/duration match the context.\n"
                "2. Every endpoint in the context has a sampler; POST/PUT carry realistic bodies.\n"
                "3. Assertions exist for status codes and at least one key field.\n"
                "4. Auth flow is coherent when auth is enabled.\n"
                "5. Valid JMX structure (XML declaration, balanced tags, jmeterTestPlan root).\n"
                "6. Cleanup for data-creating samplers."
            )
            output_hint = (
                "Output the COMPLETE revised JMX XML only — raw XML, no markdown, no diff."
            )
        elif script_kind == "k6":
            checklist = (
                "1. options/stages match the configured VUs, duration, ramp-up and think time.\n"
                "2. Every endpoint in the context is covered; POST/PUT carry realistic bodies.\n"
                "3. check() assertions on res.status and at least one key field.\n"
                "4. thresholds exist (http_req_duration / http_req_failed).\n"
                "5. Auth flow is coherent when auth is enabled.\n"
                "6. No hardcoded host/secrets; setup/teardown clean up test data."
            )
            output_hint = (
                "Output the COMPLETE revised k6 JavaScript only — raw code, no markdown, no diff."
            )
        else:
            checklist = (
                "1. Target URL and requirement coverage match the context.\n"
                "2. Robust locators (get_by_role / get_by_label / get_by_text) over CSS/XPath.\n"
                "3. Machine-checkable expect() assertions on observable outcomes.\n"
                "4. Explicit waits, no brittle fixed sleeps as the only synchronization.\n"
                "5. Test data cleanup and proper fixtures.\n"
                "6. Complete, syntactically valid, runnable Python."
            )
            output_hint = (
                "Output the COMPLETE revised Python script only — raw code, no markdown, no diff."
            )
        return f"""Review the following {script_kind} test script and produce a COMPLETE revised version.

## Review Context (the script MUST stay consistent with this)
{context_text}

## Current Script
{script}

## Review Checklist
{checklist}

{output_hint}
Preserve the original structure and naming where possible. If nothing needs to change, return the original script unchanged."""

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
        lang_hint = self.language_instruction(output_language, code_context=True)
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


# ----------------------------------------------------------------------
# Slim continuation context (plan v10 §6 / v8 方案 A)
# ----------------------------------------------------------------------

#: Section header(s) that carry the requirement text in rendered prompts.
_REQUIREMENT_SECTION_RE = re.compile(
    r"## Requirements(?: Context)?\s*\n(.*?)(?=\n## |\n---|\Z)", re.S
)


def _format_param(name: str, schema: Any, required: bool) -> str:
    """Format one parameter as ``name(type,req|opt[,enum:a|b])``."""
    t = schema.get("type", "?") if isinstance(schema, dict) else "?"
    extra = ""
    if isinstance(schema, dict) and schema.get("enum"):
        extra = ",enum:" + "|".join(str(e) for e in schema["enum"])
    return f"{name}({t},{'req' if required else 'opt'}{extra})"


def endpoints_to_signature(endpoints: list[APIEndpoint]) -> str:
    """Compact endpoint signature (plan v8 §1.2, dict-safe access).

    Keeps name/type/required/enum for both parameters and the request body
    schema's top-level properties; drops descriptions/examples. This is the
    constraint source boundary/negative cases need, at a fraction of the
    full ``endpoints_to_text`` size.
    """
    lines: list[str] = []
    for ep in endpoints:
        parts: list[str] = []
        for p in ep.parameters or []:
            if not isinstance(p, dict):
                continue
            parts.append(
                _format_param(
                    str(p.get("name", "")), p.get("schema") or {}, bool(p.get("required", False))
                )
            )
        line = f"- {ep.method} {ep.path}"
        if parts:
            line += f" params:[{', '.join(sorted(parts))}]"
        body = ep.request_body or {}
        props = (body.get("schema") or {}).get("properties", {}) if isinstance(body, dict) else {}
        if isinstance(props, dict) and props:
            req_set = set((body.get("schema") or {}).get("required") or [])
            bparts = [_format_param(k, v or {}, k in req_set) for k, v in props.items()]
            line += f" body:[{', '.join(sorted(bparts))}]"
        lines.append(line)
    return "\n".join(lines)


def extract_requirement_summary(user_prompt: str, max_chars: int) -> str | None:
    """Extract the requirements section from a rendered prompt (plan A).

    Templates delimit requirement text with a ``## Requirements`` /
    ``## Requirements Context`` section header, so the summary is surgical:
    the full Rules region / worked example / historical cases are left out.
    Returns ``None`` when no section is found (caller falls back to the
    legacy full-prompt continuation).
    """
    match = _REQUIREMENT_SECTION_RE.search(user_prompt)
    if not match:
        return None
    text = match.group(1).strip()
    if not text:
        return None
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [truncated]"
    return text


def _truncate_marker(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    cut = max(0, limit - len("\n... [truncated]"))
    return text[:cut] + "\n... [truncated]"
