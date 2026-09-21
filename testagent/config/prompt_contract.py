"""Shared prompt contract text and composition rules.

Why this module exists (plan-k B6b/B7 finding): the contract fragments below
are appended to EVERY generation call's system prompt, and the task-package
chain has to send byte-identical prompts to the legacy one (migration is
"只迁移不增强"). With the fragments living inside ``engine/prompt_builder``
the pipeline could not reach them — ``tests/test_pipeline_e2e.py::
TestArchitectureGate`` forbids ``pipeline → engine.prompt_builder`` — and the
first version of the new chain therefore shipped system prompts WITHOUT the
error contract. That is a silent quality regression, not a formatting detail:
the contract is what stops the two phases inventing contradictory status codes.

So the fragments and the composition ORDER live here, in a layer both chains
may import, and each chain keeps exactly one call site for them.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from testagent.config.models import TestCase

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
#: pass. Defining it ONCE (injected into every system prompt) is what prevents
#: Phase 1 and Phase 2 from inventing two different, contradictory status-code /
#: error-code conventions (the "spec inconsistency" defect).
ERROR_CONTRACT = (
    " ERROR CONTRACT (FALLBACK - applies ONLY where the requirement text or the "
    "Authoritative Value Table of this run already specifies the status code; "
    "spec and requirements always win over these defaults; both phases and "
    "review must agree): "
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

#: System-prompt bases for the three task packages.
TESTCASE_SYSTEM_BASE = (
    "You are a senior QA engineer. Generate comprehensive, well-structured "
    "test cases from requirements and API specifications. Output only valid JSON. "
    "Keep the total output within the model's token limit: prefer a focused set "
    "of high-value cases with concise descriptions and steps over exhaustive "
    "coverage, so the response is never cut off mid-JSON."
)
TESTCASE_HISTORICAL_SUFFIX = (
    " Historical test cases are provided as a baseline — generate ONLY "
    "net-new or updated cases that are NOT already covered by the baseline."
)
API_SYSTEM_BASE = (
    "You are a senior QA engineer specializing in API testing. Generate "
    "boundary, security, and integration test cases. Output only valid JSON."
)
REVIEW_SYSTEM_BASE = (
    "You are a meticulous senior QA reviewer. You detect gaps, inconsistencies and "
    "weak assertions in generated test cases, then produce a complete, improved list. "
    "Output only valid JSON."
)
K6_SYSTEM_BASE = (
    "You are an expert k6 performance engineer. Output only complete, runnable k6 JavaScript."
)
JMETER_SYSTEM_BASE = (
    "You are an expert JMeter performance engineer. Output only complete, valid JMeter JMX XML."
)
GUI_SYSTEM_BASE = (
    "You are a senior QA automation engineer specializing in Playwright "
    "and pytest. Generate complete, runnable, maintainable GUI test "
    "scripts. Output only valid Python code."
)
#: Links L3b (v15 §6): one unit per selected path contract. New text, so it
#: lives here with every other system base rather than in a second holder.
L3B_SYSTEM_BASE = (
    "You are a senior QA engineer writing end-to-end integration test cases. "
    "Produce cases that FULFIL the given path contract: execute every listed "
    "endpoint in order in a single case, passing real values between them. "
    "Output only valid JSON."
)

#: ``system_prompt: "generation:<name>"`` keys in the task manifests. The manifest
#: names a base; the TEXT and the composition order stay here, so a package can
#: not drift from the chain it migrated from.
SYSTEM_BASES: dict[str, str] = {
    "testcase": TESTCASE_SYSTEM_BASE,
    "api": API_SYSTEM_BASE,
    "review": REVIEW_SYSTEM_BASE,
    "k6": K6_SYSTEM_BASE,
    "jmeter": JMETER_SYSTEM_BASE,
    "gui": GUI_SYSTEM_BASE,
    "l3b": L3B_SYSTEM_BASE,
}


def system_base(name: str) -> str:
    """Look up a declared system-prompt base (fail loud on a typo)."""
    try:
        return SYSTEM_BASES[name]
    except KeyError:
        raise ValueError(
            f"unknown system prompt base {name!r}; declared bases: {sorted(SYSTEM_BASES)}"
        ) from None


#: Bases whose system prompt tells the model about a historical baseline. Only
#: the case-generation base does: Phase 2 learns what exists through its user
#: prompt (``already_covered``), and the legacy chain never added the suffix
#: there — a suffix on the wrong stage is a prompt diff, i.e. a behavior change.
HISTORICAL_AWARE_BASES = frozenset({"testcase"})

#: The two prompt families differ exactly as the legacy builders did:
#: case-generation prompts carry the shared ERROR CONTRACT (and the JSON-mode
#: wrapper), script-generation prompts carry only a CODE-CONTEXT language hint.
#: Collapsing them would put an HTTP status-code contract into a Playwright
#: prompt, or take it out of a test-case prompt.
SCRIPT_BASES = frozenset({"k6", "jmeter", "gui"})


def compose_named_system_prompt(
    name: str,
    *,
    output_language: str = "english",
    json_mode: bool = False,
    historical_cases: str = "",
) -> str:
    """Resolve a manifest's ``system_prompt: "generation:<name>"`` entry."""
    if name in SCRIPT_BASES:
        # No error contract, no JSON wrapper: the artifact is a script.
        return compose_system_prompt(
            system_base(name),
            output_language=output_language,
            code_context=True,
            with_error_contract=False,
        )
    return compose_system_prompt(
        system_base(name),
        output_language=output_language,
        json_mode=json_mode,
        with_historical_suffix=bool(historical_cases) and name in HISTORICAL_AWARE_BASES,
    )


def language_instruction(output_language: str, code_context: bool = False) -> str:
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


def compose_system_prompt(
    base: str,
    *,
    output_language: str = "english",
    json_mode: bool = False,
    with_historical_suffix: bool = False,
    code_context: bool = False,
    with_error_contract: bool = True,
) -> str:
    """The generation system prompt, in the ONE order every chain must use.

    base → [historical baseline suffix] → [language hint] → error contract →
    [JSON-mode wrapper instruction]. The legacy ``build_*`` builders and the
    task-package chain both call this; appending the same fragments in a
    different order changes the bytes and therefore the request fingerprint.

    ``with_error_contract=False`` is the script family: a k6/Playwright script
    has no HTTP status codes to agree on, and the legacy builders never sent
    one there.
    """
    prompt = base
    if with_historical_suffix:
        prompt += TESTCASE_HISTORICAL_SUFFIX
    hint = language_instruction(output_language, code_context=code_context)
    if hint:
        prompt += " " + hint
    if with_error_contract:
        prompt += ERROR_CONTRACT
    if json_mode:
        prompt += JSON_MODE_TEST_CASES_INSTRUCTION
    return prompt


def append_authoritative_table(user_prompt: str, table: str) -> str:
    """T9: attach the run's authoritative value table to the user prompt.

    Empty tables (no findings / feature not wired) leave the prompt
    byte-identical, keeping construction sites without the table on the
    legacy behavior.
    """
    if not table.strip():
        return user_prompt
    return (
        f"{user_prompt}\n\n---\n"
        "AUTHORITATIVE VALUE TABLE (this run - these decisions override the "
        "generic error contract below; rows marked conflict_unresolved must "
        f"stay explicitly unresolved in the cases):\n{table.rstrip()}\n"
    )


def case_coverage_text(cases: list[TestCase] | None) -> str:
    """Compact "what is already covered" summary for prompt injection.

    Serves BOTH historical baseline cases (Phase 1) and the Phase 1 output fed
    to Phase 2, which is why the legacy chain had one helper for two call
    sites: including id, title, endpoint, test type and a truncated
    description keeps the model from regenerating the same scenario.
    """
    if not cases:
        return ""
    lines = [f"Total existing cases: {len(cases)}", ""]
    for tc in cases:
        line = f"- [{tc.id}] {tc.title} | {tc.endpoint.full_path} | {tc.test_type.value}"
        if tc.description:
            desc = tc.description[:120]
            if len(tc.description) > 120:
                desc += "..."
            line += f" | {desc}"
        lines.append(line)
    return "\n".join(lines)


def coverage_block(
    cases: list[TestCase] | None, identities: list[dict[str, str]] | None = None
) -> str:
    """What a later phase must not regenerate: the summary plus, when the cases
    declare scenario identities (T8), the structured identity list.

    Identities are passed in rather than derived here: this module owns prompt
    TEXT, the scenario identity rule belongs to
    ``pipeline.scenario.covered_identities``.
    """
    text = case_coverage_text(cases)
    if not identities:
        return text
    return (
        f"{text}\n\nAlready-covered scenario identities "
        "(do NOT regenerate these operation+scene+variant combinations):\n"
        + json.dumps(identities, ensure_ascii=False)
    )
