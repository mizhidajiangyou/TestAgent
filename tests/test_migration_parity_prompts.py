"""FH2.7 (plan-k B7.2) conversation prompt Parity.

The conversational chain used to build its generation prompts with the legacy
``PromptBuilder.build_{testcase,performance,gui_test}_prompt`` seams. B7.2 moves
it onto the task-package templates (``tasks/<pkg>/prompts``), with constants
from ``config/constants`` — and it must stay a pure rewiring: the prompt bytes
are a frozen surface, so any change here would be an enhancement smuggled into
a migration (plan-k §12 F2).

So the (system, user) pairs are recorded from the OLD code before the switch and
replayed afterwards with ``minimal_diff`` = {} — the same discipline as the web
contract cells. ``refine`` / ``validate`` templates are untouched by B7.2 and
are covered by ``tests/test_conversation.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from testagent.engine.conversation import ConversationSession
from testagent.engine.prompt_builder import PromptBuilder
from tests.parity_harness import Fixture, ensure_fixture, minimal_diff

TASK = "conversation_prompts"

_ENDPOINTS_TEXT = "GET /users (page,opt)\nPOST /users (name,req)"
_REQUIREMENTS_TEXT = "# User Management\nAcceptance Criteria:\n- User can register\n"


SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "test-cases",
        "artifact_type": "test_cases",
        "user_message": "cover login and registration",
        "ctx": {},
    },
    {
        "name": "performance-k6",
        "artifact_type": "performance_script",
        "user_message": "load test the user endpoints",
        "ctx": {"script_format": "k6", "perf_config": {"base_url": "https://api.example.com"}},
    },
    {
        "name": "performance-jmeter",
        "artifact_type": "performance_script",
        "user_message": "load test the user endpoints",
        "ctx": {"script_format": "jmeter", "perf_config": {}},
    },
    {
        "name": "gui-with-url",
        "artifact_type": "gui_script",
        "user_message": "login flow",
        "ctx": {"gui_url": "https://app.example.com"},
    },
    {
        "name": "gui-default-url",
        "artifact_type": "gui_script",
        "user_message": "login flow",
        "ctx": {},
    },
]


def _session() -> ConversationSession:
    session = ConversationSession(
        session_id="parity",
        llm_client=MagicMock(),
        prompt_builder=PromptBuilder(),
    )
    session._endpoints_text = _ENDPOINTS_TEXT
    session._requirements_text = _REQUIREMENTS_TEXT
    return session


def _observe(scn: dict[str, Any]) -> Fixture:
    system_prompt, user_prompt = _session()._build_generate_prompt(
        scn["artifact_type"], scn["user_message"], dict(scn["ctx"])
    )
    return Fixture(
        name=scn["name"],
        task=TASK,
        input={
            "artifact_type": scn["artifact_type"],
            "user_message": scn["user_message"],
            "ctx": scn["ctx"],
            "endpoints_text": _ENDPOINTS_TEXT,
            "requirements_text": _REQUIREMENTS_TEXT,
        },
        request_trace=[],
        artifact={"system": system_prompt, "user": user_prompt},
        failure_semantics={"observable": "ok", "non_empty": bool(system_prompt and user_prompt)},
        events=[],
        meta={"layer": "prompt-bytes"},
    )


@pytest.fixture(scope="module")
def baselines() -> dict[str, Fixture]:
    out: dict[str, Fixture] = {}
    for scn in SCENARIOS:
        out[scn["name"]] = ensure_fixture(TASK, scn["name"], lambda s=scn: _observe(s))
    return out


@pytest.mark.parametrize("scn", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
def test_prompt_bytes_replay_diff_zero(scn: dict[str, Any], baselines: dict[str, Fixture]) -> None:
    replayed = _observe(scn)
    diff = minimal_diff(baselines[scn["name"]], replayed)
    assert diff == {}, f"{scn['name']} prompt drifted: {json.dumps(diff)[:500]}"


#: artifact_type -> task package whose prompts must be the source.
_PACKAGE = {"test_cases": "testcase", "performance_script": "perf", "gui_script": "gui"}


@pytest.mark.parametrize("scn", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
def test_prompts_render_from_the_package_not_templates(scn: dict[str, Any]) -> None:
    """B7.2's whole point is the SOURCE of the text, which byte equality alone
    cannot prove (``templates/`` holds the same bytes until B7.3 deletes it).

    So disable the legacy fallback: if the prompt still renders, it came out of
    ``tasks/<pkg>/prompts``.
    """
    package = _PACKAGE[scn["artifact_type"]]
    prompts = Path("tasks") / package / "prompts"
    assert list(prompts.glob("*.j2")), f"{prompts} has no templates"

    builder = PromptBuilder()
    monkey_calls: list[str] = []

    def _no_legacy_template(name: str, **context: Any) -> str | None:
        monkey_calls.append(name)
        return None

    builder.render_template = _no_legacy_template  # type: ignore[method-assign]
    session = ConversationSession(
        session_id="parity", llm_client=MagicMock(), prompt_builder=builder
    )
    session._endpoints_text = _ENDPOINTS_TEXT
    session._requirements_text = _REQUIREMENTS_TEXT
    system_prompt, user_prompt = session._build_generate_prompt(
        scn["artifact_type"], scn["user_message"], dict(scn["ctx"])
    )
    assert monkey_calls == [], f"fell back to templates/ for {scn['name']}"
    assert user_prompt and system_prompt
