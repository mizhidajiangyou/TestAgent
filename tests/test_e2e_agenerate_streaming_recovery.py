"""End-to-end reproduction of the empty-streaming + blocking-recovery bug.

This drives the REAL generation stack through the bug path:

    TestCaseGenerator.agenerate
        -> MultiModelLLMClient.achat          (async, via asyncio.to_thread)
            -> OpenAIClient.chat              (the method we fixed)
                -> _complete
                    -> _stream_completion     (REQ-002/003 return EMPTY + length)
                    -> _blocking_create      (all requirements recover here)

The mocked OpenAI SDK ``create`` returns, **per requirement**:

  * streaming (``stream=True``): an EMPTY chunk with ``finish_reason="length"``
    for REQ-002/REQ-003 (the field bug), valid content for REQ-001.
  * blocking (``stream=False``): valid test-case JSON for ALL requirements —
    the robust channel the fix falls back to on retry.

This proves the actual root-cause fix (``chat()`` retries on the blocking
channel when streaming returns an empty ``finish_reason=length`` response)
works through the full ``agenerate`` chain — not just the ``OpenAIClient.chat``
unit. It reproduces the user's failing run exactly: 1/3 requirements succeed
on the stream, 2/3 come back empty at the same moment and must be rescued by
the blocking channel so all three requirements end up with test cases.

See ``output/bugfix-llm-empty-response.md`` for the root-cause analysis.
"""

from __future__ import annotations

import json
from typing import cast
from unittest.mock import MagicMock

import pytest

from testagent.config.models import (
    APIEndpoint,
    RequirementItem,
    TestCase,
    TestCaseGenInput,
)
from testagent.engine.llm_client import MultiModelLLMClient, OpenAIClient
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.testcase_generator import TestCaseGenerator

#: Two valid test cases returned by the (mocked) blocking endpoint so that a
#: recovered requirement yields exactly 2 TestCase objects.
_VALID_CASES = json.dumps(
    [
        {
            "id": "TC-A",
            "title": "Valid case A",
            "description": "recovered via blocking channel",
            "test_type": "functional",
            "priority": "high",
            "steps": ["step one"],
            "expected_results": ["ok"],
        },
        {
            "id": "TC-B",
            "title": "Valid case B",
            "description": "recovered via blocking channel",
            "test_type": "functional",
            "priority": "medium",
            "steps": ["step two"],
            "expected_results": ["ok2"],
        },
    ]
)


def _stream_chunk(
    content: str, finish_reason: str = "stop", usage: MagicMock | None = None
) -> MagicMock:
    """Build one streaming chunk mock carrying ``content`` / ``finish_reason``."""
    if usage is None:
        usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=content), finish_reason=finish_reason)]
    chunk.usage = usage
    return chunk


def _blocking_response(content: str, finish_reason: str = "stop") -> MagicMock:
    """Build a blocking ``ChatCompletion``-like response carrying ``content``."""
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content), finish_reason=finish_reason)]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return resp


def _requirement_from_prompt(user_prompt: str) -> str | None:
    """Identify which requirement a prompt belongs to (id substring)."""
    for rid in ("REQ-001", "REQ-002", "REQ-003"):
        if rid in user_prompt:
            return rid
    return None


def _make_openai_client(
    record: list[dict[str, object]] | None = None,
    *,
    blocking_also_empty: bool = False,
) -> OpenAIClient:
    """OpenAIClient whose mocked SDK reproduces the empty-streaming bug.

    Args:
        record: optional list that receives one dict per ``create`` call
            describing ``req`` / ``stream`` / ``finish`` — used by assertions
            to prove the streaming->blocking switch actually happened.
        blocking_also_empty: when True, the blocking channel ALSO returns an
            empty + length response, simulating a genuinely-down provider (the
            case the fix cannot rescue — the run must degrade, not crash).
    """
    inner = MagicMock()

    def create(**kwargs: object) -> object:
        msgs = cast("list[dict[str, object]]", kwargs.get("messages") or [])
        user_prompt = str(msgs[1]["content"]) if len(msgs) > 1 else ""
        stream = bool(kwargs.get("stream"))
        req = _requirement_from_prompt(user_prompt)
        rec: dict[str, object] = {"req": req, "stream": stream, "finish": None}
        if record is not None:
            record.append(rec)

        if stream:
            if req == "REQ-001":
                # REQ-001: streaming endpoint behaves normally.
                rec["finish"] = "stop"
                return [_stream_chunk(content=_VALID_CASES, finish_reason="stop")]
            # REQ-002/003: streaming endpoint drops the stream -> 0 chars + length.
            rec["finish"] = "length"
            return [_stream_chunk(content="", finish_reason="length")]

        # Blocking channel: the robust path.
        if blocking_also_empty:
            rec["finish"] = "length"
            return _blocking_response(content="", finish_reason="length")
        rec["finish"] = "stop"
        return _blocking_response(content=_VALID_CASES, finish_reason="stop")

    inner.chat.completions.create.side_effect = create
    client = OpenAIClient(client=inner, model="mock-model", timeout=1.0, max_output_tokens=100)
    client.set_stream_enabled(True)
    return client


def _three_requirements() -> list[RequirementItem]:
    return [
        RequirementItem(
            id="REQ-001",
            title="Good",
            description="succeeds on stream",
            module="m1",
            acceptance_criteria=[],
        ),
        RequirementItem(
            id="REQ-002",
            title="DroppedA",
            description="empty stream under load",
            module="m2",
            acceptance_criteria=[],
        ),
        RequirementItem(
            id="REQ-003",
            title="DroppedB",
            description="empty stream under load",
            module="m3",
            acceptance_criteria=[],
        ),
    ]


def _make_generator(client: MultiModelLLMClient) -> TestCaseGenerator:
    return TestCaseGenerator(
        llm_client=client,
        prompt_builder=PromptBuilder(),
        verify_model=False,
    )


class TestE2EStreamingEmptyRecovery:
    """Full-chain reproduction: 2/3 streaming empty, blocking rescues all three."""

    async def test_e2e_all_three_requirements_produce_cases(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-001 streams OK; REQ-002/003 stream empty+length but recover on
        the blocking channel — every requirement ends with 2 cases (6 total).
        """
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0)
        record: list[dict[str, object]] = []
        sdk_client = _make_openai_client(record=record)
        gen = _make_generator(MultiModelLLMClient(clients=[sdk_client]))

        cases = await gen.agenerate(
            TestCaseGenInput(requirements=_three_requirements(), endpoints=[])
        )

        # All three requirements recovered -> 3 * 2 cases.
        assert len(cases) == 6
        assert all(isinstance(c, TestCase) for c in cases)
        # Re-numbered sequentially by agenerate.
        assert {c.id for c in cases} == {f"TC-{i:03d}" for i in range(1, 7)}

    async def test_e2e_recovery_used_blocking_channel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The smoking gun: for REQ-002/003 the failed ``stream=True`` call is
        followed by a successful ``stream=False`` (blocking) call — proving the
        fix's streaming->blocking fallback fired end-to-end.
        """
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0)
        record: list[dict[str, object]] = []
        sdk_client = _make_openai_client(record=record)
        gen = _make_generator(MultiModelLLMClient(clients=[sdk_client]))

        await gen.agenerate(TestCaseGenInput(requirements=_three_requirements(), endpoints=[]))

        # REQ-001: exactly one stream=True call (it succeeded on the stream).
        req1 = [r for r in record if r["req"] == "REQ-001"]
        assert len(req1) == 1
        assert req1[0]["stream"] == True  # noqa: E712
        assert req1[0]["finish"] == "stop"

        # REQ-002/003: stream=True (empty+length) FIRST, then stream=False (ok).
        for rid in ("REQ-002", "REQ-003"):
            seq = [r for r in record if r["req"] == rid]
            assert len(seq) >= 2
            assert seq[0]["stream"] == True  # noqa: E712
            assert seq[0]["finish"] == "length"
            assert seq[-1]["stream"] == False  # noqa: E712
            assert seq[-1]["finish"] == "stop"

    async def test_e2e_synchronous_mirror_also_recovers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sync ``generate`` path (same ``OpenAIClient.chat`` fix) also
        rescues 2/3 empty streams via the blocking channel.
        """
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0)
        sdk_client = _make_openai_client()
        gen = _make_generator(MultiModelLLMClient(clients=[sdk_client]))

        cases = gen.generate(TestCaseGenInput(requirements=_three_requirements(), endpoints=[]))

        assert len(cases) == 6


class TestE2EGenuinelyDown:
    """Boundary: when the blocking channel is ALSO empty, the fix cannot help —
    those requirements degrade to [] but the run must NOT crash, and REQ-001's
    cases still come through. This proves the fix only rescues the recoverable
    case and that the generator fails gracefully rather than aborting.
    """

    async def test_e2e_blocking_also_empty_degrades_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-002/003 fail on BOTH channels (provider truly down) -> [] each;
        REQ-001 (stream OK) still yields its 2 cases. No exception escapes.
        """
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0)
        sdk_client = _make_openai_client(blocking_also_empty=True)
        gen = _make_generator(MultiModelLLMClient(clients=[sdk_client]))

        # Must not raise — degrade to partial results like the async path.
        cases = await gen.agenerate(
            TestCaseGenInput(requirements=_three_requirements(), endpoints=[])
        )

        # Only REQ-001 survived.
        assert len(cases) == 2
        assert all(isinstance(c, TestCase) for c in cases)


class TestE2EWithAPIEndpoints:
    """Same recovery, but with endpoints present so Phase 2 (API-specific) also runs.

    Phase 1 fans out 3 requirements concurrently (REQ-002/003 dropped on the
    stream, recovered via blocking); Phase 2 then generates API-specific cases.
    The whole two-phase run must complete with every requirement's cases.
    """

    async def test_e2e_two_phase_recovers_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0)
        sdk_client = _make_openai_client()
        gen = _make_generator(MultiModelLLMClient(clients=[sdk_client]))

        endpoints = [
            APIEndpoint(method="GET", path="/users", summary="List users"),
            APIEndpoint(method="POST", path="/users", summary="Create user"),
        ]
        cases = await gen.agenerate(
            TestCaseGenInput(requirements=_three_requirements(), endpoints=endpoints)
        )

        # Phase 1: 3 requirements * 2 = 6. Phase 2: 1 endpoint batch * 2 = 2.
        assert len(cases) == 8
