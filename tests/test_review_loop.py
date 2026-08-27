"""Tests for the generic ReviewLoop (engine/review.py).

Covers the behavioral contract extracted from TestCaseGenerator's private
``_review_and_refine`` / ``_areview_and_refine`` (plan v2 §4.1):

- Odd rounds use the review (secondary) client, even rounds the primary.
- A failed round keeps the previous artifact and the loop continues.
- All rounds failing returns the original artifact (``used_review=False``).
- ``ReviewResult`` carries round counters for observability.
- ``arun`` is true async: it drives ``achat``, never ``chat``.
- Without ``call_llm``/``acall_llm`` the loop calls the clients itself and
  parses via the injected ``parse`` hook (perf/gui path), with one internal
  retry per round.

Also closes the R1 audit gap: existing testcase-generator tests never covered
"one round fails mid-loop -> previous artifact is kept and the NEXT round runs".
"""

from __future__ import annotations

import dataclasses
import logging

import pytest

from testagent.engine.review import ReviewLoop, ReviewResult


class FakeClient:
    """Minimal sync+async LLM double.

    Deliberately does NOT define ``chat_with_meta``/``achat_with_meta`` so the
    loop falls back to the plain string contract (``chat``/``achat``).
    """

    def __init__(self, responses: list[str], name: str = "fake") -> None:
        self.responses = list(responses)
        self.model_name = name
        self.calls: list[tuple[str, str]] = []
        self.async_calls: list[tuple[str, str]] = []

    def _next(self) -> str:
        return self.responses.pop(0) if self.responses else ""

    def chat(self, system_prompt: str, user_prompt: str, **_: object) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._next()

    async def achat(self, system_prompt: str, user_prompt: str, **_: object) -> str:
        self.async_calls.append((system_prompt, user_prompt))
        return self._next()


class MetaClient(FakeClient):
    """Client exposing ``chat_with_meta`` for token-aware logging (M2)."""

    def __init__(self, responses: list[str], tokens: list[int]) -> None:
        super().__init__(responses, name="meta-model")
        self.tokens = list(tokens)

    def chat_with_meta(self, system_prompt: str, user_prompt: str, **_: object) -> object:
        from testagent.engine.llm_client import LLMResponse

        self.calls.append((system_prompt, user_prompt))
        text = self._next()
        tok = self.tokens.pop(0) if self.tokens else None
        return LLMResponse(text=text, finish_reason="stop", completion_tokens=tok)


def make_loop(
    primary: FakeClient,
    secondary: FakeClient,
    *,
    max_rounds: int = 2,
) -> ReviewLoop[str]:
    return ReviewLoop[str](
        primary_llm=primary,  # type: ignore[arg-type]
        review_llm=secondary,  # type: ignore[arg-type]
        prompt_builder=None,  # type: ignore[arg-type]
        max_rounds=max_rounds,
    )


class TestReviewLoopSync:
    def test_rounds_alternate_secondary_then_primary(self) -> None:
        """Odd rounds -> review client, even rounds -> primary (M-plan §4.1)."""
        primary = FakeClient([])
        secondary = FakeClient([])
        loop = make_loop(primary, secondary, max_rounds=2)

        seen_clients: list[object] = []

        def call_llm(system: str, user: str, client: object, round_idx: int) -> list[str]:
            seen_clients.append(client)
            return [f"refined-{len(seen_clients)}"]

        result = loop.run(
            ["original"],
            build_prompt=lambda artifact, round_idx: (
                f"sys-{round_idx}",
                f"user-{round_idx}",
            ),
            parse=None,
            call_llm=call_llm,
        )

        assert seen_clients == [secondary, primary]
        assert result.artifact == ["refined-2"]
        assert result.rounds_executed == 2
        assert result.rounds_succeeded == 2
        assert result.used_review is True

    def test_failed_round_keeps_previous_and_next_round_receives_it(self) -> None:
        """R1 gap: round-2 failure must not discard round-1's improvement."""
        primary = FakeClient([])
        secondary = FakeClient([])

        outcomes = [["v1"], [], ["v3"]]
        prompts_seen: list[tuple[list[str], int]] = []

        def call_llm(system: str, user: str, client: object, round_idx: int) -> list[str]:
            return outcomes.pop(0)

        def build_prompt(artifact: list[str], round_idx: int) -> tuple[str, str]:
            prompts_seen.append((artifact, round_idx))
            return ("s", "u")

        loop2 = ReviewLoop[list[str]](
            primary_llm=primary,  # type: ignore[arg-type]
            review_llm=secondary,  # type: ignore[arg-type]
            prompt_builder=None,  # type: ignore[arg-type]
            max_rounds=3,
        )
        result = loop2.run(
            ["original"],
            build_prompt=build_prompt,
            parse=None,
            call_llm=call_llm,
        )

        # Round 3 must receive round-1's artifact (round 2 produced nothing).
        assert prompts_seen[2] == (["v1"], 3)
        assert result.artifact == ["v3"]
        assert result.rounds_executed == 3
        assert result.rounds_succeeded == 2
        assert result.used_review is True

    def test_shrunk_list_result_is_rejected_and_previous_kept(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A round that salvages/returns a drastically smaller list (truncated
        response) must NOT replace the accumulated artifact — it is treated as
        a failed round and the next round receives the previous artifact."""
        outcomes: list[list[str]] = [["a", "b", "c", "d"], ["only-1"]]
        prompts_seen: list[tuple[list[str], int]] = []

        def call_llm(system: str, user: str, client: object, round_idx: int) -> list[str]:
            return outcomes.pop(0)

        def build_prompt(artifact: list[str], round_idx: int) -> tuple[str, str]:
            prompts_seen.append((artifact, round_idx))
            return ("s", "u")

        loop2 = ReviewLoop[list[str]](
            primary_llm=FakeClient([]),  # type: ignore[arg-type]
            review_llm=FakeClient([]),  # type: ignore[arg-type]
            prompt_builder=None,  # type: ignore[arg-type]
            max_rounds=2,
        )
        with caplog.at_level(logging.WARNING):
            result = loop2.run(
                ["original"],
                build_prompt=build_prompt,
                parse=None,
                call_llm=call_llm,
            )

        # Round 2 must have seen round-1's 4-item artifact and returned 1 item;
        # the shrink is rejected, so the final artifact stays round-1's list.
        assert prompts_seen[1] == (["a", "b", "c", "d"], 2)
        assert result.artifact == ["a", "b", "c", "d"]
        assert result.rounds_succeeded == 1
        assert result.used_review is True
        assert any("shrank the artifact" in rec.message for rec in caplog.records)

    def test_shrink_at_exactly_half_is_accepted(self) -> None:
        """Boundary: exactly 50% retention is legitimate dedup, not regression."""
        outcomes: list[list[str]] = [["a", "b", "c", "d"], ["a", "b"]]
        loop2 = ReviewLoop[list[str]](
            primary_llm=FakeClient([]),  # type: ignore[arg-type]
            review_llm=FakeClient([]),  # type: ignore[arg-type]
            prompt_builder=None,  # type: ignore[arg-type]
            max_rounds=2,
        )
        result = loop2.run(
            ["original"],
            build_prompt=lambda a, i: ("s", "u"),
            parse=None,
            call_llm=lambda s, u, c, i: outcomes.pop(0),
        )
        assert result.artifact == ["a", "b"]
        assert result.rounds_succeeded == 2

    def test_string_artifact_unaffected_by_shrink_guard(self) -> None:
        """Script artifacts (str) have no length-regression semantics."""
        loop = make_loop(FakeClient([]), FakeClient([]), max_rounds=1)
        result = loop.run(
            "a very long original script " * 10,
            build_prompt=lambda a, i: ("s", "u"),
            parse=None,
            call_llm=lambda s, u, c, i: "short",
        )
        assert result.artifact == "short"
        assert result.rounds_succeeded == 1

    def test_all_rounds_fail_returns_original_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        primary = FakeClient([])
        secondary = FakeClient([])
        loop = make_loop(primary, secondary, max_rounds=2)

        with caplog.at_level(logging.WARNING):
            result = loop.run(
                ["original"],
                build_prompt=lambda a, i: ("s", "u"),
                parse=None,
                call_llm=lambda s, u, c, i: [],
            )

        assert result.artifact == ["original"]
        assert result.used_review is False
        assert result.rounds_succeeded == 0
        assert any("all 2 rounds failed" in rec.message for rec in caplog.records)

    def test_build_prompt_receives_round_index(self) -> None:
        """M3: build_prompt signature carries the 1-based round index."""
        loop = make_loop(FakeClient([]), FakeClient([]), max_rounds=2)
        seen_idx: list[int] = []

        def build_prompt(artifact: str, round_idx: int) -> tuple[str, str]:
            seen_idx.append(round_idx)
            return ("s", "u")

        loop.run(
            "v0",
            build_prompt=build_prompt,
            parse=None,
            call_llm=lambda s, u, c, i: "v-next",
        )
        assert seen_idx == [1, 2]

    def test_without_call_llm_uses_own_chat_and_parse(self) -> None:
        """perf/gui path: no call_llm -> loop calls chat() and parse() itself."""
        primary = FakeClient(["raw-2"])
        secondary = FakeClient(["raw-1"])
        loop = make_loop(primary, secondary, max_rounds=2)

        result = loop.run(
            "script-v0",
            build_prompt=lambda a, i: (f"sys:{a}", "user"),
            parse=lambda raw: f"PARSED({raw})" if raw else None,
        )

        assert secondary.calls == [("sys:script-v0", "user")]
        assert primary.calls == [("sys:PARSED(raw-1)", "user")]
        assert result.artifact == "PARSED(raw-2)"
        assert result.rounds_succeeded == 2
        assert result.used_review is True

    def test_without_call_llm_retries_once_on_parse_failure(self) -> None:
        """Built-in simple retry (R2): one retry per round, no injection."""
        primary = FakeClient([])
        secondary = FakeClient(["garbage", "ok-text"])
        loop = make_loop(primary, secondary, max_rounds=1)

        result = loop.run(
            "v0",
            build_prompt=lambda a, i: ("s", "u"),
            parse=lambda raw: raw.upper() if "ok" in raw else None,
        )

        assert len(secondary.calls) == 2  # initial attempt + one retry
        assert result.artifact == "OK-TEXT"
        assert result.used_review is True

    def test_without_call_llm_all_rounds_fail_keeps_original(self) -> None:
        primary = FakeClient(["junk"])
        secondary = FakeClient(["junk"])
        loop = make_loop(primary, secondary, max_rounds=2)

        result = loop.run(
            "v0",
            build_prompt=lambda a, i: ("s", "u"),
            parse=lambda raw: None,
        )

        assert result.artifact == "v0"
        assert result.used_review is False
        assert result.rounds_succeeded == 0

    def test_zero_rounds_returns_original_without_any_call(self) -> None:
        primary = FakeClient([])
        secondary = FakeClient([])
        loop = make_loop(primary, secondary, max_rounds=0)

        result = loop.run(
            "v0",
            build_prompt=lambda a, i: ("s", "u"),
            parse=lambda raw: raw,
        )

        assert result.artifact == "v0"
        assert result.rounds_executed == 0
        assert result.used_review is False
        assert secondary.calls == [] and primary.calls == []

    def test_structured_round_log_format(self, caplog: pytest.LogCaptureFixture) -> None:
        """M2: each round logs round idx / model / label / parse outcome."""
        primary = FakeClient([], name="primary-model")
        secondary = MetaClient(["fine"], tokens=[42])
        loop = make_loop(primary, secondary, max_rounds=1)

        with caplog.at_level(logging.INFO, logger="testagent.engine.review"):
            loop.run(
                "v0",
                build_prompt=lambda a, i: ("s", "u"),
                parse=lambda raw: raw,
                label="perf-script",
            )

        round_logs = [r.message for r in caplog.records if "review round 1/1" in r.message]
        assert len(round_logs) == 1
        msg = round_logs[0]
        assert "meta-model" in msg
        assert "perf-script" in msg
        assert "parse=ok" in msg
        assert "tokens_out=42" in msg

    def test_model_label_falls_back_to_primary_model(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MultiModelLLMClient exposes ``primary_model`` only; the round log
        must still show a real model name instead of ``unknown``."""
        secondary = FakeClient(["fine"], name="secondary-model")
        del secondary.model_name
        secondary.primary_model = "primary-model"  # type: ignore[attr-defined]
        loop = make_loop(FakeClient([]), secondary, max_rounds=1)

        with caplog.at_level(logging.INFO, logger="testagent.engine.review"):
            loop.run("v0", build_prompt=lambda a, i: ("s", "u"), parse=lambda raw: raw)

        round_logs = [r.message for r in caplog.records if "review round 1/1" in r.message]
        assert len(round_logs) == 1
        assert "primary-model" in round_logs[0]

    def test_review_result_is_frozen_and_slotted(self) -> None:
        result = ReviewResult[str](
            artifact="x", rounds_executed=1, rounds_succeeded=1, used_review=True
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.artifact = "y"  # type: ignore[misc]
        assert not hasattr(result, "__dict__")

    def test_truncation_policy_reserved_slot(self) -> None:
        """Decision 3: constructor accepts a truncation_policy placeholder."""
        sentinel = object()
        loop = ReviewLoop[str](
            primary_llm=FakeClient([]),  # type: ignore[arg-type]
            review_llm=FakeClient([]),  # type: ignore[arg-type]
            prompt_builder=None,  # type: ignore[arg-type]
            truncation_policy=sentinel,
        )
        assert loop._truncation_policy is sentinel


class TestReviewLoopAsync:
    async def test_arun_alternates_and_uses_achat_not_chat(self) -> None:
        """M1: true async — rounds drive achat(); chat() must never run."""
        primary = FakeClient([])
        secondary = FakeClient([])
        loop = make_loop(primary, secondary, max_rounds=2)

        seen_clients: list[object] = []

        async def acall_llm(system: str, user: str, client: object, round_idx: int) -> str:
            seen_clients.append(client)
            return f"a-{len(seen_clients)}"

        result = await loop.arun(
            "v0",
            build_prompt=lambda a, i: ("s", "u"),
            parse=None,
            acall_llm=acall_llm,
        )

        assert seen_clients == [secondary, primary]
        assert result.artifact == "a-2"
        assert result.used_review is True
        # No sync path was touched.
        assert primary.calls == [] and secondary.calls == []

    async def test_arun_all_rounds_fail_returns_original(self) -> None:
        loop = make_loop(FakeClient([]), FakeClient([]), max_rounds=2)

        async def acall_llm(system: str, user: str, client: object, round_idx: int) -> str:
            return ""

        result = await loop.arun(
            "v0",
            build_prompt=lambda a, i: ("s", "u"),
            parse=None,
            acall_llm=acall_llm,
        )

        assert result.artifact == "v0"
        assert result.used_review is False

    async def test_arun_without_acall_llm_uses_achat_and_parse(self) -> None:
        primary = FakeClient(["raw-2"])
        secondary = FakeClient(["raw-1"])
        loop = make_loop(primary, secondary, max_rounds=2)

        result = await loop.arun(
            "v0",
            build_prompt=lambda a, i: ("s", "u"),
            parse=lambda raw: raw.upper() if raw else None,
        )

        assert secondary.async_calls and not secondary.calls
        assert primary.async_calls and not primary.calls
        assert result.artifact == "RAW-2"
        assert result.used_review is True

    async def test_arun_shrunk_list_result_is_rejected(self) -> None:
        """Async mirror of the shrink guard: salvaged 1-item round must not
        replace a larger list artifact."""
        outcomes: list[list[str]] = [["a", "b", "c", "d"], ["only-1"]]

        async def acall_llm(system: str, user: str, client: object, round_idx: int) -> list[str]:
            return outcomes.pop(0)

        loop2 = ReviewLoop[list[str]](
            primary_llm=FakeClient([]),  # type: ignore[arg-type]
            review_llm=FakeClient([]),  # type: ignore[arg-type]
            prompt_builder=None,  # type: ignore[arg-type]
            max_rounds=2,
        )
        result = await loop2.arun(
            ["original"],
            build_prompt=lambda a, i: ("s", "u"),
            parse=None,
            acall_llm=acall_llm,
        )

        assert result.artifact == ["a", "b", "c", "d"]
        assert result.rounds_succeeded == 1
        assert result.used_review is True

    async def test_arun_retries_once_on_parse_failure(self) -> None:
        secondary = FakeClient(["bad", "ok-data"])
        loop = make_loop(FakeClient([]), secondary, max_rounds=1)

        result = await loop.arun(
            "v0",
            build_prompt=lambda a, i: ("s", "u"),
            parse=lambda raw: raw if "ok" in raw else None,
        )

        assert len(secondary.async_calls) == 2
        assert result.artifact == "ok-data"
