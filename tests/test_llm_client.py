"""Tests for MultiModelLLMClient fallback and secondary client behavior."""

from unittest.mock import MagicMock

import pytest

from testagent.engine.llm_client import MultiModelLLMClient, OpenAIClient, TokenUsage


def _make_client(model_name: str, chat_return: str | Exception = "ok") -> OpenAIClient:
    """Build an OpenAIClient with a mocked underlying client."""
    inner = MagicMock()
    if isinstance(chat_return, Exception):
        inner.chat.completions.create.side_effect = chat_return
    else:
        inner.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=chat_return), finish_reason="stop")],
            usage=MagicMock(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
    return OpenAIClient(client=inner, model=model_name, timeout=1.0, max_output_tokens=100)


class TestMultiModelLLMClient:
    """Multi-model fallback + secondary client behavior."""

    def test_chat_uses_primary_first(self) -> None:
        """Primary model is used when it succeeds."""
        primary = _make_client("gpt-4o-mini", "primary-response")
        secondary = _make_client("gpt-4o", "secondary-response")
        client = MultiModelLLMClient(clients=[primary, secondary])

        result = client.chat("sys", "usr")
        assert result == "primary-response"
        # Secondary never called
        assert secondary.usage.request_count == 0

    def test_chat_falls_back_on_failure(self) -> None:
        """When primary raises, the next model is tried."""
        primary = _make_client("gpt-4o-mini", RuntimeError("primary down"))
        secondary = _make_client("gpt-4o", "secondary-response")
        client = MultiModelLLMClient(clients=[primary, secondary])

        result = client.chat("sys", "usr")
        assert result == "secondary-response"

    def test_chat_all_models_fail_raises(self) -> None:
        """All models failing raises a RuntimeError listing the count."""
        primary = _make_client("gpt-4o-mini", RuntimeError("primary down"))
        secondary = _make_client("gpt-4o", RuntimeError("secondary down"))
        client = MultiModelLLMClient(clients=[primary, secondary])

        with pytest.raises(RuntimeError, match="All 2 model\\(s\\) failed"):
            client.chat("sys", "usr")

    def test_available_models_and_primary(self) -> None:
        """Introspection properties expose the model list and primary."""
        primary = _make_client("gpt-4o-mini")
        secondary = _make_client("gpt-4o")
        client = MultiModelLLMClient(clients=[primary, secondary])

        assert client.available_models == ["gpt-4o-mini", "gpt-4o"]
        assert client.primary_model == "gpt-4o-mini"
        assert client.has_secondary is True

    def test_has_secondary_false_for_single_model(self) -> None:
        """Single-model client reports has_secondary=False."""
        client = MultiModelLLMClient(clients=[_make_client("gpt-4o-mini")])
        assert client.has_secondary is False

    def test_secondary_client_returns_non_primary(self) -> None:
        """With multiple models, secondary_client returns the second one."""
        primary = _make_client("gpt-4o-mini")
        secondary = _make_client("gpt-4o")
        client = MultiModelLLMClient(clients=[primary, secondary])

        assert client.secondary_client() is secondary

    def test_secondary_client_falls_back_to_primary_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Single-model client returns primary and logs a warning."""
        primary = _make_client("gpt-4o-mini")
        client = MultiModelLLMClient(clients=[primary])

        with caplog.at_level("WARNING"):
            result = client.secondary_client()
        assert result is primary
        assert any(
            "Only one model" in rec.message and "gpt-4o-mini" in rec.message
            for rec in caplog.records
        )

    def test_usage_aggregates_across_sub_clients(self) -> None:
        """Aggregated usage sums prompt/completion/total tokens and requests.

        Scenario:
          - client has primary (fails) + secondary (succeeds).
          - One chat call: primary's usage stays 0 (it raised before
            ``_update_usage``), secondary's usage is 1 request / 15 tokens.
          - Aggregate should reflect secondary's contribution only.
        """
        primary = _make_client("gpt-4o-mini", RuntimeError("primary down"))
        secondary = _make_client("gpt-4o", "secondary-response")
        client = MultiModelLLMClient(clients=[primary, secondary])

        result = client.chat("sys", "usr")
        assert result == "secondary-response"

        usage = client.usage
        # Only the secondary sub-client recorded a successful call.
        assert usage.request_count == 1
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15

    def test_usage_sums_multiple_successful_calls(self) -> None:
        """Multiple successful primary calls accumulate in the aggregate."""
        primary = _make_client("gpt-4o-mini", "primary")
        secondary = _make_client("gpt-4o", "secondary")
        client = MultiModelLLMClient(clients=[primary, secondary])

        # Two calls — both go to primary (it succeeds), secondary untouched.
        client.chat("sys", "usr")
        client.chat("sys", "usr")

        usage = client.usage
        assert usage.request_count == 2
        assert usage.prompt_tokens == 20  # 10 per call
        assert usage.completion_tokens == 10  # 5 per call
        assert usage.total_tokens == 30  # 15 per call
        # Secondary was never called
        assert secondary.usage.request_count == 0

    def test_empty_clients_rejected(self) -> None:
        """Constructor rejects an empty client list."""
        with pytest.raises(ValueError, match="at least one"):
            MultiModelLLMClient(clients=[])

    def test_token_usage_add(self) -> None:
        """TokenUsage.add accumulates fields."""
        a = TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3, request_count=1)
        b = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, request_count=4)
        a.add(b)
        assert a.prompt_tokens == 11
        assert a.completion_tokens == 22
        assert a.total_tokens == 33
        assert a.request_count == 5
