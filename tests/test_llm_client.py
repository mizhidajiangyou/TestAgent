"""Tests for MultiModelLLMClient fallback and secondary client behavior."""

from unittest.mock import MagicMock

import pytest

from testagent.engine.llm_client import (
    LLMOutputTooLongError,
    ModelUnavailableError,
    MultiModelLLMClient,
    OpenAIClient,
    TokenUsage,
)


def _chunk(content: str, finish_reason: str = "stop", usage: MagicMock | None = None) -> MagicMock:
    """Build a single streaming chunk mock carrying ``content``/``finish_reason``."""
    if usage is None:
        usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=content), finish_reason=finish_reason)]
    chunk.usage = usage
    return chunk


def _make_client(model_name: str, chat_return: str | Exception = "ok") -> OpenAIClient:
    """Build an OpenAIClient whose mocked ``create`` yields a token stream."""
    inner = MagicMock()
    if isinstance(chat_return, Exception):
        inner.chat.completions.create.side_effect = chat_return
    else:
        # Streaming response: one chunk carrying the full content.
        inner.chat.completions.create.return_value = [
            _chunk(content=chat_return, finish_reason="stop")
        ]
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


class TestOpenAIClientTruncation:
    """Truncation handling: empty truncated output fails fast and clearly."""

    @staticmethod
    def _client(
        stream_content: str,
        stream_finish: str,
        block_content: str,
        block_finish: str,
    ) -> OpenAIClient:
        """Client whose mocked ``create`` returns streaming vs blocking payloads.

        ``stream=True`` yields a token stream (one chunk); ``stream=False``
        yields a blocking ``ChatCompletion``-like response. This lets tests
        exercise the streaming→blocking fallback independently.
        """
        inner = MagicMock()

        def _create(**kwargs: object) -> object:
            if kwargs.get("stream"):
                return [_chunk(content=stream_content, finish_reason=stream_finish)]
            resp = MagicMock()
            resp.choices = [
                MagicMock(
                    message=MagicMock(content=block_content),
                    finish_reason=block_finish,
                )
            ]
            resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            return resp

        inner.chat.completions.create.side_effect = _create
        return OpenAIClient(client=inner, model="m", timeout=1.0, max_output_tokens=100)

    def test_empty_stream_recovers_via_blocking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty streaming response is retried on the blocking channel and recovers.

        This reproduces the field bug: the streaming endpoint returned 0 chars
        + finish_reason=length, failing 2/3 requirements. The fix retries on the
        blocking channel, which succeeds.
        """
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0)
        client = self._client(
            stream_content="",
            stream_finish="length",
            block_content="RECOVERED",
            block_finish="stop",
        )
        assert client.chat("sys", "usr") == "RECOVERED"

    def test_empty_both_channels_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When streaming AND blocking both return empty, raise after retries."""
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0)
        client = self._client(
            stream_content="",
            stream_finish="length",
            block_content="",
            block_finish="length",
        )
        with pytest.raises(LLMOutputTooLongError):
            client.chat("sys", "usr")

    def test_nonempty_truncation_returns_partial(self) -> None:
        """A length-truncated NON-empty response returns partial content."""
        client = self._client(
            stream_content='[{"id":"TC-1"}',
            stream_finish="length",
            block_content='[{"id":"TC-1"}',
            block_finish="length",
        )
        assert client.chat("sys", "usr") == '[{"id":"TC-1"}'

    def test_normal_stream_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A normal streaming success is returned unchanged."""
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0)
        client = self._client(
            stream_content="hello",
            stream_finish="stop",
            block_content="hello",
            block_finish="stop",
        )
        assert client.chat("sys", "usr") == "hello"

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


class TestOpenAIClientAsync:
    """Async chat (``achat``) wraps the sync path via the executor."""

    async def test_achat_returns_content(self) -> None:
        """achat returns the same content as chat."""
        client = _make_client("gpt-4o-mini", "hello")
        result = await client.achat("sys", "usr")
        assert result == "hello"


class TestMultiModelLLMClientAsync:
    """Async fallback mirrors the sync fallback semantics."""

    async def test_achat_uses_primary_first(self) -> None:
        """Primary model is used when it succeeds."""
        primary = _make_client("gpt-4o-mini", "primary-response")
        secondary = _make_client("gpt-4o", "secondary-response")
        client = MultiModelLLMClient(clients=[primary, secondary])

        result = await client.achat("sys", "usr")
        assert result == "primary-response"
        # Secondary never called
        assert secondary.usage.request_count == 0

    async def test_achat_falls_back_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When primary raises, the next model is tried."""
        # Speed up: disable the retry backoff sleeps for this test.
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0)
        primary = _make_client("gpt-4o-mini", RuntimeError("primary down"))
        secondary = _make_client("gpt-4o", "secondary-response")
        client = MultiModelLLMClient(clients=[primary, secondary])

        result = await client.achat("sys", "usr")
        assert result == "secondary-response"

    async def test_achat_all_models_fail_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All models failing raises a RuntimeError listing the count."""
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0)
        primary = _make_client("gpt-4o-mini", RuntimeError("primary down"))
        secondary = _make_client("gpt-4o", RuntimeError("secondary down"))
        client = MultiModelLLMClient(clients=[primary, secondary])

        with pytest.raises(RuntimeError, match="All 2 model\\(s\\) failed"):
            await client.achat("sys", "usr")


class TestModelVerification:
    """Zero-token pre-flight verification (``verify`` / ``averify``)."""

    def _force_openai_path(self, client: OpenAIClient) -> None:
        """Make a MagicMock-backed client take the non-Azure verify path."""
        # MagicMock auto-creates attributes, so ``getattr(_client, 'azure_endpoint')``
        # would otherwise be truthy and short-circuit the Azure skip. Pin it to None.
        client._client.azure_endpoint = None  # type: ignore[attr-defined]

    def test_verify_succeeds_when_model_present(self) -> None:
        """verify() returns silently when ``models.retrieve`` succeeds."""
        client = _make_client("gpt-4o-mini")
        self._force_openai_path(client)
        client._client.models.retrieve.return_value = MagicMock(id="gpt-4o-mini")
        client.verify()  # no raise

    def test_verify_raises_on_unavailable(self) -> None:
        """verify() raises ModelUnavailableError when the model cannot be fetched."""
        client = _make_client("gpt-4o-mini")
        self._force_openai_path(client)
        client._client.models.retrieve.side_effect = RuntimeError("401 auth")
        with pytest.raises(ModelUnavailableError, match="not available"):
            client.verify()

    def test_verify_skips_azure(self) -> None:
        """Azure clients skip the ``/models`` call (no false negative)."""
        client = _make_client("deployment-1")
        client._client.azure_endpoint = "https://x.openai.azure.com"  # type: ignore[attr-defined]
        client.verify()
        client._client.models.retrieve.assert_not_called()

    async def test_averify_mirrors_verify_success(self) -> None:
        """averify() runs the sync verify (success path) in a thread."""
        client = _make_client("gpt-4o-mini")
        self._force_openai_path(client)
        client._client.models.retrieve.return_value = MagicMock(id="gpt-4o-mini")
        await client.averify()  # no raise

    async def test_averify_propagates_error(self) -> None:
        """averify() propagates ModelUnavailableError from the sync verify."""
        client = _make_client("gpt-4o-mini")
        self._force_openai_path(client)
        client._client.models.retrieve.side_effect = RuntimeError("401")
        with pytest.raises(ModelUnavailableError):
            await client.averify()


class TestMultiModelVerification:
    """MultiModelLLMClient.verify: primary failure aborts, fallback failure warns."""

    def _force_openai_path(self, client: OpenAIClient) -> None:
        client._client.azure_endpoint = None  # type: ignore[attr-defined]

    def test_primary_failure_raises(self) -> None:
        """A bad primary model raises ModelUnavailableError (fast fail)."""
        primary = _make_client("gpt-4o-mini")
        secondary = _make_client("gpt-4o")
        self._force_openai_path(primary)
        self._force_openai_path(secondary)
        primary._client.models.retrieve.side_effect = RuntimeError("401")
        client = MultiModelLLMClient(clients=[primary, secondary])
        with pytest.raises(ModelUnavailableError):
            client.verify()

    def test_fallback_failure_only_warns(self) -> None:
        """A bad fallback model must NOT abort generation (primary is fine)."""
        primary = _make_client("gpt-4o-mini")
        secondary = _make_client("gpt-4o")
        self._force_openai_path(primary)
        self._force_openai_path(secondary)
        primary._client.models.retrieve.return_value = MagicMock(id="gpt-4o-mini")
        secondary._client.models.retrieve.side_effect = RuntimeError("401")
        client = MultiModelLLMClient(clients=[primary, secondary])
        client.verify()  # no raise — only the fallback failed
