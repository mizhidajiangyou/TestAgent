"""
Unified LLM client supporting OpenAI and Azure OpenAI.

Features:
- Multi-model fallback: try the primary model, on failure try subsequent models.
- Per-model ``OpenAIClient`` instances with their own retry/timeout/usage.
- ``MultiModelLLMClient`` aggregates token usage across all sub-clients.
- ``secondary_client()`` returns a non-primary client for review; falls back
  to the primary with a warning when only one model is configured.
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from openai import OpenAI
from openai.types.chat import ChatCompletion

from testagent.config.settings import Settings

if TYPE_CHECKING:
    from openai import AzureOpenAI

logger = logging.getLogger(__name__)

#: Default max output tokens (can be overridden via OPENAI_MAX_OUTPUT_TOKENS).
DEFAULT_MAX_OUTPUT_TOKENS = 16000

#: Number of retry attempts for LLM calls.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


@dataclass
class TokenUsage:
    """Cumulative token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0

    def summary(self) -> str:
        """Return a human-readable usage summary."""
        return (
            f"requests={self.request_count}, prompt_tokens={self.prompt_tokens}, "
            f"completion_tokens={self.completion_tokens}, total_tokens={self.total_tokens}"
        )

    def add(self, other: "TokenUsage") -> None:
        """Accumulate another usage record into this one."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.request_count += other.request_count


class LLMClient(Protocol):
    """LLM client interface."""

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request.

        Args:
            system_prompt: System message.
            user_prompt: User message.

        Returns:
            Model response text.
        """
        ...


class OpenAIClient:
    """OpenAI-compatible LLM client with retry, timeout and usage tracking."""

    def __init__(
        self,
        client: "OpenAI | AzureOpenAI",
        model: str,
        timeout: float = 300.0,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout
        self._max_output_tokens = max_output_tokens
        self.usage = TokenUsage()

    @property
    def model_name(self) -> str:
        """Return the model name used by this client."""
        return self._model

    @property
    def max_output_tokens(self) -> int:
        """Return configured max output tokens."""
        return self._max_output_tokens

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request with retry and backoff.

        If the response is truncated (``finish_reason == "length"``), the
        truncated content is still returned (the caller can attempt to
        salvage it). A warning is logged so the caller knows the output
        may be incomplete.
        """
        last_error = ""
        truncated_hint = (
            " [IMPORTANT: Your previous response was truncated. "
            "Please make each case MORE COMPACT: shorter descriptions, "
            "fewer steps, concise expected_results. Fit within the token limit.]"
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                effective_user_prompt = user_prompt
                if attempt > 1:
                    effective_user_prompt = user_prompt + truncated_hint

                response = self._client.chat.completions.create(
                    model=self._model,
                    temperature=0,
                    max_tokens=self._max_output_tokens,
                    timeout=self._timeout,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": effective_user_prompt},
                    ],
                )
                self._update_usage(response)

                choice = response.choices[0]
                content = choice.message.content or ""

                if choice.finish_reason == "length":
                    logger.warning(
                        "Response truncated at max_tokens=%d (attempt %d/%d). "
                        "Returning partial content for salvage.",
                        self._max_output_tokens,
                        attempt,
                        MAX_RETRIES,
                    )
                    # Return the partial content - the caller's _extract_json
                    # will attempt to salvage it by closing brackets.
                    if content.strip():
                        return content.strip()
                    last_error = "Empty response after truncation"
                elif not content.strip():
                    last_error = "Empty response from model"
                    logger.warning("Attempt %d/%d: %s", attempt, MAX_RETRIES, last_error)
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                else:
                    logger.debug("LLM raw response (%d chars): %.500s", len(content), content)
                    return content.strip()
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, last_error)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        raise RuntimeError(
            f"Failed to get LLM response after {MAX_RETRIES} attempts. Last error: {last_error}"
        )

    def _update_usage(self, response: ChatCompletion) -> None:
        """Accumulate token usage from response."""
        self.usage.request_count += 1
        usage = response.usage
        if usage is None:
            return
        self.usage.prompt_tokens += usage.prompt_tokens or 0
        self.usage.completion_tokens += usage.completion_tokens or 0
        self.usage.total_tokens += usage.total_tokens or 0


class MultiModelLLMClient:
    """LLM client that tries multiple models in order with automatic fallback.

    The first model in ``clients`` is the primary. If ``chat()`` raises after
    all internal retries, the next client is tried. Token usage is aggregated
    across all sub-clients so callers see a single total.

    ``secondary_client()`` returns the first non-primary client (preferred for
    review cross-validation). When only one model is configured, the primary
    is returned and a warning is logged so the operator knows the
    "cross-validation" is effectively single-model.
    """

    def __init__(self, clients: list[OpenAIClient]) -> None:
        if not clients:
            raise ValueError("MultiModelLLMClient requires at least one OpenAIClient")
        self._clients: list[OpenAIClient] = list(clients)
        # Aggregate usage snapshot taken lazily to avoid double-counting.
        self._aggregate_usage = TokenUsage()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def available_models(self) -> list[str]:
        """Return the ordered list of model names configured."""
        return [c.model_name for c in self._clients]

    @property
    def primary_model(self) -> str:
        """Return the primary (first) model name."""
        return self._clients[0].model_name

    @property
    def has_secondary(self) -> bool:
        """Return True when at least one non-primary model is configured."""
        return len(self._clients) > 1

    @property
    def clients(self) -> list[OpenAIClient]:
        """Return the underlying per-model clients (read-only view)."""
        return list(self._clients)

    # ------------------------------------------------------------------
    # Aggregated usage
    # ------------------------------------------------------------------

    @property
    def usage(self) -> TokenUsage:
        """Aggregate token usage across all sub-clients.

        Each sub-client already accumulates its own usage; we sum them on
        every access so callers always see a fresh total. The internal
        ``_aggregate_usage`` is kept only for API compatibility with code
        that mutates ``usage`` directly.
        """
        total = TokenUsage()
        for c in self._clients:
            total.add(c.usage)
        # Preserve identity so external mutations still work in tests
        self._aggregate_usage.prompt_tokens = total.prompt_tokens
        self._aggregate_usage.completion_tokens = total.completion_tokens
        self._aggregate_usage.total_tokens = total.total_tokens
        self._aggregate_usage.request_count = total.request_count
        return self._aggregate_usage

    # ------------------------------------------------------------------
    # Chat with fallback
    # ------------------------------------------------------------------

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Call the primary model; on failure, fall back to subsequent models.

        Each sub-client already performs internal retries with backoff. We
        only escalate to the next model once a sub-client exhausts its
        retries and raises ``RuntimeError``.
        """
        last_error: Exception | None = None
        for idx, client in enumerate(self._clients):
            label = "primary" if idx == 0 else f"fallback #{idx}"
            try:
                logger.debug("Trying %s model: %s", label, client.model_name)
                return client.chat(system_prompt, user_prompt)
            except Exception as exc:
                last_error = exc
                if idx < len(self._clients) - 1:
                    logger.warning(
                        "%s model '%s' failed (%s); falling back to next model.",
                        label,
                        client.model_name,
                        exc,
                    )
                else:
                    logger.error(
                        "%s model '%s' failed (%s); no more fallback candidates.",
                        label,
                        client.model_name,
                        exc,
                    )

        raise RuntimeError(
            f"All {len(self._clients)} model(s) failed. Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Secondary client for review
    # ------------------------------------------------------------------

    def secondary_client(self) -> OpenAIClient:
        """Return a non-primary client for cross-validation review.

        Falls back to the primary client (with a warning) when only one
        model is configured, so review can still proceed. Callers should
        inspect ``has_secondary`` or check the returned client's
        ``model_name`` against ``primary_model`` to detect the fallback.
        """
        if len(self._clients) > 1:
            return self._clients[1]
        logger.warning(
            "Only one model ('%s') is configured; review will reuse the primary "
            "model. Cross-validation will be single-model only. "
            "Set OPENAI_MODEL to a comma-separated list (e.g. "
            "'gpt-4o-mini,gpt-4o') to enable true multi-model review.",
            self._clients[0].model_name,
        )
        return self._clients[0]


def create_llm_client(settings: Settings) -> MultiModelLLMClient:
    """Factory function to create a multi-model LLM client.

    Builds one ``OpenAIClient`` per model name in ``settings.llm.models``.
    All clients share the same api_key/base_url (OpenAI-compatible) unless
    Azure is enabled (in which case a single Azure deployment is used and
    fallback across multiple OpenAI models is disabled — Azure users
    should configure the deployment name list explicitly if needed).

    Args:
        settings: Application settings.

    Returns:
        Configured multi-model LLM client.
    """
    timeout = float(settings.llm.timeout)
    max_tokens = settings.llm.max_output_tokens

    if settings.azure_llm.enabled:
        from openai import AzureOpenAI

        azure_client: OpenAI | AzureOpenAI = AzureOpenAI(
            api_key=settings.azure_llm.api_key,
            azure_endpoint=settings.azure_llm.endpoint,
            api_version=settings.azure_llm.api_version,
        )
        # Azure uses deployment names, not model names. If the user lists
        # multiple models, treat each as a separate deployment name.
        deployments = settings.llm.models
        if len(deployments) == 1 and deployments[0] == "gpt-4o-mini":
            # Default: use the configured Azure deployment
            deployments = [settings.azure_llm.deployment]
        clients = [
            OpenAIClient(
                client=azure_client,
                model=dep,
                timeout=timeout,
                max_output_tokens=max_tokens,
            )
            for dep in deployments
        ]
        logger.info("Using Azure OpenAI deployments: %s", deployments)
    else:
        client = OpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )
        clients = [
            OpenAIClient(
                client=client,
                model=model_name,
                timeout=timeout,
                max_output_tokens=max_tokens,
            )
            for model_name in settings.llm.models
        ]
        logger.info("Using OpenAI models (in fallback order): %s", settings.llm.models)

    logger.info("Max output tokens: %d", max_tokens)
    return MultiModelLLMClient(clients=clients)
