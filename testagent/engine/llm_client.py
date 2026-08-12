"""
Unified LLM client supporting OpenAI and Azure OpenAI.

Features: retry with backoff, configurable timeout, token usage tracking.
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

MAX_OUTPUT_TOKENS = 16000
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

    def __init__(self, client: "OpenAI | AzureOpenAI", model: str, timeout: float = 300.0) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout
        self.usage = TokenUsage()

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request with retry and backoff."""
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    temperature=0,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    timeout=self._timeout,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                self._update_usage(response)

                choice = response.choices[0]
                content = choice.message.content or ""

                if choice.finish_reason == "length":
                    logger.warning(
                        "Response truncated at max_tokens=%d; output may be incomplete",
                        MAX_OUTPUT_TOKENS,
                    )

                if not content.strip():
                    last_error = "Empty response from model"
                    logger.warning("Attempt %d/%d: %s", attempt, MAX_RETRIES, last_error)
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue

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


def create_llm_client(settings: Settings) -> OpenAIClient:
    """Factory function to create the appropriate LLM client.

    Args:
        settings: Application settings.

    Returns:
        Configured LLM client.
    """
    timeout = float(settings.llm.timeout)

    if settings.azure_llm.enabled:
        from openai import AzureOpenAI

        client: OpenAI | AzureOpenAI = AzureOpenAI(
            api_key=settings.azure_llm.api_key,
            azure_endpoint=settings.azure_llm.endpoint,
            api_version=settings.azure_llm.api_version,
        )
        model = settings.azure_llm.deployment
        logger.info("Using Azure OpenAI deployment: %s", model)
    else:
        client = OpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )
        model = settings.llm.model
        logger.info("Using OpenAI model: %s", model)

    return OpenAIClient(client=client, model=model, timeout=timeout)
