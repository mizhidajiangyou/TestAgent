"""
Layered configuration with pydantic-settings.

Priority: environment variables > .env file > defaults.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Valid logging levels accepted by ``LOG_LEVEL``. Kept as the single source
#: of truth so both pydantic validation and ``setup_logging`` agree.
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class LLMSettings(BaseSettings):
    """LLM provider configuration.

    ``OPENAI_MODEL`` accepts a comma-separated list of model names. The first
    model is the primary; the rest are fallback candidates used when the
    primary call fails (and as the preferred reviewer when ``REVIEW_ENABLED``
    is true).

    The raw env value is stored as ``model`` (a string, for backward
    compatibility with single-model configs). The parsed list is exposed via
    the ``models`` property.
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    # Stored as the raw string (env var ``OPENAI_MODEL``) for backward
    # compatibility. Use the ``models`` property to get the parsed list.
    model: str = "gpt-4o-mini"
    timeout: int = 300
    max_output_tokens: int = 16000
    # Opt-in JSON mode (OpenAI ``response_format={"type":"json_object"}``).
    # When true, test-case generation wraps its output in
    # ``{"test_cases": [...]}`` so the model is forced to emit valid JSON.
    # Leave disabled (default) when the backend is a non-OpenAI compatible
    # provider that does not support ``response_format`` (e.g. some
    # OpenAI-compatible gateways). The default-off posture keeps the
    # existing bare-array contract and never breaks those endpoints.
    json_mode: bool = Field(default=False, alias="OPENAI_JSON_MODE")
    # Max number of LLM batches generated concurrently in the async path
    # (``TestCaseGenerator.agenerate``). Bounds the asyncio semaphore so we
    # never flood the provider with simultaneous requests (which would trip
    # 429 rate limits). Default 5.
    max_concurrency: int = Field(default=5, alias="OPENAI_MAX_CONCURRENCY")
    # Run a zero-token model availability check (GET /v1/models/{model}) before
    # generation starts, so a misconfigured key / endpoint / model name fails
    # fast (with a clear error) instead of hanging for minutes. Set to False
    # only if your provider does not implement the OpenAI /models API. Default True.
    verify_model: bool = Field(default=True, alias="OPENAI_VERIFY_MODEL")
    # Stream tokens for real-time progress logs during generation. When the
    # provider does not support streaming (or ``stream_options``), the client
    # transparently falls back to a blocking call. Default True.
    stream: bool = Field(default=True, alias="OPENAI_STREAM")
    # Explicit model profile override (plan v10 §3.3): empty = auto-match by
    # model name, falling back to the generic profile. An unknown name fails
    # fast at client construction listing the available profiles.
    model_profile: str = Field(default="", alias="OPENAI_MODEL_PROFILE")
    # First-round effort intent tier (e.g. "low" / "medium" / "disabled");
    # empty = model default. This is an INTENT, not a raw parameter: the
    # resolved profile translates it into the model's dialect.
    reasoning_effort: str = Field(default="", alias="OPENAI_REASONING_EFFORT")
    # One-shot downgrade effort tier for budget-exhausted recovery (plan v10
    # §7). Empty = use the profile's own continuation_intent (deepseek:
    # disabled, qwen: low, openai-reasoning: low).
    continuation_reasoning_effort: str = Field(
        default="", alias="OPENAI_CONTINUATION_REASONING_EFFORT"
    )
    # Qwen-only: thinking_budget cap applied when effort tier "low" is
    # translated for the qwen3.8 profile (UNVERIFIED until diagnosed).
    continuation_thinking_budget: int = Field(
        default=4096, alias="OPENAI_CONTINUATION_THINKING_BUDGET"
    )
    # Hard wall-clock limit (seconds) for a single blocking (non-streaming)
    # LLM call (plan-c B1.1/B2.2). An EXPLICIT value is honoured verbatim
    # (must be >= OPENAI_TIMEOUT, else startup fails); unset means
    # max(OPENAI_TIMEOUT * 2, 600). Exceeding it raises LLMCallTimeoutError
    # into a bounded retry path instead of waiting forever on a wedged
    # provider. Default 0 = unset.
    blocking_hard_timeout: int = Field(default=0, alias="OPENAI_BLOCKING_HARD_TIMEOUT")
    # Hard wall-clock limit (seconds) for a single STREAMING call. The blocking
    # path has had a hard timeout since plan-c B1.1, but a stream that keeps
    # emitting chunks (reasoning-heavy models stream thinking tokens
    # continuously) never trips the transport read timeout and had NO cap at
    # all — a single call could therefore occupy the wall clock far beyond
    # OPENAI_TIMEOUT with zero visible progress. Exceeding this limit raises
    # LLMCallTimeoutError into the same bounded retry path as the blocking
    # case. Unset means max(OPENAI_TIMEOUT * 2, 600); an explicit value must
    # be >= OPENAI_TIMEOUT (validated at client construction).
    stream_hard_timeout: int = Field(default=0, alias="OPENAI_STREAM_HARD_TIMEOUT")
    # Escape hatch for models that are NOT in the profile registry: a raw
    # ``extra_body`` JSON object merged into every request. The registry
    # translates the ``OPENAI_REASONING_EFFORT`` intent into each family's
    # dialect, so an unregistered model silently receives NO thinking control
    # at all (its effort tier is dropped). Rather than guessing a vendor's
    # dialect, the operator supplies the exact fragment their provider
    # documents, e.g. '{"enable_thinking": false}' or
    # '{"thinking_budget": 4096}'. Empty = disabled.
    # Example: OPENAI_EXTRA_BODY_JSON='{"enable_thinking": false}'
    extra_body_json: str = Field(default="", alias="OPENAI_EXTRA_BODY_JSON")

    @property
    def models(self) -> list[str]:
        """Return the parsed list of model names.

        Splits the raw ``model`` string on commas, strips whitespace, and
        drops empty parts. Always returns at least one entry (falls back to
        ``["gpt-4o-mini"]`` when the field is empty).
        """
        raw = self.model or ""
        parts = [p.strip() for p in raw.split(",")]
        cleaned = [p for p in parts if p]
        return cleaned or ["gpt-4o-mini"]

    @property
    def primary_model(self) -> str:
        """Return the primary (first) model name."""
        return self.models[0]


class AzureLLMSettings(BaseSettings):
    """Azure OpenAI configuration."""

    model_config = SettingsConfigDict(
        env_prefix="AZURE_OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = False
    api_key: str = ""
    endpoint: str = ""
    api_version: str = "2024-10-21"
    deployment: str = "gpt-4o-mini"


class PerfSettings(BaseSettings):
    """Performance test configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PERF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = "https://api.example.com"
    virtual_users: int = 100
    duration_seconds: int = 300
    ramp_up_seconds: int = 60
    think_time_ms: int = 500
    auth_type: str = "none"


class Settings(BaseSettings):
    """Root application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    azure_llm: AzureLLMSettings = Field(default_factory=AzureLLMSettings)
    perf: PerfSettings = Field(default_factory=PerfSettings)

    output_dir: str = Field(default="./output", alias="OUTPUT_DIR")
    # Root log level for the ``testagent`` logger. Drives ``setup_logging``
    # (case-insensitive). The CLI ``-v/--verbose`` flag overrides this to
    # DEBUG for a one-off run. Invalid values fail fast at startup.
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        """Normalize and validate the log level early.

        Rejects typos / unknown levels at startup (clearer than a silent
        fallback to INFO) and makes the value case-insensitive so
        ``LOG_LEVEL=debug`` works as expected in ``.env``.
        """
        normalized = (v or "").strip().upper()
        if normalized not in VALID_LOG_LEVELS:
            raise ValueError(f"Invalid LOG_LEVEL={v!r}; expected one of {sorted(VALID_LOG_LEVELS)}")
        return normalized

    script_format: Literal["k6", "jmeter"] = Field(default="k6", alias="SCRIPT_FORMAT")
    #: Directory scanned for task packages (manifest.json + templates), each
    #: subdirectory becoming a ``testagent run <name>`` command (plan-c B4.1).
    #: Underscore-prefixed dirs (e.g. ``_example``) are loadable but hidden
    #: from the default command listing.
    tasks_dir: str = Field(default="./tasks", alias="TASKS_DIR")

    #: Whether to run a second-pass LLM review after generating test cases.
    review_enabled: bool = Field(default=False, alias="REVIEW_ENABLED")
    #: Max number of cross-validation rounds when review is enabled.
    #: Each round alternates between the primary and a secondary model.
    #: Default 2 = primary generates, secondary reviews once.
    review_max_rounds: int = Field(default=2, alias="REVIEW_MAX_ROUNDS")
    #: Output language for generated content and reports.
    output_language: Literal["english", "chinese"] = Field(
        default="chinese", alias="OUTPUT_LANGUAGE"
    )
    #: Conversation session persistence backend. ``"file"`` persists each
    #: session as ``<output_dir>/conversations/<id>.json`` so it survives
    #: process restarts and can be resumed; ``"memory"`` keeps sessions
    #: in-process only (lost on restart). Default ``"file"``.
    session_store: Literal["file", "memory"] = Field(default="file", alias="SESSION_STORE")

    # --- quality: T1 (fix-plan §3.6; plan-l L-1 physical-write protocol) ---
    #: Dump every raw LLM response of the generation chain to
    #: ``<output_dir>/sessions/<sid>/*.raw.txt`` plus a reconciliation table
    #: (``reconciliation.json``) whose merge rows sum to the artifact count.
    #: ``false`` restores the pre-T1 silent behavior (rollback switch).
    audit_dump_enabled: bool = Field(default=True, alias="AUDIT_DUMP_ENABLED")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
