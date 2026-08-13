"""
Layered configuration with pydantic-settings.

Priority: environment variables > .env file > defaults.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    script_format: Literal["k6", "jmeter"] = Field(default="k6", alias="SCRIPT_FORMAT")

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
