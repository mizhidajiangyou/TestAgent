"""
Layered configuration with pydantic-settings.

Priority: environment variables > .env file > defaults.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout: int = 300


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
    #: Output language for generated content and reports.
    output_language: Literal["english", "chinese"] = Field(
        default="chinese", alias="OUTPUT_LANGUAGE"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
