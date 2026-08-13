"""Tests for Settings configuration."""

import pytest

from testagent.config.settings import Settings


class TestSettings:
    """Settings defaults and env overrides."""

    def test_defaults(self) -> None:
        """Test default values for pipeline settings."""
        settings = Settings(_env_file=None)
        assert settings.review_enabled is False
        assert settings.review_max_rounds == 2
        assert settings.output_language == "chinese"
        assert settings.output_dir == "./output"

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test env vars override defaults via alias."""
        monkeypatch.setenv("REVIEW_ENABLED", "true")
        monkeypatch.setenv("REVIEW_MAX_ROUNDS", "3")
        monkeypatch.setenv("OUTPUT_LANGUAGE", "english")
        monkeypatch.setenv("SCRIPT_FORMAT", "jmeter")
        settings = Settings(_env_file=None)
        assert settings.review_enabled is True
        assert settings.review_max_rounds == 3
        assert settings.output_language == "english"
        assert settings.script_format == "jmeter"

    def test_invalid_language_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test invalid output language is rejected."""
        monkeypatch.setenv("OUTPUT_LANGUAGE", "french")
        with pytest.raises(ValueError):
            Settings(_env_file=None)

    def test_openai_model_single(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single model name produces a one-element list."""
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        settings = Settings(_env_file=None)
        assert settings.llm.model == "gpt-4o-mini"
        assert settings.llm.models == ["gpt-4o-mini"]
        assert settings.llm.primary_model == "gpt-4o-mini"

    def test_openai_model_comma_separated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Comma-separated model list is parsed and trimmed."""
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini, gpt-4o, gpt-4-turbo")
        settings = Settings(_env_file=None)
        assert settings.llm.models == ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
        assert settings.llm.primary_model == "gpt-4o-mini"  # primary

    def test_openai_model_comma_separated_with_trailing_comma(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Trailing/empty parts are stripped."""
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini,, gpt-4o,")
        settings = Settings(_env_file=None)
        assert settings.llm.models == ["gpt-4o-mini", "gpt-4o"]

    def test_openai_model_empty_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty value falls back to the default single-model list."""
        monkeypatch.setenv("OPENAI_MODEL", "")
        settings = Settings(_env_file=None)
        assert settings.llm.models == ["gpt-4o-mini"]
