"""Tests for Settings configuration."""

import pytest

from testagent.config.settings import Settings


class TestSettings:
    """Settings defaults and env overrides."""

    def test_defaults(self) -> None:
        """Test default values for pipeline settings."""
        settings = Settings(_env_file=None)
        assert settings.review_enabled is False
        assert settings.output_language == "chinese"
        assert settings.output_dir == "./output"

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test env vars override defaults via alias."""
        monkeypatch.setenv("REVIEW_ENABLED", "true")
        monkeypatch.setenv("OUTPUT_LANGUAGE", "english")
        monkeypatch.setenv("SCRIPT_FORMAT", "jmeter")
        settings = Settings(_env_file=None)
        assert settings.review_enabled is True
        assert settings.output_language == "english"
        assert settings.script_format == "jmeter"

    def test_invalid_language_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test invalid output language is rejected."""
        monkeypatch.setenv("OUTPUT_LANGUAGE", "french")
        with pytest.raises(ValueError):
            Settings(_env_file=None)
