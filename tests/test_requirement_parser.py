"""Tests for RequirementParser."""

import json
import tempfile
from pathlib import Path

import pytest

from testagent.config.models import TestPriority
from testagent.parsers.requirement_parser import RequirementParser

SAMPLE_REQUIREMENTS = {
    "requirements": [
        {
            "id": "REQ-001",
            "title": "User Registration",
            "description": "Users should be able to register with email and password",
            "module": "auth",
            "priority": "high",
            "acceptance_criteria": [
                "User can register with valid email and password",
                "Password must be at least 8 characters",
                "Email must be unique",
            ],
        },
        {
            "id": "REQ-002",
            "title": "User Login",
            "description": "Users should be able to login with credentials",
            "module": "auth",
            "priority": "high",
        },
    ]
}

SAMPLE_MARKDOWN = """# User Registration

Users should be able to register with email and password.

Acceptance Criteria:
- User can register with valid email and password
- Password must be at least 8 characters

# User Login

Users should be able to login with credentials.
"""


class TestRequirementParser:
    """Test suite for RequirementParser."""

    def setup_method(self) -> None:
        self.parser = RequirementParser()

    def test_parse_from_dict(self) -> None:
        """Test parsing from a dict source."""
        items = self.parser.parse(SAMPLE_REQUIREMENTS)
        assert len(items) == 2
        assert items[0].id == "REQ-001"
        assert items[0].title == "User Registration"
        assert items[0].priority == TestPriority.HIGH
        assert len(items[0].acceptance_criteria) == 3

    def test_parse_from_json_file(self) -> None:
        """Test parsing from a JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(SAMPLE_REQUIREMENTS, f)
            f.flush()
            path = f.name

        try:
            items = self.parser.parse(path)
            assert len(items) == 2
        finally:
            Path(path).unlink()

    def test_parse_markdown(self) -> None:
        """Test parsing from Markdown."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(SAMPLE_MARKDOWN)
            f.flush()
            path = f.name

        try:
            items = self.parser.parse(path)
            assert len(items) == 2
            assert items[0].title == "User Registration"
            assert len(items[0].acceptance_criteria) == 2
        finally:
            Path(path).unlink()

    def test_parse_text(self) -> None:
        """Test parsing from plain text."""
        text = "User Registration\nUsers can register with email.\n\nUser Login\nUsers can login."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            f.flush()
            path = f.name

        try:
            items = self.parser.parse(path)
            assert len(items) == 2
            assert items[0].title == "User Registration"
        finally:
            Path(path).unlink()

    def test_requirements_to_text(self) -> None:
        """Test text conversion of requirements."""
        items = self.parser.parse(SAMPLE_REQUIREMENTS)
        text = RequirementParser.requirements_to_text(items)

        assert "REQ-001" in text
        assert "User Registration" in text
        assert "module: auth" in text
        assert "Acceptance Criteria:" in text

    def test_file_not_found(self) -> None:
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            self.parser.parse("/nonexistent/path.json")

    def test_default_priority(self) -> None:
        """Test default priority assignment."""
        data = {"requirements": [{"title": "Test", "description": "Test req"}]}
        items = self.parser.parse(data)
        assert items[0].priority == TestPriority.MEDIUM
