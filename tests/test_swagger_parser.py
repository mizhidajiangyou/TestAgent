"""Tests for SwaggerParser."""

import json
import tempfile
from pathlib import Path

import pytest

from testagent.parsers.swagger_parser import SwaggerParser

SAMPLE_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Test API", "version": "1.0.0"},
    "paths": {
        "/users": {
            "get": {
                "summary": "List users",
                "parameters": [
                    {"name": "page", "in": "query", "schema": {"type": "integer"}},
                    {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                ],
                "responses": {"200": {"description": "Success"}},
                "tags": ["users"],
            },
            "post": {
                "summary": "Create user",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "email": {"type": "string"},
                                },
                            }
                        }
                    }
                },
                "responses": {"201": {"description": "Created"}},
                "tags": ["users"],
            },
        },
        "/users/{id}": {
            "get": {
                "summary": "Get user by ID",
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}},
                ],
                "responses": {
                    "200": {"description": "Success"},
                    "404": {"description": "Not found"},
                },
                "tags": ["users"],
            },
        },
    },
}


class TestSwaggerParser:
    """Test suite for SwaggerParser."""

    def setup_method(self) -> None:
        self.parser = SwaggerParser()

    def test_parse_from_dict(self) -> None:
        """Test parsing from a dict source."""
        endpoints = self.parser.parse(SAMPLE_SPEC)
        assert len(endpoints) == 3

    def test_parse_from_file(self) -> None:
        """Test parsing from a JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(SAMPLE_SPEC, f)
            f.flush()
            path = f.name

        try:
            endpoints = self.parser.parse(path)
            assert len(endpoints) == 3
        finally:
            Path(path).unlink()

    def test_endpoint_fields(self) -> None:
        """Test that endpoint fields are correctly extracted."""
        endpoints = self.parser.parse(SAMPLE_SPEC)

        get_users = next(e for e in endpoints if e.method == "GET" and e.path == "/users")
        assert get_users.summary == "List users"
        assert len(get_users.parameters) == 2
        assert get_users.request_body is None
        assert "200" in get_users.responses
        assert "users" in get_users.tags

        post_users = next(e for e in endpoints if e.method == "POST" and e.path == "/users")
        assert post_users.summary == "Create user"
        assert post_users.request_body is not None
        assert post_users.request_body["media_type"] == "application/json"

    def test_endpoints_to_text(self) -> None:
        """Test text conversion of endpoints."""
        endpoints = self.parser.parse(SAMPLE_SPEC)
        text = SwaggerParser.endpoints_to_text(endpoints)

        assert "GET /users" in text
        assert "POST /users" in text
        assert "GET /users/{id}" in text
        assert "List users" in text
        assert "page, limit" in text

    def test_file_not_found(self) -> None:
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            self.parser.parse("/nonexistent/path.json")

    def test_full_path_property(self) -> None:
        """Test APIEndpoint.full_path property."""
        endpoints = self.parser.parse(SAMPLE_SPEC)
        for ep in endpoints:
            assert ep.full_path == f"{ep.method} {ep.path}"
