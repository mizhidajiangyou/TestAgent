"""Tests for SwaggerParser."""

import json
import tempfile
from pathlib import Path

import pytest

from testagent.engine.prompt_builder import (
    endpoints_to_rich_signature,
    endpoints_to_signature,
)
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


# ----------------------------------------------------------------------
# T3 (fix-plan RC-4): response-schema extraction + $ref resolution
# ----------------------------------------------------------------------

REF_SPEC = {
    "openapi": "3.0.2",
    "info": {"title": "Ref API", "version": "1.0.0"},
    "components": {
        "schemas": {
            "User": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "integer"},
                    "profile": {"$ref": "#/components/schemas/Profile"},
                },
            },
            "Profile": {
                "type": "object",
                "properties": {"bio": {"type": "string"}},
            },
            "Error": {
                "type": "object",
                "required": ["message"],
                "properties": {"message": {"type": "string"}},
            },
        }
    },
    "paths": {
        "/users/{id}": {
            "get": {
                "summary": "Get user",
                "parameters": [
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/User"}}
                        },
                    },
                    "404": {
                        "description": "Missing",
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/Error"}}
                        },
                    },
                    "204": {"description": "No content"},
                },
            }
        }
    },
}

SWAGGER2_SPEC = {
    "swagger": "2.0",
    "info": {"title": "Legacy API", "version": "1.0.0"},
    "definitions": {"User": {"type": "object", "properties": {"id": {"type": "integer"}}}},
    "paths": {
        "/users": {
            "get": {
                "summary": "List users",
                "responses": {
                    "200": {
                        "description": "OK",
                        "schema": {"$ref": "#/definitions/User"},
                    }
                },
            }
        }
    },
}


class TestResponseSchemaExtraction:
    """T3: $ref-resolved response schemas; Swagger 2.0 explicit degradation."""

    def setup_method(self) -> None:
        self.parser = SwaggerParser()

    def test_openapi3_resolves_response_refs(self) -> None:
        endpoints = self.parser.parse(REF_SPEC)
        assert len(endpoints) == 1
        ep = endpoints[0]
        user = ep.response_schemas["200"]
        # Nested $ref resolved all the way down.
        assert user["properties"]["id"]["type"] == "integer"
        assert user["properties"]["profile"]["properties"]["bio"]["type"] == "string"
        assert "$ref" not in json.dumps(user)
        error = ep.response_schemas["404"]
        assert error["required"] == ["message"]

    def test_description_only_response_absent(self) -> None:
        """A documented-but-schema-less code (204) must NOT pretend to have
        a schema — renderers distinguish it from documented shapes."""
        endpoints = self.parser.parse(REF_SPEC)
        assert "204" not in endpoints[0].response_schemas
        assert set(endpoints[0].response_schemas) == {"200", "404"}

    def test_swagger2_degrades_to_empty(self) -> None:
        """Swagger 2.0 explicitly degrades to empty (honest blindness)."""
        endpoints = self.parser.parse(SWAGGER2_SPEC)
        assert endpoints[0].response_schemas == {}
        assert endpoints[0].responses == ["200"]  # codes themselves kept

    def test_unresolvable_ref_kept_visible(self) -> None:
        spec = {
            "openapi": "3.0.0",
            "paths": {
                "/x": {
                    "get": {
                        "responses": {
                            "200": {
                                "description": "OK",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/Missing"}
                                    }
                                },
                            }
                        }
                    }
                }
            },
        }
        ep = self.parser.parse(spec)[0]
        assert ep.response_schemas["200"] == {"$ref": "#/components/schemas/Missing"}

    def test_ref_cycle_guarded(self) -> None:
        spec = {
            "openapi": "3.0.0",
            "components": {
                "schemas": {
                    "Node": {
                        "type": "object",
                        "properties": {"child": {"$ref": "#/components/schemas/Node"}},
                    }
                }
            },
            "paths": {
                "/n": {
                    "get": {
                        "responses": {
                            "200": {
                                "description": "OK",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/Node"}
                                    }
                                },
                            }
                        }
                    }
                }
            },
        }
        ep = self.parser.parse(spec)[0]  # must not recurse forever
        assert ep.response_schemas["200"]["type"] == "object"

    def test_request_body_ref_resolved(self) -> None:
        spec = {
            "openapi": "3.0.0",
            "components": {
                "schemas": {
                    "UserIn": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    }
                }
            },
            "paths": {
                "/users": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/UserIn"}
                                }
                            }
                        },
                        "responses": {"201": {"description": "Created"}},
                    }
                }
            },
        }
        ep = self.parser.parse(spec)[0]
        body_schema = (ep.request_body or {}).get("schema", {})
        assert set(body_schema["properties"]) == {"name", "age"}


# ----------------------------------------------------------------------
# T3: rich signature rendering contract (main chain) vs frozen signature
# ----------------------------------------------------------------------


class TestRichSignature:
    """Main-chain contract: bounds/defaults/responses + undefined marker.

    The frozen :func:`endpoints_to_signature` (perf parity + continuation
    prompts) must stay byte-identical for the same inputs.
    """

    def setup_method(self) -> None:
        self.parser = SwaggerParser()
        self.endpoints = self.parser.parse(REF_SPEC)

    def test_gate_constraints_present(self) -> None:
        """fix-plan T3 gate facts: name/age required with bounds, limit
        default visible, response schemas rendered with status codes."""
        text = endpoints_to_rich_signature(self.endpoints)
        assert "id(integer,req)" in text
        assert "message(string,req)" in text
        # Compactness contract: response props render one level deep —
        # profile shows as an object property, its inner fields do not.
        assert "profile(object,opt)" in text
        assert "responses:[200:object{" in text
        assert "404:object{message(string,req)}" in text

    def test_bounds_and_defaults_rendered(self) -> None:
        endpoints = self.parser.parse(
            {
                "openapi": "3.0.0",
                "paths": {
                    "/items": {
                        "get": {
                            "parameters": [
                                {
                                    "name": "limit",
                                    "in": "query",
                                    "schema": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 100,
                                        "default": 20,
                                    },
                                },
                                {
                                    "name": "age",
                                    "in": "query",
                                    "required": True,
                                    "schema": {"type": "integer", "minimum": 0},
                                },
                            ],
                            "responses": {"200": {"description": "OK"}},
                        }
                    }
                },
            }
        )
        text = endpoints_to_rich_signature(endpoints)
        assert "limit(integer,opt,min=1,max=100,default=20)" in text
        assert "age(integer,req,min=0)" in text

    def test_no_schema_marker(self) -> None:
        """Endpoints without documented response schemas carry the explicit
        honesty marker (no envelope / pagination guessing)."""
        endpoints = self.parser.parse(SAMPLE_SPEC)  # description-only responses
        text = endpoints_to_rich_signature(endpoints)
        assert text.count("response schema undefined - do not assume envelope shape") == 3

    def test_swagger2_params_render_top_level_types(self) -> None:
        text = endpoints_to_rich_signature(self.parser.parse(SWAGGER2_SPEC))
        assert "- GET /users" in text
        assert "response schema undefined" in text

    def test_frozen_signature_unchanged(self) -> None:
        """The frozen compact signature must NOT gain bounds/defaults/
        responses (perf parity surface + continuation prompts)."""
        text = endpoints_to_signature(self.endpoints)
        assert "min=" not in text and "max=" not in text and "default=" not in text
        assert "responses:" not in text
        assert "response schema undefined" not in text
        assert text.startswith("- GET /users/{id} params:[id(integer,req)]")
