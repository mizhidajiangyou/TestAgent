"""Parsers package for API specs and requirement documents."""

from testagent.parsers.requirement_parser import RequirementParser
from testagent.parsers.swagger_parser import SwaggerParser

__all__ = ["RequirementParser", "SwaggerParser"]
