"""Web GUI for TestAgent (FastAPI).

Provides a browser interface for generating test cases from requirement
documents, plus a JSON API. Designed to be embeddable in other platforms
via ``<iframe>`` (``frame-ancestors`` is configurable, default ``*``).
"""

from testagent.web.app import create_app

__all__ = ["create_app"]
