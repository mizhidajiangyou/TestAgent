"""Abstract base class for parsers."""

from abc import ABC, abstractmethod
from typing import Any

#: Type alias for parser input sources (file path, URL, or raw data).
ParseSource = str | dict[str, Any]


class BaseParser(ABC):
    """Abstract parser interface."""

    @abstractmethod
    def parse(self, source: ParseSource) -> list[Any]:
        """Parse a source and return a list of structured items.

        Args:
            source: Input source (file path, URL, or raw data).

        Returns:
            List of parsed items.
        """
        ...
