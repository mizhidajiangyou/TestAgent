"""Abstract base class for report generators."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

#: Input data type for a report generator.
T_in = TypeVar("T_in")


class BaseReport(ABC, Generic[T_in]):
    """Abstract report interface.

    Subclasses bind their concrete input type via the generic parameter,
    e.g. ``BaseReport[TestCaseReportInput]``.
    """

    @abstractmethod
    def generate(self, data: T_in) -> str:
        """Generate report content.

        Args:
            data: Report input payload.

        Returns:
            Report content as string.
        """
        ...

    @abstractmethod
    def save(self, content: str, output_path: Path) -> Path:
        """Save report to file.

        Args:
            content: Report content.
            output_path: Target file path.

        Returns:
            Saved file path.
        """
        ...
