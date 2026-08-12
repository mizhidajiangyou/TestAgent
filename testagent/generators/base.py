"""Abstract base class for generators."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

#: Input data type for a generator.
T_in = TypeVar("T_in")
#: Output data type produced by a generator.
T_out = TypeVar("T_out")


class BaseGenerator(ABC, Generic[T_in, T_out]):
    """Abstract generator interface.

    Subclasses bind their concrete input/output types via the generic
    parameters, e.g. ``BaseGenerator[TestCaseGenInput, list[TestCase]]``.
    """

    @abstractmethod
    def generate(self, data: T_in) -> T_out:
        """Generate output based on input data.

        Args:
            data: Generator input payload.

        Returns:
            Generated content.
        """
        ...

    @abstractmethod
    def save(self, output: T_out, output_path: Path) -> Path:
        """Save generated output to file.

        Args:
            output: Generated content.
            output_path: Target file path.

        Returns:
            Actual saved file path.
        """
        ...
