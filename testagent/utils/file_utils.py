"""File utility functions."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating if necessary.

    Args:
        path: Directory path.

    Returns:
        The same path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, content: str) -> Path:
    """Write text content to file, creating parent dirs if needed.

    Args:
        path: Target file path.
        content: Text content to write.

    Returns:
        The file path.
    """
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Wrote %d chars to %s", len(content), path)
    return path
