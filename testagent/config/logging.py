"""Logging configuration for TestAgent."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure application-wide logging.

    Idempotent: calling it repeatedly (CLI startup, web startup, tests)
    replaces the handler instead of stacking duplicate ones.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Case-insensitive. Unknown values fall back to INFO.
    """
    log_level = getattr(logging, level.strip().upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger("testagent")
    root.setLevel(log_level)
    # Replace any prior app handler so re-setup (web/tests) does not duplicate
    # log lines. Disable propagation so WARNING+ messages are not also emitted
    # by the root logger's last-resort handler (which would double-print).
    root.handlers = [handler]
    root.propagate = False
