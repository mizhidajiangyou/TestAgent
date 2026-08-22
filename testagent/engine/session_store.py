"""Conversation session persistence (checkpointer-style store).

Mirrors langgraph's checkpointer pattern: a :class:`SessionStore` persists
conversation state (messages, artifacts, feedback, context cache) keyed by
``session_id`` so a session survives process restarts and can be resumed.

Two backends:

- :class:`InMemoryStore` — process-local dict, the default when no store is
  injected (keeps tests and single-run CLIs dependency-free).
- :class:`FileStore` — one JSON file per session under a base directory
  (default ``output/conversations/``), transparent and grep-able.

The active backend is selected by the ``SESSION_STORE`` setting (``file`` |
``memory``); see :func:`create_session_store`.
"""

import contextlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: A safe session id is alphanumeric with ``-``/``_``. Anything else is rejected
#: to prevent path traversal when the id is used as a file name.
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]+$")

__all__ = [
    "FileStore",
    "InMemoryStore",
    "SessionStore",
    "create_session_store",
]


def _validate_session_id(session_id: str) -> None:
    """Raise ``ValueError`` if the id is not a safe file-name component."""
    if not session_id or not _SAFE_SESSION_ID.match(session_id):
        raise ValueError(
            f"Invalid session id {session_id!r}: must be alphanumeric with "
            "'-' or '_' only (used as a file name)."
        )


@runtime_checkable
class SessionStore(Protocol):
    """Persistence backend for conversation session snapshots.

    A snapshot is a plain ``dict`` produced by
    :meth:`ConversationSession.to_snapshot`. The store is agnostic to the
    snapshot's internal shape; it only keys blobs by ``session_id``.
    """

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Return the snapshot for ``session_id`` or ``None`` if absent."""
        ...

    def save(self, session_id: str, data: dict[str, Any]) -> None:
        """Persist (create or replace) the snapshot for ``session_id``."""
        ...

    def delete(self, session_id: str) -> None:
        """Remove the snapshot for ``session_id`` (no-op if absent)."""
        ...

    def list_ids(self) -> list[str]:
        """Return all persisted session ids."""
        ...


class InMemoryStore:
    """Process-local dict store (no persistence across restarts)."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def load(self, session_id: str) -> dict[str, Any] | None:
        return self._data.get(session_id)

    def save(self, session_id: str, data: dict[str, Any]) -> None:
        self._data[session_id] = data

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    def list_ids(self) -> list[str]:
        return list(self._data.keys())


class FileStore:
    """One JSON file per session, stored under a base directory.

    Files are named ``<session_id>.json`` and written with UTF-8 + indent for
    human readability (so ``output/conversations/<sid>.json`` can be inspected
    or grepped). The base dir is created lazily on first save.
    """

    def __init__(self, base_dir: Path | str) -> None:
        self._base = Path(base_dir)

    def _path(self, session_id: str) -> Path:
        _validate_session_id(session_id)
        return self._base / f"{session_id}.json"

    def load(self, session_id: str) -> dict[str, Any] | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load session %s: %s", session_id, exc)
            return None
        if not isinstance(data, dict):
            logger.warning("Session %s snapshot is not a JSON object", session_id)
            return None
        return data

    def save(self, session_id: str, data: dict[str, Any]) -> None:
        self._base.mkdir(parents=True, exist_ok=True)
        path = self._path(session_id)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()

    def list_ids(self) -> list[str]:
        if not self._base.exists():
            return []
        return sorted(p.stem for p in self._base.glob("*.json"))


def create_session_store(backend: str, output_dir: str = "output") -> SessionStore:
    """Build a :class:`SessionStore` from a backend name.

    Args:
        backend: ``"file"`` for :class:`FileStore`, ``"memory"`` for
            :class:`InMemoryStore`. Any other value falls back to memory with
            a warning (so a misconfigured env var never crashes the app).
        output_dir: Application output directory; file-backed sessions live
            under ``<output_dir>/conversations/<id>.json``.

    Returns:
        The store instance.
    """
    base_dir = Path(output_dir) / "conversations"
    if backend == "file":
        return FileStore(base_dir)
    if backend == "memory":
        return InMemoryStore()
    logger.warning("Unknown SESSION_STORE=%r; falling back to in-memory store.", backend)
    return InMemoryStore()
