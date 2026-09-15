"""Raw-response audit dumping (T1, fix-plan §3.6).

Domain-free file writer wired between the truncation engine's optional
``raw_sink`` hook and the host's session directory. The engine emits
plain-dict records (see ``TruncationEngine``); this class turns them into:

- ``<sessions_dir>/<sid>/<label>-r<round>.raw.txt`` — one file per LLM
  response (the raw ``LLMResponse.text``, byte-for-byte);
- ``<sessions_dir>/<sid>/raw_calls.jsonl`` — one metadata line per record
  (raw and merge kinds; text excluded, it lives in the .raw.txt files);
- ``<sessions_dir>/<sid>/reconciliation.json`` — written once by the host
  after generation: merge rows and their sum against the artifact count
  (``rows_sum == artifact_count`` is the T1 acceptance invariant).

Concurrency: batches fan out through ``gather_with_concurrency``, so the
sink can be called from multiple threads/tasks — every mutating operation
is lock-guarded (shared-mutable-state rule, engine/experience R05).
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

__all__ = ["RawResponseDumper"]

#: Same session-id shape the session store enforces (keeps directory names
#: filesystem-safe regardless of who created the id).
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]+$")

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]")


class RawResponseDumper:
    """Collect raw/merge records for one session and write the audit files."""

    def __init__(self, sessions_dir: Path | str, session_id: str) -> None:
        if not session_id or not _SAFE_SESSION_ID.match(session_id):
            raise ValueError(f"invalid session id for raw dump: {session_id!r}")
        self._session_id = session_id
        self._dir = Path(sessions_dir) / session_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._calls_path = self._dir / "raw_calls.jsonl"
        self._lock = threading.Lock()
        self._merge_rows: list[dict[str, Any]] = []
        self._raw_calls = 0

    def sink(self, record: dict[str, Any]) -> None:
        """Consume one engine record (``kind`` = ``raw`` | ``merge`` | ``drop``)."""
        kind = record.get("kind")
        if kind == "raw":
            self._dump_raw(record)
        elif kind == "merge":
            self._record_merge(record)
        elif kind == "drop":
            # T2/T7: structured scope-filter drop report (consumed via the
            # session journal; the WARNING log is the human channel).
            with self._lock:
                self._append_meta(record)
        else:
            raise ValueError(f"unknown raw-dump record kind: {kind!r}")

    def _dump_raw(self, record: dict[str, Any]) -> None:
        label = str(record.get("label", ""))
        round_no = int(record.get("round", 0))
        text = str(record.get("text", ""))
        stem = _UNSAFE_FILENAME.sub("_", label) or "call"
        path = self._dir / f"{stem}-r{round_no}.raw.txt"
        meta = {k: v for k, v in record.items() if k != "text"}
        with self._lock:
            path.write_text(text, encoding="utf-8")
            self._raw_calls += 1
            self._append_meta(meta)

    def _record_merge(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._merge_rows.append(dict(record))
            self._append_meta(record)

    def _append_meta(self, meta: dict[str, Any]) -> None:
        # Caller holds the lock.
        with self._calls_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False, sort_keys=True) + "\n")

    def write_report(self, filename: str, content: str) -> Path:
        """Write an auxiliary markdown report (e.g. the T4/T9 consistency
        gap report) into the session directory and return its path."""
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", filename):
            raise ValueError(f"unsafe report filename: {filename!r}")
        path = self._dir / filename
        with self._lock:
            path.write_text(content, encoding="utf-8")
        return path

    def write_reconciliation(self, artifact_count: int) -> Path:
        """Write ``reconciliation.json`` and return its path.

        ``rows_sum`` is the sum of all merge rows (each row's ``added``);
        the T1 contract is ``rows_sum == artifact_count`` — every artifact
        must be traceable to exactly one audited merge.
        """
        with self._lock:
            rows = [dict(r) for r in self._merge_rows]
            raw_calls = self._raw_calls
        rows_sum = sum(int(r.get("added", 0)) for r in rows)
        payload = {
            "session_id": self._session_id,
            "artifact_count": artifact_count,
            "rows_sum": rows_sum,
            "match": rows_sum == artifact_count,
            "raw_calls": raw_calls,
            "rows": rows,
        }
        path = self._dir / "reconciliation.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path
