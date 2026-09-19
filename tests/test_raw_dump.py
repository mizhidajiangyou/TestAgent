"""T1 raw-response audit tests (fix-plan §3.6).

Covers:
- ``RawResponseDumper``: raw file layout, filename sanitization, meta
  journal, reconciliation invariant (``rows_sum == artifact_count``),
  session-id validation, concurrency safety.
- Engine ``raw_sink`` emission through the real generator path (fake LLM):
  raw files exist for every call, merge rows sum to the artifact count.
- Wiring defaults: dumping off for direct constructions, on via container
  params.
"""

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from testagent.config.models import APIEndpoint, TestCaseGenInput
from testagent.engine.prompt_builder import PromptBuilder
from testagent.engine.raw_dump import RawResponseDumper
from testagent.generators.testcase_generator import TestCaseGenerator
from tests.test_testcase_generator import MOCK_LLM_RESPONSE, FakeAsyncLLM

_ENDPOINTS = [
    APIEndpoint(method="GET", path="/users", summary="List users"),
    APIEndpoint(method="POST", path="/users", summary="Create user"),
]


class TestRawResponseDumper:
    def test_writes_raw_file_and_meta(self, tmp_path: Path) -> None:
        dumper = RawResponseDumper(tmp_path, "sess0001abc")
        dumper.sink(
            {
                "kind": "raw",
                "label": "REQ-001/3",
                "round": 1,
                "text": "HELLO RAW",
                "finish_reason": "stop",
                "completion_tokens": 12,
            }
        )
        raw_file = tmp_path / "sess0001abc" / "REQ-001_3-r1.raw.txt"
        assert raw_file.read_text(encoding="utf-8") == "HELLO RAW"
        journal = (tmp_path / "sess0001abc" / "raw_calls.jsonl").read_text(encoding="utf-8")
        meta = json.loads(journal.splitlines()[0])
        assert meta["kind"] == "raw" and meta["round"] == 1 and meta["finish_reason"] == "stop"
        assert "text" not in meta  # the payload lives in the .raw.txt file

    def test_reconciliation_match(self, tmp_path: Path) -> None:
        dumper = RawResponseDumper(tmp_path, "s1")
        dumper.sink({"kind": "raw", "label": "a", "round": 1, "text": "x"})
        dumper.sink({"kind": "merge", "label": "a", "added": 3})
        dumper.sink({"kind": "merge", "label": "a", "added": 2})
        path = dumper.write_reconciliation(5)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["rows_sum"] == 5
        assert payload["match"] is True
        assert payload["artifact_count"] == 5
        assert payload["raw_calls"] == 1

    def test_reconciliation_mismatch_flagged(self, tmp_path: Path) -> None:
        dumper = RawResponseDumper(tmp_path, "s2")
        dumper.sink({"kind": "merge", "label": "a", "added": 3})
        payload = json.loads(dumper.write_reconciliation(4).read_text(encoding="utf-8"))
        assert payload["match"] is False and payload["rows_sum"] == 3

    def test_reconciliation_full_chain(self, tmp_path: Path) -> None:
        """Full-chain contract: merges - dedup - budget trims == artifacts."""
        dumper = RawResponseDumper(tmp_path, "s5")
        dumper.sink({"kind": "merge", "label": "a", "added": 10})
        payload = json.loads(
            dumper.write_reconciliation(8, dedup_removed=2, budget_trimmed=0).read_text(
                encoding="utf-8"
            )
        )
        assert payload["match"] is True
        assert payload["dedup_removed"] == 2

    def test_invalid_session_id_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            RawResponseDumper(tmp_path, "bad/id")

    def test_unknown_kind_rejected(self, tmp_path: Path) -> None:
        dumper = RawResponseDumper(tmp_path, "s3")
        with pytest.raises(ValueError):
            dumper.sink({"kind": "nope"})

    def test_concurrent_sinks_are_lock_guarded(self, tmp_path: Path) -> None:
        dumper = RawResponseDumper(tmp_path, "s4")

        def worker(i: int) -> None:
            dumper.sink({"kind": "merge", "label": f"batch-{i}", "added": 1})

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        payload = json.loads(dumper.write_reconciliation(24).read_text(encoding="utf-8"))
        assert payload["rows_sum"] == 24 and len(payload["rows"]) == 24


class TestGeneratorAuditWiring:
    def _generator(self, audit_dump_dir: str | None, enabled: bool = True) -> TestCaseGenerator:
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MOCK_LLM_RESPONSE
        return TestCaseGenerator(
            llm_client=mock_llm,
            prompt_builder=PromptBuilder(),
            audit_dump_enabled=enabled,
            audit_dump_dir=audit_dump_dir,
        )

    def _async_generator(
        self, audit_dump_dir: str | None, enabled: bool = True
    ) -> TestCaseGenerator:
        return TestCaseGenerator(
            llm_client=FakeAsyncLLM(MOCK_LLM_RESPONSE),
            prompt_builder=PromptBuilder(),
            audit_dump_enabled=enabled,
            audit_dump_dir=audit_dump_dir,
        )

    @staticmethod
    def _input() -> TestCaseGenInput:
        return TestCaseGenInput(requirements=[], endpoints=list(_ENDPOINTS))

    def test_generate_dumps_raw_and_reconciles(self, tmp_path: Path) -> None:
        generator = self._generator(str(tmp_path))
        cases = generator.generate(self._input(), session_id="auditcase01")
        session_dir = tmp_path / "sessions" / "auditcase01"
        assert session_dir.is_dir()
        raw_files = list(session_dir.glob("*.raw.txt"))
        assert raw_files, "at least one raw response must be dumped"
        # Every raw file carries the exact LLM text.
        assert any(f.read_text(encoding="utf-8") == MOCK_LLM_RESPONSE for f in raw_files)
        payload = json.loads((session_dir / "reconciliation.json").read_text(encoding="utf-8"))
        assert payload["artifact_count"] == len(cases) == 2
        assert payload["rows_sum"] == payload["artifact_count"]
        assert payload["match"] is True
        assert payload["raw_calls"] >= 1

    def test_dumping_off_when_disabled_or_without_dir(self, tmp_path: Path) -> None:
        # Explicitly disabled with a dir configured.
        generator = self._generator(str(tmp_path), enabled=False)
        generator.generate(self._input(), session_id="audutoff01")
        assert not (tmp_path / "sessions" / "audutoff01").exists()
        # Enabled but no dir (direct constructions without container wiring).
        generator2 = self._generator(None)
        generator2.generate(self._input(), session_id="audutoff02")
        assert not (tmp_path / "sessions" / "audutoff02").exists()

    def test_agenerate_dumps_too(self, tmp_path: Path) -> None:
        generator = self._async_generator(str(tmp_path))
        cases = asyncio.run(generator.agenerate(self._input(), session_id="auditasync1"))
        session_dir = tmp_path / "sessions" / "auditasync1"
        payload = json.loads((session_dir / "reconciliation.json").read_text(encoding="utf-8"))
        assert payload["artifact_count"] == len(cases) == 2
        assert payload["match"] is True
