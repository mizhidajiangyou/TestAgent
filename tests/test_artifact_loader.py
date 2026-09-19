"""M3 tests: strict loading, chunking, reassembly, serialization (V05-V07)."""

import json
from pathlib import Path

import pytest

from testagent.artifact import (
    ArtifactLoadError,
    load_artifact,
    plan_chunks,
    reassemble_text,
    serialize_json,
)
from testagent.artifact.models import ArtifactFormat


def _write(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


class TestJsonLoading:
    def test_plain_array(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "cases.json", '[{"id": "1"}, {"id": "2"}]')
        artifact = load_artifact(src)
        assert artifact.format is ArtifactFormat.JSON
        assert len(artifact.items) == 2
        assert artifact.json_root_is_envelope is False

    def test_envelope_meta_preserved(self, tmp_path: Path) -> None:
        src = _write(
            tmp_path,
            "cases.json",
            json.dumps({"meta": {"v": 1}, "test_cases": [{"id": "1"}]}),
        )
        artifact = load_artifact(src)
        assert artifact.json_root_is_envelope is True
        assert artifact.envelope_meta == {"meta": {"v": 1}}

    def test_rejects_non_object_entries(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "cases.json", '[{"id": "1"}, "str-entry"]')
        with pytest.raises(ArtifactLoadError, match="not an object"):
            load_artifact(src)

    def test_rejects_duplicate_keys(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "cases.json", '[{"id": "1", "id": "2"}]')
        with pytest.raises(ArtifactLoadError, match="duplicate"):
            load_artifact(src)

    def test_rejects_non_finite(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "cases.json", '[{"id": "1", "score": NaN}]')
        with pytest.raises(ArtifactLoadError, match=r"finite|invalid"):
            load_artifact(src)

    def test_rejects_unknown_root(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "cases.json", '{"foo": 1}')
        with pytest.raises(ArtifactLoadError, match="envelope"):
            load_artifact(src)

    def test_rejects_binary_extension(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "cases.docx", "PK")
        with pytest.raises(ArtifactLoadError, match="extension"):
            load_artifact(src)

    def test_empty_array_loads(self, tmp_path: Path) -> None:
        src = _write(tmp_path, "cases.json", "[]")
        assert load_artifact(src).items == []


class TestTextLoading:
    def test_md_lossless_segmentation(self, tmp_path: Path) -> None:
        content = "# Title\n\nPara one.\n\nPara two with\nwrapped lines.\n\n- list a\n- list b\n"
        src = _write(tmp_path, "doc.md", content)
        artifact = load_artifact(src)
        assert reassemble_text(artifact, {}) == content

    def test_fenced_code_kept_whole(self, tmp_path: Path) -> None:
        content = "Intro para.\n\n```python\nx = 1\n\ny = 2\n```\n\nAfter para.\n"
        src = _write(tmp_path, "doc.md", content)
        artifact = load_artifact(src)
        fenced = [s for s in artifact.text_segments if "```python" in s]
        assert len(fenced) == 1
        assert "y = 2" in fenced[0]
        assert reassemble_text(artifact, {}) == content

    def test_unclosed_fence_rejected(self, tmp_path: Path) -> None:
        content = "Para.\n\n```python\nx = 1\n"
        src = _write(tmp_path, "doc.md", content)
        with pytest.raises(ArtifactLoadError, match="fence"):
            load_artifact(src)

    def test_bom_preserved(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.md"
        p.write_bytes(b"\xef\xbb\xbf# T\n\nBody.")
        artifact = load_artifact(str(p))
        assert artifact.bom == b"\xef\xbb\xbf"
        assert artifact.text_segments[0].startswith("# T")

    def test_crlf_survives(self, tmp_path: Path) -> None:
        content = "Para one.\r\n\r\nPara two.\r\n"
        src = _write(tmp_path, "doc.txt", content)
        artifact = load_artifact(src)
        assert reassemble_text(artifact, {}) == content


class TestChunking:
    def _json_artifact(self, n: int, tmp_path: Path):
        src = _write(
            tmp_path,
            "cases.json",
            json.dumps([{"id": str(i), "title": f"t{i}", "steps": ["s" * 10]} for i in range(n)]),
        )
        return load_artifact(src)

    def test_pack_respects_entry_cap(self, tmp_path: Path) -> None:
        artifact = self._json_artifact(7, tmp_path)
        plan = plan_chunks(
            artifact, chunk_size=3, char_budget=100000, system_prompt_chars=100, reference_chars=0
        )
        assert [c.item_end - c.item_start for c in plan.chunks] == [3, 3, 1]

    def test_oversized_atomic_no_call(self, tmp_path: Path) -> None:
        big = {"id": "big", "steps": ["x" * 50000]}
        src = _write(tmp_path, "cases.json", json.dumps([{"id": "1"}, big, {"id": "3"}]))
        artifact = load_artifact(src)
        plan = plan_chunks(
            artifact, chunk_size=20, char_budget=1000, system_prompt_chars=50, reference_chars=0
        )
        assert 1 in plan.oversized_atomic
        assert all(c.item_start != 1 for c in plan.chunks)

    def test_reference_over_budget_empty_plan(self, tmp_path: Path) -> None:
        artifact = self._json_artifact(3, tmp_path)
        plan = plan_chunks(
            artifact, chunk_size=20, char_budget=100, system_prompt_chars=50, reference_chars=400
        )
        assert plan.chunks == []

    def test_serialize_envelope_and_bytes(self, tmp_path: Path) -> None:
        src = _write(
            tmp_path, "cases.json", json.dumps({"meta": "keep", "test_cases": [{"id": "1"}]})
        )
        artifact = load_artifact(src)
        assert serialize_json(artifact, {}) == artifact.raw_bytes
        rewritten = serialize_json(artifact, {0: {"id": "1", "reviewed": True}})
        data = json.loads(rewritten.decode("utf-8"))
        assert data["meta"] == "keep"
        assert data["test_cases"][0]["reviewed"] is True

    def test_reassemble_replaces_only_target(self, tmp_path: Path) -> None:
        content = "One.\n\nTwo.\n\nThree.\n"
        src = _write(tmp_path, "doc.txt", content)
        artifact = load_artifact(src)
        out = reassemble_text(artifact, {1: "Two (reviewed)."})
        assert out == "One.\n\nTwo (reviewed).\n\nThree.\n"
