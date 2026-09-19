"""M5 tests: publication paths, no-clobber, backup, failure semantics (V13/V14)."""

import json
from pathlib import Path

import pytest

from testagent.artifact import load_artifact
from testagent.artifact.loader import ArtifactLoadError
from testagent.artifact.models import (
    ChunkReviewResult,
    ChunkStatus,
    DocumentReviewOutcome,
    DocumentStatus,
)
from testagent.artifact.publication import (
    publish_outcome,
    resolve_publication_paths,
)


def _src(tmp_path: Path, name: str = "cases.json", n: int = 2) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps([{"id": str(i)} for i in range(n)]), encoding="utf-8")
    return p


def _outcome(status: DocumentStatus, adopted: int = 0) -> DocumentReviewOutcome:
    o = DocumentReviewOutcome()
    o.status = status
    o.adopted_chunks = adopted
    o.used_review = adopted > 0
    r = ChunkReviewResult(chunk_index=0)
    r.status = ChunkStatus.REVIEWED if adopted else ChunkStatus.REVIEW_FAILED
    o.chunk_results.append(r)
    return o


class TestPathResolution:
    def test_default_target_reviewed_suffix(self, tmp_path: Path) -> None:
        src = _src(tmp_path)
        target, report = resolve_publication_paths(src, None, False)
        assert target.name == "cases.reviewed.json"
        assert report.name == "cases.reviewed.review.json"

    def test_explicit_output(self, tmp_path: Path) -> None:
        src = _src(tmp_path)
        target, report = resolve_publication_paths(src, str(tmp_path / "final.json"), False)
        assert target.name == "final.json"
        assert report.name == "final.review.json"

    def test_output_equals_source_rejected(self, tmp_path: Path) -> None:
        src = _src(tmp_path)
        with pytest.raises(ValueError, match="in-place"):
            resolve_publication_paths(src, str(src), False)

    def test_report_collision_foo_review_json(self, tmp_path: Path) -> None:
        """Source foo.review.json + output foo.json -> report == source -> reject."""
        src = _src(tmp_path, "foo.review.json")
        with pytest.raises(ValueError, match="collides"):
            resolve_publication_paths(src, str(tmp_path / "foo.json"), False)

    def test_default_for_same_source_has_no_collision(self, tmp_path: Path) -> None:
        """Same source with default target: foo.review.reviewed.json — no clash."""
        src = _src(tmp_path, "foo.review.json")
        target, report = resolve_publication_paths(src, None, False)
        assert target.name == "foo.review.reviewed.json"
        assert report.name == "foo.review.reviewed.review.json"

    def test_existing_target_rejected(self, tmp_path: Path) -> None:
        src = _src(tmp_path)
        (tmp_path / "out.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="no-clobber"):
            resolve_publication_paths(src, str(tmp_path / "out.json"), False)

    def test_existing_report_rejected(self, tmp_path: Path) -> None:
        src = _src(tmp_path)
        (tmp_path / "out.review.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="no-clobber"):
            resolve_publication_paths(src, str(tmp_path / "out.json"), False)

    def test_in_place_bypasses_target_exists(self, tmp_path: Path) -> None:
        src = _src(tmp_path)
        target, _report = resolve_publication_paths(src, None, True)
        assert target == src.resolve()


class TestPublication:
    def _artifact(self, tmp_path: Path):
        src = _src(tmp_path)
        return load_artifact(str(src)), src

    def test_reviewed_writes_artifact_then_report(self, tmp_path: Path) -> None:
        artifact, src = self._artifact(tmp_path)
        target, report = resolve_publication_paths(src, str(tmp_path / "out.json"), False)
        reviewed = json.dumps({"test_cases": [{"id": "reviewed"}]}).encode("utf-8")
        result, report_data = publish_outcome(
            _outcome(DocumentStatus.REVIEWED, adopted=1),
            artifact,
            src,
            target,
            report,
            in_place=False,
            reviewed_bytes=reviewed,
        )
        assert result.artifact_committed and result.report_committed
        assert json.loads(target.read_text(encoding="utf-8"))["test_cases"][0]["id"] == "reviewed"
        assert json.loads(report.read_text(encoding="utf-8"))["status"] == "REVIEWED"
        assert report_data is not None

    def test_rejected_publishes_report_only(self, tmp_path: Path) -> None:
        artifact, src = self._artifact(tmp_path)
        target, report = resolve_publication_paths(src, None, False)
        result, _ = publish_outcome(
            _outcome(DocumentStatus.REVIEW_REJECTED),
            artifact,
            src,
            target,
            report,
            in_place=False,
            reviewed_bytes=None,
        )
        assert not result.artifact_committed and result.report_committed
        assert not target.exists()
        assert json.loads(report.read_text(encoding="utf-8"))["status"] == "REVIEW_REJECTED"
        assert src.read_text() == json.dumps([{"id": "0"}, {"id": "1"}])  # source untouched

    def test_in_place_creates_backup(self, tmp_path: Path) -> None:
        artifact, src = self._artifact(tmp_path)
        target, report = resolve_publication_paths(src, None, True)
        reviewed = json.dumps([{"id": "reviewed"}]).encode("utf-8")
        result, _ = publish_outcome(
            _outcome(DocumentStatus.REVIEWED, adopted=1),
            artifact,
            src,
            target,
            report,
            in_place=True,
            reviewed_bytes=reviewed,
        )
        assert result.backup_path is not None
        assert Path(result.backup_path).exists()
        assert json.loads(target.read_text(encoding="utf-8"))[0]["id"] == "reviewed"

    def test_artifact_failure_report_has_error_exit_path(self, tmp_path: Path) -> None:
        artifact, src = self._artifact(tmp_path)
        target, report = resolve_publication_paths(src, str(tmp_path / "out.json"), False)
        # Pre-create the target to force a no-clobber artifact failure.
        (tmp_path / "out.json").write_text("{}", encoding="utf-8")
        # resolve would reject; call publish directly to simulate a race.
        result, _report_data = publish_outcome(
            _outcome(DocumentStatus.REVIEWED, adopted=1),
            artifact,
            src,
            target,
            report,
            in_place=False,
            reviewed_bytes=json.dumps([{"id": "x"}]).encode("utf-8"),
        )
        assert result.artifact_committed is False
        assert "artifact-failed" in result.error
        # Report with publication.error still written (best effort).
        assert result.report_committed is True
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["publication"]["artifact_committed"] is False

    def test_bom_loss_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.md"
        p.write_bytes(b"\xef\xbb\xbf# T\n\nBody.")
        artifact = load_artifact(str(p))
        target, report = resolve_publication_paths(p, str(tmp_path / "out.md"), False)
        result, _ = publish_outcome(
            _outcome(DocumentStatus.REVIEWED, adopted=1),
            artifact,
            p,
            target,
            report,
            in_place=False,
            reviewed_bytes=b"# T\n\nBody reviewed.",  # BOM dropped
        )
        assert result.artifact_committed is False
        assert "serialization-invalid" in result.error

    def test_load_errors(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactLoadError, match="not found"):
            load_artifact(str(tmp_path / "missing.json"))
        d = tmp_path / "dir.json"
        d.mkdir()
        with pytest.raises(ArtifactLoadError, match="directory"):
            load_artifact(str(d))
