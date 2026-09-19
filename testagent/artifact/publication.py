"""Publication layer (plan-modular-capabilities-v4 §7).

Path rules, no-clobber atomic commit, in-place backup, and the
report-after-artifact order. Semantic review state and on-disk state are
independent: a REVIEWED outcome can still fail to publish.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from testagent.artifact.models import (
    DocumentReviewOutcome,
    DocumentStatus,
    LoadedArtifact,
    PublicationResult,
    sha256_bytes,
)

__all__ = ["publish_outcome", "render_review_report", "resolve_publication_paths"]

_SOURCE_EXT_ALIASES = {".md": ".md", ".markdown": ".markdown"}


def _report_path_for(target: Path) -> Path:
    """Report = target minus its last extension + '.review.json'
    (v4 §7.1: cases.reviewed.json -> cases.reviewed.review.json)."""
    stem = target.name.rsplit(".", 1)[0] if target.suffix else target.name
    return target.with_name(stem + ".review.json")


def resolve_publication_paths(
    source: Path,
    output: str | None,
    in_place: bool,
) -> tuple[Path, Path]:
    """(target, report) resolution per v4 §7.1. Raises ValueError on any
    forbidden collision. Report = target.with_suffix('.review.json')."""
    source = source.resolve()
    if in_place:
        target = source
    elif output:
        target = Path(output).resolve()
        if target == source:
            raise ValueError(f"output equals source ({source}); use --in-place for in-place review")
    else:
        # Replace the LAST extension: cases.json -> cases.reviewed.json;
        # foo.review.json -> foo.review.reviewed.json (v4 §7.1 examples).
        target = source.with_suffix(".reviewed.json")
    report = _report_path_for(target)
    if report == source and not in_place:
        raise ValueError(
            f"report path {report} collides with the source file; choose a different --output"
        )
    if report == target:
        raise ValueError(f"report path collides with target: {report}")
    if target.exists() and not in_place:
        raise ValueError(f"target already exists (no-clobber): {target}")
    if report.exists():
        raise ValueError(f"report already exists (no-clobber): {report}")
    if target.parent != source.parent and not target.parent.exists():
        raise ValueError(f"target parent directory missing: {target.parent}")
    return target, report


def render_review_report(
    run_id: str,
    outcome: DocumentReviewOutcome,
    source: LoadedArtifact,
    prospective_target: Path,
    publication: PublicationResult,
) -> bytes:
    """Single authoritative report: report.status IS the review status."""
    payload = {
        "report_version": 1,
        "run_id": run_id,
        "status": outcome.status.value,
        "source_path": source.source_path,
        "source_sha256": source.source_sha256,
        "prospective_target": str(prospective_target),
        "adopted_chunks": outcome.adopted_chunks,
        "used_review": outcome.used_review,
        "partial": outcome.partial,
        "candidate_reviewed_chunks": outcome.candidate_reviewed_chunks,
        "chunks": [
            {
                "index": r.chunk_index,
                "status": r.status.value,
                "original_count": r.original_count,
                "final_count": r.final_count,
                "parse_ok": r.parse_ok,
                "reason": r.reason,
            }
            for r in outcome.chunk_results
        ],
        "diagnostics": outcome.diagnostics,
        "publication": {
            "artifact_committed": publication.artifact_committed,
            "report_committed": publication.report_committed,
            "backup_path": publication.backup_path,
            "artifact_sha256": publication.artifact_sha256,
            "error": publication.error,
        },
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes, *, exclusive: bool) -> None:
    """Single-file atomic publish; exclusive=True refuses existing targets
    (no-clobber for concurrent runs)."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_bytes(data)
    with tmp.open("rb+") as fh:
        fh.flush()
        os.fsync(fh.fileno())
    try:
        if exclusive and path.exists():
            tmp.unlink(missing_ok=True)
            raise FileExistsError(f"target appeared concurrently: {path}")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _validate_serialized(data: bytes, artifact: LoadedArtifact) -> None:
    """Re-parse what we are about to publish (v4 §7.2 step 1)."""
    import json

    text = data.decode("utf-8")
    parsed = json.loads(text)
    if artifact.format.value == "json":
        if isinstance(parsed, dict):
            assert isinstance(parsed.get("test_cases"), list)
        else:
            assert isinstance(parsed, list)
    if artifact.bom:
        raise ValueError("BOM lost in serialization")


def publish_outcome(
    outcome: DocumentReviewOutcome,
    artifact: LoadedArtifact,
    source_path: Path,
    target: Path,
    report: Path,
    *,
    in_place: bool,
    reviewed_bytes: bytes | None,
) -> tuple[PublicationResult, bytes | None]:
    """Publish per v4 §7.2 ordering. Returns (result, report_bytes).

    Only REVIEWED prepares reviewed artifact bytes. REJECTED/FAILED/
    DISABLED publish the report only. On artifact failure, a report with
    publication.error is still attempted (best effort).
    """
    run_id = uuid.uuid4().hex[:12]
    result = PublicationResult(run_id=run_id, target_path=str(target), report_path=str(report))

    if outcome.status is not DocumentStatus.REVIEWED:
        report_data = render_review_report(run_id, outcome, artifact, target, result)
        try:
            _atomic_write(report, report_data, exclusive=True)
            result.report_committed = True
            result.report_sha256 = sha256_bytes(report_data)
            return result, report_data
        except Exception as exc:
            result.error = f"report-failed:{type(exc).__name__}"
            return result, None

    if reviewed_bytes is None:
        result.error = "no-reviewed-bytes"
        report_data = render_review_report(run_id, outcome, artifact, target, result)
        try:
            _atomic_write(report, report_data, exclusive=True)
            result.report_committed = True
        except Exception as exc:
            result.error += f";report-failed:{type(exc).__name__}"
        return result, None

    # 1. serialize + validate
    try:
        _validate_serialized(reviewed_bytes, artifact)
    except Exception as exc:
        result.error = f"serialization-invalid:{type(exc).__name__}"
        report_data = render_review_report(run_id, outcome, artifact, target, result)
        try:
            _atomic_write(report, report_data, exclusive=True)
            result.report_committed = True
        except Exception as exc2:
            result.error += f";report-failed:{type(exc2).__name__}"
        return result, None

    # 2. backup for in-place
    backup: Path | None = None
    if in_place:
        try:
            backup = source_path.with_name(f"{source_path.name}.bak.{run_id}")
            backup.write_bytes(artifact.raw_bytes)
            with backup.open("rb+") as fh:
                os.fsync(fh.fileno())
            result.backup_path = str(backup)
        except Exception as exc:
            result.error = f"backup-failed:{type(exc).__name__}"
            report_data = render_review_report(run_id, outcome, artifact, target, result)
            try:
                _atomic_write(report, report_data, exclusive=True)
                result.report_committed = True
            except Exception as exc2:
                result.error += f";report-failed:{type(exc2).__name__}"
            return result, None

    # 3. artifact commit (in-place replace / exclusive publish)
    try:
        if in_place:
            _atomic_write(target, reviewed_bytes, exclusive=False)
        else:
            _atomic_write(target, reviewed_bytes, exclusive=True)
        result.artifact_committed = True
        result.artifact_sha256 = sha256_bytes(reviewed_bytes)
    except Exception as exc:
        result.error = f"artifact-failed:{type(exc).__name__}"
        report_data = render_review_report(run_id, outcome, artifact, target, result)
        try:
            _atomic_write(report, report_data, exclusive=True)
            result.report_committed = True
        except Exception as exc2:
            result.error += f";report-failed:{type(exc2).__name__}"
        return result, None

    # 4. report after artifact
    report_data = render_review_report(run_id, outcome, artifact, target, result)
    try:
        _atomic_write(report, report_data, exclusive=True)
        result.report_committed = True
        result.report_sha256 = sha256_bytes(report_data)
    except Exception as exc:
        result.error = f"report-failed-after-artifact:{type(exc).__name__}"
        return result, None
    return result, report_data
