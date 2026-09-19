"""M4 tests: review service status matrix, retention, retries (V08-V12)."""

import json
from pathlib import Path
from typing import ClassVar

from testagent.artifact import load_artifact, plan_chunks
from testagent.artifact.models import ChunkStatus, DocumentStatus
from testagent.review import (
    DocumentReviewService,
    parse_model_envelope,
)


class FakeLLM:
    """Rich fake with scripted per-call responses (one per achat_with_meta)."""

    models: ClassVar[list[str]] = ["fake-primary", "fake-secondary"]

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def achat_with_meta(
        self, system_prompt, user_prompt, response_format=None, max_tokens=None
    ):
        self.calls += 1
        from testagent.engine.llm_client import LLMResponse

        text = self._responses.pop(0) if self._responses else "[]"
        return LLMResponse(text=text, finish_reason="stop", completion_tokens=10)

    async def achat(self, system_prompt, user_prompt, response_format=None, max_tokens=None):
        meta = await self.achat_with_meta(system_prompt, user_prompt)
        return meta.text

    def chat_with_meta(self, system_prompt, user_prompt, response_format=None, max_tokens=None):
        self.calls += 1
        from testagent.engine.llm_client import LLMResponse

        text = self._responses.pop(0) if self._responses else "[]"
        return LLMResponse(text=text, finish_reason="stop", completion_tokens=10)

    def chat(self, system_prompt, user_prompt, response_format=None, max_tokens=None):
        meta = self.chat_with_meta(system_prompt, user_prompt)
        return meta.text


class NoopBuilder:
    def render_template(self, name, **ctx):
        return None


def _service(primary, secondary, rounds=2, ratio=0.5):
    return DocumentReviewService(
        primary_llm=primary,
        review_llm=secondary,
        prompt_builder=NoopBuilder(),
        chunk_size=20,
        max_prompt_chars=48000,
        rounds=rounds,
        max_chunk_failure_ratio=ratio,
    )


def _one_chunk(tmp_path: Path, n_items: int = 10):
    src = tmp_path / "cases.json"
    src.write_text(
        json.dumps([{"id": str(i), "title": f"t{i}"} for i in range(n_items)]), encoding="utf-8"
    )
    artifact = load_artifact(str(src))
    plan = plan_chunks(
        artifact, chunk_size=20, char_budget=10**6, system_prompt_chars=100, reference_chars=0
    )
    return artifact, plan.chunks


def _envelope(items: list[dict]) -> str:
    return json.dumps({"items": items}, ensure_ascii=False)


class TestEnvelopeParsing:
    def test_complete_json(self) -> None:
        assert parse_model_envelope('{"items": [1, 2]}') == [1, 2]

    def test_single_full_fence(self) -> None:
        text = '```json\n{"items": [{"a": 1}]}\n```'
        assert parse_model_envelope(text) == [{"a": 1}]

    def test_prose_and_garbage_rejected(self) -> None:
        assert parse_model_envelope('Here you go:\n{"items": []}') is None
        assert parse_model_envelope('{"items": []} hope that helps') is None
        assert parse_model_envelope('{"results": []}') is None
        assert parse_model_envelope("") is None
        assert parse_model_envelope('{"items": "not-a-list"}') is None
        assert parse_model_envelope('{"items": []}') == []


class TestChunkMatrix:
    def test_ten_to_seven_accepted(self, tmp_path: Path) -> None:
        artifact, chunks = _one_chunk(tmp_path, 10)
        secondary = FakeLLM([_envelope([{"id": str(i)} for i in range(7)])])
        outcome = _service(FakeLLM([]), secondary).review_document(artifact, chunks, grounded=False)
        assert outcome.status is DocumentStatus.REVIEWED
        assert outcome.adopted_chunks == 1
        assert outcome.used_review is True

    def test_ten_to_six_rejected(self, tmp_path: Path) -> None:
        artifact, chunks = _one_chunk(tmp_path, 10)
        secondary = FakeLLM([_envelope([{"id": str(i)} for i in range(6)])])
        outcome = _service(primary=FakeLLM([]), secondary=secondary).review_document(
            artifact, chunks, grounded=False
        )
        assert outcome.status is DocumentStatus.REVIEW_REJECTED
        assert outcome.chunk_results[0].status is ChunkStatus.REVIEW_REJECTED

    def test_verbatim_still_used_review(self, tmp_path: Path) -> None:
        artifact, chunks = _one_chunk(tmp_path, 10)
        verbatim = _envelope([{"id": str(i), "title": f"t{i}"} for i in range(10)])
        outcome = _service(primary=FakeLLM([]), secondary=FakeLLM([verbatim])).review_document(
            artifact, chunks, grounded=False
        )
        assert outcome.status is DocumentStatus.REVIEWED
        assert outcome.used_review is True

    def test_all_parse_failures_chunk_failed(self, tmp_path: Path) -> None:
        artifact, chunks = _one_chunk(tmp_path, 10)
        outcome = _service(
            primary=FakeLLM([]), secondary=FakeLLM(["total garbage"])
        ).review_document(artifact, chunks, grounded=False)
        assert outcome.chunk_results[0].status is ChunkStatus.REVIEW_FAILED
        assert outcome.status is DocumentStatus.REVIEW_FAILED

    def test_threshold_strict_gt(self, tmp_path: Path) -> None:
        """1/2 failure with ratio 0.5: NOT > 0.5 -> still REVIEWED (partial)."""
        src = tmp_path / "cases.json"
        src.write_text(json.dumps([{"id": str(i)} for i in range(2)]), encoding="utf-8")
        artifact = load_artifact(str(src))
        plan = plan_chunks(
            artifact, chunk_size=1, char_budget=10**6, system_prompt_chars=10, reference_chars=0
        )
        secondary = FakeLLM(["garbage", _envelope([{"id": "1"}])])
        outcome = _service(primary=FakeLLM([]), secondary=secondary, ratio=0.5).review_document(
            artifact, plan.chunks, grounded=False
        )
        assert outcome.status is DocumentStatus.REVIEWED
        assert outcome.partial is True
        assert outcome.adopted_chunks == 1

    def test_two_of_three_rolls_back(self, tmp_path: Path) -> None:
        src = tmp_path / "cases.json"
        src.write_text(json.dumps([{"id": str(i)} for i in range(3)]), encoding="utf-8")
        artifact = load_artifact(str(src))
        plan = plan_chunks(
            artifact, chunk_size=1, char_budget=10**6, system_prompt_chars=10, reference_chars=0
        )
        secondary = FakeLLM(["garbage", "garbage", _envelope([{"id": "2"}])])
        outcome = _service(primary=FakeLLM([]), secondary=secondary, ratio=0.5).review_document(
            artifact, plan.chunks, grounded=False
        )
        assert outcome.status is DocumentStatus.REVIEW_FAILED
        assert outcome.adopted_chunks == 0
        assert outcome.used_review is False

    def test_disabled_paths(self, tmp_path: Path) -> None:
        artifact, chunks = _one_chunk(tmp_path, 4)
        outcome = _service(primary=FakeLLM([]), secondary=FakeLLM([]), rounds=0).review_document(
            artifact, chunks, grounded=False
        )
        assert outcome.status is DocumentStatus.REVIEW_DISABLED
        outcome = _service(primary=FakeLLM([]), secondary=FakeLLM([])).review_document(
            artifact, chunks, grounded=False, enabled=False
        )
        assert outcome.status is DocumentStatus.REVIEW_DISABLED

    def test_exception_isolated_to_chunk(self, tmp_path: Path) -> None:
        artifact, chunks = _one_chunk(tmp_path, 10)

        class Exploding(FakeLLM):
            def chat_with_meta(self, *a, **k):
                raise RuntimeError("provider down")

            async def achat_with_meta(self, *a, **k):
                raise RuntimeError("provider down")

        outcome = _service(primary=FakeLLM([]), secondary=Exploding([])).review_document(
            artifact, chunks, grounded=False
        )
        assert outcome.chunk_results[0].status is ChunkStatus.REVIEW_FAILED
        assert "RuntimeError" in outcome.chunk_results[0].reason
        assert outcome.status is DocumentStatus.REVIEW_FAILED

    def test_rounds_zero_means_disabled_not_error(self, tmp_path: Path) -> None:
        artifact, chunks = _one_chunk(tmp_path, 4)
        outcome = _service(primary=FakeLLM([]), secondary=FakeLLM([]), rounds=0).review_document(
            artifact, chunks, grounded=False
        )
        assert outcome.status is DocumentStatus.REVIEW_DISABLED
