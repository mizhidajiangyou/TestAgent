"""Generic multi-round cross-validation review loop (plan v2 §4.1).

Extracted from ``TestCaseGenerator``'s private ``_review_and_refine`` /
``_areview_and_refine`` mirrors so every generator (testcase / performance /
GUI) can share one review implementation.

Behavioral contract (strictly preserved from the original testcase version):

- Odd rounds use the review (secondary) client, even rounds the primary
  client — true cross-validation when two models are configured.
- Each round runs in a fresh conversation (no prior context).
- A round that fails (empty/unparseable result) keeps the previous round's
  artifact and the loop continues, so a transient failure never discards
  accumulated refinement.
- All rounds failing returns the original artifact with ``used_review=False``.

Consumers inject the artifact-specific behavior:

- ``build_prompt(artifact, round_idx)`` — per-round prompt pair.
- ``parse(raw)`` — turns raw LLM text into the artifact (script generators).
- ``call_llm`` / ``acall_llm`` — optional delegation of the LLM call itself
  (only ``TestCaseGenerator`` passes these, reusing its retry-heavy
  ``_generate_with_retry``; script generators rely on the loop's built-in
  single retry instead — plan v2 R2).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from testagent.engine.llm_client import LLMClient, LLMResponse

if TYPE_CHECKING:
    from testagent.engine.prompt_builder import PromptBuilder
    from testagent.engine.truncation import TruncationPolicy

logger = logging.getLogger(__name__)

#: A review round whose list-shaped artifact shrinks below this fraction of
#: the input is treated as a FAILURE (previous artifact kept). Truncated
#: responses salvaged down to a handful of cases must never replace a
#: complete list — observed 2026-08-23: a salvaged 1-case round silently
#: replaced 30+ cases. Legitimate dedup rarely exceeds 50% reduction.
MIN_RETENTION_RATIO = 0.5


def _coverage_regression(current: object, refined: object) -> bool:
    """True when a list-shaped round result shrank catastrophically.

    Only applies to sized artifacts (lists); scripts (str) are unaffected.
    Empty ``current`` never regresses (first-round semantics).
    """
    if not isinstance(current, list) or not isinstance(refined, list):
        return False
    if len(current) == 0:
        return False
    return len(refined) < len(current) * MIN_RETENTION_RATIO


@dataclass(frozen=True, slots=True)
class ReviewResult[T]:
    """Outcome of a review run (plan v2 M4).

    Attributes:
        artifact: Final artifact (last successful round's output, or the
            original input when every round failed).
        rounds_executed: Number of rounds actually attempted.
        rounds_succeeded: Number of rounds that produced a valid artifact.
        used_review: True when at least one round succeeded; consumers use
            this to mark outputs as reviewed (or not).
    """

    artifact: T
    rounds_executed: int
    rounds_succeeded: int
    used_review: bool


class ReviewLoop[T]:
    """Multi-round alternating-model review loop over an artifact of type T.

    Args:
        primary_llm: Primary (preferred) client; reviews on even rounds.
        review_llm: Review client (usually the secondary model); odd rounds.
        prompt_builder: Shared prompt builder (kept for symmetry with the
            generators; prompt construction itself is injected per call via
            ``build_prompt``).
        max_rounds: Total review rounds. 0 disables the loop entirely.
        truncation_policy: Reserved slot (plan v2 decision 3); unused in this
            phase — wired in when script-review truncation protection lands.
    """

    def __init__(
        self,
        primary_llm: LLMClient,
        review_llm: LLMClient,
        prompt_builder: PromptBuilder,
        max_rounds: int = 2,
        truncation_policy: TruncationPolicy | None = None,
    ) -> None:
        self._llm = primary_llm
        self._review_llm = review_llm
        self._prompt_builder = prompt_builder
        self._max_rounds = max_rounds
        self._truncation_policy = truncation_policy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        artifact: T,
        *,
        build_prompt: Callable[[T, int], tuple[str, str]],
        parse: Callable[[str], T | None] | None = None,
        call_llm: Callable[[str, str, LLMClient, int], T] | None = None,
        label: str = "artifact",
    ) -> ReviewResult[T]:
        """Run the review loop synchronously.

        Args:
            artifact: Initial artifact to review/refine.
            build_prompt: ``(current_artifact, round_idx) -> (system, user)``.
            parse: Raw-text -> artifact parser (used only when ``call_llm``
                is not provided). Returning ``None`` marks the attempt failed.
            call_llm: Optional delegated LLM call
                ``(system, user, client, round_idx) -> artifact`` (TestCaseGenerator
                path). When omitted, the loop calls the client itself with one
                built-in retry per round.
            label: Artifact label for structured logs.
        """
        original = artifact
        current = artifact
        rounds_executed = 0
        rounds_succeeded = 0
        # FH1.4 short-circuit: when the previous round was rejected by the
        # shrink guard AND this round would use the SAME client (single-model
        # fallback), the identical input re-run is skipped — same input to
        # the same model repeats the same failure (修复4, a.log case).
        prev_rejected = False
        prev_client: LLMClient | None = None

        for round_idx in range(1, self._max_rounds + 1):
            rounds_executed += 1
            use_secondary = round_idx % 2 == 1
            # Odd rounds: secondary (non-primary) model; even rounds: primary.
            client = self._review_llm if use_secondary else self._llm
            if prev_rejected and client is prev_client:
                logger.info(
                    "review round %d/%d skipped_after_rejection | label=%s "
                    "(same client, input unchanged after shrink-guard rejection)",
                    round_idx,
                    self._max_rounds,
                    label,
                )
                prev_rejected = True
                prev_client = client
                continue
            system_prompt, user_prompt = build_prompt(current, round_idx)

            tokens_out: int | None = None
            if call_llm is not None:
                refined: T | None = call_llm(system_prompt, user_prompt, client, round_idx)
            else:
                refined, tokens_out = self._call_and_parse_sync(
                    client, system_prompt, user_prompt, parse
                )

            if not refined:
                self._log_round(round_idx, client, label, ok=False, tokens_out=tokens_out)
                prev_rejected = False
                prev_client = client
                continue

            if _coverage_regression(current, refined):
                logger.warning(
                    "review round %d/%d shrank the artifact %d -> %d items "
                    "(< %.0f%% retention, likely a truncated/salvaged response); "
                    "keeping the previous artifact",
                    round_idx,
                    self._max_rounds,
                    len(current),  # type: ignore[arg-type]
                    len(refined),  # type: ignore[arg-type]
                    MIN_RETENTION_RATIO * 100,
                )
                self._log_round(round_idx, client, label, ok=False, tokens_out=tokens_out)
                prev_rejected = True
                prev_client = client
                continue

            rounds_succeeded += 1
            current = refined
            prev_rejected = False
            prev_client = client
            self._log_round(round_idx, client, label, ok=True, tokens_out=tokens_out)

        if self._max_rounds > 0 and rounds_succeeded == 0:
            logger.warning(
                "review all %d rounds failed, returning original artifact",
                self._max_rounds,
            )

        return ReviewResult(
            artifact=current if rounds_succeeded else original,
            rounds_executed=rounds_executed,
            rounds_succeeded=rounds_succeeded,
            used_review=rounds_succeeded > 0,
        )

    async def arun(
        self,
        artifact: T,
        *,
        build_prompt: Callable[[T, int], tuple[str, str]],
        parse: Callable[[str], T | None] | None = None,
        acall_llm: Callable[[str, str, LLMClient, int], Awaitable[T]] | None = None,
        label: str = "artifact",
    ) -> ReviewResult[T]:
        """Async mirror of :meth:`run` (plan v2 M1).

        True async: rounds drive the client's native ``achat``/
        ``achat_with_meta`` path instead of wrapping sync calls in a thread,
        so concurrent reviews do not consume executor threads.
        """
        original = artifact
        current = artifact
        rounds_executed = 0
        rounds_succeeded = 0
        # FH1.4 short-circuit (async mirror): see the sync ``run`` comment.
        prev_rejected = False
        prev_client: LLMClient | None = None

        for round_idx in range(1, self._max_rounds + 1):
            rounds_executed += 1
            use_secondary = round_idx % 2 == 1
            client = self._review_llm if use_secondary else self._llm
            if prev_rejected and client is prev_client:
                logger.info(
                    "review round %d/%d skipped_after_rejection | label=%s "
                    "(same client, input unchanged after shrink-guard rejection)",
                    round_idx,
                    self._max_rounds,
                    label,
                )
                prev_rejected = True
                prev_client = client
                continue
            system_prompt, user_prompt = build_prompt(current, round_idx)

            tokens_out: int | None = None
            if acall_llm is not None:
                refined: T | None = await acall_llm(system_prompt, user_prompt, client, round_idx)
            else:
                refined, tokens_out = await self._call_and_parse_async(
                    client, system_prompt, user_prompt, parse
                )

            if not refined:
                self._log_round(round_idx, client, label, ok=False, tokens_out=tokens_out)
                prev_rejected = False
                prev_client = client
                continue

            if _coverage_regression(current, refined):
                logger.warning(
                    "review round %d/%d shrank the artifact %d -> %d items "
                    "(< %.0f%% retention, likely a truncated/salvaged response); "
                    "keeping the previous artifact",
                    round_idx,
                    self._max_rounds,
                    len(current),  # type: ignore[arg-type]
                    len(refined),  # type: ignore[arg-type]
                    MIN_RETENTION_RATIO * 100,
                )
                self._log_round(round_idx, client, label, ok=False, tokens_out=tokens_out)
                prev_rejected = True
                prev_client = client
                continue

            rounds_succeeded += 1
            current = refined
            prev_rejected = False
            prev_client = client
            self._log_round(round_idx, client, label, ok=True, tokens_out=tokens_out)

        if self._max_rounds > 0 and rounds_succeeded == 0:
            logger.warning(
                "review all %d rounds failed, returning original artifact",
                self._max_rounds,
            )

        return ReviewResult(
            artifact=current if rounds_succeeded else original,
            rounds_executed=rounds_executed,
            rounds_succeeded=rounds_succeeded,
            used_review=rounds_succeeded > 0,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_and_parse_sync(
        self,
        client: LLMClient,
        system_prompt: str,
        user_prompt: str,
        parse: Callable[[str], T | None] | None,
    ) -> tuple[T | None, int | None]:
        """Built-in sync call with one retry per round (plan v2 R2).

        Used by script generators that do not inject ``call_llm``. A failed
        attempt (empty response or ``parse`` returning ``None``/falsy) is
        retried once with the same prompt; a second failure marks the round
        as failed so the caller keeps the previous artifact.
        """
        tokens_out: int | None = None
        for _attempt in range(2):
            text, meta_tokens = self._raw_call_sync(client, system_prompt, user_prompt)
            if meta_tokens is not None:
                tokens_out = meta_tokens
            if not text or not text.strip():
                continue
            parsed: T | None = parse(text) if parse is not None else cast(T, text)
            if not parsed:
                continue
            return parsed, tokens_out
        return None, tokens_out

    async def _call_and_parse_async(
        self,
        client: LLMClient,
        system_prompt: str,
        user_prompt: str,
        parse: Callable[[str], T | None] | None,
    ) -> tuple[T | None, int | None]:
        """Async mirror of :meth:`_call_and_parse_sync` (native ``achat``)."""
        tokens_out: int | None = None
        for _attempt in range(2):
            text, meta_tokens = await self._raw_call_async(client, system_prompt, user_prompt)
            if meta_tokens is not None:
                tokens_out = meta_tokens
            if not text or not text.strip():
                continue
            parsed: T | None = parse(text) if parse is not None else cast(T, text)
            if not parsed:
                continue
            return parsed, tokens_out
        return None, tokens_out

    @staticmethod
    def _raw_call_sync(
        client: LLMClient, system_prompt: str, user_prompt: str
    ) -> tuple[str, int | None]:
        """Call the client, preferring ``chat_with_meta`` when it is real.

        Capability detection (vs. an ``isinstance`` on the client) keeps test
        doubles that only implement ``chat`` working: anything that is not a
        genuine :class:`LLMResponse` falls back to the plain string contract.
        """
        meta_fn = getattr(client, "chat_with_meta", None)
        if callable(meta_fn):
            resp = meta_fn(system_prompt, user_prompt)
            if isinstance(resp, LLMResponse):
                return resp.text, resp.completion_tokens
        return client.chat(system_prompt, user_prompt), None

    @staticmethod
    async def _raw_call_async(
        client: LLMClient, system_prompt: str, user_prompt: str
    ) -> tuple[str, int | None]:
        """Async mirror of :meth:`_raw_call_sync` (``achat_with_meta``)."""
        meta_fn = getattr(client, "achat_with_meta", None)
        if callable(meta_fn):
            resp = await meta_fn(system_prompt, user_prompt)
            if isinstance(resp, LLMResponse):
                return resp.text, resp.completion_tokens
        return await client.achat(system_prompt, user_prompt), None

    def _log_round(
        self,
        round_idx: int,
        client: LLMClient,
        label: str,
        *,
        ok: bool,
        tokens_out: int | None,
    ) -> None:
        """Structured per-round log line (plan v2 M2)."""
        logger.info(
            "review round %d/%d | model=%s | label=%s | parse=%s | tokens_out=%s",
            round_idx,
            self._max_rounds,
            self._model_name(client),
            label,
            "ok" if ok else "fallback_prev",
            tokens_out if tokens_out is not None else "n/a",
        )

    @staticmethod
    def _model_name(client: LLMClient) -> str:
        """Best-effort model label for logs (test doubles may lack it).

        ``MultiModelLLMClient`` exposes ``primary_model`` rather than
        ``model_name`` (it aggregates several sub-clients), so probe both.
        """
        for attr in ("model_name", "primary_model"):
            name = getattr(client, attr, None)
            if isinstance(name, str):
                return name
        return "unknown"
