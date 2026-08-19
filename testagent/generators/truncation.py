"""Truncation-aware generation loop (v6).

Implements the state machine from ``output/long_return_truncation_plan_v6.md``:
a single ``while`` loop that keeps calling the LLM until every endpoint's
expected-case quota is covered, the call/time budget is exhausted, or the
scope cannot shrink any further. Truncation is detected from the rich
:class:`~testagent.engine.llm_client.LLMResponse` (``finish_reason`` /
``completion_tokens``) with a character-heuristic fallback when the provider
reports neither.

Design notes:
- The loop is **async-native** (``TruncationEngine.arun``) because the async
  path is the production driver; the sync ``run`` wrapper executes it via
  ``asyncio.run`` (the sync ``generate`` path never runs inside an event loop).
- The engine owns only loop mechanics + pure helpers. JSON extraction /
  salvage / conversion / re-ask prompt building stay on the generator and are
  injected as hooks, so this module has no dependency on prompt templates.
"""

import asyncio
import inspect
import json
import logging
import time
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from testagent.config.models import APIEndpoint, TestCase
from testagent.engine.llm_client import (
    JSON_OBJECT_FORMAT,
    LLMClient,
    LLMOutputTooLongError,
    LLMResponse,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TruncationEngine",
    "TruncationPolicy",
    "build_continue_prompt",
    "chars_per_token_for",
    "filter_to_scope",
    "is_truncated",
    "recompute_covered_pending",
    "shrink_scope",
]


@dataclass(frozen=True)
class TruncationPolicy:
    """Knobs for the truncation-aware generation loop (see plan v6 §5.4)."""

    #: Master switch: when False the generator falls back to the legacy
    #: v2 fixed-retry loop (kept for A/B comparison and emergency rollback).
    enable_v4_resume: bool = True
    #: Per-call output token cap (should mirror OPENAI_MAX_OUTPUT_TOKENS).
    output_token_cap: int = 16000
    #: Ratio of the cap above which a response is considered "near cap" when
    #: the provider did not report finish_reason/completion_tokens.
    heuristic_token_ratio: float = 0.95
    #: Character-to-token ratios used by the fallback heuristic.
    chars_per_token_cjk: float = 1.5
    chars_per_token_lat: float = 4.0
    chars_per_token_mix: float = 2.0
    #: Consecutive zero-progress rounds (in truncated mode) before shrinking.
    stall_threshold: int = 2
    #: Minimum endpoints kept when shrinking scope (the floor).
    min_scope: int = 1
    #: Extra rounds allowed after the scope floor is reached.
    max_single_scope_rounds: int = 2
    #: Hard cap on total LLM calls per batch (budget guard).
    max_total_calls: int = 24
    #: Hard cap on wall time per batch, seconds.
    max_wall_time: float = 600.0
    #: Expected cases generated per endpoint (drives the pending quota).
    default_expected_cases_per_endpoint: int = 2
    #: Consecutive empty responses tolerated before giving up.
    max_empty_streak: int = 3
    #: Minimum remaining time (seconds) to justify another LLM call.
    min_call_budget: float = 30.0


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


def _is_wide(ch: str) -> bool:
    """Return True for fullwidth/CJK characters (roughly 1 token each 1.5 chars)."""
    return unicodedata.east_asian_width(ch) in ("W", "F")


def chars_per_token_for(text: str, policy: TruncationPolicy) -> float:
    """Estimate chars-per-token by the CJK density of the text."""
    if not text:
        return policy.chars_per_token_mix
    wide = sum(1 for ch in text if _is_wide(ch))
    ratio = wide / len(text)
    if ratio > 0.7:
        return policy.chars_per_token_cjk
    if ratio < 0.1:
        return policy.chars_per_token_lat
    return policy.chars_per_token_mix


def _try_parse_ok(text: str) -> bool:
    """Return True when the text parses as complete JSON (array or object)."""
    stripped = text.strip()
    if not stripped:
        return False
    try:
        json.loads(stripped)
        return True
    except json.JSONDecodeError:
        return False


def _looks_truncated(text: str) -> bool:
    """Syntactic truncation heuristic: JSON started, never usefully closed.

    Mirrors the legacy ``_classify_failure`` semantics so responses from
    metadata-less clients (or plain mocks) still route to the truncated
    (salvage + continue) path instead of the generic non-parseable re-ask.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if _try_parse_ok(stripped):
        return False
    return stripped.startswith("[") or stripped.startswith("{")


def is_truncated(result: LLMResponse, policy: TruncationPolicy) -> bool:
    """Decide whether a rich LLM result is (probably) truncated. Plan v6 §5.1."""
    fr = result.finish_reason
    cap = policy.output_token_cap
    text = (result.text or "").rstrip()

    if fr == "length":
        return True

    if fr in (None, "", "unknown"):
        # Degraded mode: no finish_reason reported. Prefer reported tokens,
        # fall back to the script-aware character estimate. Additionally, a
        # syntactic check catches truncated JSON on legacy/mocked clients
        # that report no metadata at all (v2 _classify_failure parity).
        if _looks_truncated(text):
            return True
        if result.completion_tokens:
            near_cap = result.completion_tokens >= cap * policy.heuristic_token_ratio
        elif cap:
            cpt = chars_per_token_for(text, policy)
            near_cap = len(text) >= cap * cpt * policy.heuristic_token_ratio
        else:
            near_cap = False
        return near_cap and not (text.endswith("}") or text.endswith("]"))

    if fr == "stop":
        # Syntactic guard against a "fake stop": providers sometimes report
        # stop on a truncated stream. A parseable payload is trusted.
        if _try_parse_ok(text):
            return False
        return not (text.endswith("}") or text.endswith("]"))

    return False


def filter_to_scope(
    items: list[dict[str, Any]],
    batch_set: set[str],
    expected: dict[str, int],
) -> list[dict[str, Any]]:
    """Single filtering entry: drop items whose endpoint is outside this batch
    (avoiding the downstream ``fallback_ep`` remap pollution) and cap each
    endpoint at its expected quota. Operates on RAW items before conversion.
    """
    kept: list[dict[str, Any]] = []
    count: dict[str, int] = defaultdict(int)
    for it in items:
        ep = str(it.get("endpoint", "") or "")
        if not ep:
            # No endpoint declared: keep it and let the downstream converter's
            # fallback mapping assign one (v2 parity for review responses and
            # models that omit the field).
            kept.append(it)
            continue
        if ep not in batch_set:
            continue
        if count[ep] >= expected.get(ep, 0):
            continue
        count[ep] += 1
        kept.append(it)
    return kept


def recompute_covered_pending(
    expected: dict[str, int],
    produced: list[TestCase],
    covered: dict[str, int],
    pending: dict[str, int],
) -> None:
    """Recompute per-endpoint coverage and remaining quota (in place).

    ``covered`` aggregates produced cases by ``endpoint.full_path``;
    ``pending[ep] = max(0, expected[ep] - covered[ep])``.
    """
    covered.clear()
    for tc in produced:
        ep = tc.endpoint.full_path
        covered[ep] = covered.get(ep, 0) + 1
    for ep in expected:
        pending[ep] = max(0, expected[ep] - covered.get(ep, 0))


def shrink_scope(
    scope: list[str],
    expected: dict[str, int],
    produced: list[TestCase],
    covered: dict[str, int],
    policy: TruncationPolicy,
) -> tuple[list[str], set[str], dict[str, int], bool]:
    """Halve the scope, keeping the best-covered endpoints first.

    Returns ``(new_scope, new_batch_set, new_pending, floor_reached)``.
    """
    del produced  # kept in the signature to mirror the plan; ranking uses covered
    new_n = max(policy.min_scope, len(scope) // 2)
    ranked = sorted(scope, key=lambda ep: covered.get(ep, 0), reverse=True)
    new_scope = ranked[:new_n]
    new_pending = {ep: max(0, expected.get(ep, 0) - covered.get(ep, 0)) for ep in new_scope}
    floor = len(new_scope) <= policy.min_scope
    return new_scope, set(new_scope), new_pending, floor


def compress_fingerprint(produced: list[TestCase], max_items: int = 30) -> str:
    """Compress already-produced cases into a short fingerprint for continue prompts."""
    if not produced:
        return "(none yet)"
    lines = [f"- {tc.title} @ {tc.endpoint.full_path}" for tc in produced[:max_items]]
    if len(produced) > max_items:
        lines.append(f"... and {len(produced) - max_items} more")
    return "\n".join(lines)


def build_continue_prompt(
    base_prompt: str,
    fingerprint: str,
    label: str,
    pending: dict[str, int],
) -> str:
    """Build the CONTINUATION prompt after truncation (plan v6 §4)."""
    pending_text = ", ".join(f"{ep} x{max(0, n)}" for ep, n in pending.items() if n > 0) or "(none)"
    return (
        f"{base_prompt}\n\n---\n"
        f"CONTINUATION for '{label}': your previous response was truncated.\n"
        f"Cases already produced (do NOT repeat these):\n{fingerprint}\n\n"
        f"Still needed (endpoint xcount): {pending_text}\n"
        "Generate ONLY the still-needed cases. Keep each case compact and "
        "return a complete, properly closed JSON array."
    )


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------


class _SyncPreferredAdapter:
    """Wraps an LLM client so the sync ``run`` entry prefers sync methods.

    The engine loop is async-native; when executed via ``run`` (the sync
    ``generate`` path, in a worker thread with its own loop) we bridge every
    call back to the client's sync ``chat`` / ``chat_with_meta`` contract so
    sync-only doubles (legacy fakes) and call-count assertions behave exactly
    like the pre-v2 sync path.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def achat_with_meta(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Any = None,
        max_tokens: Any = None,
    ) -> LLMResponse:
        cm = getattr(self._inner, "chat_with_meta", None)
        if cm is not None:
            result = await asyncio.to_thread(
                cm,
                system_prompt,
                user_prompt,
                response_format=response_format,
                max_tokens=max_tokens,
            )
            if isinstance(result, LLMResponse):
                return result
            # Not a real rich result (e.g. an auto-created MagicMock attribute
            # on a plain double): fall through to the configured sync ``chat``.
        chat = getattr(self._inner, "chat", None)
        if chat is not None:
            text = await asyncio.to_thread(
                chat,
                system_prompt,
                user_prompt,
                response_format=response_format,
                max_tokens=max_tokens,
            )
            return LLMResponse(text=text)
        # Pure-async double (no sync contract at all): delegate unchanged.
        result = await self._inner.achat_with_meta(
            system_prompt, user_prompt, response_format, max_tokens
        )
        if isinstance(result, LLMResponse):
            return result
        return LLMResponse(text=str(result))

    async def achat(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Any = None,
        max_tokens: Any = None,
    ) -> str:
        return await asyncio.to_thread(
            self._inner.chat, system_prompt, user_prompt, response_format, max_tokens
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@dataclass
class _EngineHooks:
    """Callables provided by the host generator (keeps this module pure)."""

    extract_json: Callable[[str], list[Any] | None]
    salvage_truncated: Callable[[str], list[Any] | None]
    to_test_cases: Callable[[list[Any], list[APIEndpoint]], list[TestCase]]
    build_reask: Callable[[str, str, str, str], str]
    case_dedup_key: Callable[[TestCase], str]


class TruncationEngine:
    """Drives the truncation-aware generation loop for one batch.

    The engine is stateless across batches; all mutable state lives inside
    :meth:`arun`. Both the sync (``run``) and async (``arun``) entry points
    execute the same async-native loop — ``run`` wraps it in ``asyncio.run``
    because the sync ``generate`` path is never invoked from an event loop.
    """

    def __init__(
        self,
        policy: TruncationPolicy,
        json_mode: bool,
        hooks: _EngineHooks,
    ) -> None:
        self._policy = policy
        self._json_mode = json_mode
        self._hooks = hooks

    # -- sync entry point ------------------------------------------------

    def run(
        self,
        llm: LLMClient,
        system_prompt: str,
        user_prompt: str,
        endpoints: list[APIEndpoint],
        label: str,
    ) -> list[TestCase]:
        """Run the loop synchronously.

        Normally executes ``asyncio.run`` directly. When called from inside a
        running event loop (e.g. parity tests comparing the sync and async
        paths), falls back to a worker thread with its own loop instead of
        raising — the caller stays blocked either way.
        """
        wrapped: Any = _SyncPreferredAdapter(llm)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(wrapped, system_prompt, user_prompt, endpoints, label))
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run, self.arun(wrapped, system_prompt, user_prompt, endpoints, label)
            ).result()

    # -- async-native loop -------------------------------------------------

    async def _call_llm(
        self,
        llm: LLMClient,
        system_prompt: str,
        prompt: str,
        fmt: dict[str, object] | None,
        cap: int,
    ) -> LLMResponse:
        """Call the LLM preferring the rich API, degrading to ``achat``.

        Compatibility layer: legacy clients and plain test doubles that only
        implement the ``str`` contract (``achat``) keep working — when
        ``achat_with_meta`` is missing or returns a non-awaitable (plain
        ``MagicMock`` attribute), we transparently fall back to ``achat`` and
        wrap the text in an :class:`LLMResponse` with no metadata (the
        truncation heuristic then degrades to the character estimate).
        """
        rich = getattr(llm, "achat_with_meta", None)
        if rich is not None:
            maybe = rich(system_prompt, prompt, response_format=fmt, max_tokens=cap)
            if inspect.isawaitable(maybe):
                result = await maybe
                if isinstance(result, LLMResponse):
                    return result
                return LLMResponse(text=str(result))
        text = await llm.achat(system_prompt, prompt, response_format=fmt, max_tokens=cap)
        return LLMResponse(text=text)

    async def arun(
        self,
        llm: LLMClient,
        system_prompt: str,
        user_prompt: str,
        endpoints: list[APIEndpoint],
        label: str,
    ) -> list[TestCase]:
        """Run the v6 truncation-aware loop (plan v6 §4)."""
        policy = self._policy
        hooks = self._hooks
        fmt = JSON_OBJECT_FORMAT if self._json_mode else None

        scope = [ep.full_path for ep in endpoints]
        batch_set = set(scope)
        expected = {ep.full_path: policy.default_expected_cases_per_endpoint for ep in endpoints}
        covered: dict[str, int] = {}
        pending = dict(expected)
        produced: list[TestCase] = []
        seen_keys: set[str] = set()
        stall = 0
        empty_streak = 0
        single_scope_rounds = 0
        scope_floor_reached = len(scope) <= policy.min_scope
        forced: str | None = None
        last_raw = ""
        deadline = time.monotonic() + policy.max_wall_time
        calls = 0
        transient_streak = 0

        def _merge(new_cases: list[TestCase]) -> int:
            """Merge new cases into produced (dedup by key); return added count."""
            added = 0
            for tc in new_cases:
                key = hooks.case_dedup_key(tc)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                produced.append(tc)
                added += 1
            return added

        def _absorb(raw: str, *, salvage: bool) -> int:
            """Extract/salvage items, scope-filter, convert and merge."""
            items = hooks.salvage_truncated(raw) if salvage else None
            if items is None:
                items = hooks.extract_json(raw)
            if items is None:
                return 0
            # With no endpoint context (requirements-only batches) the scope
            # filter is bypassed: there is no batch set to clip against and
            # the behaviour matches the legacy v2 loop (accept everything).
            filtered = filter_to_scope(items, batch_set, expected) if batch_set else items
            added = _merge(hooks.to_test_cases(filtered, endpoints))
            recompute_covered_pending(expected, produced, covered, pending)
            return added

        while True:
            # ---- global budget guards ----
            if calls >= policy.max_total_calls:
                logger.warning(
                    "[%s] truncation budget: call cap %d reached (produced=%d)",
                    label,
                    policy.max_total_calls,
                    len(produced),
                )
                return produced
            remaining = deadline - time.monotonic()
            if remaining <= 0 or remaining < policy.min_call_budget:
                logger.warning(
                    "[%s] truncation budget: wall time exhausted (produced=%d)",
                    label,
                    len(produced),
                )
                return produced
            calls += 1

            # ---- prompt selection ----
            if forced == "truncated":
                prompt = build_continue_prompt(
                    user_prompt, compress_fingerprint(produced), label, pending
                )
            elif forced:
                prompt = hooks.build_reask(user_prompt, last_raw, label, forced)
            else:
                prompt = user_prompt

            # ---- LLM call (rich result) ----
            try:
                result = await self._call_llm(
                    llm, system_prompt, prompt, fmt, policy.output_token_cap
                )
            except LLMOutputTooLongError:
                # Empty truncation. With an endpoint scope this means the
                # scope is beyond the model's real output capacity — shrink
                # it and continue. Without endpoints (requirements-only
                # batches) there is nothing to shrink: treat it as a
                # retryable empty response (v2 parity) instead of bailing out.
                logger.warning("[%s] empty truncation (scope=%d)", label, len(scope))
                if not expected:
                    empty_streak += 1
                    if empty_streak >= policy.max_empty_streak:
                        logger.warning(
                            "[%s] empty streak exhausted after empty truncation (produced=%d)",
                            label,
                            len(produced),
                        )
                        return produced
                    forced, last_raw = "empty", ""
                    continue
                if scope_floor_reached:
                    return self._salvage_and_return(produced, last_raw)
                scope, batch_set, pending, scope_floor_reached = shrink_scope(
                    scope, expected, produced, covered, policy
                )
                stall = 0
                empty_streak = 0
                single_scope_rounds = 0
                forced, last_raw = "truncated", ""
                continue
            except Exception as exc:
                # Transient transport failure: short backoff, then retry. The
                # client already retries internally (MAX_RETRIES per model),
                # so three consecutive engine-level failures mean the request
                # is really dead — return what we have rather than burning
                # the whole call budget (keeps tests and real runs fast).
                transient_streak += 1
                logger.debug("[%s] transient error (streak=%d): %s", label, transient_streak, exc)
                if transient_streak >= 3:
                    logger.warning(
                        "[%s] 3 consecutive transient errors; returning %d cases",
                        label,
                        len(produced),
                    )
                    return produced
                await asyncio.sleep(0.5)
                continue

            raw_response = result.text
            empty_return = not (raw_response and raw_response.strip())
            truncated = is_truncated(result, policy)

            # ---- truncated with partial content: salvage + continue ----
            if truncated and not empty_return:
                added = _absorb(raw_response, salvage=True)
                stall = 0 if added else stall + 1
                empty_streak = 0

                if stall >= policy.stall_threshold:
                    if scope_floor_reached:
                        single_scope_rounds += 1
                        stall = 0
                        if single_scope_rounds >= policy.max_single_scope_rounds:
                            logger.warning(
                                "[%s] scope floor + %d stall rounds; returning %d cases",
                                label,
                                policy.max_single_scope_rounds,
                                len(produced),
                            )
                            return self._salvage_and_return(produced, raw_response)
                    else:
                        scope, batch_set, pending, scope_floor_reached = shrink_scope(
                            scope, expected, produced, covered, policy
                        )
                        stall = 0
                        empty_streak = 0
                        single_scope_rounds = 0
                    forced, last_raw = "truncated", raw_response
                    continue

                if all(v <= 0 for v in pending.values()):
                    logger.info(
                        "[%s] coverage complete: %d cases in %d calls",
                        label,
                        len(produced),
                        calls,
                    )
                    return produced
                forced, last_raw = "truncated", raw_response
                continue

            # ---- empty response ----
            if empty_return:
                empty_streak += 1
                logger.debug("[%s] empty return (streak=%d)", label, empty_streak)
                if empty_streak >= policy.max_empty_streak:
                    logger.warning(
                        "[%s] empty streak exhausted (produced=%d)",
                        label,
                        len(produced),
                    )
                    return produced
                forced, last_raw = "empty", raw_response
                continue

            # ---- complete response: parse and finish ----
            items = hooks.extract_json(raw_response)
            if items is not None:
                added = _absorb(raw_response, salvage=False)
                if added or produced:
                    logger.info("[%s]: %d cases (calls=%d)", label, len(produced), calls)
                    return produced
                # Everything was filtered out by the scope quota: if the batch
                # has zero coverage there is nothing more to ask for.
                if all(v <= 0 for v in pending.values()):
                    return produced
                forced, last_raw = "truncated", raw_response
                continue

            forced, last_raw = "non_parseable", raw_response
            logger.warning(
                "[%s] call %d: cannot parse JSON. Preview: %.200s",
                label,
                calls,
                raw_response,
            )

    def _salvage_and_return(self, produced: list[TestCase], raw: str) -> list[TestCase]:
        """Best-effort final salvage of the last raw response, then return."""
        if not raw.strip():
            return produced
        items = self._hooks.salvage_truncated(raw)
        if not items:
            return produced
        return produced
