"""Truncation-aware generation loop (v6 / B6a generalized).

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
  salvage / re-ask prompt building stay on the host and are injected as
  hooks (:class:`GenericHooks`), so this module has no dependency on prompt
  templates.
- **Domain-free (plan-d B6a-1 / R4)**: the engine produces and merges plain
  ``dict`` items; conversion to domain types (``TestCase`` etc.) happens in
  the host's ADAPTER outside the engine. The five data capabilities
  (extract / salvage / scope / dedup / continue) plus the re-ask prompt
  builder are supplied via hooks; generic dict-level implementations live
  in :mod:`testagent.pipeline.truncation_hooks` (B6b.1 wires them from
  manifest config).
- **Scope items are opaque (plan-d B6a-2)**: ``arun`` accepts host-shaped
  scope items and derives their keys via ``GenericHooks.scope_item_key``;
  shrink / filter / expected all run on those keys, and the engine never
  reads scope-item attributes.
- **Single-context continuation (plan-d B6a-3)**: the slim continuation
  hook receives one :class:`EngineContext` instead of five scattered
  arguments; re-ask product nouns are parameterized on the host side
  (:mod:`testagent.pipeline.truncation_hooks`).
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

from testagent.engine.llm_client import (
    CALL_LABEL,
    JSON_OBJECT_FORMAT,
    LLMClient,
    LLMOutputTooLongError,
    LLMResponse,
    ReasoningBudgetExhaustedError,
)
from testagent.engine.model_profiles import (
    ModelProfile,
    NextAction,
    Outcome,
    RecoveryState,
    RequestIntent,
    classify_response,
    recovery_plan,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EngineContext",
    "GenericHooks",
    "TruncationEngine",
    "TruncationPolicy",
    "build_continue_prompt",
    "chars_per_token_for",
    "compress_fingerprint",
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
    #: Approximate token budget for the slim continuation context (plan v10
    #: §8). Converted to a character budget (2 chars/token) inside the
    #: prompt builder.
    slim_continue_max_tokens: int = 2000
    #: Multiplier for the last-resort RAISE_BUDGET recovery action; the result
    #: is still clamped to the profile's real output cap (plan v10 §5.2).
    budget_raise_multiplier: float = 2.0  # x2, capped by the profile


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


OUT_OF_SPEC_PLACEHOLDERS = frozenset({"N/A", "NA"})


def filter_to_scope(
    items: list[dict[str, Any]],
    batch_set: set[str],
    expected: dict[str, int],
    scope_key: Callable[[dict[str, Any]], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Single filtering entry: drop items whose DECLARED scope key (the
    item's own declaration via ``scope_key``, NOT the resolved key) is
    outside this batch (avoiding the downstream ``fallback_ep`` remap
    pollution) and cap each scope key at its expected quota. Operates on
    RAW items before conversion. Items declaring no key (``""``) are kept
    unconditionally (legacy converter-fallback parity).

    Visibility contract (T2 / fix-plan D3, extended by user ruling
    2026-09-15): every drop is logged as a WARNING (endpoint, reason
    ``out_of_scope | quota | malformed``, count) and returned in
    ``dropped_report`` — aggregated ``{"endpoint", "reason", "count"}``
    entries whose counts always sum to ``len(items) - len(kept)``.
    Items with NO declared key AND items whose DECLARED key is the prompt's
    "N/A" placeholder are KEPT and marked ``out_of_spec=True`` on the raw
    dict (D3: never silently dropped, never quota-clipped; the marker makes
    "not endpoint-grounded" visible to downstream accounting while the
    converter's fallback attribution stays in charge of placement).
    """
    kept: list[dict[str, Any]] = []
    dropped: dict[tuple[str, str], int] = defaultdict(int)
    count: dict[str, int] = defaultdict(int)
    for it in items:
        if not isinstance(it, dict):
            dropped[("", "malformed")] += 1
            continue
        ep = scope_key(it)
        if not ep or ep.strip().upper() in OUT_OF_SPEC_PLACEHOLDERS:
            # Undeclared key or the prompt contract '(use "N/A" if no API
            # spec)' (D3): keep honestly, skip batch/quota. The converter's
            # fallback still attributes placement for undeclared keys.
            it["out_of_spec"] = True
            kept.append(it)
            continue
        if ep not in batch_set:
            dropped[(ep, "out_of_scope")] += 1
            continue
        if count[ep] >= expected.get(ep, 0):
            dropped[(ep, "quota")] += 1
            continue
        count[ep] += 1
        kept.append(it)
    report: list[dict[str, Any]] = []
    for (ep, reason), n in dropped.items():
        logger.warning("Scope filter dropped %d item(s) for endpoint %r: %s", n, ep, reason)
        report.append({"endpoint": ep, "reason": reason, "count": n})
    return kept, report


def recompute_covered_pending(
    expected: dict[str, int],
    produced: list[dict[str, Any]],
    covered: dict[str, int],
    pending: dict[str, int],
    scope_of: Callable[[dict[str, Any]], str],
) -> None:
    """Recompute per-scope-key coverage and remaining quota (in place).

    ``covered`` aggregates produced items by their RESOLVED scope key
    (``scope_of``); ``pending[key] = max(0, expected[key] - covered[key])``.
    """
    covered.clear()
    for it in produced:
        key = scope_of(it)
        covered[key] = covered.get(key, 0) + 1
    for key in expected:
        pending[key] = max(0, expected[key] - covered.get(key, 0))


def shrink_scope(
    scope: list[str],
    expected: dict[str, int],
    covered: dict[str, int],
    policy: TruncationPolicy,
) -> tuple[list[str], set[str], dict[str, int], bool]:
    """Halve the scope, keeping the best-covered endpoints first.

    Returns ``(new_scope, new_batch_set, new_pending, floor_reached)``.
    """
    new_n = max(policy.min_scope, len(scope) // 2)
    ranked = sorted(scope, key=lambda ep: covered.get(ep, 0), reverse=True)
    new_scope = ranked[:new_n]
    new_pending = {ep: max(0, expected.get(ep, 0) - covered.get(ep, 0)) for ep in new_scope}
    floor = len(new_scope) <= policy.min_scope
    return new_scope, set(new_scope), new_pending, floor


def compress_fingerprint(
    produced: list[dict[str, Any]],
    scope_of: Callable[[dict[str, Any]], str],
    max_items: int = 30,
) -> str:
    """Compress already-produced items into a short fingerprint for continue
    prompts (``- <title> @ <resolved scope key>`` per item)."""
    if not produced:
        return "(none yet)"
    lines = [f"- {it.get('title', '')} @ {scope_of(it)}" for it in produced[:max_items]]
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
        intent_override: RequestIntent | None = None,
    ) -> LLMResponse:
        # Layer 4 of the intent_override chain (plan v10 §4.2, P0-2): this
        # method is EXPLICITLY defined, so the ``__getattr__`` fallback does
        # NOT apply — a new parameter must be added here or it never reaches
        # the inner client (the TypeError root cause found in review).
        # The kwarg is only forwarded when set, so legacy inner clients with
        # fixed signatures stay compatible.
        call_kwargs: dict[str, Any] = {"response_format": response_format, "max_tokens": max_tokens}
        if intent_override is not None:
            call_kwargs["intent_override"] = intent_override

        cm = getattr(self._inner, "chat_with_meta", None)
        if cm is not None:
            result = await asyncio.to_thread(cm, system_prompt, user_prompt, **call_kwargs)
            if isinstance(result, LLMResponse):
                return result
            # Not a real rich result (e.g. an auto-created MagicMock attribute
            # on a plain double): fall through to the configured sync ``chat``.
        chat = getattr(self._inner, "chat", None)
        if chat is not None:
            text = await asyncio.to_thread(chat, system_prompt, user_prompt, **call_kwargs)
            return LLMResponse(text=text)
        # Pure-async double (no sync contract at all): delegate unchanged.
        result = await self._inner.achat_with_meta(system_prompt, user_prompt, **call_kwargs)
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


@dataclass(frozen=True)
class EngineContext:
    """The single input of the continuation-context hook (plan-d B6a-3).

    Converges the five scattered ``build_continue_context`` arguments
    (user_prompt / scope_items / fingerprint / label / pending) into one
    immutable object, so extending what a continuation sees never changes
    the hook signature again. ``scope_items`` stay opaque (B6a-2): the host
    renders them (e.g. endpoint signatures for the legacy testcase host).
    """

    #: The batch's original user prompt (full-prompt fallback source).
    user_prompt: str
    #: The run's opaque scope items (endpoints for the legacy host).
    scope_items: list[Any]
    #: Compressed ``- <title> @ <scope key>`` fingerprint of produced items.
    fingerprint: str
    #: The batch's logical label (e.g. "Req REQ-001/3").
    label: str
    #: Remaining quota per scope key AT CONSTRUCTION TIME — the engine
    #: passes a snapshot copy, so a stored context never mutates under
    #: the host's feet (keys > 0 are what continuation needs).
    pending: dict[str, int]


@dataclass
class GenericHooks:
    """The engine's data capabilities, supplied by the host (plan-d R4).

    Five core capabilities (extract / salvage / scope / dedup / continue)
    plus the re-ask prompt builder. The engine produces and merges plain
    ``dict`` items — **conversion to domain types is NOT an engine
    capability**: the host adapter converts the returned items after
    ``arun``/``run`` (``to_test_cases`` used to live here; B6a-1 moved it
    out to keep the engine domain-free).

    - ``extract`` / ``salvage``: raw response text -> item list (``None``
      marks the attempt failed; salvage parses truncated text).
    - ``scope_key``: item -> its DECLARED scope key (``""`` when the item
      declares none — the engine then attributes it to the run's primary
      scope, i.e. the first entry of the entry scope: legacy converter
      parity, so scope-less items still count toward coverage).
    - ``dedup_key``: ``(item, resolved_scope_key) -> str`` — the resolved
      scope key is passed in so hosts that fold it into the identity get
      the same attribution the coverage accounting uses.
    - ``scope_item_key``: scope item -> its scope key (B6a-2). ``arun``
      accepts OPAQUE scope items (endpoints for the legacy host, any
      host-shaped item elsewhere); this hook derives the keys that drive
      the scope/batch-set/expected/shrink machinery — the engine itself
      never inspects scope-item attributes.
    - ``build_reask``: targeted re-ask prompt.
    - ``build_continue_context``: optional slim continuation context
      (plan v10 §6 / v8 方案 A); ``None`` -> the engine's legacy
      full-prompt continuation (:func:`build_continue_prompt`). Receives a
      single :class:`EngineContext` (B6a-3) carrying the original prompt,
      the opaque scope items, the produced-items fingerprint, the label and
      the pending quota — hosts render what they need from it.
    """

    extract: Callable[[str], list[Any] | None]
    salvage: Callable[[str], list[Any] | None]
    scope_key: Callable[[dict[str, Any]], str]
    dedup_key: Callable[[dict[str, Any], str], str]
    scope_item_key: Callable[[Any], str]
    build_reask: Callable[[str, str, str, str], str]
    build_continue_context: Callable[[EngineContext], str] | None = None
    # T5/T7 (fix-plan §3.1/§3.5): obligation-driven quota floor. Receives the
    # batch scope keys, returns the expected-cases map. ``None`` (default and
    # every legacy host) keeps the ``default_expected_cases_per_endpoint``
    # quota — byte-identical legacy behaviour.
    expected_for: Callable[[list[str]], dict[str, int]] | None = None


@dataclass(frozen=True)
class EngineEvent:
    """Behavior event emitted at the engine's decision points (plan-e E3 / B6a-0).

    Events are BEHAVIOR markers in three categories — ``request`` (call),
    ``recovery`` (salvage / continue / reask / budget_exhausted / downgrade /
    split / raise_budget) and ``terminal`` (done / fail) — they do NOT carry
    final-state semantics. ``round`` counts the LLM requests issued so far in
    this batch (request events carry their own 1-based number; recovery
    actions issue no request and keep the previous round). Recovery events
    snapshot the POST-action state (e.g. a ``split`` event reports the
    shrunk scope). Frozen-field discipline (plan-d v3 R5): golden files
    diff on every field.
    """

    event: str
    scope: str  # deterministic snapshot: ",".join of current scope keys
    label: str
    round: int
    chars: int
    pending: int
    covered: int
    budget: int


#: Observer contract: receives every EngineEvent of one ``arun`` batch.
EngineObserver = Callable[[EngineEvent], None]

#: T1 raw-audit sink (fix-plan §3.6): receives plain-dict records —
#: ``{"kind": "raw", "label", "round", "text", "finish_reason",
#: "completion_tokens"}`` per LLM response and ``{"kind": "merge", "label",
#: "added"}`` per merge into ``produced``. Pure observation; merge rows sum
#: to the artifact count (session reconciliation contract).
RawSink = Callable[[dict[str, Any]], None]


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
        hooks: GenericHooks,
        observer: EngineObserver | None = None,
        raw_sink: RawSink | None = None,
    ) -> None:
        self._policy = policy
        self._json_mode = json_mode
        self._hooks = hooks
        # B6a-0 (plan-e E3): pure observation — when None (the default and
        # every existing construction site) behaviour is bit-for-bit
        # unchanged; the full suite proves the zero-control-flow-change.
        self._observer = observer
        # T1 (fix-plan §3.6): raw-response audit hook — same pure-observation
        # discipline as ``observer``; None (default) changes nothing.
        self._raw_sink = raw_sink

    # -- sync entry point ------------------------------------------------

    def run(
        self,
        llm: LLMClient,
        system_prompt: str,
        user_prompt: str,
        scope_items: list[Any],
        label: str,
    ) -> list[dict[str, Any]]:
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
            return asyncio.run(self.arun(wrapped, system_prompt, user_prompt, scope_items, label))
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run, self.arun(wrapped, system_prompt, user_prompt, scope_items, label)
            ).result()

    # -- async-native loop -------------------------------------------------

    async def _call_llm(
        self,
        llm: LLMClient,
        system_prompt: str,
        prompt: str,
        fmt: dict[str, object] | None,
        cap: int,
        intent: RequestIntent | None = None,
    ) -> LLMResponse:
        """Call the LLM preferring the rich API, degrading to ``achat``.

        Compatibility layer: legacy clients and plain test doubles that only
        implement the ``str`` contract (``achat``) keep working — when
        ``achat_with_meta`` is missing or returns a non-awaitable (plain
        ``MagicMock`` attribute), we transparently fall back to ``achat`` and
        wrap the text in an :class:`LLMResponse` with no metadata (the
        truncation heuristic then degrades to the character estimate).

        ``intent`` is the recovery effort (plan v10 §4.2, P0-2). It is only
        forwarded when the client advertises ``intent_capable`` — capability
        probing rather than duck-typing the kwarg keeps the 316-test legacy
        fake population free of TypeErrors.
        """
        call_kwargs: dict[str, Any] = {"response_format": fmt, "max_tokens": cap}
        if intent is not None and getattr(llm, "intent_capable", False):
            call_kwargs["intent_override"] = intent
        rich = getattr(llm, "achat_with_meta", None)
        if rich is not None:
            maybe = rich(system_prompt, prompt, **call_kwargs)
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
        scope_items: list[Any],
        label: str,
    ) -> list[dict[str, Any]]:
        """Run the v6/v10 truncation-aware loop (plan v6 §4 + v10 §5).

        ``scope_items`` are OPAQUE to the engine (B6a-2): their keys come
        from ``GenericHooks.scope_item_key`` and drive the scope/batch-set/
        expected/shrink machinery; the items themselves are only passed
        through to ``build_continue_context``. Returns the produced items
        as plain dicts (B6a-1): the host adapter converts them to domain
        types after the call.
        """
        policy = self._policy
        hooks = self._hooks
        fmt = JSON_OBJECT_FORMAT if self._json_mode else None

        # Tag every client-side log line for this batch (e.g. "[sid][Req
        # REQ-001/3] streaming: ..."). Concurrent fan-out otherwise makes
        # interleaved streaming logs indistinguishable — it LOOKS sequential.
        # Task-scoped: each gathered arun runs in its own asyncio Task with a
        # copied context, and asyncio.to_thread propagates it to the worker,
        # so labels never cross between concurrent batches.
        CALL_LABEL.set(label)

        # ---- capability detection (plan v10 §4.2 / §5) ----
        # The profile and continuation intent live on the client; plain fakes
        # (no attributes) degrade to the legacy behaviour transparently.
        raw_profile = getattr(llm, "profile", None)
        profile = raw_profile if isinstance(raw_profile, ModelProfile) else None
        raw_cont = getattr(llm, "continuation_intent", None)
        cont_effort = raw_cont if isinstance(raw_cont, str) and raw_cont else None
        intent_capable = bool(getattr(llm, "intent_capable", False))
        downgrade_possible = cont_effort is not None and intent_capable

        # B6a-2: scope keys derive from the opaque scope items through the
        # host hook — the engine never inspects scope-item attributes.
        scope = [str(hooks.scope_item_key(it)) for it in scope_items]
        batch_set = set(scope)
        if hooks.expected_for is not None:
            # T5/T7: obligation-driven quota floor supplied by the host.
            expected = dict(hooks.expected_for(scope))
            for key in scope:
                expected.setdefault(key, policy.default_expected_cases_per_endpoint)
        else:
            expected = {key: policy.default_expected_cases_per_endpoint for key in scope}
        # B6a-1: the run's PRIMARY scope key — items that declare no scope
        # key are attributed to it (legacy converter parity: the host's
        # converter used to remap scope-less items to the first endpoint).
        primary_scope = scope[0] if scope else ""
        covered: dict[str, int] = {}
        pending = dict(expected)
        produced: list[dict[str, Any]] = []
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
        # Budget-exhausted recovery state (plan v10 §5.2).
        downgrade_used = False
        budget_raised = False
        downgrade_effort: str | None = None
        current_cap = policy.output_token_cap

        observer = self._observer

        def _scope_of(item: dict[str, Any]) -> str:
            """Resolved scope key: the item's DECLARED key, else the run's
            primary scope (see ``GenericHooks.scope_key``)."""
            return str(hooks.scope_key(item) or "") or primary_scope

        def _emit(event: str, *, chars: int = 0) -> None:
            """Emit one behavior event (no-op without an observer, B6a-0).

            Reads the LIVE loop state, so recovery events snapshot the
            post-action scope/pending/budget; request events carry their own
            1-based round (``calls`` was incremented just before emission).
            """
            if observer is None:
                return
            observer(
                EngineEvent(
                    event=event,
                    scope=",".join(scope),
                    label=label,
                    round=calls,
                    chars=chars,
                    pending=sum(max(0, v) for v in pending.values()),
                    covered=sum(covered.values()),
                    budget=current_cap,
                )
            )

        def _merge(new_items: list[Any]) -> int:
            """Merge new items into produced (dedup by key); return added count."""
            added = 0
            for it in new_items:
                if not isinstance(it, dict):
                    # Legacy converter parity: non-dict items were dropped
                    # during conversion (they can never become domain items).
                    continue
                key = hooks.dedup_key(it, _scope_of(it))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                produced.append(it)
                added += 1
            return added

        def _absorb(raw: str, *, salvage: bool) -> int:
            """Extract/salvage items, scope-filter and merge (as dicts)."""
            items = hooks.salvage(raw) if salvage else None
            if items is None:
                items = hooks.extract(raw)
            if items is None:
                return 0
            # With no scope context (requirements-only batches) the scope
            # filter is bypassed: there is no batch set to clip against and
            # the behaviour matches the legacy v2 loop (accept everything).
            # Drops inside the filter are WARNING-logged there (T2); the
            # structured report is consumed by the budget task (T7).
            if batch_set:
                filtered, _dropped = filter_to_scope(items, batch_set, expected, hooks.scope_key)
            else:
                filtered = items
            added = _merge(filtered)
            recompute_covered_pending(expected, produced, covered, pending, _scope_of)
            if self._raw_sink is not None:
                self._raw_sink({"kind": "merge", "label": label, "added": added})
            return added

        def _handle_budget_exhausted() -> list[dict[str, Any]] | None:
            """Recovery decision for BUDGET_EXHAUSTED (plan v10 §5.2).

            Sequence: one-shot DOWNGRADE → SPLIT (shrink scope) →
            RAISE_BUDGET (clamped to the profile cap) → FAIL (salvage).
            Returns the final item list when recovery is exhausted, else
            ``None`` to continue the loop with the action applied.
            """
            nonlocal downgrade_used, budget_raised, downgrade_effort, current_cap
            nonlocal scope, batch_set, pending, scope_floor_reached
            nonlocal stall, empty_streak, single_scope_rounds, forced, last_raw

            state = RecoveryState(
                downgrade_used=downgrade_used or not downgrade_possible,
                pending_total=sum(max(0, v) for v in pending.values()),
                scope_size=len(scope),
                budget_raised=budget_raised,
                floor_reached=scope_floor_reached or not expected,
            )
            action = recovery_plan(profile, state)

            if action is NextAction.DOWNGRADE:
                # One-shot effort downgrade (deepseek: disable thinking → zero
                # reasoning overhead + temperature=0 determinism, P1-2). The
                # downgraded effort stays sticky for the rest of the batch.
                downgrade_used = True
                downgrade_effort = cont_effort
                logger.warning(
                    "[%s] budget exhausted; downgrading effort to '%s' (one-shot)",
                    label,
                    cont_effort,
                )
                forced, last_raw = "truncated", ""
                _emit("downgrade")
                return None

            if action is NextAction.SPLIT:
                # Main recovery: shrink the scope (v6 machinery reuse).
                logger.warning(
                    "[%s] budget exhausted; splitting scope %d -> %d",
                    label,
                    len(scope),
                    max(policy.min_scope, len(scope) // 2),
                )
                scope, batch_set, pending, scope_floor_reached = shrink_scope(
                    scope, expected, covered, policy
                )
                stall = 0
                empty_streak = 0
                single_scope_rounds = 0
                forced, last_raw = "truncated", ""
                _emit("split")
                return None

            if action is NextAction.RAISE_BUDGET:
                # Last resort, clamped to the model's real output cap (P0-4).
                raw_cap = getattr(llm, "max_output_cap", None)
                hard_cap = raw_cap if isinstance(raw_cap, int) and raw_cap > 0 else None
                new_cap = int(current_cap * policy.budget_raise_multiplier)
                if hard_cap is not None:
                    new_cap = min(new_cap, hard_cap)
                if new_cap > current_cap:
                    budget_raised = True
                    current_cap = new_cap
                    logger.warning(
                        "[%s] budget exhausted; raising output budget to %d (last resort)",
                        label,
                        new_cap,
                    )
                    forced, last_raw = "truncated", ""
                    _emit("raise_budget")
                    return None

            logger.warning(
                "[%s] budget exhausted; recovery options exhausted, returning %d cases",
                label,
                len(produced),
            )
            _emit("fail", chars=len(last_raw))
            return self._salvage_and_return(
                produced, last_raw, batch_set, expected, seen_keys, _emit, _scope_of, label=label
            )

        while True:
            # ---- global budget guards ----
            if calls >= policy.max_total_calls:
                logger.warning(
                    "[%s] truncation budget: call cap %d reached (produced=%d)",
                    label,
                    policy.max_total_calls,
                    len(produced),
                )
                _emit("fail", chars=len(last_raw))
                return produced
            remaining = deadline - time.monotonic()
            if remaining <= 0 or remaining < policy.min_call_budget:
                logger.warning(
                    "[%s] truncation budget: wall time exhausted (produced=%d)",
                    label,
                    len(produced),
                )
                _emit("fail", chars=len(last_raw))
                return produced

            # ---- prompt selection ----
            if forced == "truncated":
                fingerprint = compress_fingerprint(produced, _scope_of)
                if hooks.build_continue_context is not None:
                    # Slim continuation context (plan v10 §6 / v8 方案 A):
                    # endpoint signatures + requirement summary instead of
                    # re-sending the full prompt with the 9KB Rules region.
                    # Single-context call (B6a-3); scope items stay opaque.
                    prompt = hooks.build_continue_context(
                        EngineContext(
                            user_prompt=user_prompt,
                            scope_items=scope_items,
                            fingerprint=fingerprint,
                            label=label,
                            # Snapshot copy (EngineContext contract): the live
                            # ``pending`` dict is mutated in place by later
                            # absorb rounds; a stored context must not see them.
                            pending=dict(pending),
                        )
                    )
                else:
                    prompt = build_continue_prompt(user_prompt, fingerprint, label, pending)
                _emit("continue", chars=len(prompt))
            elif forced:
                prompt = hooks.build_reask(user_prompt, last_raw, label, forced)
                _emit("reask", chars=len(prompt))
            else:
                prompt = user_prompt

            # ---- LLM call (rich result) ----
            intent = (
                RequestIntent(budget=current_cap, effort=downgrade_effort)
                if downgrade_effort
                else None
            )
            calls += 1
            _emit("call")
            try:
                result = await self._call_llm(
                    llm, system_prompt, prompt, fmt, current_cap, intent=intent
                )
            except ReasoningBudgetExhaustedError as exc:
                # P0-3 main path: the client short-circuited the identical
                # retries and every fallback model already failed. Recovery:
                # one downgrade → split → raise budget → fail.
                logger.warning("[%s] reasoning budget exhausted: %s", label, exc)
                _emit("budget_exhausted")
                final = _handle_budget_exhausted()
                if final is not None:
                    return final
                continue
            except LLMOutputTooLongError:
                # Transient empty exhaustion (stream + blocking both empty).
                # With an endpoint scope this means the scope is beyond the
                # model's real output capacity — shrink it and continue.
                # Without endpoints (requirements-only batches) there is
                # nothing to shrink: treat it as a retryable empty response
                # (v2 parity) instead of bailing out.
                logger.warning("[%s] empty truncation (scope=%d)", label, len(scope))
                if not expected:
                    empty_streak += 1
                    if empty_streak >= policy.max_empty_streak:
                        logger.warning(
                            "[%s] empty streak exhausted after empty truncation (produced=%d)",
                            label,
                            len(produced),
                        )
                        _emit("fail", chars=len(last_raw))
                        return produced
                    forced, last_raw = "empty", ""
                    continue
                if scope_floor_reached:
                    _emit("fail", chars=len(last_raw))
                    return self._salvage_and_return(
                        produced,
                        last_raw,
                        batch_set,
                        expected,
                        seen_keys,
                        _emit,
                        _scope_of,
                        label=label,
                    )
                scope, batch_set, pending, scope_floor_reached = shrink_scope(
                    scope, expected, covered, policy
                )
                stall = 0
                empty_streak = 0
                single_scope_rounds = 0
                forced, last_raw = "truncated", ""
                _emit("split")
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
                    _emit("fail", chars=len(last_raw))
                    return produced
                await asyncio.sleep(0.5)
                continue

            raw_response = result.text
            empty_return = not (raw_response and raw_response.strip())
            truncated = is_truncated(result, policy)

            if self._raw_sink is not None:
                # T1 audit: one record per LLM response (raw text + metadata).
                self._raw_sink(
                    {
                        "kind": "raw",
                        "label": label,
                        "round": calls,
                        "text": raw_response or "",
                        "finish_reason": result.finish_reason,
                        "completion_tokens": result.completion_tokens,
                    }
                )

            # ---- truncated with partial content: salvage + continue ----
            if truncated and not empty_return:
                added = _absorb(raw_response, salvage=True)
                _emit("salvage", chars=len(raw_response))
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
                            _emit("fail", chars=len(raw_response))
                            return self._salvage_and_return(
                                produced,
                                raw_response,
                                batch_set,
                                expected,
                                seen_keys,
                                _emit,
                                _scope_of,
                                label=label,
                            )
                    else:
                        scope, batch_set, pending, scope_floor_reached = shrink_scope(
                            scope, expected, covered, policy
                        )
                        stall = 0
                        empty_streak = 0
                        single_scope_rounds = 0
                        _emit("split")
                    forced, last_raw = "truncated", raw_response
                    continue

                if all(v <= 0 for v in pending.values()):
                    logger.info(
                        "[%s] coverage complete: %d cases in %d calls",
                        label,
                        len(produced),
                        calls,
                    )
                    _emit("done", chars=len(raw_response))
                    return produced
                forced, last_raw = "truncated", raw_response
                continue

            # ---- empty response ----
            if empty_return:
                # Compatibility path (plan v10 §4.3): production clients raise
                # instead of returning empty bodies, so this branch serves
                # legacy clients and test fakes. Classify with the profile
                # when available so a budget-exhausted empty body takes the
                # recovery ladder here too.
                if profile is not None and (
                    classify_response(
                        profile, text=raw_response, finish_reason=result.finish_reason
                    )
                    is Outcome.BUDGET_EXHAUSTED
                ):
                    final = _handle_budget_exhausted()
                    if final is not None:
                        return final
                    continue
                empty_streak += 1
                logger.debug("[%s] empty return (streak=%d)", label, empty_streak)
                if empty_streak >= policy.max_empty_streak:
                    logger.warning(
                        "[%s] empty streak exhausted (produced=%d)",
                        label,
                        len(produced),
                    )
                    _emit("fail", chars=len(raw_response))
                    return produced
                forced, last_raw = "empty", raw_response
                continue

            # ---- complete response: parse and finish ----
            items = hooks.extract(raw_response)
            if items is not None:
                added = _absorb(raw_response, salvage=False)
                if added or produced:
                    logger.info("[%s]: %d cases (calls=%d)", label, len(produced), calls)
                    _emit("done", chars=len(raw_response))
                    return produced
                # Everything was filtered out by the scope quota: if the batch
                # has zero coverage there is nothing more to ask for.
                if all(v <= 0 for v in pending.values()):
                    _emit("done", chars=len(raw_response))
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

    def _salvage_and_return(
        self,
        produced: list[dict[str, Any]],
        raw: str,
        batch_set: set[str],
        expected: dict[str, int],
        seen_keys: set[str],
        emit: Callable[..., None] | None = None,
        scope_of: Callable[[dict[str, Any]], str] | None = None,
        label: str = "",
    ) -> list[dict[str, Any]]:
        """Best-effort final salvage of the last raw response, then return.

        v7-review bug fix: the salvaged items used to be discarded (both
        branches returned ``produced`` untouched). They now go through the
        same scope filter + dedup merge as in-loop absorptions (as dicts,
        B6a-1 — conversion is the host adapter's job).

        ``emit`` is the caller's B6a-0 event emitter (terminal ``fail`` was
        already emitted by the caller; a successful final salvage emits its
        own ``salvage`` event here). ``scope_of`` is the caller's resolved
        scope-key function (required for dedup attribution).
        """
        if not raw.strip() or scope_of is None:
            return produced
        items = self._hooks.salvage(raw)
        if not items:
            return produced
        if batch_set:
            filtered, _dropped = filter_to_scope(items, batch_set, expected, self._hooks.scope_key)
        else:
            filtered = items
        merged = list(produced)
        keys = set(seen_keys)
        added = 0
        for it in filtered:
            if not isinstance(it, dict):
                continue
            key = self._hooks.dedup_key(it, scope_of(it))
            if key in keys:
                continue
            keys.add(key)
            merged.append(it)
            added += 1
        if added:
            if self._raw_sink is not None:
                self._raw_sink(
                    {"kind": "merge", "label": label or "(final-salvage)", "added": added}
                )
            logger.info("Final salvage merged %d additional cases", added)
            if emit is not None:
                emit("salvage", chars=len(raw))
        return merged
