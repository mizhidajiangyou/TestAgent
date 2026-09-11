"""B6a-1/B6a-2 tests: GenericHooks protocol, generic dict hooks, adapter keys.

Gate (plan-d B6a-1): EngineEvent golden replay diff = 0 (covered by
test_engine_event_baseline, unchanged goldens) + correctness + full suite.
This file pins the NEW pieces:

- the dict-level adapter keys are byte-equal to the legacy TestCase keys
  (the equivalence claim that makes the golden replay meaningful);
- ``generic_reask`` is byte-equal to the frozen legacy builder (drift in
  either direction fails);
- the generic hooks factory produces a working engine wiring, including
  the scope-less-item → primary-scope attribution rule;
- the engine module itself is domain-free (zero TestCase / APIEndpoint
  references — B6a-3's gate, satisfied early and pinned here).
"""

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from testagent.config.models import APIEndpoint
from testagent.engine.llm_client import LLMResponse
from testagent.engine.prompt_builder import PromptBuilder
from testagent.engine.truncation import (
    EngineContext,
    GenericHooks,
    TruncationEngine,
    TruncationPolicy,
    build_continue_prompt,
)
from testagent.generators.testcase_generator import (
    TestCaseGenerator,
    _engine_dedup_key,
    _item_scope_key,
    _scope_item_key,
)
from testagent.pipeline.truncation_hooks import (
    dict_dedup_key,
    dict_scope_key,
    generic_reask,
    make_dict_hooks,
)

REPO = Path(__file__).parents[1]

_EPS = [
    APIEndpoint(method="GET", path="/users", summary="list"),
    APIEndpoint(method="POST", path="/users", summary="create"),
]


def _item(i: int, endpoint: str | None, **overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": f"TC-{i:03d}",
        "title": f" Case {i} ",
        "description": f"desc {i}",
        "endpoint": endpoint,
        "test_type": "functional",
        "priority": "high",
        "preconditions": [],
        "steps": [f"step {i}"],
        "expected_results": ["Status 200"],
    }
    item.update(overrides)
    return item


# ----------------------------------------------------------------------
# Adapter key equivalence (the equivalence claim behind golden replay)
# ----------------------------------------------------------------------


class TestAdapterKeyEquivalence:
    def _legacy_key(self, item: dict[str, Any], resolved_scope: str) -> str:
        """The pre-B6a key: convert through the frozen generator, then
        ``_case_dedup_key`` — with the endpoint remapped exactly as the
        legacy converter did (declared-in-scope, else first endpoint)."""
        gen = TestCaseGenerator(
            llm_client=_NoCallClient(),
            prompt_builder=PromptBuilder(),
        )
        # Feed ONE item through the frozen converter with an endpoints list
        # whose first entry is the resolved scope (the fallback target).
        primary = next((ep for ep in _EPS if ep.full_path == resolved_scope), _EPS[0])
        cases = gen._to_test_cases([item], [primary])
        return TestCaseGenerator._case_dedup_key(cases[0])

    def test_declared_endpoint_item(self) -> None:
        item = _item(1, "GET /users")
        resolved = _item_scope_key(item) or _EPS[0].full_path
        assert _engine_dedup_key(item, resolved) == self._legacy_key(item, resolved)

    def test_scopeless_item_falls_back_to_primary(self) -> None:
        # The legacy converter remapped no-endpoint items to endpoints[0];
        # the engine's primary-scope rule must reproduce that attribution.
        item = _item(2, None)
        assert _item_scope_key(item) == ""
        resolved = _EPS[0].full_path
        assert _engine_dedup_key(item, resolved) == self._legacy_key(item, resolved)
        # And the resolved key IS the primary scope (what the engine passes).
        assert resolved == "GET /users"

    def test_invalid_test_type_normalizes_to_functional(self) -> None:
        item = _item(3, "POST /users", test_type="not-a-type")
        resolved = _item_scope_key(item)
        key = _engine_dedup_key(item, resolved)
        assert key.endswith("|functional")
        assert key == self._legacy_key(item, resolved)

    def test_scope_item_key_adapter_reads_full_path(self) -> None:
        """B6a-2: the generator's scope-item adapter is exactly the legacy
        ``ep.full_path`` scope the engine used to read itself."""
        for ep in _EPS:
            assert _scope_item_key(ep) == ep.full_path


class _NoCallClient:
    """Client double that must never be called (key tests only)."""

    async def achat(self, *a: Any, **k: Any) -> str:
        raise AssertionError("no LLM call expected")


# ----------------------------------------------------------------------
# generic_reask: byte-equal to the frozen legacy builder
# ----------------------------------------------------------------------


class TestGenericReaskEquivalence:
    def test_all_error_types_match_legacy_builder(self) -> None:
        gen = TestCaseGenerator(llm_client=_NoCallClient(), prompt_builder=PromptBuilder())
        for error_type in ("empty", "truncated", "non_parseable"):
            legacy = gen._build_reask_prompt("BASE PROMPT", '{"partial":', "batch 1/2", error_type)
            generic = generic_reask("BASE PROMPT", '{"partial":', "batch 1/2", error_type)
            assert generic == legacy, f"drift for error_type={error_type}"

    def test_long_failed_output_truncated_marker(self) -> None:
        failed = "x" * 3000
        prompt = generic_reask("BASE", failed, "label", "non_parseable")
        assert "x" * 2000 in prompt
        assert "[truncated]" in prompt

    def test_product_nouns_parameterized(self) -> None:
        """B6a-3: non-testcase hosts re-word the re-ask through the noun
        parameters; defaults stay byte-equal to legacy (test above)."""
        prompt = generic_reask(
            "BASE",
            '{"partial":',
            "batch 1/2",
            "truncated",
            product="entries",
            item="entry",
            items="entries",
            compactness_hint="shorter text",
        )
        assert "Generate FEWER entries" in prompt
        assert "Make each entry more compact: shorter text." in prompt
        assert "test cases" not in prompt
        empty_prompt = generic_reask("BASE", "", "label", "empty", item="entry", items="entries")
        assert "Keep each entry compact" in empty_prompt
        assert "high-value entries" in empty_prompt
        # non_parseable carries no product nouns (already generic).
        np_prompt = generic_reask("BASE", "junk", "label", "non_parseable", product="entries")
        assert "Return ONLY a valid JSON array" in np_prompt


# ----------------------------------------------------------------------
# Generic dict hooks
# ----------------------------------------------------------------------


class TestDictHooks:
    def test_dict_scope_key_reads_field(self) -> None:
        assert dict_scope_key({"endpoint": "GET /users"}) == "GET /users"
        assert dict_scope_key({}) == ""
        assert dict_scope_key({"endpoint": None}) == ""
        assert dict_scope_key({"region": "eu"}, field="region") == "eu"

    def test_dict_dedup_key_joins_and_lowercases(self) -> None:
        item = {"title": " Case A ", "test_type": "Functional"}
        assert dict_dedup_key(item, "GET /users") == "case a|get /users|functional"
        # lower=False keeps the raw casing.
        assert dict_dedup_key(item, "GET /users", lower=False) == "Case A|GET /users|Functional"

    def test_make_dict_hooks_assembles_closures(self) -> None:
        hooks = make_dict_hooks(
            extract=lambda raw: json.loads(raw),
            salvage=lambda raw: None,
            scope_field="endpoint",
        )
        assert isinstance(hooks, GenericHooks)
        assert hooks.scope_key({"endpoint": "GET /users"}) == "GET /users"
        assert hooks.dedup_key({"title": "T"}, "GET /users") == "t|get /users|"
        # build_continue_context stays None -> engine's built-in continuation.
        assert hooks.build_continue_context is None


class _ScriptClient:
    """Minimal engine-level double (rich async contract)."""

    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = list(script)
        self.calls = 0

    async def achat_with_meta(self, *a: Any, **k: Any) -> LLMResponse:
        item = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return item

    async def achat(self, *a: Any, **k: Any) -> str:
        raise AssertionError("rich client driven via achat_with_meta")


class TestGenericHooksEndToEnd:
    async def test_scopeless_item_attributed_to_primary_scope(self) -> None:
        """The engine's primary-scope rule: an item declaring no scope key
        counts toward the FIRST scope entry, so the quota for it closes
        and the loop finishes in one call (legacy converter parity).
        Scope items are plain strings here (B6a-2: opaque to the engine,
        keys via ``scope_item_key`` — strings are their own key)."""
        llm = _ScriptClient(
            [LLMResponse(text=json.dumps([_item(1, None), _item(2, "GET /users")]))]
        )
        hooks = make_dict_hooks(extract=lambda raw: json.loads(raw), salvage=lambda raw: None)
        engine = TruncationEngine(TruncationPolicy(), False, hooks)
        produced = await engine.arun(llm, "sys", "user", [ep.full_path for ep in _EPS], "batch 1/1")
        # Quota: 2 per endpoint; the scope-less item lands on GET /users
        # (primary) plus the declared one — 2 there, 0 for POST -> done.
        assert [it["id"] for it in produced] == ["TC-001", "TC-002"]
        assert llm.calls == 1

    async def test_engine_returns_plain_dicts(self) -> None:
        llm = _ScriptClient([LLMResponse(text=json.dumps([_item(1, "GET /users")]))])
        hooks = make_dict_hooks(extract=lambda raw: json.loads(raw), salvage=lambda raw: None)
        engine = TruncationEngine(TruncationPolicy(), False, hooks)
        produced = await engine.arun(llm, "sys", "user", ["GET /users"], "batch 1/1")
        assert all(isinstance(it, dict) for it in produced)
        assert produced[0]["title"] == " Case 1 "

    async def test_explicit_scope_item_key_for_object_items(self) -> None:
        """B6a-2: non-str/dict scope items need an explicit ``scope_item_key``
        — here the legacy adapter shape (endpoint objects keyed by
        ``full_path``), proving the engine drives off the hook alone."""
        llm = _ScriptClient([LLMResponse(text=json.dumps([_item(1, "GET /users")]))])
        hooks = make_dict_hooks(
            extract=lambda raw: json.loads(raw),
            salvage=lambda raw: None,
            scope_item_key=lambda ep: ep.full_path,
        )
        engine = TruncationEngine(TruncationPolicy(), False, hooks)
        produced = await engine.arun(llm, "sys", "user", _EPS[:1], "batch 1/1")
        assert [it["id"] for it in produced] == ["TC-001"]
        assert llm.calls == 1

    async def test_default_scope_item_key_rejects_unknown_types(self) -> None:
        hooks = make_dict_hooks(extract=lambda raw: None, salvage=lambda raw: None)
        with pytest.raises(TypeError, match="scope_item_key"):
            hooks.scope_item_key(_EPS[0])


# ----------------------------------------------------------------------
# EngineContext convergence (B6a-3)
# ----------------------------------------------------------------------


class _RecordingClient:
    """Rich-contract double recording every (system, user) prompt pair."""

    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = list(script)
        self.prompts: list[tuple[str, str]] = []

    async def achat_with_meta(
        self, system_prompt: str, user_prompt: str, **kwargs: Any
    ) -> LLMResponse:
        self.prompts.append((system_prompt, user_prompt))
        idx = min(len(self.prompts) - 1, len(self._script) - 1)
        return self._script[idx]

    async def achat(self, *a: Any, **k: Any) -> str:
        raise AssertionError("rich client driven via achat_with_meta")


class TestEngineContextConvergence:
    """B6a-3: the slim continuation hook receives ONE EngineContext."""

    async def test_continue_hook_receives_full_context(self) -> None:
        captured: list[EngineContext] = []

        def _continue(ctx: EngineContext) -> str:
            captured.append(ctx)
            return "SLIM-CONTINUATION"

        # Round 1 truncates mid-array but salvages; round 2 completes.
        partial = json.dumps([_item(1, "GET /users")])[:-1]  # unclosed array
        full = json.dumps([_item(2, "POST /users")])
        llm = _RecordingClient(
            [
                LLMResponse(text=partial, finish_reason="length"),
                LLMResponse(text=full),
            ]
        )
        hooks = make_dict_hooks(
            extract=lambda raw: json.loads(raw) if raw.rstrip().endswith("]") else None,
            salvage=lambda raw: json.loads(raw + "]") if not raw.endswith("]") else None,
            scope_item_key=lambda ep: ep.full_path,
            build_continue_context=_continue,
        )
        engine = TruncationEngine(TruncationPolicy(), False, hooks)
        produced = await engine.arun(llm, "sys", "user", _EPS, "batch 1/1")

        assert [it["id"] for it in produced] == ["TC-001", "TC-002"]
        assert len(captured) == 1
        ctx = captured[0]
        # The five scattered legacy arguments all arrive via the context.
        assert ctx.user_prompt == "user"
        assert ctx.label == "batch 1/1"
        assert ctx.scope_items == _EPS  # opaque pass-through (B6a-2)
        assert "Case 1" in ctx.fingerprint and "GET /users" in ctx.fingerprint
        # pending is a point-in-time SNAPSHOT: after round 2 the engine's
        # live dict moved to {GET: 1, POST: 1}, but the stored ctx keeps
        # the construction-time values.
        assert ctx.pending == {"GET /users": 1, "POST /users": 2}
        # The engine sent the hook's output as the next request.
        assert llm.prompts[0] == ("sys", "user")
        assert llm.prompts[1] == ("sys", "SLIM-CONTINUATION")

    def test_legacy_builtin_continuation_still_applies_without_hook(self) -> None:
        """build_continue_context=None keeps the engine's built-in
        full-prompt continuation (unchanged five-field content)."""
        hooks = make_dict_hooks(extract=lambda raw: None, salvage=lambda raw: None)
        assert hooks.build_continue_context is None
        prompt = build_continue_prompt("BASE", "- T1 @ GET /users", "batch 1/1", {"GET /users": 1})
        assert "CONTINUATION for 'batch 1/1'" in prompt
        assert "GET /users x1" in prompt


# ----------------------------------------------------------------------
# Domain-freedom gate (B6a-3's grep+AST check, satisfied early)
# ----------------------------------------------------------------------


class TestEngineDomainFreedom:
    def test_engine_module_has_no_domain_references(self) -> None:
        """The engine must not import or annotate TestCase / APIEndpoint
        (plan-d B6a-3 gate — B6a-1 already satisfies it; pinned so it
        stays satisfied)."""
        src = (REPO / "testagent" / "engine" / "truncation.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name not in ("TestCase", "APIEndpoint"), (
                        f"engine imports domain type {alias.name}"
                    )
                assert node.module != "testagent.config.models", (
                    "engine imports the domain models module"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "testagent.config.models"
        # No domain-typed annotations either.
        assert "TestCase" not in src.replace("``TestCase``", "")
        assert "APIEndpoint" not in src
        # B6a-2: scope items are opaque — the engine must not read
        # scope-item attributes (keys come from the scope_item_key hook).
        assert "full_path" not in src
