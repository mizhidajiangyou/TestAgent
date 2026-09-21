"""Dependency injection container (dependency-injector).

Services are declared as :class:`providers.Singleton` so each is created lazily
on first ``container.<name>()`` call and reused thereafter. The wiring graph is
expressed declaratively via provider dependencies and resolved automatically by
the framework (replacing the previous hand-written lazy-property container).

Access pattern: call the provider to obtain the instance, e.g.
``container.llm_client()`` returns the shared :class:`MultiModelLLMClient`.
"""

from dependency_injector import containers, providers

from testagent.config.settings import Settings, get_settings
from testagent.engine.conversation import ConversationManager
from testagent.engine.llm_client import create_llm_client
from testagent.engine.prompt_builder import PromptBuilder, endpoints_to_rich_signature
from testagent.engine.session_store import create_session_store
from testagent.engine.truncation import TruncationPolicy
from testagent.generators.gui_test_generator import GUITestGenerator
from testagent.generators.performance_generator import PerformanceGenerator
from testagent.generators.testcase_generator import TestCaseGenerator
from testagent.parsers.document_parser import DocumentParser
from testagent.parsers.requirement_parser import RequirementParser
from testagent.parsers.swagger_parser import SwaggerParser
from testagent.pipeline.executor import PipelineExecutor
from testagent.pipeline.registry import get_registry
from testagent.pipeline.runtime import build_engine_generate_unit, build_review_runner
from testagent.reports.performance_report import PerformanceReport
from testagent.reports.testcase_report import TestCaseReport


class Container(containers.DeclarativeContainer):
    """Declarative DI container.

    All services are singletons. ``container.<name>()`` resolves and caches the
    instance on first call. The optional ``settings`` constructor argument lets
    tests inject a custom :class:`Settings` (otherwise the cached global one is
    used), preserving the previous ``Container(settings=...)`` contract.
    """

    # ---- configuration -------------------------------------------------
    # ``get_settings`` is @lru_cache'd, so this singleton is the global one.
    settings = providers.Singleton(get_settings)

    # ---- core services -------------------------------------------------
    llm_client = providers.Singleton(
        create_llm_client,
        settings=settings,
    )
    # Prompt text lives in the task packages (plan-k B7.2); the builder needs the
    # configured TASKS_DIR, not a cwd-relative guess.
    prompt_builder = providers.Singleton(PromptBuilder, tasks_dir=settings.provided.tasks_dir)
    swagger_parser = providers.Singleton(SwaggerParser)
    requirement_parser = providers.Singleton(RequirementParser)
    document_parser = providers.Singleton(DocumentParser)
    testcase_report = providers.Singleton(TestCaseReport)
    performance_report = providers.Singleton(PerformanceReport)

    # The review client is the multi-model client's non-primary sub-client
    # (falls back to primary with a warning when only one model is configured).
    # ``secondary_client()`` returns an already-constructed internal instance,
    # so caching it as a singleton is correct.
    review_client = providers.Singleton(llm_client.provided.secondary_client.call())

    # Truncation policy mirrors OPENAI_MAX_OUTPUT_TOKENS so the engine's
    # per-call cap and the client's configured budget stay in sync (plan
    # v8 §4.4: the policy default previously hardcoded 16000 and silently
    # diverged from a custom setting).
    truncation_policy = providers.Singleton(
        TruncationPolicy,
        output_token_cap=settings.provided.llm.max_output_tokens,
    )

    # Task-package pipeline (plan-c B4.6/B4.7): registry scans TASKS_DIR,
    # the executor orchestrates, and the unit generator + review runner are
    # INJECTED (arch gate: pipeline modules never import the legacy
    # generators; the wiring lives here at the composition root).
    task_registry = providers.Singleton(get_registry, base_dir=settings.provided.tasks_dir)
    pipeline_executor = providers.Singleton(
        PipelineExecutor,
        llm_client=llm_client,
        settings=settings,
        # Engine-backed recovery, opt-in per package via manifest
        # pipeline.truncation.enabled (tasks/testcase declares it; perf/gui do
        # not and are delegated to the single-call path byte-for-byte). The cap
        # is injected here — a pipeline module reading the settings singleton
        # would ignore every override the composition root applies.
        generate_unit=providers.Callable(
            build_engine_generate_unit,
            llm=llm_client.provided,
            output_token_cap=settings.provided.llm.max_output_tokens,
            # Same reason as the cap: the engine decides ``response_format``
            # from this flag, so a hardcoded value would silently never ask for
            # JSON mode even with OPENAI_JSON_MODE=true.
            json_mode=settings.provided.llm.json_mode,
        ),
        review_runner=providers.Callable(
            build_review_runner,
            llm=llm_client.provided,
        ),
        # Injected because pipeline modules may not import the legacy prompt
        # layer (B4.11); the links L0 index renders endpoints with the same
        # rich signature the generation prompts use.
        links_signature_fn=endpoints_to_rich_signature,
    )

    # Factory (defect ⑨, 2026-09-19 review): agenerate writes per-session
    # instance state (_raw_dumper/_session_case_count/_obligation_registry/
    # _conflict_table/_dedup_*), and web/app.py serves concurrent requests
    # from one container — a Singleton generator would cross-pollute
    # sessions (raw dumps into the wrong session dir, shared budget
    # counters). CLI constructs once per process anyway, so Factory is
    # behavior-neutral there.
    testcase_generator = providers.Factory(
        TestCaseGenerator,
        llm_client=llm_client,
        prompt_builder=prompt_builder,
        review_enabled=settings.provided.review_enabled,
        review_llm_client=review_client,
        review_max_rounds=settings.provided.review_max_rounds,
        output_language=settings.provided.output_language,
        json_mode=settings.provided.llm.json_mode,
        max_concurrency=settings.provided.llm.max_concurrency,
        verify_model=settings.provided.llm.verify_model,
        truncation_policy=truncation_policy,
        audit_dump_enabled=settings.provided.audit_dump_enabled,
        audit_dump_dir=settings.provided.output_dir,
        conflict_policy=settings.provided.conflict_policy,
        cases_budget=settings.provided.cases_budget,
    )

    gui_generator = providers.Singleton(
        GUITestGenerator,
        llm_client=llm_client,
        prompt_builder=prompt_builder,
        output_language=settings.provided.output_language,
        review_enabled=settings.provided.review_enabled,
        review_llm_client=review_client,
        review_max_rounds=settings.provided.review_max_rounds,
    )

    performance_generator = providers.Singleton(
        PerformanceGenerator,
        llm_client=llm_client,
        prompt_builder=prompt_builder,
        script_format=settings.provided.script_format,
        output_language=settings.provided.output_language,
        review_enabled=settings.provided.review_enabled,
        review_llm_client=review_client,
        review_max_rounds=settings.provided.review_max_rounds,
    )

    # Conversation session persistence backend, selected by SESSION_STORE.
    # Sessions are stored under <output_dir>/conversations/<id>.json when
    # "file" (default), or kept in-memory only when "memory".
    session_store = providers.Singleton(
        create_session_store,
        backend=settings.provided.session_store,
        output_dir=settings.provided.output_dir,
    )

    conversation_manager = providers.Singleton(
        ConversationManager,
        llm_client=llm_client,
        prompt_builder=prompt_builder,
        max_iterations=settings.provided.review_max_rounds,
        output_language=settings.provided.output_language,
        store=session_store,
    )

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the container, optionally overriding the settings source.

        Args:
            settings: Optional explicit :class:`Settings` instance. When
                provided, it replaces the default global settings singleton so
                all dependent providers resolve against it.
        """
        super().__init__()
        if settings is not None:
            self.settings.override(settings)
