"""
Dependency injection container.

Provides a lightweight DI container for wiring up components.
"""

from testagent.config.settings import Settings, get_settings
from testagent.engine.conversation import ConversationManager
from testagent.engine.llm_client import MultiModelLLMClient, create_llm_client
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.performance_generator import PerformanceGenerator
from testagent.generators.testcase_generator import TestCaseGenerator
from testagent.parsers.requirement_parser import RequirementParser
from testagent.parsers.swagger_parser import SwaggerParser
from testagent.reports.performance_report import PerformanceReport
from testagent.reports.testcase_report import TestCaseReport


class Container:
    """Simple dependency injection container."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm_client: MultiModelLLMClient | None = None
        self._prompt_builder: PromptBuilder | None = None
        self._swagger_parser: SwaggerParser | None = None
        self._requirement_parser: RequirementParser | None = None
        self._testcase_generator: TestCaseGenerator | None = None
        self._performance_generator: PerformanceGenerator | None = None
        self._testcase_report: TestCaseReport | None = None
        self._performance_report: PerformanceReport | None = None
        self._conversation_manager: ConversationManager | None = None

    @property
    def settings(self) -> Settings:
        """Return application settings."""
        return self._settings

    @property
    def llm_client(self) -> MultiModelLLMClient:
        """Return multi-model LLM client (lazy init)."""
        if self._llm_client is None:
            self._llm_client = create_llm_client(self._settings)
        return self._llm_client

    @property
    def prompt_builder(self) -> PromptBuilder:
        """Return prompt builder (lazy init)."""
        if self._prompt_builder is None:
            self._prompt_builder = PromptBuilder()
        return self._prompt_builder

    @property
    def swagger_parser(self) -> SwaggerParser:
        """Return Swagger parser (lazy init)."""
        if self._swagger_parser is None:
            self._swagger_parser = SwaggerParser()
        return self._swagger_parser

    @property
    def requirement_parser(self) -> RequirementParser:
        """Return requirement parser (lazy init)."""
        if self._requirement_parser is None:
            self._requirement_parser = RequirementParser()
        return self._requirement_parser

    @property
    def testcase_generator(self) -> TestCaseGenerator:
        """Return test case generator (lazy init)."""
        if self._testcase_generator is None:
            # Review uses a non-primary model when available, so cross-validation
            # happens between two different models. The MultiModelLLMClient
            # logs a warning and falls back to the primary when only one model
            # is configured.
            review_client = self.llm_client.secondary_client()
            self._testcase_generator = TestCaseGenerator(
                llm_client=self.llm_client,
                prompt_builder=self.prompt_builder,
                review_enabled=self._settings.review_enabled,
                review_llm_client=review_client,
                review_max_rounds=self._settings.review_max_rounds,
                output_language=self._settings.output_language,
            )
        return self._testcase_generator

    @property
    def performance_generator(self) -> PerformanceGenerator:
        """Return performance generator (lazy init)."""
        if self._performance_generator is None:
            self._performance_generator = PerformanceGenerator(
                llm_client=self.llm_client,
                prompt_builder=self.prompt_builder,
                script_format=self._settings.script_format,
                output_language=self._settings.output_language,
            )
        return self._performance_generator

    @property
    def testcase_report(self) -> TestCaseReport:
        """Return test case report generator (lazy init)."""
        if self._testcase_report is None:
            self._testcase_report = TestCaseReport()
        return self._testcase_report

    @property
    def performance_report(self) -> PerformanceReport:
        """Return performance report generator (lazy init)."""
        if self._performance_report is None:
            self._performance_report = PerformanceReport()
        return self._performance_report

    @property
    def conversation_manager(self) -> ConversationManager:
        """Return conversation manager (lazy init).

        Reuses ``review_max_rounds`` as the max refine iterations per turn.
        """
        if self._conversation_manager is None:
            self._conversation_manager = ConversationManager(
                llm_client=self.llm_client,
                prompt_builder=self.prompt_builder,
                max_iterations=self._settings.review_max_rounds,
                output_language=self._settings.output_language,
            )
        return self._conversation_manager
