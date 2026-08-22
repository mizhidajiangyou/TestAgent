"""
Performance test script generator using LLM.

Generates k6 or JMeter scripts from API endpoints.
"""

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from testagent.config.models import APIEndpoint, PerfGenInput, PerformanceConfig
from testagent.engine.llm_client import LLMClient
from testagent.engine.prompt_builder import PromptBuilder, endpoints_to_signature
from testagent.engine.review import ReviewLoop, ReviewResult
from testagent.generators.base import BaseGenerator
from testagent.parsers.swagger_parser import SwaggerParser

logger = logging.getLogger(__name__)


class PerformanceGenerator(BaseGenerator[PerfGenInput, str]):
    """Generate performance test scripts from API specs."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder,
        script_format: str = "k6",
        output_language: str = "english",
        review_enabled: bool = False,
        review_llm_client: LLMClient | None = None,
        review_max_rounds: int = 2,
    ) -> None:
        self._llm = llm_client
        self._prompt_builder = prompt_builder
        self._format = script_format
        self._output_language = output_language
        # Review is opt-in at construction time (plan v2 decision 5): pure
        # code callers never trigger an extra LLM pass by accident; the
        # container wires settings.review_enabled in for CLI/web use.
        self._review_enabled = review_enabled
        self._review_loop = ReviewLoop[str](
            primary_llm=llm_client,
            review_llm=review_llm_client or llm_client,
            prompt_builder=prompt_builder,
            max_rounds=review_max_rounds,
        )
        # Review outcome of the last generate() call (None when review did
        # not run); consumed by save() to persist the .meta.json marker.
        self._last_review: ReviewResult[str] | None = None

    def set_script_format(self, script_format: str) -> None:
        """Runtime override of the script format (CLI ``--format``).

        Public setter so DI-resolved singletons can be reconfigured without
        callers touching private attributes.
        """
        self._format = script_format

    def set_review_enabled(self, enabled: bool) -> None:
        """Runtime override of the review switch (CLI ``--review/--no-review``)."""
        self._review_enabled = enabled

    def generate(self, data: PerfGenInput) -> str:
        """Generate a performance test script.

        Args:
            data: Input payload containing endpoints and config.

        Returns:
            Generated script content as string.
        """
        endpoints_text = SwaggerParser.endpoints_to_text(data.endpoints)
        perf_config = data.config or PerformanceConfig()

        config_dict: dict[str, Any] = {
            "base_url": perf_config.base_url,
            "virtual_users": perf_config.virtual_users,
            "duration_seconds": perf_config.duration_seconds,
            "ramp_up_seconds": perf_config.ramp_up_seconds,
            "think_time_ms": perf_config.think_time_ms,
            "auth_type": perf_config.auth_type,
        }

        system_prompt, user_prompt = self._prompt_builder.build_performance_prompt(
            endpoints_text=endpoints_text,
            config=config_dict,
            script_format=self._format,
            output_language=self._output_language,
        )

        logger.info("Generating %s performance script via LLM...", self._format)
        raw_response = self._llm.chat(system_prompt, user_prompt)

        script = self._extract_script(raw_response)

        self._last_review = None
        if self._review_enabled and script:
            script, self._last_review = self._review_script(script, perf_config, data.endpoints)

        if self._format == "jmeter":
            self._validate_jmx(script)

        logger.info("Generated %s script (%d chars)", self._format, len(script))
        return script

    def save(self, output: str, output_path: Path) -> Path:
        """Save script to file.

        When the last generate() ran a review pass, a sibling
        ``<stem>.meta.json`` records the review outcome (plan v2: consumers
        use ``reviewed`` to tell refined scripts from first-pass ones).

        Args:
            output: Script content.
            output_path: Target file path.

        Returns:
            Saved file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
        logger.info("Saved %s script to %s", self._format, output_path)

        if self._last_review is not None:
            meta_path = output_path.with_suffix(".meta.json")
            meta = {
                "generator": "performance",
                "script_format": self._format,
                "reviewed": self._last_review.used_review,
                "rounds_executed": self._last_review.rounds_executed,
                "rounds_succeeded": self._last_review.rounds_succeeded,
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Saved review metadata to %s", meta_path)

        return output_path

    def _review_script(
        self,
        script: str,
        perf_config: PerformanceConfig,
        endpoints: list[APIEndpoint],
    ) -> tuple[str, ReviewResult[str]]:
        """Run the shared ReviewLoop over the generated script (no call_llm).

        The loop calls the clients itself and validates each candidate via
        ``_extract_script`` + ``_validate_jmx`` (plan v2 R2 / §4.3): a
        candidate that fails validation is treated as a failed round, so a
        bad review answer can never corrupt a good first-pass script.
        """
        context_text = (
            "## Load Configuration\n"
            f"- base_url: {perf_config.base_url}\n"
            f"- virtual_users: {perf_config.virtual_users}\n"
            f"- duration_seconds: {perf_config.duration_seconds}\n"
            f"- ramp_up_seconds: {perf_config.ramp_up_seconds}\n"
            f"- think_time_ms: {perf_config.think_time_ms}\n"
            f"- auth_type: {perf_config.auth_type}\n\n"
            "## Endpoint Signatures\n"
            f"{endpoints_to_signature(endpoints)}"
        )

        def build_prompt(current: str, round_idx: int) -> tuple[str, str]:
            return self._prompt_builder.build_script_review_prompt(
                script_kind=self._format,
                script=current,
                context_text=context_text,
                output_language=self._output_language,
            )

        def parse(raw: str) -> str | None:
            candidate = self._extract_script(raw)
            if not candidate:
                return None
            if self._format == "jmeter":
                try:
                    self._validate_jmx(candidate)
                except ValueError:
                    return None
            return candidate

        result = self._review_loop.run(
            script,
            build_prompt=build_prompt,
            parse=parse,
            label=f"{self._format}-script",
        )
        return result.artifact, result

    def _extract_script(self, raw: str) -> str:
        """Strip markdown code fences if present."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.startswith("```")]
            cleaned = "\n".join(lines).strip()
        return cleaned

    def _validate_jmx(self, xml_text: str) -> None:
        """Validate JMX XML structure."""
        if not xml_text.startswith("<?xml"):
            raise ValueError("XML declaration missing from JMX output")

        if not xml_text.strip().endswith("</jmeterTestPlan>"):
            raise ValueError("JMX missing closing </jmeterTestPlan> tag")

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise ValueError(f"JMX XML parse error: {exc}") from exc

        if root.tag != "jmeterTestPlan":
            raise ValueError(f"Root tag is '{root.tag}', expected 'jmeterTestPlan'")
