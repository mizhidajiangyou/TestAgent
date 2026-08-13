"""
Performance test script generator using LLM.

Generates k6 or JMeter scripts from API endpoints.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from testagent.config.models import PerfGenInput, PerformanceConfig
from testagent.engine.llm_client import LLMClient
from testagent.engine.prompt_builder import PromptBuilder
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
    ) -> None:
        self._llm = llm_client
        self._prompt_builder = prompt_builder
        self._format = script_format
        self._output_language = output_language

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

        if self._format == "jmeter":
            self._validate_jmx(script)

        logger.info("Generated %s script (%d chars)", self._format, len(script))
        return script

    def save(self, output: str, output_path: Path) -> Path:
        """Save script to file.

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
        return output_path

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
