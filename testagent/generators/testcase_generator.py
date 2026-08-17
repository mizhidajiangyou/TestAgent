"""
Test case generator using LLM.

Two-phase strategy:
  Phase 1: Generate from requirements, batched by functional module.
  Phase 2 (optional): If API endpoints exist, generate API-specific cases
                       (boundary, security, integration) per endpoint batch.
Merge and re-number all cases.

Inspired by guardrails' targeted re-ask: when JSON parse fails, the retry
prompt tells the LLM exactly what went wrong and includes the failed output.

Optional multi-round cross-validation review:
  When ``review_enabled`` is true, the generator runs ``review_max_rounds``
  refinement passes. Odd rounds use the secondary LLM (non-primary model),
  even rounds use the primary model. This way two different models
  cross-validate each other. If only one model is configured, the secondary
  falls back to the primary and a warning is logged by the client.
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from testagent.config.models import (
    APIEndpoint,
    RequirementItem,
    TestCase,
    TestCaseGenInput,
    TestPriority,
    TestType,
)
from testagent.engine.llm_client import JSON_OBJECT_FORMAT, LLMClient
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.base import BaseGenerator
from testagent.parsers.requirement_parser import RequirementParser
from testagent.parsers.swagger_parser import SwaggerParser

logger = logging.getLogger(__name__)

#: Keys that may wrap a test-case list when JSON mode is enabled.
_TEST_CASES_WRAPPER_KEYS = ("test_cases", "cases", "data")

#: Max parse retries per batch (guardrails-style targeted re-ask).
MAX_PARSE_RETRIES = 3

#: Max requirements per module batch (keeps each LLM response small).
MAX_REQUIREMENTS_PER_BATCH = 3

#: Max endpoints per batch in phase 2.
MAX_ENDPOINTS_PER_BATCH = 2

#: Default cross-validation rounds when review is enabled.
DEFAULT_REVIEW_MAX_ROUNDS = 2

#: CSV column order.
CSV_COLUMNS = [
    "id",
    "title",
    "description",
    "endpoint",
    "test_type",
    "priority",
    "preconditions",
    "steps",
    "expected_results",
    "tags",
]


class TestCaseGenerator(BaseGenerator[TestCaseGenInput, list[TestCase]]):
    """Generate test cases from requirements (and optionally API endpoints)."""

    __test__ = False

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder,
        review_enabled: bool = False,
        review_llm_client: LLMClient | None = None,
        review_max_rounds: int = DEFAULT_REVIEW_MAX_ROUNDS,
        output_language: str = "english",
        json_mode: bool = False,
    ) -> None:
        self._llm = llm_client
        self._prompt_builder = prompt_builder
        self._review_enabled = review_enabled
        # Review client defaults to the primary client when not provided.
        # The container should pass a secondary (non-primary) client so that
        # cross-validation actually alternates between two models.
        self._review_llm: LLMClient = review_llm_client or llm_client
        # Clamp to at least 1 round; 0 effectively disables review even if
        # review_enabled is true (we log a warning in that case).
        self._review_max_rounds = max(1, review_max_rounds)
        self._output_language = output_language
        # When true, every generation/review call requests OpenAI JSON mode and
        # the model wraps its cases in {"test_cases": [...]}. Opt-in only — see
        # settings.OPENAI_JSON_MODE; disabled backends (non-OpenAI compatible)
        # must keep this False.
        self._json_mode = json_mode
        # JSON mode sends response_format={"type":"json_object"} to the LLM.
        # Endpoints that silently ignore this param (some "OpenAI-compatible"
        # proxies, qwen/glm, etc.) will NOT be forced into JSON and may return
        # free-form text -> parse failures. The caller must confirm the backend
        # genuinely enforces JSON mode before enabling it.
        if self._json_mode:
            logger.warning(
                "OPENAI_JSON_MODE is enabled: every generation/review call will "
                "send response_format={'type':'json_object'} to the LLM. Only "
                "backends that truly enforce JSON mode (e.g. real OpenAI) are "
                "safe here. OpenAI-compatible proxies that silently ignore this "
                "param (qwen/glm, some gateways) may return non-JSON and fail to "
                "parse. Keep OPENAI_JSON_MODE=false on such endpoints."
            )

        # Detect whether review will actually use a different model.
        self._review_is_cross_model = self._review_llm is not self._llm
        if review_enabled and not self._review_is_cross_model:
            logger.warning(
                "Review is enabled but the review LLM client is the same as the "
                "primary client. Cross-validation will be single-model only. "
                "Configure multiple models in OPENAI_MODEL to enable true "
                "multi-model cross-validation."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, data: TestCaseGenInput) -> list[TestCase]:
        """Generate test cases using two-phase batch strategy.

        Phase 1: Requirements → module-batched generation.
        Phase 2: Endpoints (if any) → endpoint-batched API-specific cases.

        When ``data.historical_cases`` is non-empty, the historical cases are
        used as a baseline: the LLM is asked to produce **only net-new or
        updated** cases for the new requirements (the historical context is
        injected into the prompt so the LLM avoids duplicating coverage).
        Historical cases are then merged with the new cases and re-numbered.
        """
        requirements = data.requirements
        endpoints = data.endpoints
        historical_cases = data.historical_cases

        all_cases: list[TestCase] = []
        phase1_cases: list[TestCase] = []

        # --- Phase 1: Requirements-driven generation ---
        if requirements:
            phase1_cases = self._generate_from_requirements(
                requirements, endpoints, historical_cases
            )
            all_cases.extend(phase1_cases)
        elif endpoints:
            # No requirements but have endpoints — generate from endpoints alone
            phase1_cases = self._generate_from_endpoints(endpoints, "")
            all_cases.extend(phase1_cases)
        else:
            logger.warning("No requirements or endpoints provided; nothing to generate.")
            return []

        # --- Phase 2: API-specific enhancement (only if both req + endpoints) ---
        if requirements and endpoints:
            # Feed Phase 1 coverage into Phase 2 so it does NOT regenerate the
            # same scenarios (kills cross-phase duplication / inconsistency).
            covered_text = self._historical_cases_to_text(phase1_cases)
            api_cases = self._generate_api_specific(
                endpoints, requirements, already_covered=covered_text
            )
            all_cases.extend(api_cases)

        # --- Merge historical cases (baseline) with newly generated cases ---
        if historical_cases:
            merged = self._merge_historical_cases(historical_cases, all_cases)
            all_cases = merged

        # Re-number sequentially
        for idx, tc in enumerate(all_cases, 1):
            tc.id = f"TC-{idx:03d}"

        logger.info("Total: %d test cases", len(all_cases))

        if self._review_enabled and all_cases:
            all_cases = self._review_and_refine(all_cases, endpoints, requirements)

        return all_cases

    # ------------------------------------------------------------------
    # Phase 1: Requirements-driven (module-batched)
    # ------------------------------------------------------------------

    def _generate_from_requirements(
        self,
        requirements: list[RequirementItem],
        endpoints: list[APIEndpoint],
        historical_cases: list[TestCase] | None = None,
    ) -> list[TestCase]:
        """Generate test cases from requirements, batched by module.

        When ``historical_cases`` is provided, a summary of the historical
        coverage is injected into the prompt so the LLM generates only
        net-new or updated cases (avoiding duplicates).
        """
        batches = self._split_requirement_batches(requirements)
        endpoints_text = SwaggerParser.endpoints_to_text(endpoints) if endpoints else ""
        historical_text = (
            self._historical_cases_to_text(historical_cases) if historical_cases else ""
        )

        all_cases: list[TestCase] = []

        for i, batch in enumerate(batches, 1):
            req_text = RequirementParser.requirements_to_text(batch)
            system_prompt, user_prompt = self._prompt_builder.build_testcase_prompt(
                endpoints_text=endpoints_text,
                requirements_text=req_text,
                output_language=self._output_language,
                extra_context={"historical_cases": historical_text} if historical_text else None,
                json_mode=self._json_mode,
            )
            logger.info(
                "Phase 1 - Batch %d/%d (%d requirements)...",
                i,
                len(batches),
                len(batch),
            )
            cases = self._generate_with_retry(
                system_prompt, user_prompt, endpoints, f"Req batch {i}/{len(batches)}"
            )
            all_cases.extend(cases)

        return all_cases

    # ------------------------------------------------------------------
    # Phase 2: API-specific (endpoint-batched)
    # ------------------------------------------------------------------

    def _generate_api_specific(
        self,
        endpoints: list[APIEndpoint],
        requirements: list[RequirementItem],
        already_covered: str = "",
    ) -> list[TestCase]:
        """Generate API-specific cases (boundary, security, integration).

        ``already_covered`` is a text summary of the Phase 1 cases, injected so
        Phase 2 avoids regenerating scenarios already produced.
        """
        batches = self._split_endpoint_batches(endpoints)
        req_text = RequirementParser.requirements_to_text(requirements)
        all_cases: list[TestCase] = []

        for i, batch in enumerate(batches, 1):
            ep_text = SwaggerParser.endpoints_to_text(batch)
            system_prompt, user_prompt = self._prompt_builder.build_api_prompt(
                endpoints_text=ep_text,
                requirements_text=req_text,
                output_language=self._output_language,
                json_mode=self._json_mode,
                already_covered=already_covered,
            )
            logger.info(
                "Phase 2 - Batch %d/%d (%d endpoints, API-specific)...",
                i,
                len(batches),
                len(batch),
            )
            cases = self._generate_with_retry(
                system_prompt, user_prompt, batch, f"API batch {i}/{len(batches)}"
            )
            all_cases.extend(cases)

        return all_cases

    def _generate_from_endpoints(
        self, endpoints: list[APIEndpoint], requirements_text: str
    ) -> list[TestCase]:
        """Generate from endpoints only (no requirements)."""
        batches = self._split_endpoint_batches(endpoints)
        all_cases: list[TestCase] = []

        for i, batch in enumerate(batches, 1):
            ep_text = SwaggerParser.endpoints_to_text(batch)
            system_prompt, user_prompt = self._prompt_builder.build_testcase_prompt(
                endpoints_text=ep_text,
                requirements_text=requirements_text or "No specific requirements.",
                output_language=self._output_language,
                json_mode=self._json_mode,
            )
            logger.info(
                "Endpoint batch %d/%d (%d endpoints)...",
                i,
                len(batches),
                len(batch),
            )
            cases = self._generate_with_retry(
                system_prompt, user_prompt, batch, f"EP batch {i}/{len(batches)}"
            )
            all_cases.extend(cases)

        return all_cases

    # ------------------------------------------------------------------
    # Batch splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_requirement_batches(
        requirements: list[RequirementItem],
    ) -> list[list[RequirementItem]]:
        """Split requirements into module-aware batches."""
        if not requirements:
            return []
        # Group by module, then split large modules
        modules: dict[str, list[RequirementItem]] = {}
        for req in requirements:
            key = req.module or "default"
            modules.setdefault(key, []).append(req)

        batches: list[list[RequirementItem]] = []
        for module_reqs in modules.values():
            for i in range(0, len(module_reqs), MAX_REQUIREMENTS_PER_BATCH):
                batches.append(module_reqs[i : i + MAX_REQUIREMENTS_PER_BATCH])
        return batches

    @staticmethod
    def _split_endpoint_batches(
        endpoints: list[APIEndpoint],
    ) -> list[list[APIEndpoint]]:
        """Split endpoints into small batches."""
        if not endpoints:
            return []
        return [
            endpoints[i : i + MAX_ENDPOINTS_PER_BATCH]
            for i in range(0, len(endpoints), MAX_ENDPOINTS_PER_BATCH)
        ]

    # ------------------------------------------------------------------
    # LLM call with targeted re-ask (guardrails-inspired)
    # ------------------------------------------------------------------

    def _generate_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        endpoints: list[APIEndpoint],
        step_label: str,
        client: LLMClient | None = None,
    ) -> list[TestCase]:
        """Call LLM with targeted re-ask on parse failure.

        Instead of blindly retrying with the same prompt, the re-ask tells
        the LLM exactly what went wrong (parse error / truncation) and
        includes the failed output so it can fix it.

        Inspired by guardrails' NonParseableReAsk / SkeletonReAsk split:
        we classify the failure and give the LLM a targeted fix hint.

        Args:
            client: Optional LLM client override (used by review rounds to
                alternate between primary and secondary models). Defaults to
                the primary client.
        """
        llm = client or self._llm
        last_raw = ""

        for attempt in range(1, MAX_PARSE_RETRIES + 1):
            if attempt == 1:
                effective_prompt = user_prompt
            else:
                # Targeted re-ask: classify the failure and tell the LLM
                error_type = self._classify_failure(last_raw)
                effective_prompt = self._build_reask_prompt(
                    user_prompt, last_raw, step_label, error_type
                )

            raw_response = llm.chat(
                system_prompt,
                effective_prompt,
                response_format=JSON_OBJECT_FORMAT if self._json_mode else None,
            )
            last_raw = raw_response

            items = self._extract_json(raw_response)
            if items is None:
                items = self._salvage_truncated_json(raw_response)

            if items is not None:
                test_cases = self._to_test_cases(items, endpoints)
                logger.info("%s: %d cases (attempt %d)", step_label, len(test_cases), attempt)
                return test_cases

            logger.warning(
                "%s attempt %d/%d: cannot parse JSON. Preview: %.200s",
                step_label,
                attempt,
                MAX_PARSE_RETRIES,
                raw_response,
            )
            self._dump_debug_response(raw_response)

        logger.error("%s gave up after %d attempts", step_label, MAX_PARSE_RETRIES)
        return []

    @staticmethod
    def _classify_failure(raw: str) -> str:
        """Classify why the LLM output could not be parsed.

        Returns ``"truncated"`` when the output looks like an incomplete JSON
        array/object (started but never closed), otherwise ``"non_parseable"``
        for outputs with no usable JSON structure. This mirrors guardrails'
        distinction between NonParseableReAsk and truncation handling.
        """
        text = raw.strip()
        has_array_start = "[" in text
        has_array_end = "]" in text
        has_obj_start = "{" in text
        # Heuristic: a JSON array/object was started but never closed
        if (has_array_start and not has_array_end) or (
            has_obj_start and not has_array_end and "}" not in text
        ):
            return "truncated"
        if has_array_start or has_obj_start:
            # Some JSON structure exists but still failed to parse
            return "truncated"
        return "non_parseable"

    @staticmethod
    def _build_reask_prompt(
        original_prompt: str,
        failed_output: str,
        label: str,
        error_type: str = "non_parseable",
    ) -> str:
        """Build a targeted re-ask prompt (guardrails pattern).

        Tells the LLM its previous output was not usable, shows the failed
        output, and asks it to fix and return only valid JSON. The hint is
        tailored to the failure type so the LLM knows what to fix.
        """
        # Truncate failed output to avoid token bloat
        truncated = failed_output[:2000]
        if len(failed_output) > 2000:
            truncated += "\n... [truncated]"

        if error_type == "truncated":
            diagnosis = (
                f"Your previous response for '{label}' was TRUNCATED: the JSON "
                "array/object was started but never closed, so it could not be parsed."
            )
            fixes = (
                "Generate FEWER test cases so the output fits within the token limit. "
                "Make each case more compact: shorter descriptions, fewer steps, "
                "concise expected_results. Ensure every object and array is properly closed."
            )
        else:
            diagnosis = (
                f"Your previous response for '{label}' was NOT valid JSON and could not be parsed."
            )
            fixes = (
                "Return ONLY a valid JSON array. Common fixes:\n"
                "- Remove any text before [ or after ]\n"
                "- Remove markdown code fences (```)\n"
                "- Ensure all strings are properly escaped (no unescaped quotes)\n"
                "- Ensure all objects and arrays are properly closed"
            )

        return (
            f"{original_prompt}\n\n"
            f"---\n"
            f"IMPORTANT: {diagnosis}\n\n"
            f"Here is what you returned:\n\n"
            f"{truncated}\n\n"
            f"{fixes}\n"
        )

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------

    def _review_and_refine(
        self,
        test_cases: list[TestCase],
        endpoints: list[APIEndpoint],
        requirements: list[RequirementItem],
    ) -> list[TestCase]:
        """Run multi-round cross-validation review.

        Round flow (``review_max_rounds`` total):
          - Round 1: secondary model reviews the generated cases.
          - Round 2: primary model reviews round-1 output.
          - Round 3: secondary reviews round-2 output.
          - ... alternating until ``review_max_rounds`` exhausted.

        Each round runs in a fresh conversation with no prior context. If a
        round fails to parse, the previous round's output is kept and the
        loop continues (so a transient parse failure does not discard
        accumulated refinement). If every round fails, the original input
        is returned unchanged.
        """
        endpoints_text = SwaggerParser.endpoints_to_text(endpoints) if endpoints else ""
        requirements_text = RequirementParser.requirements_to_text(requirements)

        current_cases = test_cases
        for round_idx in range(1, self._review_max_rounds + 1):
            # Odd rounds: secondary (non-primary) model.
            # Even rounds: primary model.
            # This gives true cross-validation when two models are configured.
            use_secondary = round_idx % 2 == 1
            if use_secondary:
                client = self._review_llm
                client_label = "secondary" if self._review_is_cross_model else "primary(same)"
            else:
                client = self._llm
                client_label = "primary"

            current_json = json.dumps(
                [self._testcase_to_dict(tc) for tc in current_cases],
                ensure_ascii=False,
            )
            system_prompt, user_prompt = self._prompt_builder.build_review_prompt(
                endpoints_text=endpoints_text,
                requirements_text=requirements_text,
                test_cases_json=current_json,
                output_language=self._output_language,
                json_mode=self._json_mode,
            )

            logger.info(
                "Review round %d/%d using %s model (%d cases in)...",
                round_idx,
                self._review_max_rounds,
                client_label,
                len(current_cases),
            )
            refined = self._generate_with_retry(
                system_prompt,
                user_prompt,
                endpoints,
                f"Review round {round_idx}",
                client=client,
            )

            if not refined:
                logger.warning(
                    "Review round %d returned nothing; keeping previous %d cases.",
                    round_idx,
                    len(current_cases),
                )
                continue

            logger.info("Review round %d: %d -> %d", round_idx, len(current_cases), len(refined))
            current_cases = refined

        return current_cases

    # ------------------------------------------------------------------
    # Historical case merging
    # ------------------------------------------------------------------

    @staticmethod
    def _historical_cases_to_text(cases: list[TestCase] | None) -> str:
        """Render historical cases as a compact text summary for prompt injection.

        Includes id, title, endpoint, test_type and a one-line description so
        the LLM can see what's already covered and avoid regenerating the same
        scenarios.
        """
        if not cases:
            return ""
        lines = [f"Total existing cases: {len(cases)}", ""]
        for tc in cases:
            line = f"- [{tc.id}] {tc.title} | {tc.endpoint.full_path} | {tc.test_type.value}"
            if tc.description:
                # Truncate long descriptions to keep the prompt compact.
                desc = tc.description[:120]
                if len(tc.description) > 120:
                    desc += "..."
                line += f" | {desc}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _merge_historical_cases(
        historical: list[TestCase], new_cases: list[TestCase]
    ) -> list[TestCase]:
        """Merge historical baseline cases with newly generated cases.

        De-duplicates by a fuzzy key (title + endpoint + test_type) so that if
        the LLM regenerated a case that already exists historically, the
        historical version is kept (preserving its original detail) and the
        duplicate new case is dropped.

        Returns the merged list with historical cases first, then net-new cases.
        """
        merged: list[TestCase] = []
        seen_keys: set[str] = set()

        # Historical cases form the baseline.
        for tc in historical:
            key = TestCaseGenerator._case_dedup_key(tc)
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(tc)

        # Append only net-new cases (not already in the baseline).
        new_count = 0
        for tc in new_cases:
            key = TestCaseGenerator._case_dedup_key(tc)
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(tc)
                new_count += 1

        logger.info(
            "Merged %d historical + %d net-new = %d total (dedup removed %d duplicates)",
            len(historical),
            new_count,
            len(merged),
            len(new_cases) - new_count,
        )
        return merged

    @staticmethod
    def _case_dedup_key(tc: TestCase) -> str:
        """Build a fuzzy de-duplication key for a test case.

        Uses lowercased title + endpoint + test_type so minor formatting
        differences (case, trailing spaces) don't cause false duplicates.
        """
        return f"{tc.title.strip().lower()}|{tc.endpoint.full_path.lower()}|{tc.test_type.value}"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, output: list[TestCase], output_path: Path) -> Path:
        """Save to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [self._testcase_to_dict(tc) for tc in output]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d test cases to %s", len(output), output_path)
        return output_path

    def save_csv(self, output: list[TestCase], output_path: Path) -> Path:
        """Save to CSV (UTF-8 BOM for Excel)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for tc in output:
                row = self._testcase_to_dict(tc)
                for key in ("preconditions", "steps", "expected_results", "tags"):
                    row[key] = "; ".join(str(v) for v in row[key])
                writer.writerow(row)
        logger.info("Saved %d test cases to %s (csv)", len(output), output_path)
        return output_path

    @staticmethod
    def load_historical_cases(path: str | Path) -> list[TestCase]:
        """Load previously generated test cases from a JSON file.

        Supports the JSON array format produced by :meth:`save`. Each element
        must have at least ``id``, ``title``, ``endpoint``, ``test_type`` and
        ``priority``; missing optional fields default to empty lists.

        Args:
            path: Path to the historical test cases JSON file.

        Returns:
            List of :class:`TestCase` objects. Returns an empty list when the
            file cannot be parsed (with a warning logged).
        """
        file_path = Path(path)
        if not file_path.exists():
            logger.warning("Historical test cases file not found: %s", path)
            return []
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse historical cases from %s: %s", path, exc)
            return []
        if not isinstance(data, list):
            logger.warning("Historical cases file %s is not a JSON array", path)
            return []

        cases: list[TestCase] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            tc = TestCaseGenerator._dict_to_testcase(item)
            if tc is not None:
                cases.append(tc)
        logger.info("Loaded %d historical test cases from %s", len(cases), path)
        return cases

    @staticmethod
    def _dict_to_testcase(item: dict[str, Any]) -> TestCase | None:
        """Convert a dict (from JSON) back to a TestCase object.

        Returns ``None`` when required fields are missing or invalid.
        """
        try:
            endpoint_str = str(item.get("endpoint", "N/A N/A"))
            parts = endpoint_str.split(None, 1)
            method = parts[0] if len(parts) >= 1 else "N/A"
            path = parts[1] if len(parts) >= 2 else "N/A"
            endpoint = APIEndpoint(method=method, path=path)

            test_type_str = str(item.get("test_type", "functional")).lower()
            try:
                test_type = TestType(test_type_str)
            except ValueError:
                test_type = TestType.FUNCTIONAL

            priority_str = str(item.get("priority", "medium")).lower()
            try:
                priority = TestPriority(priority_str)
            except ValueError:
                priority = TestPriority.MEDIUM

            return TestCase(
                id=str(item.get("id", "")),
                title=str(item.get("title", "")),
                description=str(item.get("description", "")),
                endpoint=endpoint,
                test_type=test_type,
                priority=priority,
                preconditions=list(item.get("preconditions", [])),
                steps=list(item.get("steps", [])),
                expected_results=list(item.get("expected_results", [])),
                tags=list(item.get("tags", [])),
            )
        except (KeyError, TypeError) as exc:
            logger.warning("Failed to convert dict to TestCase: %s", exc)
            return None

    # ------------------------------------------------------------------
    # JSON parsing & salvage
    # ------------------------------------------------------------------

    def _extract_json(self, raw: str) -> list[Any] | None:
        """Extract JSON array from raw LLM response."""
        text = raw.strip()
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            parsed = json.loads(text)
            # Unify the bare-array and {"test_cases": [...]} envelope contracts
            # so both default and JSON-mode output normalize to a flat list.
            return self._unwrap_test_cases(parsed)
        except json.JSONDecodeError:
            pass

        for open_ch, close_ch in (("[", "]"), ("{", "}")):
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start != -1 and end > start:
                try:
                    parsed = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
                return self._unwrap_test_cases(parsed)
        return None

    @staticmethod
    def _unwrap_test_cases(parsed: Any) -> list[Any] | None:
        """Normalize a parsed JSON value into a list of test-case dicts.

        Handles both the bare-array contract (default) and the
        ``{"test_cases": [...]}`` envelope produced when JSON mode is enabled.
        A single bare object (legacy/edge case) is wrapped in a list.
        """
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in _TEST_CASES_WRAPPER_KEYS:
                if key in parsed and isinstance(parsed[key], list):
                    cases: list[Any] = parsed[key]
                    return cases
            return [parsed]
        return None

    @staticmethod
    def _salvage_truncated_json(raw: str) -> list[Any] | None:
        """Salvage a truncated JSON array by closing brackets."""
        text = raw.strip()
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        bracket_start = text.find("[")
        if bracket_start == -1:
            return None

        depth = 0
        last_complete_obj_end = -1
        in_string = False
        escape = False

        for i in range(bracket_start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_complete_obj_end = i
            elif ch == "]" and depth == 0:
                break

        if last_complete_obj_end == -1:
            return None

        salvaged = text[: last_complete_obj_end + 1] + "]"
        salvaged = re.sub(r",\s*\]$", "]", salvaged)
        try:
            parsed = json.loads(salvaged)
            if isinstance(parsed, list) and len(parsed) > 0:
                logger.info("Salvaged %d cases from truncated response", len(parsed))
                return parsed
        except json.JSONDecodeError:
            pass
        return None

    @staticmethod
    def _dump_debug_response(raw: str) -> None:
        """Persist unparseable response for debugging."""
        try:
            debug_path = Path("output/.debug_last_llm_response.txt")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(raw, encoding="utf-8")
            logger.info("Raw response saved to %s", debug_path)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _to_test_cases(self, items: list[Any], endpoints: list[APIEndpoint]) -> list[TestCase]:
        """Convert parsed JSON items into TestCase objects."""
        endpoint_map = {ep.full_path: ep for ep in endpoints}
        # Fallback endpoint for requirement-only cases (no API spec)
        fallback_ep = endpoints[0] if endpoints else APIEndpoint(method="N/A", path="N/A")

        test_cases: list[TestCase] = []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            ep_key = item.get("endpoint", "")
            endpoint = endpoint_map.get(ep_key, fallback_ep)
            try:
                test_type = TestType(item.get("test_type", "functional"))
            except ValueError:
                test_type = TestType.FUNCTIONAL
            try:
                priority = TestPriority(item.get("priority", "medium"))
            except ValueError:
                priority = TestPriority.MEDIUM

            test_cases.append(
                TestCase(
                    id=item.get("id", f"TC-{idx:03d}"),
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    endpoint=endpoint,
                    test_type=test_type,
                    priority=priority,
                    preconditions=item.get("preconditions", []),
                    steps=item.get("steps", []),
                    expected_results=item.get("expected_results", []),
                    tags=item.get("tags", []),
                )
            )
        return test_cases

    @staticmethod
    def _testcase_to_dict(tc: TestCase) -> dict[str, Any]:
        """Convert TestCase to serializable dict."""
        return {
            "id": tc.id,
            "title": tc.title,
            "description": tc.description,
            "endpoint": tc.endpoint.full_path,
            "test_type": tc.test_type.value,
            "priority": tc.priority.value,
            "preconditions": tc.preconditions,
            "steps": tc.steps,
            "expected_results": tc.expected_results,
            "tags": tc.tags,
        }
