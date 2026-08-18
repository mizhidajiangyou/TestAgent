"""
Conversational refinement engine.

Inspired by langgraph's ``StateGraph`` + ``MessagesState`` + checkpointer
pattern:

- **State management**: :class:`ConversationState` tracks messages, artifacts
  and feedback across turns (analogous to langgraph's ``MessagesState`` with
  the ``add_messages`` reducer).
- **Checkpoint / memory**: :class:`ConversationManager` keeps an in-memory
  store of sessions keyed by ``session_id`` (analogous to langgraph's
  ``InMemorySaver`` keyed by ``thread_id``). State persists across turns
  within the same session.
- **Iterative refine loop**: each user message can trigger a
  generate → validate → reflect → refine cycle (analogous to langgraph's
  ReAct loop with conditional edges), iterating up to ``max_iterations``
  until the artifact passes validation.
"""

import ast
import json
import logging
import re
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from testagent.engine.llm_client import LLMClient
from testagent.engine.prompt_builder import PromptBuilder
from testagent.engine.session_store import InMemoryStore, SessionStore
from testagent.generators.gui_test_generator import DEFAULT_TARGET_URL

logger = logging.getLogger(__name__)

#: Default max refine iterations per :meth:`ConversationSession.send` call.
DEFAULT_MAX_ITERATIONS = 3

#: Supported conversation actions.
VALID_ACTIONS = {"generate", "refine", "validate", "chat"}

#: Required fields for a test-case artifact.
TEST_CASE_REQUIRED_FIELDS = ("id", "title", "steps", "expected_results")

#: Keywords that hint an ``expected_result`` entry is machine-checkable.
ASSERTION_KEYWORDS = (
    "status",
    "code",
    "equals",
    "equal",
    "contains",
    "contain",
    "match",
    "present",
    "absent",
    "exists",
    "==",
    "!=",
    ">=",
    "<=",
    ">",
    "<",
    "返回",
    "状态码",
    "包含",
    "等于",
    "存在",
    "不包含",
)

#: Regex for ``METHOD /path`` endpoint tokens inside an endpoints spec.
_ENDPOINT_RE = re.compile(r"(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+/\S*", re.IGNORECASE)

__all__ = [
    "Artifact",
    "ConversationManager",
    "ConversationMessage",
    "ConversationSession",
    "ConversationState",
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def _new_artifact_id() -> str:
    """Generate a short unique artifact id."""
    return uuid.uuid4().hex[:12]


def _extract_json(raw: str) -> Any | None:
    """Extract the first valid JSON object/array from an LLM response.

    Strips markdown code fences, then tries a direct :func:`json.loads`. If
    that fails, locates the first ``[``...``]`` or ``{``...``}`` span and
    parses it. Returns the parsed value (list or dict) or ``None`` when no
    JSON can be extracted.
    """
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from a script/code response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [ln for ln in lines if not ln.startswith("```")]
        cleaned = "\n".join(lines).strip()
    return cleaned


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------


@dataclass
class ConversationMessage:
    """A single message in a conversation turn."""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Artifact:
    """A versioned generated artifact (test cases, script, code, ...).

    The ``parent_id`` links an artifact to the previous version in a
    refinement chain, enabling full lineage tracking.
    """

    id: str
    type: str  # "test_cases" | "performance_script" | "gui_script" | "code"
    content: str
    version: int
    created_at: str
    parent_id: str | None = None


@dataclass
class ConversationState:
    """Snapshot of a conversation session's state."""

    session_id: str
    messages: list[ConversationMessage] = field(default_factory=list)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    feedback: str = ""
    iteration: int = 0
    status: str = "idle"  # "idle" | "generating" | "validating" | "refining" | "completed"


# ----------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------


class ConversationSession:
    """A single conversation session with memory of messages and artifacts.

    Inspired by langgraph's checkpointer pattern: state persists across turns
    within the same ``session_id``. Each user message triggers a
    generate → validate → refine cycle that can iterate until quality is
    satisfactory.
    """

    def __init__(
        self,
        session_id: str,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        output_language: str = "english",
        persist_callback: Callable[[ConversationSession], None] | None = None,
    ) -> None:
        self._session_id = session_id
        self._llm = llm_client
        self._prompt_builder = prompt_builder
        self._max_iterations = max(1, max_iterations)
        self._output_language = output_language

        self._messages: list[ConversationMessage] = []
        self._artifacts: dict[str, Artifact] = {}
        self._feedback: str = ""
        self._iteration: int = 0
        self._status: str = "idle"

        # Inputs cached from the most recent context for generation/validation.
        self._endpoints_text: str = ""
        self._requirements_text: str = ""

        # Optional persistence hook: when set (by ConversationManager), each
        # ``send()`` writes the resulting snapshot to the configured store so
        # the session survives process restarts. ``None`` keeps the session
        # purely in-memory (used by unit tests that construct a session directly).
        self._persist_callback = persist_callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """Return this session's id."""
        return self._session_id

    @property
    def persist_callback(self) -> Callable[[ConversationSession], None] | None:
        """Return the persistence hook (or ``None`` when persistence is off)."""
        return self._persist_callback

    @persist_callback.setter
    def persist_callback(self, cb: Callable[[ConversationSession], None] | None) -> None:
        """Set or clear the persistence hook (used by ConversationManager)."""
        self._persist_callback = cb

    def to_snapshot(self) -> dict[str, Any]:
        """Serialize the session state for persistence.

        Captures messages, artifacts, feedback, iteration, status and the
        cached context (endpoints/requirements text). The ``llm_client`` and
        ``prompt_builder`` are runtime-only dependencies and are NOT included;
        they are re-injected when the session is restored by the manager.
        """
        return {
            "session_id": self._session_id,
            "messages": [asdict(m) for m in self._messages],
            "artifacts": {k: asdict(v) for k, v in self._artifacts.items()},
            "feedback": self._feedback,
            "iteration": self._iteration,
            "status": self._status,
            "endpoints_text": self._endpoints_text,
            "requirements_text": self._requirements_text,
        }

    def _apply_snapshot(self, data: dict[str, Any]) -> None:
        """Restore in-memory state from a snapshot produced by ``to_snapshot``.

        Overwrites all session fields; used by ConversationManager when a
        session is resumed from the store.
        """
        self._messages = [ConversationMessage(**m) for m in data.get("messages", [])]
        self._artifacts = {k: Artifact(**v) for k, v in data.get("artifacts", {}).items()}
        self._feedback = data.get("feedback", "")
        self._iteration = data.get("iteration", 0)
        self._status = data.get("status", "idle")
        self._endpoints_text = data.get("endpoints_text", "")
        self._requirements_text = data.get("requirements_text", "")

    @property
    def iteration(self) -> int:
        """Return the current refine iteration count for this turn."""
        return self._iteration

    def set_max_iterations(self, value: int) -> None:
        """Override the max refine iterations for this session.

        Used by the CLI to apply a per-invocation ``--max-iterations`` override
        without recreating the session.
        """
        self._max_iterations = max(1, value)

    def send(self, user_message: str, context: dict[str, Any] | None = None) -> str:
        """Send a user message and get a response.

        The ``context`` dict may include:

        - ``"action"``: ``"generate" | "refine" | "validate" | "chat"``
          (inferred from the message when omitted).
        - ``"artifacts"``: list of artifact dicts to load as a starting point.
        - ``"endpoints"``: API endpoints text.
        - ``"requirements"``: requirements text.
        - ``"artifact_type"``: type to generate (inferred from the message
          when omitted).
        - ``"perf_config"``: dict for performance script generation.
        - ``"script_format"``: ``"k6" | "jmeter"`` for performance scripts.

        Returns the assistant's response text.
        """
        ctx = context or {}
        self._update_context(ctx)
        self._load_seed_artifacts(ctx)
        self._iteration = 0

        self._add_message("user", user_message)
        action = self._determine_action(user_message, ctx)

        active_status = {
            "generate": "generating",
            "refine": "refining",
            "validate": "validating",
            "chat": "idle",
        }[action]
        self._status = active_status

        try:
            if action == "generate":
                response = self._handle_generate(user_message, ctx)
            elif action == "refine":
                response = self._handle_refine(user_message, ctx)
            elif action == "validate":
                response = self._handle_validate(user_message, ctx)
            else:
                response = self._handle_chat(user_message, ctx)
        finally:
            self._status = "idle" if action == "chat" else "completed"

        self._add_message(
            "assistant",
            response,
            {"action": action, "iteration": self._iteration},
        )
        self._persist()
        return response

    def _persist(self) -> None:
        """Invoke the persistence hook (if any) to flush the current state."""
        if self._persist_callback is not None:
            self._persist_callback(self)

    def get_state(self) -> ConversationState:
        """Return a snapshot of the current session state."""
        return ConversationState(
            session_id=self._session_id,
            messages=list(self._messages),
            artifacts=dict(self._artifacts),
            feedback=self._feedback,
            iteration=self._iteration,
            status=self._status,
        )

    def get_artifacts(self) -> list[Artifact]:
        """Return all artifacts in this session (insertion order)."""
        return list(self._artifacts.values())

    def get_latest_artifact(self, artifact_type: str | None = None) -> Artifact | None:
        """Return the latest artifact, optionally filtered by type.

        "Latest" is the most recently inserted matching artifact, which is
        also the highest version within its refinement chain.
        """
        matching = [
            a for a in self._artifacts.values() if artifact_type is None or a.type == artifact_type
        ]
        return matching[-1] if matching else None

    def get_history(self) -> list[ConversationMessage]:
        """Return the conversation history (copy)."""
        return list(self._messages)

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _determine_action(self, user_message: str, ctx: dict[str, Any]) -> str:
        action = str(ctx.get("action", "")).strip().lower()
        if action in VALID_ACTIONS:
            return action
        return self._infer_action(user_message)

    @staticmethod
    def _infer_action(user_message: str) -> str:
        text = user_message.lower()
        # Refine takes priority: explicit improvement intent on existing work.
        refine_kw = (
            "refine",
            "improve",
            "fix",
            "update",
            "modify",
            "adjust",
            "优化",
            "改进",
            "修改",
            "调整",
        )
        generate_kw = (
            "generate",
            "create",
            "produce",
            "生成",
            "创建",
        )
        validate_kw = (
            "validate",
            "check",
            "review",
            "verify",
            "验证",
            "检查",
            "审查",
        )
        for kw in refine_kw:
            if kw in text:
                return "refine"
        for kw in generate_kw:
            if kw in text:
                return "generate"
        for kw in validate_kw:
            if kw in text:
                return "validate"
        return "chat"

    @staticmethod
    def _infer_artifact_type(user_message: str) -> str:
        text = user_message.lower()
        if any(k in text for k in ("perf", "performance", "load", "压力", "性能", "负载")):
            return "performance_script"
        if any(k in text for k in ("gui", "ui", "界面", "前端")):
            return "gui_script"
        return "test_cases"

    # ------------------------------------------------------------------
    # Context & seeding
    # ------------------------------------------------------------------

    def _update_context(self, ctx: dict[str, Any]) -> None:
        if "endpoints" in ctx:
            self._endpoints_text = str(ctx["endpoints"] or "")
        if "requirements" in ctx:
            self._requirements_text = str(ctx["requirements"] or "")

    def _load_seed_artifacts(self, ctx: dict[str, Any]) -> None:
        seeds = ctx.get("artifacts")
        if not isinstance(seeds, list):
            return
        for item in seeds:
            if not isinstance(item, dict):
                continue
            raw_parent = item.get("parent_id")
            artifact = Artifact(
                id=str(item.get("id") or _new_artifact_id()),
                type=str(item.get("type") or "code"),
                content=str(item.get("content") or ""),
                version=int(item.get("version") or 1),
                created_at=str(item.get("created_at") or _now_iso()),
                parent_id=str(raw_parent) if raw_parent else None,
            )
            self._artifacts[artifact.id] = artifact
            logger.debug(
                "Loaded seed artifact %s (%s v%d)",
                artifact.id,
                artifact.type,
                artifact.version,
            )

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    def _handle_generate(self, user_message: str, ctx: dict[str, Any]) -> str:
        artifact_type = str(ctx.get("artifact_type") or "").strip().lower()
        if not artifact_type:
            artifact_type = self._infer_artifact_type(user_message)

        system_prompt, user_prompt = self._build_generate_prompt(artifact_type, user_message, ctx)
        raw = self._llm.chat(system_prompt, user_prompt)
        content = self._extract_artifact_content(raw, artifact_type)

        artifact = Artifact(
            id=_new_artifact_id(),
            type=artifact_type,
            content=content,
            version=1,
            created_at=_now_iso(),
            parent_id=None,
        )
        self._artifacts[artifact.id] = artifact

        feedback = self._validate_artifact(artifact)
        self._feedback = feedback

        summary = self._summarize_artifact(artifact)
        summary += "\n\n" + self._validation_footer(feedback)
        return summary

    def _build_generate_prompt(
        self, artifact_type: str, user_message: str, ctx: dict[str, Any]
    ) -> tuple[str, str]:
        if artifact_type == "test_cases":
            return self._prompt_builder.build_testcase_prompt(
                endpoints_text=self._endpoints_text,
                requirements_text=self._requirements_text or user_message,
                output_language=self._output_language,
            )
        if artifact_type == "performance_script":
            script_format = str(ctx.get("script_format") or "k6")
            perf_config = ctx.get("perf_config") or {}
            if not isinstance(perf_config, dict):
                perf_config = {}
            return self._prompt_builder.build_performance_prompt(
                endpoints_text=self._endpoints_text,
                config=perf_config,
                script_format=script_format,
                output_language=self._output_language,
            )
        if artifact_type == "gui_script":
            # GUI scripts are Playwright Python and must target a real URL.
            # The caller supplies it via context ``gui_url``; fall back to the
            # generator's default when absent so interactive chat still works.
            url = str(ctx.get("gui_url") or "").strip() or DEFAULT_TARGET_URL
            requirements_text = self._requirements_text or user_message
            return self._prompt_builder.build_gui_test_prompt(
                url=url,
                requirements_text=requirements_text,
                endpoints_text=self._endpoints_text,
                output_language=self._output_language,
            )
        # code (and any other type): generic inline prompt.
        system_prompt = (
            "You are a senior test automation engineer. Generate the requested artifact."
        )
        user_prompt = (
            f"## Request\n{user_message}\n\n"
            f"## Context\n"
            f"Endpoints: {self._endpoints_text or 'N/A'}\n"
            f"Requirements: {self._requirements_text or 'N/A'}\n\n"
            f"Output ONLY the complete {artifact_type} content. No markdown fences."
        )
        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # refine
    # ------------------------------------------------------------------

    def _handle_refine(self, user_message: str, ctx: dict[str, Any]) -> str:
        artifact_type = str(ctx.get("artifact_type") or "").strip().lower() or None
        current = self.get_latest_artifact(artifact_type)
        if current is None:
            logger.info("Refine requested but no artifact found; generating instead.")
            return self._handle_generate(user_message, ctx)

        for i in range(1, self._max_iterations + 1):
            self._iteration = i
            system_prompt, user_prompt = self._build_refine_prompt(
                user_message=user_message,
                feedback=self._feedback,
                artifact=current,
            )
            raw = self._llm.chat(system_prompt, user_prompt)
            content = self._extract_artifact_content(raw, current.type)

            refined = Artifact(
                id=_new_artifact_id(),
                type=current.type,
                content=content,
                version=current.version + 1,
                created_at=_now_iso(),
                parent_id=current.id,
            )
            self._artifacts[refined.id] = refined
            current = refined

            feedback = self._validate_artifact(refined)
            self._feedback = feedback
            logger.info(
                "Refine iteration %d/%d: feedback=%d chars",
                i,
                self._max_iterations,
                len(feedback),
            )
            if not feedback:
                break

        summary = self._summarize_artifact(current)
        summary += f"\n\nRefined over {self._iteration} iteration(s)."
        summary += "\n\n" + self._validation_footer(self._feedback)
        return summary

    def _build_refine_prompt(
        self, user_message: str, feedback: str, artifact: Artifact
    ) -> tuple[str, str]:
        rendered = self._prompt_builder.render_template(
            "conversation_refine.j2",
            user_message=user_message,
            feedback=feedback or "(none)",
            artifact_version=artifact.version,
            artifact_content=artifact.content,
            artifact_type=artifact.type,
        )
        if rendered:
            system_prompt = (
                "You are a senior QA engineer refining a previously generated artifact. "
                "Output only the refined artifact as requested."
            )
            return system_prompt, rendered
        return self._build_inline_refine_prompt(user_message, feedback, artifact)

    def _build_inline_refine_prompt(
        self, user_message: str, feedback: str, artifact: Artifact
    ) -> tuple[str, str]:
        if artifact.type == "test_cases":
            type_hint = "Output ONLY a valid JSON array of test cases."
        elif artifact.type == "performance_script":
            type_hint = "Output ONLY the complete script code."
        else:
            type_hint = "Output the complete refined content."
        system_prompt = "You are a senior QA engineer refining a previously generated artifact."
        user_prompt = (
            f"## User Request\n{user_message}\n\n"
            f"## Previous Feedback\n{feedback or '(none)'}\n\n"
            f"## Current Artifact (version {artifact.version})\n{artifact.content}\n\n"
            f"## Instructions\n"
            f"Apply the user's requested changes. Preserve what works, fix what's "
            f"broken, add what's missing. Output the complete refined artifact "
            f"(not just the diff).\n\n"
            f"{type_hint}"
        )
        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # validate
    # ------------------------------------------------------------------

    def _handle_validate(self, user_message: str, ctx: dict[str, Any]) -> str:
        artifact_type = str(ctx.get("artifact_type") or "").strip().lower() or None
        latest = self.get_latest_artifact(artifact_type)
        if latest is None:
            return "No artifact available to validate. Generate one first."

        programmatic = self._validate_artifact(latest)
        llm_feedback = self._llm_validate(latest)

        parts: list[str] = []
        if programmatic:
            parts.append(f"## Programmatic checks\n{programmatic}")
        if llm_feedback:
            parts.append(f"## LLM review\n{llm_feedback}")
        feedback = "\n\n".join(parts) if parts else "PASS: No issues found. Artifact looks good."

        # Programmatic issues feed back into the refine loop.
        self._feedback = programmatic
        return feedback

    def _llm_validate(self, artifact: Artifact) -> str:
        rendered = self._prompt_builder.render_template(
            "conversation_validate.j2",
            artifact_type=artifact.type,
            artifact_content=artifact.content,
        )
        if rendered:
            system_prompt = "You are a QA validator. Output ONLY valid JSON."
            user_prompt = rendered
        else:
            system_prompt, user_prompt = self._build_inline_validate_prompt(artifact)

        raw = self._llm.chat(system_prompt, user_prompt)
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            return raw.strip()

        passed = parsed.get("passed")
        issues = parsed.get("issues") or []
        suggestions = parsed.get("suggestions") or []
        lines: list[str] = []
        lines.append(f"passed: {'true' if passed else 'false'}")
        if issues:
            lines.append("issues:")
            for it in issues:
                lines.append(f"  - {it}")
        if suggestions:
            lines.append("suggestions:")
            for sg in suggestions:
                lines.append(f"  - {sg}")
        return "\n".join(lines)

    def _build_inline_validate_prompt(self, artifact: Artifact) -> tuple[str, str]:
        if artifact.type == "test_cases":
            checklist = (
                "1. Is the JSON valid and parseable?\n"
                "2. Are all required fields present (id, title, steps, expected_results)?\n"
                "3. Are expected_results machine-checkable (status codes, field checks)?\n"
                "4. Do endpoints match the API spec?\n"
                "5. Is there adequate coverage (happy path, boundary, negative, security)?"
            )
        elif artifact.type == "performance_script":
            checklist = (
                "1. Is the script syntactically valid?\n"
                "2. Are all endpoints from the spec included?\n"
                "3. Are load parameters (VU, duration, ramp-up) correct?\n"
                "4. Are assertions/checks included?"
            )
        else:
            checklist = "1. Is the content syntactically valid?\n2. Does it meet the request?"
        system_prompt = "You are a QA validator. Output ONLY valid JSON."
        user_prompt = (
            f"## Artifact Type\n{artifact.type}\n\n"
            f"## Artifact Content\n{artifact.content}\n\n"
            f"## Validation Checklist\n{checklist}\n\n"
            f"## Output\n"
            f"Return a JSON object: "
            f'{{"passed": true/false, "issues": ["..."], "suggestions": ["..."]}}\n'
            f"Output ONLY the JSON. No markdown."
        )
        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------

    def _handle_chat(self, user_message: str, ctx: dict[str, Any]) -> str:
        del ctx  # chat does not consume extra context beyond history/artifacts.
        recent = "\n".join(f"{m.role}: {m.content[:500]}" for m in self._messages[-6:])
        system_prompt = (
            "You are a helpful QA assistant discussing generated test artifacts. "
            "Answer the user's question concisely using the conversation context."
        )
        user_prompt = (
            f"## Conversation so far\n{recent or '(start of conversation)'}\n\n"
            f"{self._artifacts_summary()}"
            f"## User question\n{user_message}\n\n"
            "Answer the question. If it asks for changes, briefly explain what "
            "should be done and suggest using the 'refine' action."
        )
        return self._llm.chat(system_prompt, user_prompt)

    # ------------------------------------------------------------------
    # Validation (programmatic)
    # ------------------------------------------------------------------

    def _validate_artifact(self, artifact: Artifact) -> str:
        """Run programmatic validation checks and return a feedback string.

        Returns an empty string when no issues are found. The checks cover:
        JSON parseability (test cases), script syntax (scripts/code),
        required-field completeness, endpoint consistency and whether
        expected_results are machine-checkable.
        """
        issues: list[str] = []
        if artifact.type == "test_cases":
            self._validate_test_cases(artifact, issues)
        elif artifact.type == "performance_script":
            self._validate_script(artifact, issues, script_kind="performance")
        elif artifact.type == "gui_script":
            self._validate_script(artifact, issues, script_kind="gui")
        elif artifact.type == "code":
            self._validate_code(artifact, issues)
        else:
            issues.append(f"Unknown artifact type '{artifact.type}'.")
        return "\n".join(issues)

    def _validate_test_cases(self, artifact: Artifact, issues: list[str]) -> None:
        parsed = _extract_json(artifact.content)
        if parsed is None:
            issues.append("Test cases content is not valid JSON.")
            return
        if not isinstance(parsed, list):
            issues.append("Test cases content is not a JSON array.")
            return
        if not parsed:
            issues.append("Test cases array is empty.")
            return

        endpoint_set = self._known_endpoints()
        for idx, item in enumerate(parsed, 1):
            if not isinstance(item, dict):
                issues.append(f"Case #{idx}: not an object.")
                continue
            for fld in TEST_CASE_REQUIRED_FIELDS:
                val = item.get(fld)
                if val is None or val == "" or val == []:
                    issues.append(f"Case #{idx}: missing required field '{fld}'.")
            if endpoint_set:
                ep = str(item.get("endpoint", "")).strip()
                if ep and not self._endpoint_matches(ep, endpoint_set):
                    issues.append(f"Case #{idx}: endpoint '{ep}' not in API spec.")
            expected = item.get("expected_results") or []
            if not isinstance(expected, list):
                expected = []
            for j, exp in enumerate(expected, 1):
                if not self._is_machine_checkable(str(exp)):
                    issues.append(
                        f"Case #{idx}: expected_result #{j} may not be "
                        f"machine-checkable: '{str(exp)[:60]}'"
                    )

    def _validate_script(self, artifact: Artifact, issues: list[str], script_kind: str) -> None:
        content = _strip_code_fences(artifact.content)
        if not content.strip():
            issues.append("Script content is empty.")
            return
        # GUI scripts are Python (Playwright) — verify real syntax, which is
        # stricter than bracket balancing and catches malformed statements.
        if script_kind == "gui":
            try:
                ast.parse(content)
            except SyntaxError as exc:
                lineno = exc.lineno or "?"
                issues.append(f"Python syntax error: {exc.msg} (line {lineno}).")
            return
        # JMeter JMX is XML; everything else is treated as JS-like.
        if content.lstrip().startswith("<?xml") or "<jmeterTestPlan" in content:
            self._validate_jmx(content, issues)
            return
        self._check_bracket_balance(content, issues)
        if script_kind == "performance":
            low = content.lower()
            if "import" not in low and "http" not in low and "requests" not in low:
                issues.append("Performance script may be missing HTTP/import statements.")

    @staticmethod
    def _validate_jmx(content: str, issues: list[str]) -> None:
        if not content.lstrip().startswith("<?xml"):
            issues.append("JMX missing XML declaration.")
        if "jmeterTestPlan" not in content:
            issues.append("JMX missing <jmeterTestPlan> root element.")
            return
        try:
            root = ET.fromstring(content)
            if root.tag != "jmeterTestPlan":
                issues.append(f"JMX root tag is '{root.tag}', expected 'jmeterTestPlan'.")
        except ET.ParseError as exc:
            issues.append(f"JMX XML parse error: {exc}.")

    @staticmethod
    def _validate_code(artifact: Artifact, issues: list[str]) -> None:
        content = _strip_code_fences(artifact.content)
        if not content.strip():
            issues.append("Code content is empty.")
            return
        try:
            ast.parse(content)
        except SyntaxError as exc:
            lineno = exc.lineno or "?"
            issues.append(f"Python syntax error: {exc.msg} (line {lineno}).")

    @staticmethod
    def _check_bracket_balance(content: str, issues: list[str]) -> None:
        pairs = {"(": ")", "[": "]", "{": "}"}
        closers = set(pairs.values())
        stack: list[str] = []
        in_string = False
        quote_char = ""
        escaped = False
        for ch in content:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if in_string:
                if ch == quote_char:
                    in_string = False
                continue
            if ch in ('"', "'"):
                in_string = True
                quote_char = ch
                continue
            if ch in pairs:
                stack.append(ch)
            elif ch in closers:
                if not stack:
                    issues.append(f"Unbalanced closing bracket '{ch}'.")
                    return
                opener = stack.pop()
                if pairs[opener] != ch:
                    issues.append(f"Mismatched bracket: '{opener}' closed by '{ch}'.")
                    return
        if stack:
            issues.append(f"Unbalanced opening bracket(s): {''.join(stack)}.")

    def _known_endpoints(self) -> set[str]:
        if not self._endpoints_text:
            return set()
        endpoint_set: set[str] = set()
        for match in _ENDPOINT_RE.findall(self._endpoints_text):
            endpoint_set.add(match)
            parts = match.split(None, 1)
            if len(parts) == 2:
                endpoint_set.add(f"{parts[0].upper()} {parts[1]}")
        return endpoint_set

    @staticmethod
    def _endpoint_matches(ep: str, endpoint_set: set[str]) -> bool:
        if ep in endpoint_set:
            return True
        parts = ep.split(None, 1)
        if len(parts) == 2:
            return f"{parts[0].upper()} {parts[1]}" in endpoint_set
        return False

    @staticmethod
    def _is_machine_checkable(text: str) -> bool:
        if any(c.isdigit() for c in text):
            return True
        low = text.lower()
        return any(kw in low for kw in ASSERTION_KEYWORDS)

    # ------------------------------------------------------------------
    # Extraction & formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_artifact_content(raw: str, artifact_type: str) -> str:
        if artifact_type == "test_cases":
            parsed = _extract_json(raw)
            if parsed is not None:
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            return raw.strip()
        return _strip_code_fences(raw)

    @staticmethod
    def _summarize_artifact(artifact: Artifact) -> str:
        size = len(artifact.content)
        preview = artifact.content[:300]
        if len(artifact.content) > 300:
            preview += "... [truncated]"
        return (
            f"Artifact {artifact.id} (type={artifact.type}, "
            f"version={artifact.version}, size={size} chars):\n{preview}"
        )

    def _artifacts_summary(self) -> str:
        if not self._artifacts:
            return "## Artifacts\n(none)\n\n"
        lines = ["## Artifacts"]
        for art in self._artifacts.values():
            lines.append(f"- {art.id}: {art.type} v{art.version} ({len(art.content)} chars)")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _validation_footer(feedback: str) -> str:
        if feedback:
            return f"ISSUES:\n{feedback}"
        return "PASS: Validation passed."

    def _add_message(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        self._messages.append(
            ConversationMessage(
                role=role,
                content=content,
                timestamp=_now_iso(),
                metadata=metadata or {},
            )
        )


# ----------------------------------------------------------------------
# Manager
# ----------------------------------------------------------------------


class ConversationManager:
    """Manages multiple conversation sessions.

    Sessions are keyed by ``session_id`` (analogous to langgraph's
    ``thread_id`` in a checkpointer). State persists in memory for the
    lifetime of the manager.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        output_language: str = "english",
        store: SessionStore | None = None,
    ) -> None:
        self._llm = llm_client
        self._prompt_builder = prompt_builder
        self._max_iterations = max(1, max_iterations)
        self._output_language = output_language
        self._sessions: dict[str, ConversationSession] = {}
        # Persistence backend. Defaults to in-memory (process-local) so unit
        # tests that build a manager directly keep the old non-persistent
        # behavior; production wiring (Container) injects a FileStore so
        # sessions survive restarts.
        self._store: SessionStore = store or InMemoryStore()

    def _new_session(self, sid: str) -> ConversationSession:
        """Build a session wired to flush its state to the store on each send."""
        return ConversationSession(
            session_id=sid,
            llm_client=self._llm,
            prompt_builder=self._prompt_builder,
            max_iterations=self._max_iterations,
            output_language=self._output_language,
            persist_callback=lambda s: self._store.save(s.session_id, s.to_snapshot()),
        )

    def create_session(self, session_id: str | None = None) -> ConversationSession:
        """Create a new conversation session.

        Args:
            session_id: Optional explicit id; a short random id is generated
                when omitted. Raises ``ValueError`` if the id already exists.
        """
        sid = session_id or _new_artifact_id()
        if sid in self._sessions or self._store.load(sid) is not None:
            raise ValueError(f"Session '{sid}' already exists.")
        session = self._new_session(sid)
        self._sessions[sid] = session
        # Persist the initial (empty) snapshot so the session is resumable
        # even before the first message.
        self._store.save(sid, session.to_snapshot())
        logger.info("Created conversation session '%s'", sid)
        return session

    def get_session(self, session_id: str) -> ConversationSession | None:
        """Get an existing session by id (or ``None``).

        Hits the in-memory cache first; on miss, attempts to restore the
        session from the persistent store (so a session created in a previous
        process run can be resumed). Returns ``None`` when neither has it.
        """
        cached = self._sessions.get(session_id)
        if cached is not None:
            return cached
        snapshot = self._store.load(session_id)
        if snapshot is None:
            return None
        session = self._new_session(session_id)
        session._apply_snapshot(snapshot)
        self._sessions[session_id] = session
        logger.info("Restored conversation session '%s' from store", session_id)
        return session

    def list_sessions(self) -> list[str]:
        """Return all known session ids (in-memory union persistent store)."""
        return list(set(self._sessions.keys()) | set(self._store.list_ids()))

    def close_session(self, session_id: str) -> None:
        """Close and remove a session (no-op if unknown, with a warning).

        Drops the session from both the in-memory cache and the persistent
        store so it cannot be resumed afterwards.
        """
        in_memory = self._sessions.pop(session_id, None)
        in_store = self._store.load(session_id) is not None
        self._store.delete(session_id)
        if in_memory is None and not in_store:
            logger.warning("Cannot close unknown session '%s'", session_id)
        else:
            logger.info("Closed conversation session '%s'", session_id)
