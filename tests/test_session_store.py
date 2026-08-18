"""Tests for conversation session persistence (SessionStore + manager wiring).

Covers the three backends (InMemoryStore / FileStore / create_session_store),
session-id validation, and the ConversationManager <-> store integration that
lets a session survive a process restart (the checkpointer-style resume path).
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from testagent.engine.conversation import ConversationManager
from testagent.engine.prompt_builder import PromptBuilder
from testagent.engine.session_store import (
    FileStore,
    InMemoryStore,
    _validate_session_id,
    create_session_store,
)

# A minimal valid test-case array the mock LLM returns for "generate" actions.
_TC_JSON = '[{"id":"TC-001","title":"x","steps":["s"],"expected_results":["Status 200"]}]'


class TestInMemoryStore:
    def test_save_load_roundtrip(self) -> None:
        s = InMemoryStore()
        assert s.load("x") is None
        s.save("x", {"a": 1})
        assert s.load("x") == {"a": 1}

    def test_delete_absent_is_noop(self) -> None:
        s = InMemoryStore()
        s.delete("missing")  # must not raise

    def test_list_ids(self) -> None:
        s = InMemoryStore()
        assert s.list_ids() == []
        s.save("a", {})
        s.save("b", {})
        assert set(s.list_ids()) == {"a", "b"}


class TestFileStore:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        s = FileStore(tmp_path)
        assert s.load("sess1") is None
        s.save("sess1", {"messages": [], "feedback": ""})
        assert s.load("sess1") == {"messages": [], "feedback": ""}
        assert (tmp_path / "sess1.json").exists()

    def test_load_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        s = FileStore(tmp_path)
        (tmp_path / "bad.json").write_text("not json{", encoding="utf-8")
        assert s.load("bad") is None

    def test_load_non_object_returns_none(self, tmp_path: Path) -> None:
        s = FileStore(tmp_path)
        (tmp_path / "arr.json").write_text("[1,2,3]", encoding="utf-8")
        assert s.load("arr") is None

    def test_delete(self, tmp_path: Path) -> None:
        s = FileStore(tmp_path)
        s.save("x", {"a": 1})
        s.delete("x")
        assert s.load("x") is None
        s.delete("y")  # absent, no-op

    def test_list_ids(self, tmp_path: Path) -> None:
        s = FileStore(tmp_path)
        assert s.list_ids() == []
        s.save("a", {})
        s.save("b", {})
        assert set(s.list_ids()) == {"a", "b"}

    def test_invalid_session_id_rejected(self, tmp_path: Path) -> None:
        s = FileStore(tmp_path)
        with pytest.raises(ValueError):
            s.save("../etc/passwd", {})
        with pytest.raises(ValueError):
            s.load("a/b")

    def test_creates_nested_base_dir(self, tmp_path: Path) -> None:
        s = FileStore(tmp_path / "nested" / "deep")
        s.save("x", {"a": 1})
        assert (tmp_path / "nested" / "deep" / "x.json").exists()


class TestCreateSessionStore:
    def test_file_backend_lands_under_conversations(self, tmp_path: Path) -> None:
        s = create_session_store("file", str(tmp_path))
        assert isinstance(s, FileStore)
        s.save("x", {})
        assert (tmp_path / "conversations" / "x.json").exists()

    def test_memory_backend(self) -> None:
        assert isinstance(create_session_store("memory"), InMemoryStore)

    def test_unknown_falls_back_to_memory(self) -> None:
        assert isinstance(create_session_store("redis", "output"), InMemoryStore)


class TestValidateSessionId:
    def test_valid(self) -> None:
        _validate_session_id("abc123-_")
        _validate_session_id("a" * 12)

    @pytest.mark.parametrize("bad", ["", "a/b", "a\\b", "..", "a.b", "a b", "a:b"])
    def test_invalid(self, bad: str) -> None:
        with pytest.raises(ValueError):
            _validate_session_id(bad)


class TestConversationPersistence:
    """ConversationManager + store: sessions survive a restart."""

    @staticmethod
    def _manager(store: object, llm: MagicMock) -> ConversationManager:
        return ConversationManager(
            llm_client=llm,
            prompt_builder=PromptBuilder(),
            store=store,  # type: ignore[arg-type]
        )

    def test_send_persists_to_store(self, tmp_path: Path) -> None:
        llm = MagicMock()
        llm.chat.return_value = _TC_JSON
        store = FileStore(tmp_path)
        mgr = self._manager(store, llm)
        session = mgr.create_session("s1")
        session.send("generate test cases", context={"requirements": "do something"})
        snap = store.load("s1")
        assert snap is not None
        assert len(snap["messages"]) >= 2  # user + assistant
        assert snap["artifacts"]  # at least one artifact persisted
        assert snap["requirements_text"] == "do something"

    def test_session_resumes_across_manager_instances(self, tmp_path: Path) -> None:
        """A new manager reading the same store restores a prior session."""
        llm = MagicMock()
        llm.chat.return_value = _TC_JSON
        store = FileStore(tmp_path)
        mgr1 = self._manager(store, llm)
        s1 = mgr1.create_session("s1")
        s1.send("generate test cases", context={"requirements": "do something"})
        artifacts_before = s1.get_artifacts()
        assert artifacts_before

        # New manager, same store — simulates a process restart.
        mgr2 = self._manager(store, llm)
        s2 = mgr2.get_session("s1")
        assert s2 is not None
        assert s2 is not s1  # restored instance, not the cached one
        assert len(s2.get_artifacts()) == len(artifacts_before)
        assert s2.get_artifacts()[0].content == artifacts_before[0].content
        assert s2.get_history()  # messages restored

    def test_close_removes_from_store(self, tmp_path: Path) -> None:
        llm = MagicMock()
        store = FileStore(tmp_path)
        mgr = self._manager(store, llm)
        mgr.create_session("s1")
        assert store.load("s1") is not None
        mgr.close_session("s1")
        assert store.load("s1") is None

    def test_create_rejects_id_existing_in_store(self, tmp_path: Path) -> None:
        llm = MagicMock()
        store = FileStore(tmp_path)
        mgr1 = self._manager(store, llm)
        mgr1.create_session("s1")
        # A second manager has an empty in-memory cache but the store has "s1".
        mgr2 = self._manager(store, llm)
        with pytest.raises(ValueError):
            mgr2.create_session("s1")

    def test_list_sessions_unions_memory_and_store(self, tmp_path: Path) -> None:
        llm = MagicMock()
        store = FileStore(tmp_path)
        mgr1 = self._manager(store, llm)
        mgr1.create_session("a")
        mgr2 = self._manager(store, llm)  # empty memory; store has "a"
        mgr2.create_session("b")  # memory "b"; store now has "a"+"b"
        assert {"a", "b"} <= set(mgr2.list_sessions())

    def test_default_store_is_inmemory(self) -> None:
        """A manager built without an explicit store uses InMemoryStore."""
        mgr = ConversationManager(llm_client=MagicMock(), prompt_builder=PromptBuilder())
        assert isinstance(mgr._store, InMemoryStore)
