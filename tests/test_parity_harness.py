"""FH-H self-test: the harness itself can record / replay / minimal-diff
for a sample testcase-shaped and a sample perf-shaped observation."""

import json
from pathlib import Path

import pytest

from tests.parity_harness import (
    RECORD_ENV,
    Fingerprint,
    Fixture,
    ensure_fixture,
    ensure_no_credentials,
    load_fixture,
    minimal_diff,
    observable_failure_class,
    record_fixture,
    replay_diff,
)


@pytest.fixture(autouse=True)
def _recording_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests exercise the writer, and always into a tmp FIXTURE_ROOT."""
    monkeypatch.setenv(RECORD_ENV, "1")


def _sample_testcase_observation() -> Fixture:
    return Fixture(
        name="selftest-testcase",
        task="_selftest",
        input={"requirements": "req.md", "swagger": "spec.json"},
        request_trace=[
            {
                "fingerprint": Fingerprint.of(
                    model="fake-model",
                    system_prompt="SYS",
                    user_prompt="USER 1",
                    params={"max_tokens": 16000},
                    label="Req REQ-001/1",
                ).to_dict(),
                "response_text": '[{"id": "TC-XXX", "title": "t"}]',
            }
        ],
        artifact=[{"id": "TC-001", "title": "t", "endpoint": "GET /users"}],
        failure_semantics={"units": [{"label": "Req REQ-001/1", "observable": "ok"}]},
        events=[{"round": 1, "event": "call"}, {"round": 1, "event": "done"}],
        meta={"scenario": "selftest"},
    )


def _sample_perf_observation() -> Fixture:
    return Fixture(
        name="selftest-perf",
        task="_selftest",
        input={"swagger": "spec.json", "script_format": "k6"},
        request_trace=[
            {
                "fingerprint": Fingerprint.of(
                    model="fake-model",
                    system_prompt="SYS-P",
                    user_prompt="USER-P",
                    params={},
                    label="perf",
                ).to_dict(),
                "response_text": "export default function() {}",
            }
        ],
        artifact="export default function() {}",
        failure_semantics={"units": [{"label": "perf", "observable": "ok"}]},
        events=[{"round": 1, "event": "call"}, {"round": 1, "event": "done"}],
        meta={"scenario": "selftest"},
    )


class TestHarnessSelfTest:
    def test_record_then_replay_diff_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tests.parity_harness.FIXTURE_ROOT", tmp_path)
        fixture = _sample_testcase_observation()
        record_fixture(fixture)
        loaded = load_fixture(fixture.task, fixture.name)
        assert minimal_diff(loaded, _sample_testcase_observation()) == {}

    def test_replay_diff_detects_artifact_change(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tests.parity_harness.FIXTURE_ROOT", tmp_path)
        record_fixture(_sample_testcase_observation())
        drifted = _sample_testcase_observation()
        drifted.artifact = [{"id": "TC-001", "title": "CHANGED"}]
        diff = replay_diff(load_fixture("_selftest", "selftest-testcase"), drifted)
        assert "artifact" in diff
        assert "request_trace" not in diff  # unchanged elements not reported

    def test_perf_observation_round_trip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tests.parity_harness.FIXTURE_ROOT", tmp_path)
        record_fixture(_sample_perf_observation())
        loaded = load_fixture("_selftest", "selftest-perf")
        assert minimal_diff(loaded, _sample_perf_observation()) == {}

    def test_no_credentials_guard(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("tests.parity_harness.FIXTURE_ROOT", tmp_path)
        bad = _sample_testcase_observation()
        bad.input = {"api_key": "sk-123"}  # type: ignore[assignment]
        with pytest.raises(AssertionError, match="credential"):
            record_fixture(bad)

    def test_credential_guard_ignores_prompt_vocabulary(self) -> None:
        """Env-var NAMES and auth schemes are ordinary prompt text (an error
        contract teaches the model to send ``Bearer <TOKEN>``); blocking them
        made a recorded HTTP contract impossible to store. The real risk —
        token shapes and the local key value — still fails loudly."""
        ensure_no_credentials({"p": "set OPENAI_API_KEY and send Bearer <TOKEN>"})
        for bad in (
            {"p": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz1234"},
            {"p": "sk-abcdefghijklmnop"},
        ):
            with pytest.raises(AssertionError, match=r"credential|bearer"):
                ensure_no_credentials(bad)

    def test_fingerprint_excludes_volatile_fields(self) -> None:
        """I2: same (system,user,params,label) -> same fingerprint regardless
        of session id / timestamp (they are not even inputs)."""
        a = Fingerprint.of(model="m", system_prompt="s", user_prompt="u", label="l")
        b = Fingerprint.of(model="m", system_prompt="s", user_prompt="u", label="l")
        assert a == b
        payload = json.dumps(a.to_dict())
        assert "session" not in payload and "timestamp" not in payload

    def test_observable_failure_mapping(self) -> None:
        assert observable_failure_class("SUCCESS") == "ok"
        assert observable_failure_class("EMPTY") == "empty"
        assert observable_failure_class("VALIDATION_ERROR") == "validation"
        assert observable_failure_class("TIMEOUT") == "timeout"
        assert observable_failure_class("PROVIDER_ERROR") == "provider_error"

    def test_json_serializable(self, tmp_path: Path) -> None:
        payload = json.loads(json.dumps(_sample_testcase_observation().to_dict()))
        assert payload["fixture_version"] == 1


class TestRecordingIsOptIn:
    """A default run must not be able to rewrite the baseline it is checked
    against — otherwise "replay diff=0" is a self-proof, not a gate."""

    def test_record_refused_without_opt_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tests.parity_harness.FIXTURE_ROOT", tmp_path)
        monkeypatch.delenv(RECORD_ENV, raising=False)
        with pytest.raises(RuntimeError, match=RECORD_ENV):
            record_fixture(_sample_testcase_observation())
        assert list(tmp_path.rglob("*.json")) == []

    def test_ensure_fixture_loads_baseline_without_building(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tests.parity_harness.FIXTURE_ROOT", tmp_path)
        record_fixture(_sample_testcase_observation())
        monkeypatch.delenv(RECORD_ENV, raising=False)
        built: list[str] = []

        def _build() -> Fixture:
            built.append("called")
            return _sample_testcase_observation()

        loaded = ensure_fixture("_selftest", "selftest-testcase", _build)
        assert built == []
        assert minimal_diff(loaded, _sample_testcase_observation()) == {}

    def test_ensure_fixture_records_under_opt_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tests.parity_harness.FIXTURE_ROOT", tmp_path)
        built: list[str] = []

        def _build() -> Fixture:
            built.append("called")
            return _sample_testcase_observation()

        ensure_fixture("_selftest", "selftest-testcase", _build)
        assert built == ["called"]
        assert (tmp_path / "_selftest" / "selftest-testcase.json").exists()

    def test_missing_baseline_fails_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tests.parity_harness.FIXTURE_ROOT", tmp_path)
        monkeypatch.delenv(RECORD_ENV, raising=False)
        with pytest.raises(AssertionError, match=RECORD_ENV):
            ensure_fixture("_selftest", "selftest-absent", _sample_testcase_observation)
