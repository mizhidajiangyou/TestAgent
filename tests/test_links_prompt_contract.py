"""S6a tests: prompt contract constants + schema additions material."""

import json

from testagent.pipeline.links_prompt_contract import (
    BINDS_V2_CONTRACT,
    PATH_CONTRACT_INSTRUCTION,
    render_schema_additions_json,
)


class TestS6aMaterial:
    def test_binds_contract_covers_key_rules(self) -> None:
        assert "[A-Z][A-Z0-9_]{0,30}" in BINDS_V2_CONTRACT
        assert "producer" in BINDS_V2_CONTRACT
        assert "consumer" in BINDS_V2_CONTRACT
        assert "verbatim" in BINDS_V2_CONTRACT

    def test_path_contract_forbids_identity_self_report(self) -> None:
        assert "Keep `path_id` out of your output" in PATH_CONTRACT_INSTRUCTION

    def test_schema_additions_valid_json_and_keys(self) -> None:
        parsed = json.loads(render_schema_additions_json())
        assert set(parsed) == {"path_id", "source_stage", "binds", "executability"}
        # binds value schema: only producer/consumer/reason allowed
        binds_props = parsed["binds"]["additionalProperties"]["properties"]
        assert set(binds_props) == {"producer", "consumer", "reason"}
        assert parsed["binds"]["additionalProperties"]["required"] == ["producer"]
        # program-stamped fields default empty (model never owns them)
        assert parsed["path_id"]["default"] == ""
        assert parsed["source_stage"]["default"] == ""

    def test_additions_are_additive_not_required(self) -> None:
        """binds/path_id/executability must NOT be in a required set (v15
        §6.1: code identity fields are not demanded from the LLM)."""
        additions = json.loads(render_schema_additions_json())
        for field_spec in additions.values():
            assert "required" not in field_spec or not field_spec.get("required")
