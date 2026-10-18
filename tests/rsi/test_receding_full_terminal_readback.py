"""Synthetic whole reader test; NO physical/model/archived authentication evidence."""

import json

import pytest

from scripts import rsi_read_mesh_contact_native_pilot as audit
from tests.rsi.test_mesh_teacher_terminal_readback import expanded_fixture


def test_whole_receding_reader_retains_all_six_pairs_without_admission(tmp_path, monkeypatch):
    output = expanded_fixture(tmp_path, monkeypatch, receding=True)
    audit.main()
    result = json.loads(output.read_text())
    assert result["schema"] == "soccer.rsi.receding_catalog2_terminal_readback.v1"
    assert len(result["rows"]) == 10
    assert result["paired_development"]["pairs"] == 6
    assert result["reused_endpoint_count"] == 2
    assert result["archived_c7_full_numeric_reproduction"] is True
    assert result["single_high_x_receding_stratum_only"] is True
    assert not result["teacher_qualified"]
    assert not result["fresh_exam"]


@pytest.mark.parametrize(
    "tamper", ["planning_stage", "view_identity", "receipt_role", "reproduction", "pairing"]
)
def test_receding_reader_rejects_resealed_clock_role_or_reproduction_tampering(
    tmp_path, monkeypatch, tamper
):
    output = expanded_fixture(tmp_path, monkeypatch, tamper, receding=True)
    with pytest.raises(ValueError):
        audit.main()
    assert not output.exists()
