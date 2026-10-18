"""Whole synthetic 16-body workflow; numerical and upstream evidence mocked."""

import json

import pytest

from scripts import rsi_read_mesh_contact_native_pilot as audit
from tests.rsi.test_mesh_teacher_terminal_readback import expanded_fixture


def test_complete_cross_source_reader_keeps_all_pairs_and_cluster_split(tmp_path, monkeypatch):
    output = expanded_fixture(tmp_path, monkeypatch, positive_lateral=True)
    audit.main()
    report = json.loads(output.read_text())
    assert report["schema"] == "soccer.rsi.positive_lateral_physical_terminal_readback.v1"
    assert report["source_native_bodies"] == 16 and len(report["rows"]) == 16
    assert report["paired_development"]["pairs"] == 8
    assert report["distinct_derived_course_count"] == 8 and report["source_scene_clusters"] == 4
    assert report["generated_seed_binding_in_all_bodies"]
    assert not report["teacher_qualified"] and not report["neural_training_authorized"]
    assert not report["fresh_exam"] and report["new_physical_replays"] == 0
    assert len(report["prospective_split"]["train_source_seeds"]) == 3
    assert len(report["prospective_split"]["validation_source_seeds"]) == 1


@pytest.mark.parametrize(
    "tamper",
    [
        "positive_registry",
        "positive_binding",
        "planning_stage",
        "view_identity",
        "receipt_role",
        "ledger_declaration",
        "pairing",
        "positive_split",
        "positive_source_plan",
        "positive_previous",
        "positive_mixed_intervention",
    ],
)
def test_reader_rejects_resealed_cross_source_lineage_and_pairing_errors(
    tmp_path, monkeypatch, tamper
):
    output = expanded_fixture(tmp_path, monkeypatch, tamper, positive_lateral=True)
    with pytest.raises(ValueError):
        audit.main()
    assert not output.exists()
