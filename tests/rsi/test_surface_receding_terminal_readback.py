"""Synthetic protocol tests; models, upstream provenance and physics are mocked."""

import json

import pytest

from rosclaw_soccer.rsi.surface_receding_evidence import surface_receding_jobs
from scripts import rsi_read_mesh_contact_native_pilot as audit
from tests.rsi.test_release_four_full_terminal_readback import fixture


def test_new_six_profile_keeps_exact_catalog2_courses_and_three_source_clusters(
    tmp_path, monkeypatch
):
    assert audit.bounded_profile("surface-receding-six")["jobs"] == surface_receding_jobs()
    output = fixture(tmp_path, monkeypatch, surface_receding=True)
    audit.main()
    report = json.loads(output.read_text())
    assert report["schema"] == "soccer.rsi.mesh_surface_receding_terminal_readback.v1"
    assert len(report["rows"]) == 6
    assert report["distinct_derived_course_count"] == 6
    assert report["source_scene_clusters"] == 3
    assert report["all_original_pre_activation_prefixes_bit_equal"]
    assert report["reused_original_none_endpoint_count"] == 6
    assert not report["teacher_qualified"] and not report["fresh_exam"]
    assert "independent_course_count" not in report


@pytest.mark.parametrize(
    "tamper", ["intent_baseline", "prefix_hash", "upstream_unverified", "surface_nonfoot_active"]
)
def test_material_point_receding_reader_rejects_resealed_lineage_or_capture_errors(
    tmp_path, monkeypatch, tamper
):
    output = fixture(tmp_path, monkeypatch, tamper, surface_receding=True)
    with pytest.raises(ValueError):
        audit.main()
    assert not output.exists()
