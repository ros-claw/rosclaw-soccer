"""Metadata/source guards; actual FK math has separate non-mocked tests."""

import ast
from pathlib import Path

import pytest

from scripts import rsi_diagnose_six_contact_heading as reader


def fixture(kind):
    return dict(
        schema="soccer.rsi.foot_heading_first_touch_stage_pairs_readback.v1"
        if kind == "foot-heading"
        else "soccer.rsi.original_ik_first_touch_stage_pairs_readback.v1",
        configuration=dict(stage_profile=kind, mode="SIM_ONLY"),
        all_six_receipts_and_failures_retained=True,
        source_native_bodies=6,
        derived_courses=3,
        source_scene_clusters=3,
        new_physical_steps=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        receiving_skill_qualified=False,
        successor_readiness_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )


@pytest.mark.parametrize("kind", ["foot-heading", "original-ik-pulse"])
@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "schema",
        "profile",
        "mode",
        "partial",
        "truthy",
        "newphysics",
        "inference",
        "updates",
        "teacher",
        "promotion",
    ],
)
def test_only_whole_closed_six_stage_can_reach_fk(kind, mutation):
    d = fixture(kind)
    if mutation == "schema":
        d["schema"] = "soccer.rsi.pulse_first_touch_stage_pairs_readback.v1"
    elif mutation == "profile":
        d["configuration"]["stage_profile"] = "zero-reference-pulse"
    elif mutation == "mode":
        d["configuration"]["mode"] = "REAL"
    elif mutation == "partial":
        d["source_native_bodies"] = 5
    elif mutation == "truthy":
        d["new_physical_steps"] = False
    elif mutation == "newphysics":
        d["new_physical_steps"] = 3000
    elif mutation == "inference":
        d["new_neural_inferences"] = 1
    elif mutation == "updates":
        d["new_optimizer_updates"] = 1
    elif mutation in ("teacher", "promotion"):
        d["teacher_qualified" if mutation == "teacher" else "promotion_authorized"] = True
    if mutation is None:
        assert reader.stage_kind(d) == kind
    else:
        with pytest.raises(ValueError):
            reader.stage_kind(d)


def test_geometry_reader_has_no_integration_controller_or_neural_call():
    tree = ast.parse(Path(reader.__file__).read_text())
    attrs = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not attrs.intersection(
        {"mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_Euler", "rollout", "InferenceSession"}
    )
    assert "from_binary_path" in attrs
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "reconstruct_contact_orientation" in calls and "initial_contact_heading" in calls
