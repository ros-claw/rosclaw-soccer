import json

import pytest
from test_paid_fresh_protocol import source_cases

from rosclaw_soccer.rsi.paid_first_bank import (
    SCHEMA,
    STATUS,
    load_motor_exam_plan,
    original_case_with_measured_velocity,
)
from scripts.rsi_native_context_recovery_common import digest


def fixture():
    cases = source_cases()
    for c in cases:
        c["velocity_world_initial"] = [0.2, 0.0]
    d = dict(
        schema=SCHEMA,
        bank="dev",
        cases=cases,
        cases_predeclared=16,
        source_head="frozen",
        body_scene_hash="body",
        hardware_authorized=False,
        promotion_authorized=False,
    )
    r = dict(
        status=STATUS,
        bank="dev",
        source_head="frozen",
        body_scene_hash="body",
        cases_preflighted=16,
        own_native_steps=0,
        own_neural_rows=0,
        original_seven_entry_fields_exact=True,
        explicit_velocity_preserves_full_native_entry=True,
        entries=[
            dict(case=i, position=c["ball_world_initial"], velocity=c["velocity_world_initial"])
            for i, c in enumerate(cases)
        ],
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return d, r


def publish(tmp_path, d, r):
    dp = tmp_path / "declaration.json"
    dp.write_text(json.dumps(d))
    r["declaration_hash"] = digest(dp)
    rp = tmp_path / "complete.json"
    rp.write_text(json.dumps(r))
    return digest(rp)


def test_dev_fixture_has_own_complete_bank_identity_without_neural_execution(tmp_path):
    d, r = fixture()
    h = publish(tmp_path, d, r)
    rr, dd = load_motor_exam_plan(tmp_path, h)
    assert dd["bank"] == "dev" and len(dd["cases"]) == 16 and rr["own_neural_rows"] == 0


def test_legacy_dev_metadata_defaults_are_derived_from_measured_compiled_ball():
    c = source_cases()[0]
    c.pop("profile")
    c.pop("ball_mass_kg")
    out = original_case_with_measured_velocity(c, 0, [0.19, -0.02], 0.43)
    assert out["profile"] == "cached-standard" and out["ball_mass_kg"] == 0.43
    assert out["velocity_world_initial"] == [0.19, -0.02] and "profile" not in c
    with pytest.raises(ValueError):
        original_case_with_measured_velocity(c, 0, [0.19, -0.02], 0.41)


def test_local_frame_dev_case_is_explicitly_projected_from_actual_native_entry():
    source = dict(
        source="soccer-standard-001_right.npz", ball_local=[0.65, -0.2], velocity_local=[0, 0]
    )
    out = original_case_with_measured_velocity(
        source,
        0,
        [0.0, 0.0],
        0.43,
        measured_ball=[-0.2, -0.65, 0.11],
        measured_goal=[0.0, -6.0, 0.11],
    )
    assert "ball_local" not in out and out["ball_world_initial"] == [-0.2, -0.65, 0.11]
    assert out["goal_world"] == [0.0, -6.0, 0.11] and "ball_world_initial" not in source
    with pytest.raises(ValueError):
        original_case_with_measured_velocity(
            dict(source, ball_world_initial=[1.0, 0.0, 0.11]),
            0,
            [0.0, 0.0],
            0.43,
            measured_ball=[-0.2, -0.65, 0.11],
            measured_goal=[0.0, -6.0, 0.11],
        )


@pytest.mark.parametrize(
    "fault", ["missing", "entry", "bank", "body", "steps", "preservation", "velocity", "radius"]
)
def test_partial_or_changed_original_fixture_is_rejected(tmp_path, fault):
    d, r = fixture()
    if fault == "missing":
        d["cases"].pop()
    elif fault == "entry":
        r["entries"][0]["velocity"] = [0.3, 0.0]
    elif fault == "bank":
        d["bank"] = "fresh"
    elif fault == "body":
        d["body_scene_hash"] = "another"
    elif fault == "steps":
        r["own_native_steps"] = 10
    elif fault == "preservation":
        r["explicit_velocity_preserves_full_native_entry"] = False
    elif fault == "velocity":
        d["cases"][0]["velocity_world_initial"] = [float("inf"), 0]
    else:
        d["cases"][0]["ball_world_initial"][2] = 0.115
    h = publish(tmp_path, d, r)
    with pytest.raises(ValueError):
        load_motor_exam_plan(tmp_path, h)
