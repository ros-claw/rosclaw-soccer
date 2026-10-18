import copy

import pytest

from rosclaw_soccer.rsi.planner_navigation_pulse_scope import (
    GROUPS,
    PROFILE,
    assess_pulse_scope,
    pulse_scope,
    require_pulse_scope,
)
from scripts.rsi_reference_student_exam import load_inputs

PARENT = "sha256:" + "a" * 64


def config():
    return dict(
        exam_profile=PROFILE,
        mode="SIM_ONLY",
        pulse_group_order=list(GROUPS),
        required_receipt_witnesses=["intent_witness", "encoder_witness", "prefix_witness"],
        max_attempts=6,
        max_generations=1,
        plateau_generations=1,
        new_physical_budget=6,
        new_independent_cpu_budget=6,
        combined_body_substep_budget=36000,
        new_optimizer_updates=0,
        **{
            k: False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "private_fresh_accessed",
                "neural_training_authorized",
                "teacher_auto_admission",
                "use_mesh_surface_swing",
                "suppress_postcontact_swing_release",
                "use_contact_exit_torque_mpc",
                "use_directional_contact_compliance",
            )
        },
    )


def test_fixed_three_training_sources_and_two_actions_not_a_search_sweep():
    require_pulse_scope(config())
    scope = pulse_scope(PARENT)
    assert scope["source_seeds"] == [20261078, 20261093, 20261077]
    assert len(scope["jobs"]) == 6
    assert [r["lateral_delta_m_s"] for r in scope["jobs"]] == [0.0, -0.08] * 3
    assert not scope["validation_or_fresh_sources_used"]
    assert not scope["trained_policy"]


@pytest.mark.parametrize("pulse_mode", [False, True, 1])
def test_six_body_loader_is_explicit_and_never_bypasses_source_validation(tmp_path, pulse_mode):
    import json

    cfg = dict(
        config(),
        source_root=str(tmp_path),
        core_root=str(tmp_path),
        output_root=str(tmp_path / "unused"),
        source_head="not-this-source",
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    expected = (
        "immutable" if pulse_mode is True else "explicit" if type(pulse_mode) is int else "profile"
    )
    with pytest.raises(ValueError, match=expected):
        load_inputs(path, pulse_research=pulse_mode)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_attempts", 7),
        ("new_independent_cpu_budget", 0),
        ("new_optimizer_updates", True),
        ("neural_training_authorized", True),
        ("pulse_group_order", [28, 76, 92]),
        ("combined_body_substep_budget", 36001),
    ],
)
def test_budget_or_validation_scope_cannot_widen(key, value):
    c = config()
    c[key] = value
    with pytest.raises(ValueError):
        require_pulse_scope(c)


@pytest.mark.parametrize("tamper", [None, "partial", "order", "delta", "duplicate"])
def test_whole_failed_and_unsafe_results_survive_without_a_success_claim(tamper):
    scope = pulse_scope(PARENT)
    rows = [
        dict(j, receipt_hash=str(i), outcome=dict(task_pass=False, safe=i != 1))
        for i, j in enumerate(scope["jobs"])
    ]
    original = copy.deepcopy(rows)
    if tamper == "partial":
        rows.pop()
    elif tamper == "order":
        rows.reverse()
    elif tamper == "delta":
        rows[1]["lateral_delta_m_s"] = -0.04
    elif tamper == "duplicate":
        rows[1]["receipt_hash"] = rows[0]["receipt_hash"]
    if tamper is None:
        report = assess_pulse_scope(scope, rows)
        assert report["paired_results"]["newly_unsafe_pairs"] == 1
        assert report["paired_results"]["net_success_gain"] == 0
        assert rows == original and not report["teacher_qualified"]
    else:
        with pytest.raises(ValueError):
            assess_pulse_scope(scope, rows)
