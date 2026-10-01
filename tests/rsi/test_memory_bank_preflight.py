import copy

import pytest

from scripts.rsi_collect_step_motor_pilot import COURSES
from scripts.rsi_preflight_memory_bank_validation import check_cpu, check_pilot


def pilot():
    summary = dict(
        schema="soccer.rsi.online_step_motor_physical_validation.v1",
        report_hash="summary",
        independent_contexts=4,
        physical_executions=12,
        old_high_quality_loss=0,
        old_clean_foot_loss=0,
        new_out_of_play=0,
        online_high_quality=3,
        baseline_high_quality=3,
        promotion_authorized=False,
        hardware_authorized=False,
        commitment=dict(
            model_hash="candidate",
            partition="CONSUMED_PILOT",
            courses=[list(v) for v in COURSES],
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    review = dict(
        schema="soccer.rsi.step_motor_physics_independent_review.v1",
        source_summary_hash="summary",
        actual_reports_reviewed=12,
        actual_motor_actions_reconstructed=2400,
        independent_contexts=4,
        safe_pelvis=True,
        old_high_quality_loss=0,
        old_clean_foot_loss=0,
        new_out_of_play=0,
        candidate_high_quality=3,
        baseline_high_quality=3,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return summary, review


def cpu():
    meta = dict(
        schema="soccer.rsi.cpu_motor_transfer.v1",
        partition="CONSUMED_TRANSFER_DIAGNOSTIC",
        seed=11,
        lane=0,
        step_model_hash="candidate",
        report_hash="report",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    proof = dict(
        schema="soccer.rsi.cpu_motor_transfer_review.v1",
        reviewed_report_hash="report",
        physical_substeps=3000,
        actual_mujoco_dynamics_replayed=True,
        actual_pd_torque_reconstructed=True,
        neural_target_reconstructed=True,
        safety_passed=True,
        minimum_pelvis_z_m=0.7,
        maximum_lateral_excursion_m=2.0,
        high_quality=True,
        clean_foot_only=True,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    old = copy.deepcopy(meta)
    old["step_model_hash"] = "parent"
    return meta, proof, old, copy.deepcopy(proof)


def test_retained_but_not_improved_pilot_allows_only_consumed_comparison():
    check_pilot(*pilot(), "candidate")


def test_current_parent_is_required_not_only_an_old_good_baseline():
    summary, review = pilot()
    summary["commitment"]["warm_model_hash"] = "older-parent"
    with pytest.raises(ValueError, match="consumed pilot"):
        check_pilot(summary, review, "candidate", baseline_hash="current-parent")
    summary["commitment"]["warm_model_hash"] = "current-parent"
    check_pilot(summary, review, "candidate", baseline_hash="current-parent")


def test_missing_parent_binding_is_rejected_by_current_parent_gate():
    with pytest.raises(ValueError, match="consumed pilot"):
        check_pilot(*pilot(), "candidate", baseline_hash="current-parent")


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_summary_hash", "stale"),
        ("actual_motor_actions_reconstructed", 2399),
        ("old_high_quality_loss", 1),
        ("old_clean_foot_loss", 1),
        ("new_out_of_play", 1),
        ("safe_pelvis", False),
        ("promotion_authorized", True),
    ],
)
def test_partial_unbound_or_unsafe_pilot_is_rejected(key, value):
    summary, review = pilot()
    review[key] = value
    with pytest.raises(ValueError):
        check_pilot(summary, review, "candidate")


def test_all_cpu_boundaries_pass_without_claiming_gain_or_promotion():
    check_cpu(*cpu(), model_hash="candidate", baseline_hash="parent", course=(11, 0))


@pytest.mark.parametrize(
    "key,value",
    [
        ("physical_substeps", 2999),
        ("reviewed_report_hash", "another-report"),
        ("high_quality", False),
        ("clean_foot_only", False),
        ("maximum_lateral_excursion_m", 4.01),
        ("minimum_pelvis_z_m", 0.64),
        ("minimum_pelvis_z_m", float("nan")),
        ("maximum_lateral_excursion_m", float("inf")),
        ("actual_mujoco_dynamics_replayed", False),
        ("neural_target_reconstructed", False),
        ("hardware_authorized", True),
    ],
)
def test_cpu_loss_nonfinite_or_incomplete_dynamics_is_rejected(key, value):
    args = cpu()
    args[1][key] = value
    with pytest.raises(ValueError):
        check_cpu(*args, model_hash="candidate", baseline_hash="parent", course=(11, 0))


def test_cpu_cannot_substitute_a_different_model_or_course():
    with pytest.raises(ValueError):
        check_cpu(*cpu(), model_hash="wrong", baseline_hash="parent", course=(11, 0))
    with pytest.raises(ValueError):
        check_cpu(*cpu(), model_hash="candidate", baseline_hash="parent", course=(11, 4))
