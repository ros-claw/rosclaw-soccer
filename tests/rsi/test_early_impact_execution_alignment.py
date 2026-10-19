"""Synthetic prediction-vs-actual comparison, never new physical evidence."""

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS
from scripts.rsi_early_impact_execution_alignment import (
    compare_execution,
    compare_named_release_execution,
    verify_actual_torques,
)
from tests.rsi.test_early_feedback_readback import fixture


def test_all40_controls_and_three_neural_boundaries_match():
    forecast, trace, _ = fixture()
    d = compare_execution(trace, forecast, 30)
    assert d["full80ms_execution_alignment_verified"]
    assert d["physical_substeps_compared"] == 40
    assert d["actual_neural_target_updates_compared"] == 3
    assert d["actual_control_state_boundaries_compared"] == 4
    assert max(d["maximum_absolute_errors"].values()) == 0
    assert not d["dense2ms_actual_qv_captured"] and not d["teacher_qualified"]
    assert not d["independent_neural_implementation"] and d["new_physical_steps"] == 0


def named_fixture():
    # Explicit synthetic data, not a relabelled historical capture.
    forecast, synthetic, _ = fixture()
    trace = {k: v for k, v in synthetic.items() if not k.startswith("early_impact_feedback_")}
    trace["postcontact_release_requested_offset_nm"] = np.zeros((300, 1, 29))
    forecast["requested_offset_nm"] = np.zeros((40, 29))
    return forecast, trace


def test_named_release_has_its_own_namespace_and_full_bridge():
    forecast, trace = named_fixture()
    result = compare_named_release_execution(trace, forecast, 30)
    assert result["full80ms_execution_alignment_verified"]
    assert result["actual_neural_target_updates_compared"] == 3
    assert result["new_physical_steps"] == 0
    trace["postcontact_exit_requested_offset_nm"] = np.zeros((300, 1, 29))
    with pytest.raises(ValueError, match="namespace"):
        compare_named_release_execution(trace, forecast, 30)


def test_changed_requested_lease_is_not_hidden_by_equal_saturated_controls():
    forecast, trace = named_fixture()
    trace["postcontact_release_requested_offset_nm"][31, 0, 0] = 1
    with pytest.raises(ValueError, match="requested offsets"):
        compare_named_release_execution(trace, forecast, 30)


@pytest.mark.parametrize("j", [1, 2, 3])
@pytest.mark.parametrize("bad", ["neural", "target", "body"])
def test_every_actual_intermediate_update_is_checked(j, bad):
    forecast, trace, _ = fixture()
    if bad == "neural":
        trace["foundation_neural_encoder_features"][30 + j, 0, 0] = -0.0
    elif bad == "target":
        trace["joint_target_rad"][30 + j, 0, 0] = 0.1
    else:
        trace["canonical_qpos"][30 + j, 0, 0] = 0.1
    with pytest.raises(ValueError):
        compare_execution(trace, forecast, 30)


@pytest.mark.parametrize("period", range(4))
@pytest.mark.parametrize(
    "field", ["torque_nm", "pd_requested_torque_nm", "actual_actuator_force_nm", "force_n"]
)
def test_every_control_period_and_actual_contact_bin_checked(period, field):
    forecast, trace, _ = fixture()
    if field == "force_n":
        trace[field][30 + period, 0, 0] = 0.1
    else:
        trace[field][30 + period, 0, 9, 0] = 0.1
    with pytest.raises(ValueError):
        compare_execution(trace, forecast, 30)


def test_final80ms_state_not_only_three_intermediate_targets_is_checked():
    forecast, trace, _ = fixture()
    trace["canonical_qpos"][34, 0, 0] = 0.1
    with pytest.raises(ValueError):
        compare_execution(trace, forecast, 30)


@pytest.mark.parametrize("cancel", [31, 32])
def test_safety_cancellation_is_explicit_partial_not_full_model_success(cancel):
    forecast, trace, _ = fixture()
    trace["torque_nm"][cancel:34] = 0.1
    d = compare_execution(trace, forecast, 30, safety_cancel_frame=cancel)
    assert d["actual_safety_cancellation_requires_separate_review"]
    assert not d["full80ms_execution_alignment_verified"]
    assert d["physical_substeps_compared"] == (cancel - 30) * 10
    assert not d["teacher_qualified"]


@pytest.mark.parametrize("frame,cancel", [(29, None), (37, None), (30, 30), (30, 33), (True, None)])
def test_invalid_clock_or_invented_cancellation_boundary_rejected(frame, cancel):
    forecast, trace, _ = fixture()
    with pytest.raises(ValueError):
        compare_execution(trace, forecast, frame, safety_cancel_frame=cancel)


def test_actual_nonfinite_and_missing_substeps_are_not_hidden():
    forecast, trace, _ = fixture()
    trace["torque_nm"][33, 0, 9, 0] = np.nan
    with pytest.raises(ValueError):
        compare_execution(trace, forecast, 30)
    forecast, trace, _ = fixture()
    forecast["applied_control_nm"] = forecast["applied_control_nm"][:39]
    with pytest.raises(ValueError):
        compare_execution(trace, forecast, 30)


@pytest.mark.parametrize(
    "bad",
    [None, "torque_nm", "actual_actuator_force_nm", "early_impact_feedback_applied_offset_nm"],
)
def test_saturated_requested_action_does_not_invent_physical_authority(bad):
    _, trace, _ = fixture()

    limits = np.asarray(G1_HARD_TORQUE_LIMITS)
    trace["pd_requested_torque_nm"][:] = limits + 10
    trace["torque_nm"][:] = limits
    trace["actual_actuator_force_nm"][:] = limits
    trace["early_impact_feedback_requested_offset_nm"] = np.zeros((300, 1, 29))
    trace["early_impact_feedback_requested_offset_nm"][30:33] = 1
    trace["early_impact_feedback_applied_offset_nm"] = np.zeros((300, 1, 10, 29))
    if bad is None:
        verify_actual_torques(trace)
    else:
        trace[bad][30, 0, 0, 0] += 0.01
        with pytest.raises(ValueError):
            verify_actual_torques(trace)
