"""Synthetic numeric controls, not physical evidence or skill qualification."""

import copy

import numpy as np
import pytest
from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi import reference_feedback_diagnostics as diagnosis
from rosclaw_soccer.rsi.reference_student_execution import make_model
from rosclaw_soccer.rsi.reference_student_imitation import numeric_errors
from tests.rsi.test_pre_parent_reference_features import measured_trace


def model(nonzero=False):
    params = initial_parameters(135, 12, hidden_dimension=64, seed=7)
    if nonzero:
        params["head_bias"][0] = 0.4
        params["head_weight"][0][0] = 0.3
    return make_model(
        params,
        mean=np.zeros(135),
        scale=np.ones(135),
        world_hash="sha256:" + "a" * 64,
        parent_model_hash="sha256:" + "b" * 64,
    )


def test_fixed_body_zero_model_matches_old_loss_and_has_no_projection():
    trace = measured_trace()
    report = diagnosis.diagnose_feedback(
        trace, model(), joint_limits=np.tile([-10.0, 10.0], (29, 1))
    )
    assert len(diagnosis.PREVIOUS) == 12
    assert report["maximum_past_reference_input_difference_rad"] == pytest.approx(0.2)
    assert report["maximum_teacher_forced_projection_change_rad"] == 0
    assert report["maximum_self_fed_projection_change_rad"] == 0
    assert len({r["predictions_hash"] for r in report["errors"].values()}) == 1
    assert report["measured_body_navigation_and_force_inputs_bit_exact_between_paths"]
    assert not report["student_body_rollout"] and not report["promotion_authorized"]
    assert report["optimizer_updates"] == report["physical_integration_steps"] == 0


def test_teacher_raw_reproduces_numeric_fit_error_and_projection_is_separate():
    trace, fitted = measured_trace(), model(nonzero=True)
    rows = diagnosis.trace_observations(trace)
    expected = numeric_errors(
        fitted["parameters"],
        dict(
            normalized=rows["observation"][None],
            targets=rows["same_state_reference_target_rad"][None],
            partitions=["train"],
            source_seeds=[4],
            eligible=[True],
        ),
    )[0]["mean_squared_reference_error_rad2"]
    bounds = np.tile([-0.01, 0.01], (29, 1))
    report = diagnosis.diagnose_feedback(trace, fitted, joint_limits=bounds)
    assert report["errors"]["teacher_forced_raw"]["mean_squared_reference_error_rad2"] == expected
    assert report["maximum_teacher_forced_projection_change_rad"] > 0
    assert report["maximum_self_fed_projection_change_rad"] > 0
    assert (
        report["errors"]["teacher_forced_raw"]["predictions_hash"]
        != report["errors"]["self_fed_raw"]["predictions_hash"]
    )


def test_input_and_trace_owned_and_no_physics_or_learning_imports():
    import inspect

    trace, fitted = measured_trace(), model(nonzero=True)
    original_trace, original_model = copy.deepcopy(trace), copy.deepcopy(fitted)
    diagnosis.diagnose_feedback(trace, fitted, joint_limits=np.tile([-3.0, 3.0], (29, 1)))
    for k in trace:
        np.testing.assert_array_equal(trace[k], original_trace[k])
    assert fitted == original_model
    source = inspect.getsource(diagnosis)
    assert "import mujoco" not in source and "fit_recurrent" not in source


@pytest.mark.parametrize("tamper", ["shape", "nan", "float32", "reversed", "model", "trace"])
def test_rejects_invalid_original_boundaries(tamper):
    trace, fitted = measured_trace(), model()
    bounds = np.tile([-3.0, 3.0], (29, 1))
    if tamper == "shape":
        bounds = bounds[:28]
    elif tamper == "nan":
        bounds[0, 0] = np.nan
    elif tamper == "float32":
        bounds = bounds.astype(np.float32)
    elif tamper == "reversed":
        bounds[0] = [1, -1]
    elif tamper == "model":
        fitted["mean"][0] = 1
    else:
        trace["root_pose_xyzw_m"][1, 0, 6] = 0
    with pytest.raises(ValueError):
        diagnosis.diagnose_feedback(trace, fitted, joint_limits=bounds)


def test_clipping_counts_disjoint_feature_groups_without_clipping_inputs():
    x = np.zeros((300, 135))
    x[:, diagnosis.PREVIOUS] = 9
    x[:, -1] = 1
    report = diagnosis.clipping_readout(x, np.zeros(135), np.ones(135))
    assert report["previous_reference"]["clipped_cell_fraction"] == 1
    assert report["measured_body_ball_navigation"]["clipped_cell_fraction"] == 0
    assert report["remaining_time"]["clipped_cell_fraction"] == 0
    assert np.all(x[:, diagnosis.PREVIOUS] == 9)
