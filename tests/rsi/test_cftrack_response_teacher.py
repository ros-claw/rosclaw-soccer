import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_motor_authority import MOTOR_AXES
from rosclaw_soccer.rsi.cftrack_response_teacher import (
    fit_motor_response,
    propose_response_motor_plan,
    refit_current_hip_response,
)


def calibration():
    base = np.array([1.2, 0.7])
    jac = np.zeros((2, 4))
    jac[:, 1] = [1.0, -2.0]

    def metric(y, safe=True):
        return dict(
            private_safe=safe,
            post60_observed=True,
            first_foot_control=71,
            forward_60_m=float(y[0]),
            lateral_60_m=float(y[1]),
        )

    rows = [dict(name="parent", metrics=metric(base))]
    for i in range(8):
        delta = [0.0] * 29
        value = 0.15 if i % 2 else -0.15
        delta[MOTOR_AXES[i // 2][1]] = value
        rows.append(
            dict(
                name=f"arm-{i:02d}",
                metrics=metric(base + jac[:, i // 2] * value, i != 2),
                motor_plan=dict(
                    delta_hg=delta, origin_counter=0, start_control=46, peak_control=24
                ),
            )
        )
    return dict(status="CFTRACK_MOTOR_AUTHORITY_LOCAL_RESCUE", records=rows)


def propose(model, parent, limits=None):
    return propose_response_motor_plan(
        model,
        parent,
        origin_counter=0,
        donor_policy_hash="sha256:" + "b" * 64,
        parent_targets=np.zeros((160, 29)),
        limits_hg=np.tile([-1.0, 1.0], (29, 1)) if limits is None else limits,
    )


def test_fit_excludes_unsafe_direction_and_proposes_only_certified_math_not_physics():
    data = calibration()
    model = fit_motor_response(data, source_hash="sha256:" + "a" * 64)
    np.testing.assert_allclose(np.array(model.jacobian)[:, 1], [1.0, -2.0])
    assert model.lower[1] == 0 and model.fitted_arms[1] == ("arm-03",)
    plan, receipt = propose(model, data["records"][0]["metrics"])
    assert 0.24 < plan.delta_hg[7] < 0.25
    xy = receipt["predicted_outcome_xy"]
    assert xy[0] >= 1.2 and abs(xy[1]) / xy[0] <= 0.2
    assert (
        receipt["proposal_extrapolates_sampled_amplitude"]
        and receipt["private_native_validation_required"]
    )
    assert receipt["qp_certificate"]["physical_feasibility_not_implied"]


def test_actual_parent_target_range_can_make_proposal_infeasible_without_clipping():
    data = calibration()
    model = fit_motor_response(data, source_hash="sha256:" + "a" * 64)
    limits = np.tile([-1.0, 1.0], (29, 1))
    limits[7, 1] = 0.05
    with pytest.raises(ValueError, match="certified feasible"):
        propose(model, data["records"][0]["metrics"], limits)


@pytest.mark.parametrize("defect", ["axis", "unsafe_anchor", "no_safe_axis", "invalid_source"])
def test_bad_calibration_never_silently_fits_or_uses_unsafe_both_directions(defect):
    data = calibration()
    source = "sha256:" + "a" * 64
    if defect == "axis":
        data["records"][1]["motor_plan"]["delta_hg"][8] = 0.0
    elif defect == "unsafe_anchor":
        data["records"][0]["metrics"]["private_safe"] = False
    elif defect == "no_safe_axis":
        data["records"][4]["metrics"]["private_safe"] = False
    else:
        source = "unbound"
    with pytest.raises(ValueError):
        fit_motor_response(data, source_hash=source)


def test_current_safe_response_replaces_only_one_column_without_mutating_prior():
    data = calibration()
    model = fit_motor_response(data, source_hash="sha256:" + "a" * 64)
    parent = data["records"][0]["metrics"]
    support = dict(parent, forward_60_m=1.5, lateral_60_m=0.25)
    updated, receipt = refit_current_hip_response(model, parent, support)
    np.testing.assert_allclose(np.array(updated.jacobian)[:, 1], [2.0, -3.0])
    np.testing.assert_allclose(np.array(model.jacobian)[:, 1], [1.0, -2.0])
    np.testing.assert_array_equal(
        np.array(model.jacobian)[:, [0, 2, 3]], np.array(updated.jacobian)[:, [0, 2, 3]]
    )
    assert receipt["neural_optimizer_updates"] == 0 and updated.lower[1] == 0.0
    with pytest.raises(ValueError):
        refit_current_hip_response(model, parent, dict(support, private_safe=False))


def test_sparse_mode_keeps_forward_as_constraint_and_zeros_ineffective_axes():
    data = calibration()
    model = fit_motor_response(data, source_hash="sha256:" + "a" * 64)
    plan, c = propose_response_motor_plan(
        model,
        data["records"][0]["metrics"],
        origin_counter=0,
        donor_policy_hash="sha256:" + "b" * 64,
        parent_targets=np.zeros((160, 29)),
        limits_hg=np.tile([-1.0, 1.0], (29, 1)),
        minimal_intervention=True,
    )
    assert c["objective_weights"] == [0.0, 4.0] and c["l1_penalty"] == 0.25
    assert sum(v != 0.0 for v in plan.delta_hg) == 1 and plan.delta_hg[7] > 0.2
    assert c["qp_certificate"]["encoded_maximum_constraint_violation"] < 1e-9
