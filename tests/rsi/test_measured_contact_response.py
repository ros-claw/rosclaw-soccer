import numpy as np
import pytest

from rosclaw_soccer.rsi.measured_contact_response import (
    fit_measured_contact_response,
    propose_bounded_response,
)


def report():
    base = np.array([0.1, 0.05, 0.02])
    jac = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    safe = dict(complete=True, body_failed=False, runtime_failure=None, outcome={"safe": True})
    records = {"parent": dict(safe)}
    for n, axis in enumerate(range(6, 10)):
        for sign, name in ((-1, "minus"), (1, "plus")):
            records[f"hg{axis}-{name}"] = dict(
                safe,
                delta_hg_index=axis,
                delta_rad=sign * 0.08,
                witness_relative_xyz_m=(base + sign * 0.08 * jac[:, n]).tolist(),
            )
    return dict(
        status="CFTRACK_HIP_KNEE_PHYSICS_RESPONSE_CALIBRATION_NOT_PROMOTED",
        task_promoted=False,
        hardware_authorized=False,
        witness_cftrack_control=95,
        witness_parent_relative_xyz_m=base.tolist(),
        records=records,
    )


def test_centered_response_and_bounded_candidate():
    model = fit_measured_contact_response(report(), source_hash="sha256:" + "a" * 64)
    np.testing.assert_allclose(model.jacobian, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    delta, receipt = propose_bounded_response(model)
    assert np.max(np.abs(delta)) <= 0.080000001
    assert receipt["predicted_gap_m"] < receipt["baseline_gap_m"]
    assert receipt["simultaneous_axes_not_calibrated"] is True


def test_reject_unsafe_or_missing_pair():
    value = report()
    value["records"]["hg6-plus"]["outcome"] = {"safe": False}
    with pytest.raises(ValueError):
        fit_measured_contact_response(value, source_hash="sha256:" + "a" * 64)
    value = report()
    del value["records"]["hg6-plus"]
    with pytest.raises(ValueError):
        fit_measured_contact_response(value, source_hash="sha256:" + "a" * 64)
