import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_response_model import propose_contact_response


def window(x, y):
    return dict(quality_observed=True, forward_60_m=x, lateral_60_m=y)


def probes(nominal, jac, baseline):
    result = []
    for axis in range(jac.shape[1]):
        for delta in (-0.03, 0.03):
            target = nominal.copy()
            target[axis] += delta
            output = baseline + jac[:, axis] * delta
            result.append((target, window(*output)))
    return result


def test_native_data_response_learns_coupled_axes_instead_of_fk_or_scalar_gain():
    nominal = np.zeros(29)
    baseline = np.array([1.1, 0.5])
    jac = np.array([[2.0, -2.0], [4.0, 4.0]])
    target, report = propose_contact_response(
        nominal, np.tile([-2.0, 2.0], (29, 1)), window(*baseline), probes(nominal, jac, baseline)
    )
    np.testing.assert_allclose(report["response_m_per_target_rad"], jac, atol=1e-12)
    assert abs(target[:2]).max() <= 0.0300001
    assert np.count_nonzero(target[:2]) == 2
    np.testing.assert_array_equal(target[2:], nominal[2:])
    assert report["neural_actor_updates"] == 0 and not report["physically_verified"]


def test_censored_windows_do_not_become_zero_outcomes():
    nominal = np.zeros(29)
    parent = dict(quality_observed=False, forward_60_m=None, lateral_60_m=None)
    assert propose_contact_response(nominal, np.tile([-2.0, 2.0], (29, 1)), parent, [])[0] is None
    values = probes(nominal, np.eye(2), np.array([0.8, 0.4]))
    values[0] = (values[0][0], parent)
    values[1] = (values[1][0], parent)
    target, report = propose_contact_response(
        nominal, np.tile([-2.0, 2.0], (29, 1)), window(0.8, 0.4), values
    )
    assert target is None and report["status"] == "ABSTAIN_INSUFFICIENT_OBSERVED_AXES"


def test_no_inadmissible_parent_axis_is_repaired_and_observed_trust_region_is_kept():
    nominal = np.zeros(29)
    nominal[0] = 3.0
    values = probes(nominal, np.ones((2, 3)), np.array([0.8, 0.4]))
    target, report = propose_contact_response(
        nominal, np.tile([-2.0, 2.0], (29, 1)), window(0.8, 0.4), values
    )
    assert target[0] == 3.0
    assert 0 not in report["observed_axes"]
    assert abs(target[1:3]).max() <= 0.03 + 1e-12


def test_unobserved_multi_joint_probe_rejected():
    target = np.zeros(29)
    target[:2] = 0.03
    with pytest.raises(ValueError, match="single-axis"):
        propose_contact_response(
            np.zeros(29),
            np.tile([-2.0, 2.0], (29, 1)),
            window(0.8, 0.4),
            [(target, window(1.0, 0.0))],
        )
