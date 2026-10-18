import numpy as np
import pytest

from rosclaw_soccer.rsi.directional_contact_compliance import slew_stiffness
from rosclaw_soccer.rsi.directional_contact_impedance import vertical_compliance_stiffness


def test_gain_slew_contractions_and_exact_restore_without_touching_other_joints():
    rng = np.random.default_rng(202610180)
    kp = np.linspace(20, 200, 29)
    relative = np.eye(29)
    for _ in range(30):
        jac = rng.normal(size=(3, 29))
        desired, _ = vertical_compliance_stiffness(kp, jac, 0)
        matrix, relative, step = slew_stiffness(kp, relative, desired)
        assert step <= 0.1 + 1e-12
        assert np.linalg.eigvalsh(relative).min() >= 0.5 - 1e-12
        assert np.linalg.eigvalsh(relative).max() <= 1 + 1e-12
        assert np.array_equal(matrix, matrix.T)
        assert np.array_equal(matrix[6:], np.diag(kp)[6:])
    for _ in range(6):
        matrix, relative, step = slew_stiffness(kp, relative, np.diag(kp))
        assert step <= 0.1 + 1e-12
    assert matrix.tobytes() == np.diag(kp).tobytes()
    assert relative.tobytes() == np.eye(29).tobytes()


def test_stationary_inactive_gain_has_no_roundoff_drift():
    kp = np.linspace(11, 170, 29)
    previous = np.eye(29)
    for _ in range(300):
        matrix, previous, step = slew_stiffness(kp, previous, np.diag(kp))
        assert matrix.tobytes() == np.diag(kp).tobytes()
        assert step == 0


def test_steady_active_gain_never_rounds_unmodified_torso_into_a_new_gain():
    kp = np.linspace(11, 170, 29)
    jac = np.random.default_rng(18).normal(size=(3, 29))
    desired, _ = vertical_compliance_stiffness(kp, jac, 0)
    relative = np.eye(29)
    for _ in range(30):
        matrix, relative, _ = slew_stiffness(kp, relative, desired)
        assert matrix[6:].tobytes() == np.diag(kp)[6:].tobytes()


@pytest.mark.parametrize("kind", ["nan", "previous", "desired", "asymmetry"])
def test_slew_rejects_invalid_or_out_of_contract_gain_states(kind):
    kp, previous, desired = np.ones(29), np.eye(29), np.eye(29)
    if kind == "nan":
        desired[0, 0] = np.nan
    elif kind == "previous":
        previous[0, 0] = 0.4
    elif kind == "desired":
        desired[0, 0] = 1.1
    else:
        desired[0, 1] = 0.1
    with pytest.raises(ValueError):
        slew_stiffness(kp, previous, desired)
