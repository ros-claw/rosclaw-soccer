import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.directional_contact_impedance import vertical_compliance_stiffness
from rosclaw_soccer.rsi.taskspace_swing_evidence import LEG_NAMES


@pytest.mark.parametrize("side", [0, 1])
def test_contraction_preserves_static_tangential_compliance_not_dynamic_success(side):
    rng = np.random.default_rng(20261018)
    for _ in range(40):
        kp = np.exp(rng.uniform(1, 7, size=29))
        jac = rng.normal(size=(3, 29))
        before_kp, before_jac = kp.copy(), jac.copy()
        matrix, report = vertical_compliance_stiffness(kp, jac, side)
        relative = matrix / np.sqrt(kp)[:, None] / np.sqrt(kp)[None, :]
        eig = np.linalg.eigvalsh(relative)
        assert eig.min() >= 0.5 - 1e-12 and eig.max() <= 1 + 1e-12
        assert report["enabled"] is True
        old = (jac / kp) @ jac.T
        new = jac @ np.linalg.solve(matrix, jac.T)
        np.testing.assert_allclose(new[:2], old[:2], atol=1e-12, rtol=0)
        np.testing.assert_allclose(new[:, :2], old[:, :2], atol=1e-12, rtol=0)
        assert old[2, 2] <= new[2, 2] + 1e-12 <= 2 * old[2, 2] + 1e-12
        leg = [G1_DDS_JOINT_NAMES.index(n) for n in LEG_NAMES[side]]
        other = sorted(set(range(29)) - set(leg))
        assert np.array_equal(matrix[other], np.diag(kp)[other])
        assert kp.tobytes() == before_kp.tobytes() and jac.tobytes() == before_jac.tobytes()


def test_no_pressure_and_degenerate_vertical_geometry_are_exactly_inert():
    kp = np.ones(29)
    jac = np.zeros((3, 29))
    for side in (None, 0):
        matrix, report = vertical_compliance_stiffness(kp, jac, side)
        assert report["enabled"] is False
        assert matrix.tobytes() == np.diag(kp).tobytes()


@pytest.mark.parametrize("kind", ["nan", "gains", "shape", "side", "scale"])
def test_bad_inputs_rejected(kind):
    kp, jac, side = np.ones(29), np.ones((3, 29)), 0
    if kind == "nan":
        jac[0, 0] = np.nan
    elif kind == "gains":
        kp[0] = 0
    elif kind == "shape":
        jac = jac[:, :-1]
    elif kind == "side":
        side = True
    else:
        jac[0, 0] = 11
    with pytest.raises(ValueError):
        vertical_compliance_stiffness(kp, jac, side)
