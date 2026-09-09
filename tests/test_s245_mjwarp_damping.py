import os

import numpy as np
import pytest

from rosclaw_soccer.sim.mjwarp_contract import (
    qualify_mjwarp_damping,
    validate_damping_forces,
)


def test_uniform_scalar_shortcut_cannot_pass_anisotropic_force_contract():
    velocity = np.array([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]])
    correct = -velocity * np.array([0.02] * 3 + [0.00002] * 3)
    with pytest.raises(ValueError, match="per-DOF damping"):
        validate_damping_forces(correct, -velocity * 0.02)
    assert validate_damping_forces(correct, correct.astype(np.float32)) < 1e-8


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_damping_fails_closed(bad):
    with pytest.raises(ValueError, match="finite"):
        validate_damping_forces([[0.0]], [[bad]])


def test_no_empty_broadcast_or_implicit_device_qualification():
    for value in ([], [1.0], [[1.0, 2.0]]):
        with pytest.raises(ValueError):
            validate_damping_forces([[1.0]], value)
    with pytest.raises(ValueError, match="explicit CUDA"):
        qualify_mjwarp_damping(None, device="cpu")


@pytest.mark.skipif(
    os.environ.get("ROSCLAW_RUN_MJWARP_TESTS") != "1", reason="explicit GPU kernel test"
)
@pytest.mark.parametrize("joint", ["free", "ball"])
@pytest.mark.parametrize("disabled", [False, True])
def test_actual_passive_kernel_preserves_each_compiled_dof(joint, disabled):
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 1">'
        f'<joint type="{joint}"/><geom type="sphere" size=".1" mass=".4"/>'
        "</body></worldbody></mujoco>"
    )
    # First DOF zero must not mask later nonzero damping, including nonlinear terms.
    model.dof_damping[:] = np.linspace(0.0, 0.02, model.nv)
    model.dof_dampingpoly[:, 0] = np.linspace(0.0, 0.003, model.nv)
    model.dof_dampingpoly[:, 1] = np.linspace(0.0, 0.0002, model.nv)
    if disabled:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_DAMPER
    coefficients = model.dof_damping.copy()
    result = qualify_mjwarp_damping(model, device="cuda:0")
    np.testing.assert_array_equal(coefficients, model.dof_damping)
    assert result["physics_steps"] == 0
    assert result["maximum_absolute_force_error"] < 1e-7
    assert result["probes"] == 3
