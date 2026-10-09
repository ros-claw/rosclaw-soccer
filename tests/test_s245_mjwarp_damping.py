import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.sim.mjwarp_contract import (
    _put_model_checked,
    qualify_mjwarp_damping,
    validate_converted_options,
    validate_damping_forces,
)


def option_model(**updates):
    values = dict(
        timestep=0.002,
        tolerance=1e-8,
        ls_tolerance=0.01,
        integrator=0,
        solver=2,
        iterations=100,
        ls_iterations=50,
        cone=0,
        disableflags=0,
        enableflags=0,
    )
    values.update(updates)
    return SimpleNamespace(opt=SimpleNamespace(**values))


def test_silent_float32_solver_tolerance_floor_is_rejected_without_mutation():
    native, gpu = option_model(), option_model(tolerance=np.array([1e-6], dtype=np.float32))
    with pytest.raises(ValueError, match="silently changed option tolerance"):
        _put_model_checked(SimpleNamespace(put_model=lambda _: gpu), native)
    assert native.opt.tolerance == 1e-8
    assert gpu.opt.tolerance[0] == np.float32(1e-6)


def test_exact_float32_option_representation_is_allowed_and_recorded():
    native = option_model()
    gpu = option_model(
        **{
            key: np.array([np.float32(getattr(native.opt, key))])
            for key in ("timestep", "tolerance", "ls_tolerance")
        }
    )
    result = validate_converted_options(native, gpu)
    assert result["tolerance"] == float(np.float32(1e-8))
    assert result["iterations"] == 100


@pytest.mark.parametrize("field", tuple(vars(option_model().opt)))
def test_every_silent_changed_or_missing_option_fails_closed(field):
    native, gpu = option_model(), option_model()
    setattr(gpu.opt, field, getattr(gpu.opt, field) + 1)
    with pytest.raises(ValueError, match=f"silently changed option {field}"):
        validate_converted_options(native, gpu)
    delattr(gpu.opt, field)
    with pytest.raises(ValueError, match=f"converted option {field}"):
        validate_converted_options(native, gpu)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, [], [[1e-8]], [1e-8, 1e-8], True])
def test_nonfinite_or_ambiguous_option_cannot_pass(bad):
    with pytest.raises(ValueError, match="converted option tolerance"):
        validate_converted_options(option_model(), option_model(tolerance=bad))


@pytest.mark.parametrize("category", [UserWarning, RuntimeWarning, DeprecationWarning])
def test_model_conversion_warning_cannot_be_silently_qualified(category):
    sentinel = object()

    def degraded(model):
        assert model is sentinel
        warnings.warn("MULTICCD: At most 1 contact will be generated", category, stacklevel=2)
        return object()

    # Even an operator's pre-existing ignore filter cannot hide degradation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="At most 1 contact"):
            _put_model_checked(SimpleNamespace(put_model=degraded), sentinel)


def test_warning_free_conversion_preserves_exact_model_and_exception():
    model = option_model()
    converted = option_model()

    def convert(candidate):
        assert candidate is model
        return converted

    assert _put_model_checked(SimpleNamespace(put_model=convert), model) is converted

    def reject(candidate):
        raise RuntimeError("unsupported model")

    with pytest.raises(RuntimeError, match="unsupported model"):
        _put_model_checked(SimpleNamespace(put_model=reject), model)


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
    # This new primitive fixture declares the backend's precision-domain
    # tolerance explicitly; never rewrite an existing G1 scene to pass.
    model.opt.tolerance = 1e-6
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


@pytest.mark.skipif(
    os.environ.get("ROSCLAW_RUN_MJWARP_TESTS") != "1", reason="explicit GPU kernel test"
)
def test_keeper_backend_qualifies_physics_before_loading_policy(tmp_path, monkeypatch):
    import mujoco

    from rosclaw_soccer.sim import mjwarp_contract
    from rosclaw_soccer.training.goalkeeper_mjwarp import (
        GoalkeeperMJWarpBatch,
        GoalkeeperMJWarpConfig,
    )
    from rosclaw_soccer.world import field

    # A deliberately unloadable policy proves rejection precedes model rollout
    # and policy deserialization. No external robot asset is needed for this test.
    policy = tmp_path / "invalid.pt"
    policy.write_bytes(b"not a policy")
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/>"
        '<geom type="sphere" size=".1"/></body></worldbody></mujoco>'
    )
    calls = []

    def reject(candidate, *, device):
        assert candidate is model
        calls.append(device)
        raise ValueError("per-DOF damping rejected test backend")

    monkeypatch.setattr(field, "build_g1_stadium_model", lambda _: model)
    monkeypatch.setattr(mjwarp_contract, "qualify_mjwarp_damping", reject)
    with pytest.raises(ValueError, match="per-DOF damping rejected"):
        GoalkeeperMJWarpBatch(
            asset_root=Path(tmp_path),
            locomotion_policy_path=policy,
            device="cuda:0",
            config=GoalkeeperMJWarpConfig(environment_count=1),
        )
    assert calls == ["cuda:0"]
