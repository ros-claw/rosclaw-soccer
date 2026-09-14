"""Native CPU-only refinement tests; no robot assets or transport."""

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.held_control_refinement import step_with_held_control


def world():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.002"/><worldbody>'
        '<body pos="0 0 1"><joint name="slide" type="slide" axis="0 0 1"/>'
        '<geom type="sphere" size="0.1" mass="1"/></body></worldbody>'
        '<actuator><motor joint="slide"/></actuator></mujoco>'
    )
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.3
    data.qfrc_applied[:] = 0.1
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize("factor", [1, 2, 4, 8, 16])
def test_matches_explicit_fixed_control_microsteps(factor):
    model, data = world()
    reference_model, reference = world()
    reference_model.opt.timestep = 0.002 / factor
    for _ in range(factor):
        mujoco.mj_step(reference_model, reference)
    times = []
    receipt = step_with_held_control(
        model, data, refinement=factor, observer=lambda m, d: times.append(d.time)
    )
    np.testing.assert_array_equal(data.qpos, reference.qpos)
    np.testing.assert_array_equal(data.qvel, reference.qvel)
    assert len(times) == receipt.observed_microsteps == factor
    assert model.opt.timestep == 0.002
    assert data.time == pytest.approx(0.002, abs=1e-14)
    assert receipt.activation_ceiling == "SIM_ONLY" and not receipt.promotion_eligible


@pytest.mark.parametrize("factor", [True, 0, 3, 32, 2.0, None])
def test_rejects_invalid_refinement_without_advancing(factor):
    model, data = world()
    with pytest.raises(ValueError):
        step_with_held_control(model, data, refinement=factor)
    assert data.time == 0 and model.opt.timestep == 0.002


def test_observer_exception_restores_timestep_not_simulation_state():
    model, data = world()

    def fail(m, d):
        raise RuntimeError("retain failed episode")

    with pytest.raises(RuntimeError):
        step_with_held_control(model, data, refinement=4, observer=fail)
    assert model.opt.timestep == 0.002
    assert data.time == 0.0005


@pytest.mark.parametrize("field", ["ctrl", "qfrc_applied", "xfrc_applied"])
def test_observer_cannot_silently_change_held_input(field):
    model, data = world()

    def corrupt(m, d):
        getattr(d, field).flat[0] += 1

    with pytest.raises(ValueError, match="control changed"):
        step_with_held_control(model, data, refinement=4, observer=corrupt)
    assert model.opt.timestep == 0.002


@pytest.mark.parametrize("field", ["time", "timestep"])
def test_observer_clock_mutation_is_rejected(field):
    model, data = world()

    def corrupt(m, d):
        if field == "time":
            d.time += 1
        else:
            m.opt.timestep *= 2

    with pytest.raises(ValueError):
        step_with_held_control(model, data, refinement=2, observer=corrupt)
    assert model.opt.timestep == 0.002


def test_control_callback_is_refused_before_stepping():
    model, data = world()
    previous = mujoco.get_mjcb_control()
    try:
        mujoco.set_mjcb_control(lambda m, d: None)
        with pytest.raises(ValueError, match="callbacks"):
            step_with_held_control(model, data, refinement=2)
        assert data.time == 0
    finally:
        mujoco.set_mjcb_control(previous)


@pytest.mark.parametrize("field", ["qpos", "qvel", "ctrl"])
def test_nonfinite_input_refused(field):
    model, data = world()
    getattr(data, field)[:] = np.nan
    with pytest.raises(ValueError, match="finite"):
        step_with_held_control(model, data, refinement=2)
    assert data.time == 0


def test_missing_observer_is_not_complete_observation():
    model, data = world()
    receipt = step_with_held_control(model, data, refinement=2)
    assert receipt.observed_microsteps == 0


def test_same_shape_foreign_model_data_is_refused():
    model, _ = world()
    _, data = world()
    with pytest.raises(ValueError, match="exact supplied model"):
        step_with_held_control(model, data, refinement=2)
    assert data.time == 0
