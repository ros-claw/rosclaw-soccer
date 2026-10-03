import numpy as np
import pytest

from rosclaw_soccer.sim.current_kinematic_observation import (
    CurrentKinematicObserver,
    observation_contract,
    snapshot_from_contract,
)
from rosclaw_soccer.sim.root_velocity_reference import root_observation_contract


def scene():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.002"/><worldbody>'
        '<body pos="0 0 2"><freejoint/><inertial pos="0.1 0.02 -0.03" '
        'mass="2" diaginertia="0.1 0.2 0.3"/><geom type="sphere" size="0.1"/>'
        '<body pos="0 0 0.2"><joint type="hinge"/><geom type="sphere" size="0.1"/>'
        '</body></body></worldbody><actuator><motor joint="0"/></actuator></mujoco>'.replace(
            '<joint type="hinge"/>', '<joint name="hinge" type="hinge"/>'
        ).replace('joint="0"', 'joint="hinge"')
    )
    data = mujoco.MjData(model)
    data.qvel[:] = np.linspace(0.1, 0.7, model.nv)
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def test_current_view_matches_independent_forward_without_touching_live_state():
    mujoco, model, data = scene()
    observer = CurrentKinematicObserver(model)
    for _ in range(7):
        mujoco.mj_step(model, data)
    names = (
        "qpos",
        "qvel",
        "qacc",
        "qacc_warmstart",
        "ctrl",
        "xpos",
        "xquat",
        "cvel",
        "actuator_force",
        "qfrc_applied",
        "xfrc_applied",
    )
    before = {name: getattr(data, name).copy() for name in names}
    model_names = ("qpos0", "body_mass", "body_inertia", "jnt_range", "dof_damping", "dof_armature")
    model_before = {name: getattr(model, name).copy() for name in model_names}
    time = data.time
    actual = observer.sample(data)
    reference = mujoco.MjData(model)
    reference.qpos[:] = data.qpos
    reference.qvel[:] = data.qvel
    mujoco.mj_forward(model, reference)
    np.testing.assert_array_equal(actual.body_position_m, reference.xpos)
    np.testing.assert_array_equal(actual.body_quaternion_wxyz, reference.xquat)
    for body in range(model.nbody):
        expected = np.zeros(6)
        mujoco.mj_objectVelocity(model, reference, mujoco.mjtObj.mjOBJ_XBODY, body, expected, 0)
        np.testing.assert_array_equal(
            actual.body_origin_velocity_world[body], expected[[3, 4, 5, 0, 1, 2]]
        )
    for name in names:
        np.testing.assert_array_equal(getattr(data, name), before[name])
    for name in model_names:
        np.testing.assert_array_equal(getattr(model, name), model_before[name])
    assert data.time == actual.time_sec == time
    assert not np.array_equal(actual.body_position_m, before["xpos"])


def test_observation_arrays_are_owned_and_do_not_carry_state_between_samples():
    mujoco, model, data = scene()
    observer = CurrentKinematicObserver(model)
    first = observer.sample(data)
    saved = first.body_position_m.copy()
    mujoco.mj_step(model, data)
    second = observer.sample(data)
    np.testing.assert_array_equal(first.body_position_m, saved)
    assert not np.array_equal(first.body_position_m, second.body_position_m)
    assert not first.body_position_m.flags.writeable
    assert not first.body_quaternion_wxyz.flags.writeable
    assert not first.body_origin_velocity_world.flags.writeable


def test_same_shape_state_from_another_model_is_not_silently_reinterpreted():
    _, model, data = scene()
    _, other, other_data = scene()
    assert model.nq == other.nq
    with pytest.raises(ValueError, match="finite complete"):
        CurrentKinematicObserver(model).sample(other_data)


def test_observer_never_calls_forward_dynamics_or_step(monkeypatch):
    mujoco, model, data = scene()

    def forbidden(*args, **kwargs):
        raise AssertionError("observation must not run dynamics or integration")

    monkeypatch.setattr(mujoco, "mj_forward", forbidden)
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    assert CurrentKinematicObserver(model).sample(data).time_sec == data.time


@pytest.mark.parametrize("field", ["qpos", "qvel", "time"])
def test_nonfinite_state_rejected_without_mutating_source(field):
    _, model, data = scene()
    if field == "time":
        data.time = float("nan")
    else:
        getattr(data, field)[0] = float("nan")
    with pytest.raises(ValueError, match="finite complete"):
        CurrentKinematicObserver(model).sample(data)


def test_cached_contracts_are_exactly_unchanged_and_current_requires_explicit_origin():
    for reference in ("body-com", "body-origin"):
        assert observation_contract(reference, "cached") == root_observation_contract(reference)
        assert snapshot_from_contract(root_observation_contract(reference)) == (reference, "cached")
    contract = observation_contract("body-origin", "current-kinematic")
    assert snapshot_from_contract(contract) == ("body-origin", "current-kinematic")
    with pytest.raises(ValueError):
        observation_contract("body-com", "current-kinematic")
    with pytest.raises(ValueError):
        snapshot_from_contract({**contract, "force_input": "future"})
