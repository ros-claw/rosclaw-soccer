import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes


@pytest.fixture
def structural_model():
    """Fake 29-hinge topology for copy/budget tests, NOT an actual G1 asset."""
    bodies = []
    for i, name in enumerate(G1_DDS_JOINT_NAMES):
        body = (
            "left_ankle_roll_link"
            if i == 5
            else "right_ankle_roll_link"
            if i == 11
            else f"body_{i}"
        )
        bodies.append(
            f'<body name="{body}" pos="{i * 0.15} 0 0.25">'
            f'<joint name="{name}" axis="0 1 0"/>'
            '<geom type="sphere" size="0.02" mass="0.2"/></body>'
        )
    actuators = "".join(f'<motor joint="{name}"/>' for name in G1_DDS_JOINT_NAMES)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.002" integrator="Euler"/><worldbody>'
        '<geom type="plane" size="0 0 .01"/><body name="pelvis" pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size=".04" mass="5" contype="0" conaffinity="0"/>'
        + "".join(bodies)
        + '</body><body name="ball" pos="5 0 3"><freejoint/>'
        '<geom name="ball_geom" type="sphere" size=".11" mass=".43"/>'
        "</body></worldbody><actuator>" + actuators + "</actuator></mujoco>"
    )
    binary = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    return model, hash_bytes(binary.tobytes())


def forecaster(structural_model, budget=20):
    model, h = structural_model
    return G1PDTorqueForecaster(
        model,
        expected_world_hash=h,
        kp=np.full(29, 10.0),
        kd=np.full(29, 0.2),
        max_forecast_steps=budget,
    )


def test_forecast_rejects_non_g1_before_state_or_step_allocation():
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    with pytest.raises(ValueError, match="canonical"):
        G1PDTorqueForecaster(
            model,
            expected_world_hash="sha256:" + "a" * 64,
            kp=np.ones(29),
            kd=np.ones(29),
            max_forecast_steps=10,
        )


def test_complete_copy_repeatability_donor_ownership_and_budget(structural_model):
    model, _ = structural_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    f = forecaster(structural_model)
    before = data.qpos.copy()
    first = f.forecast(data, np.full(29, 0.1), np.zeros(29))
    second = f.forecast(data, np.full(29, 0.1), np.zeros(29))
    assert first.end_integration_state.tobytes() == second.end_integration_state.tobytes()
    np.testing.assert_array_equal(first.actual_actuator_force_nm, first.applied_control_nm)
    np.testing.assert_array_equal(data.qpos, before)
    assert (
        data.time == 0
        and f.attempted_steps == 20
        and f.integrated_steps == 20
        and f.forecast_attempts == 2
    )
    with pytest.raises(RuntimeError, match="budget"):
        f.forecast(data, np.full(29, 0.1), np.zeros(29))
    assert f.forecast_attempts == 2 and f.attempted_steps == 20


@pytest.mark.parametrize("kind", ["offset", "target", "nan", "dtype"])
def test_invalid_candidate_rejected_without_charging_steps(structural_model, kind):
    model, _ = structural_model
    data = mujoco.MjData(model)
    f = forecaster(structural_model)
    target = np.zeros(29)
    offset = np.zeros(29)
    if kind == "offset":
        offset[0] = 13
    elif kind == "target":
        target[0] = 11
    elif kind == "nan":
        offset[0] = np.nan
    else:
        offset = offset.astype(np.float32)
    with pytest.raises(ValueError):
        f.forecast(data, target, offset)
    assert f.integrated_steps == f.attempted_steps == f.forecast_attempts == 0


def test_nonzero_offset_changes_own_forecast_not_the_next_candidate(structural_model):
    model, _ = structural_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    f = forecaster(structural_model, budget=30)
    base = f.forecast(data, np.zeros(29), np.zeros(29))
    nonzero = f.forecast(data, np.zeros(29), np.ones(29))
    again = f.forecast(data, np.zeros(29), np.zeros(29))
    assert (
        base.canonical_qvel_after_substep.tobytes()
        != nonzero.canonical_qvel_after_substep.tobytes()
    )
    assert base.end_integration_state.tobytes() == again.end_integration_state.tobytes()
