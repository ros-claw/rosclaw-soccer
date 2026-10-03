import numpy as np
import pytest

from rosclaw_soccer.sim.root_velocity_reference import (
    reference_from_contract,
    root_observation_contract,
    root_velocity_world,
)


def test_legacy_contract_remains_absent():
    assert root_observation_contract("body-com") is None
    assert reference_from_contract(None) == "body-com"


@pytest.mark.parametrize("bad", [{}, {"root_velocity_reference": "body-origin"}, [], True])
def test_incomplete_contract_rejected(bad):
    with pytest.raises(ValueError, match="unsupported"):
        reference_from_contract(bad)


def test_origin_contract_round_trip_and_no_mutable_shared_default():
    contract = root_observation_contract("body-origin")
    assert reference_from_contract(contract) == "body-origin"
    contract["derived_state_stage"] = "post-integration-refresh"
    with pytest.raises(ValueError, match="unsupported"):
        reference_from_contract(contract)
    assert reference_from_contract(root_observation_contract("body-origin")) == "body-origin"


def test_reference_point_shift_and_no_physics_mutation():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="root">
      <freejoint/><inertial pos="0.10 0.02 0.03" mass="1" diaginertia="0.1 0.1 0.1"/>
      <geom type="sphere" size="0.05"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    d.qvel[:] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    mujoco.mj_forward(m, d)
    state_before = d.qpos.copy(), d.qvel.copy(), d.qacc_warmstart.copy(), d.time
    com = root_velocity_world(m, d, 1, "body-com")
    origin = root_velocity_world(m, d, 1, "body-origin")
    legacy = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, 1, legacy, 0)
    np.testing.assert_array_equal(com, legacy[[3, 4, 5, 0, 1, 2]])
    np.testing.assert_allclose(
        origin[:3], com[:3] + np.cross(com[3:], d.xpos[1] - d.xipos[1]), atol=1e-15
    )
    np.testing.assert_array_equal(origin[3:], com[3:])
    assert np.max(np.abs(com[:3] - origin[:3])) > 0.01
    j, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, j, jr, 1)
    np.testing.assert_allclose(origin, np.concatenate((j @ d.qvel, jr @ d.qvel)), atol=1e-15)
    for before, after in zip(state_before[:3], (d.qpos, d.qvel, d.qacc_warmstart), strict=True):
        np.testing.assert_array_equal(before, after)
    assert d.time == state_before[3]
    for reference, body in [("unknown", 1), ("body-origin", 0), ("body-com", True)]:
        with pytest.raises(ValueError):
            root_velocity_world(m, d, body, reference)
