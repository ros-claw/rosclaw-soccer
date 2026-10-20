"""Synthetic topology tests only, not genuine G1 skill qualification."""

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.rsi.measured_foot_kinematics import MeasuredFootKinematics
from rosclaw_soccer.sim.contracts import hash_bytes
from tests.physics.test_g1_feedback_forecast import world


def setup():
    model, donor, _ = world()
    binary = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    extractor = MeasuredFootKinematics(model, expected_world_hash=hash_bytes(binary.tobytes()))
    return model, donor, extractor


def test_current_private_kinematics_never_steps_or_changes_donor(monkeypatch):
    model, donor, k = setup()
    before = donor.qpos.copy(), donor.qvel.copy(), donor.qacc.copy(), donor.time

    def forbidden(*args, **kwargs):
        raise AssertionError("kinematic feature extractor must never simulate")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    q, v = donor.qpos.copy(), donor.qvel.copy()
    f = k.features(q, v)
    assert f.shape == (26,) and f.dtype == np.float32
    assert np.isfinite(f).all() and f[12] >= 0 and f[25] >= 0
    np.testing.assert_array_equal(q, donor.qpos)
    np.testing.assert_array_equal(v, donor.qvel)
    for old, current in zip(before[:3], (donor.qpos, donor.qvel, donor.qacc), strict=True):
        np.testing.assert_array_equal(old, current)
    assert donor.time == before[3]
    np.testing.assert_array_equal(f, k.features(q, v))


def test_translation_common_linear_velocity_and_quaternion_double_cover_invariant():
    _, donor, k = setup()
    q, v = donor.qpos.copy(), donor.qvel.copy()
    a = k.features(q, v)
    q[:3] += [2, -3, 0.4]
    q[36:39] += [2, -3, 0.4]
    q[3:7] *= -1
    v[:3] += [0.3, -0.2, 0.1]
    v[35:38] += [0.3, -0.2, 0.1]
    np.testing.assert_allclose(a, k.features(q, v), atol=1e-7)


def test_jacobian_velocity_matches_kinematic_finite_difference():
    model, donor, k = setup()
    q, v = donor.qpos.copy(), donor.qvel.copy()
    v[5] = 0.8
    f = k.features(q, v)
    old = k.private.xpos[k.feet[0]].copy()
    changed = q.copy()
    mujoco.mj_integratePos(model, changed, v, 1e-6)
    k.features(changed, v)
    actual = (k.private.xpos[k.feet[0]] - old) / 1e-6
    np.testing.assert_allclose(actual, f[3:6], atol=1e-6)


@pytest.mark.parametrize("bad", ["hash", "nan", "shape", "quat"])
def test_invalid_state_or_body_commitment_rejected(bad):
    model, donor, k = setup()
    if bad == "hash":
        with pytest.raises(ValueError):
            MeasuredFootKinematics(model, expected_world_hash="sha256:" + "0" * 64)
        return
    q, v = donor.qpos.copy(), donor.qvel.copy()
    if bad == "nan":
        v[3] = np.nan
    elif bad == "shape":
        q = q[:-1]
    else:
        q[3] = 2
    with pytest.raises(ValueError):
        k.features(q, v)
