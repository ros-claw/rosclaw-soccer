import math

import numpy as np
import pytest

from rosclaw_soccer.world.rolling_ball import (
    ideal_planar_rolling_velocity,
    sphere_angular_velocity_world,
)


def estimate(v=(7.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0), **kwargs):
    return ideal_planar_rolling_velocity(
        linear_velocity_world=v,
        angular_velocity_world=omega,
        **({"radius_m": 0.1, "inertia_ratio": 0.4} | kwargs),
    )


def test_pure_sliding_solid_sphere_loses_two_sevenths_speed():
    assert estimate() == pytest.approx((5, 0))


def test_pure_rolling_preserves_velocity_and_axial_spin_is_irrelevant():
    assert estimate((2.0, 3.0, 0.0), (-30.0, 20.0, 99.0)) == pytest.approx((2, 3))
    assert estimate((0.0, 0.0, 0.0), (0.0, -70.0, 0.0)) == pytest.approx((-2, 0))


def test_planar_rotation_equivariance():
    x, y = estimate((2.0, 3.0, 0.0), (4.0, 5.0, 6.0))
    assert estimate((-3.0, 2.0, 0.0), (-5.0, 4.0, 6.0)) == pytest.approx((-y, x))


def test_quaternion_rotation_and_double_cover():
    q = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    assert sphere_angular_velocity_world(q, (2.0, 3.0, 4.0)) == pytest.approx((-3, 2, 4))
    assert sphere_angular_velocity_world(tuple(-x for x in q), (2.0, 3.0, 4.0)) == pytest.approx(
        (-3, 2, 4)
    )


def test_matches_native_mujoco_object_velocity_for_free_sphere():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="ball"><freejoint/>'
        '<geom type="sphere" size=".1" mass=".4"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    rng = np.random.default_rng(750)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        local = rng.uniform(-50, 50, 3)
        data.qpos[3:7] = q
        data.qvel[3:6] = local
        mujoco.mj_forward(model, data)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, velocity, 0)
        assert sphere_angular_velocity_world(
            tuple(q.tolist()), tuple(local.tolist())
        ) == pytest.approx(velocity[:3], abs=1e-12)


@pytest.mark.parametrize(
    "v", [(float("nan"), 0.0, 0.0), (True, 0.0, 0.0), (101.0, 0.0, 0.0), [0, 0, 0], (0, 0)]
)
def test_invalid_linear_observations_rejected(v):
    with pytest.raises(ValueError):
        estimate(v=v)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius_m": 0},
        {"radius_m": True},
        {"inertia_ratio": float("inf")},
        {"inertia_ratio": -1},
        {"omega": (1001.0, 0.0, 0.0)},
    ],
)
def test_invalid_sphere_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        estimate(**kwargs)


@pytest.mark.parametrize(
    "q", [(0.0, 0.0, 0.0, 0.0), (1.1, 0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0, 0.0)]
)
def test_invalid_quaternion_rejected(q):
    with pytest.raises(ValueError):
        sphere_angular_velocity_world(q, (0.0, 0.0, 0.0))
