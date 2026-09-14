"""Private current-position geometry with no mutation of completed contact evidence."""

import math

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.providers.g1.kinematic_snapshot import CpuKinematicsSnapshot


def model_data(ball_x=0.2):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep=".002"/><worldbody>'
        f'<geom name="ball" type="sphere" size=".115" pos="{ball_x} 0 -.25"/>'
        '<body><joint type="hinge" axis="0 1 0"/>'
        '<geom name="shin" type="capsule" size=".04" '
        'fromto="0 0 0 0 0 -.3" mass="1"/></body></worldbody></mujoco>'
    )
    return model, mujoco.MjData(model)


def live_arrays(data):
    return {
        key: getattr(data, key).copy()
        for key in (
            "qpos",
            "qvel",
            "qacc",
            "qacc_warmstart",
            "geom_xpos",
            "geom_xmat",
            "xpos",
            "xmat",
            "subtree_com",
            "cdof",
            "cvel",
            "efc_force",
            "qfrc_constraint",
        )
    }


@pytest.mark.parametrize("ball_x", [0.1, 0.2])
def test_current_geometry_gradient_and_live_solver_preservation(ball_x):
    model, live = model_data(ball_x)
    mujoco.mj_forward(model, live)
    # Intentionally leave derived geometry stale, as integration can do.
    live.qpos[0] = 0.1
    live.qvel[0] = 0.2
    before = live_arrays(live)
    reader = CpuKinematicsSnapshot(model, [live])
    result = reader.read_surface_pairs(((0, 1), (1, 0)))
    assert all(np.array_equal(value, getattr(live, key)) for key, value in before.items())
    assert not result["distance_censored"].any()
    assert result["qpos"].numpy()[0, 0] == 0.1
    reference = mujoco.MjData(model)
    reference.qpos[:] = live.qpos
    mujoco.mj_forward(model, reference)
    distance = mujoco.mj_geomDistance(model, reference, 0, 1, 1.0, None)
    np.testing.assert_allclose(result["signed_surface_distance_m"], distance, atol=1e-12)
    for index in range(2):
        segment = result["surface_segment_world"].numpy()[0, index]
        normal = (segment[3:] - segment[:3]) / distance
        gradient = float(normal @ result["relative_point_jacobian"].numpy()[0, index, :, 0])
        values = []
        for offset in (-1e-6, 1e-6):
            reference.qpos[0] = live.qpos[0] + offset
            mujoco.mj_forward(model, reference)
            values.append(mujoco.mj_geomDistance(model, reference, 0, 1, 1.0, None))
        assert gradient == pytest.approx((values[1] - values[0]) / 2e-6, abs=1e-7)


def test_censored_pairs_do_not_expose_fictitious_endpoint_jacobians():
    model, live = model_data(2.0)
    result = CpuKinematicsSnapshot(model, [live]).read_surface_pairs(((0, 1),))
    assert result["distance_censored"].all()
    assert result["signed_surface_distance_m"].item() == 1.0
    assert not result["surface_segment_world"].any()
    assert not result["relative_point_jacobian"].any()


def test_result_mutation_does_not_change_later_capture_or_live_state():
    model, live = model_data()
    reader = CpuKinematicsSnapshot(model, [live])
    result = reader.read_surface_pairs(((0, 1),))
    for key in (
        "qpos",
        "signed_surface_distance_m",
        "surface_segment_world",
        "relative_point_jacobian",
    ):
        result[key].fill_(123)
    again = reader.read_surface_pairs(((0, 1),))
    assert live.qpos[0] == 0.0 and again["qpos"].item() == 0.0
    assert again["signed_surface_distance_m"].item() == pytest.approx(0.045)


def test_surface_reads_preserve_a_complete_dynamic_replay_exactly():
    model, left = model_data(0.1)
    right = mujoco.MjData(model)
    left.qvel[0] = right.qvel[0] = 0.3
    reader = CpuKinematicsSnapshot(model, [left])
    for _ in range(50):
        mujoco.mj_step(model, left)
        mujoco.mj_step(model, right)
        before = live_arrays(left)
        reader.read_surface_pairs(((0, 1),))
        assert all(np.array_equal(value, getattr(left, key)) for key, value in before.items())
        assert all(np.array_equal(value, getattr(right, key)) for key, value in before.items())


@pytest.mark.parametrize(
    "pairs",
    [(), [(0, 1)], ((0, 0),), ((-1, 1),), ((0, 2),), ((True, 0),), ((0, 1), (0, 1)), ((0,),)],
)
def test_invalid_pairs_latch_the_reader(pairs):
    model, live = model_data()
    reader = CpuKinematicsSnapshot(model, [live])
    with pytest.raises(ValueError):
        reader.read_surface_pairs(pairs)
    with pytest.raises(RuntimeError, match="invalid"):
        reader.read()


@pytest.mark.parametrize("cutoff", [0, -1, True, math.nan, math.inf, 11])
def test_invalid_cutoff_latches_the_reader(cutoff):
    model, live = model_data()
    reader = CpuKinematicsSnapshot(model, [live])
    with pytest.raises(ValueError):
        reader.read_surface_pairs(((0, 1),), maximum_distance_m=cutoff)
    with pytest.raises(RuntimeError, match="invalid"):
        reader.read_surface_pairs(((0, 1),))


def test_nonfinite_geometry_latches_without_mutating_live(monkeypatch):
    model, live = model_data()
    before = live_arrays(live)
    reader = CpuKinematicsSnapshot(model, [live])
    monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *args: float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        reader.read_surface_pairs(((0, 1),))
    assert all(np.array_equal(value, getattr(live, key)) for key, value in before.items())
    with pytest.raises(RuntimeError, match="invalid"):
        reader.read()
