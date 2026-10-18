"""Kinematic local derivatives are not future physical success evidence."""

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.convex_surface_distance import ConvexSurfaceDistance
from rosclaw_soccer.physics.mesh_sphere_contact_state import measure_mesh_sphere_contact_state


def fixture():
    vertices = np.array([(x, y, z) for x in (-0.3, 0.3) for y in (-0.2, 0.2) for z in (-0.1, 0.1)])
    asset = " ".join(str(v) for v in vertices.ravel())
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><asset><mesh name="foot" vertex="{asset}"/></asset><worldbody>'
        '<body pos="0 0 .5"><joint type="hinge" axis="0 0 1"/>'
        '<geom name="mesh" type="mesh" mesh="foot"/></body>'
        '<body pos=".8 .3 .55"><freejoint/><geom name="ball" type="sphere" size=".1"/>'
        "</body></worldbody></mujoco>"
    )
    g, s = int(model.geom("mesh").id), int(model.geom("ball").id)
    mesh = int(model.geom_dataid[g])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    compiled = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
    # The compiled rectangular box has axis-aligned principal axes. Construct
    # its six analytic faces so ordinary tests need no optional SciPy install.
    lo, hi = compiled.min(axis=0), compiled.max(axis=0)
    triangles, planes = [], []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for bound, sign in ((lo[axis], -1), (hi[axis], 1)):
            corners = []
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                point = np.zeros(3)
                point[axis] = bound
                point[other[0]] = (lo, hi)[u][other[0]]
                point[other[1]] = (lo, hi)[v][other[1]]
                corners.append(point)
            triangles.extend([[corners[i] for i in face] for face in ((0, 1, 2), (0, 2, 3))])
            plane = np.zeros(4)
            plane[axis], plane[3] = sign, -sign * bound
            planes.append(plane)
    surface = ConvexSurfaceDistance(np.asarray(triangles), np.asarray(planes))
    return model, mujoco.MjData(model), g, s, surface


def forward(model, data):
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


@pytest.mark.parametrize("angle", [0.0, 0.3, -0.4])
def test_all_dofs_gap_derivative_and_velocity(angle):
    model, data, g, s, hull = fixture()
    data.qpos[0] = angle
    data.qvel[:] = np.linspace(-0.4, 0.7, model.nv)
    forward(model, data)
    q, v = data.qpos.copy(), data.qvel.copy()
    state = measure_mesh_sphere_contact_state(model, data, g, s, hull)
    numeric = []
    epsilon = 1e-6
    for dof in range(model.nv):
        values = []
        for sign in (-1, 1):
            probe = mujoco.MjData(model)
            probe.qpos[:] = q
            tangent = np.zeros(model.nv)
            tangent[dof] = sign
            mujoco.mj_integratePos(model, probe.qpos, tangent, epsilon)
            forward(model, probe)
            values.append(measure_mesh_sphere_contact_state(model, probe, g, s, hull).gap_m)
        numeric.append((values[1] - values[0]) / (2 * epsilon))
    np.testing.assert_allclose(state.gap_jacobian, numeric, atol=2e-8, rtol=0)
    assert state.gap_rate_m_s == pytest.approx(np.dot(numeric, v), abs=2e-8)
    assert state.gap_rate_m_s == pytest.approx(
        np.dot(state.outward_normal_world, state.relative_point_velocity_world_m_s)
    )
    # A rotating offset surface is not represented by its body-origin Jacobian.
    assert abs(state.gap_jacobian[0]) > 0.01
    np.testing.assert_array_equal(data.qpos, q)
    np.testing.assert_array_equal(data.qvel, v)
    assert data.time == 0


def test_reject_wrong_geometry_and_nonfinite_velocity():
    model, data, g, s, hull = fixture()
    forward(model, data)
    with pytest.raises(ValueError, match="explicit compiled"):
        measure_mesh_sphere_contact_state(model, data, s, g, hull)
    data.qvel[0] = np.nan
    with pytest.raises(ValueError, match="finite current"):
        measure_mesh_sphere_contact_state(model, data, g, s, hull)


def test_sphere_spin_does_not_change_normal_gap_rate():
    model, data, g, s, hull = fixture()
    forward(model, data)
    baseline = measure_mesh_sphere_contact_state(model, data, g, s, hull)
    # Last three qvel entries are the free sphere's angular velocity.
    data.qvel[-3:] = [20, -15, 40]
    spinning = measure_mesh_sphere_contact_state(model, data, g, s, hull)
    assert spinning.gap_rate_m_s == baseline.gap_rate_m_s == 0
    assert spinning.gap_m == baseline.gap_m
