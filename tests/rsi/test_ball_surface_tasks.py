import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from test_shared_native_motor_world import source_xml

from rosclaw_soccer.rsi.ball_surface_tasks import measure_ball_surfaces
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene


def fixture():
    root = ET.fromstring(source_xml())
    for i, side, part in (
        (3, "left", "knee"),
        (5, "left", "ankle_roll"),
        (9, "right", "knee"),
        (11, "right", "ankle_roll"),
    ):
        body = root.find(f'.//body[@name="link{i}"]')
        body.set("name", side + "_" + part + "_link")
        geom = body.find("geom")
        geom.set("contype", "1")
        geom.set("pos", ".3 .2 -.4")
        geom.set("name", side + ("_shin" if part == "knee" else "_foot"))
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(ET.tostring(root, encoding="unicode"), prefixes=("a/",))
    )
    data = mujoco.MjData(model)
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="ball")
    return model, data, layout


def test_separated_surface_gap_gradient_matches_actual_geometry_difference():
    m, d, layout = fixture()
    before = d.qpos.copy(), d.xpos.copy()
    row = measure_ball_surfaces(m, d, layout)
    for name, surface in row["surfaces"].items():
        assert surface["smooth_separated"]
        numerical = np.zeros(29)
        for j, qi in enumerate(layout.qi):
            distances = []
            for sign in (-1, 1):
                scratch = mujoco.MjData(m)
                scratch.qpos[:] = d.qpos
                scratch.qpos[qi] += sign * 1e-6
                distances.append(
                    measure_ball_surfaces(m, scratch, layout)["surfaces"][name]["gap_m"]
                )
            numerical[j] = (distances[1] - distances[0]) / 2e-6
        np.testing.assert_allclose(surface["gap_jacobian_hg"], numerical, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(d.qpos, before[0])
    np.testing.assert_array_equal(d.xpos, before[1])
    assert d.time == 0


def test_distance_sentinel_is_not_fabricated_surface():
    m, d, layout = fixture()
    d.qpos[layout.bq : layout.bq + 3] = [10, 0, 0]
    assert all(not r["available"] for r in measure_ball_surfaces(m, d, layout)["surfaces"].values())
