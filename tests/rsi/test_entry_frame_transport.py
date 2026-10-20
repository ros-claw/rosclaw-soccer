"""Geometry-only invariants; no physics, dynamic admission or learned policy."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
from rosclaw_soccer.rsi.entry_frame_transport import transport_entry_frame


def call(roots, **kwargs):
    return transport_entry_frame(
        SkillEntryAffordance((0.2, 0.03), 150, 1.0, "sha256:" + "1" * 64),
        roots,
        original_source_frame=kwargs.get("original", 0),
        target_source_frame=kwargs.get("target", 1),
        source_motion_artifact_hash="sha256:" + "2" * 64,
    )


def fixture():
    r = np.zeros((2, 7))
    r[:, 2], r[:, 3] = 0.79, 1.0
    r[1, :2] = [-0.3, 0.2]
    return r


def test_point_transport_not_physical_body_move():
    r = fixture()
    before = r.tobytes()
    t = call(r)
    np.testing.assert_allclose(t.transported_ball_offset_xy_m, [0.5, -0.17], atol=1e-14)
    assert r.tobytes() == before
    assert not t.report()["dynamic_feasibility_proven"]
    assert not t.report()["changes_actual_body_or_ball_state"]
    assert t.report()["new_native_steps"] == 0


def test_global_source_frame_invariance():
    r = fixture()
    angle = 0.7
    rot = Rotation.from_euler("z", angle)
    changed = r.copy()
    changed[:, :3] = rot.apply(r[:, :3]) + [2.0, -3.0, 0.4]
    changed[:, 3:7] = (rot * Rotation.from_quat(r[:, [4, 5, 6, 3]])).as_quat()[:, [3, 0, 1, 2]]
    np.testing.assert_allclose(
        call(r).transported_ball_offset_xy_m, call(changed).transported_ball_offset_xy_m, atol=1e-12
    )


def test_source_yaw_change_rotates_point_not_only_translates_it():
    r = fixture()
    r[1, :2] = [1.0, 0.0]
    r[1, 3:7] = Rotation.from_euler("z", 90, degrees=True).as_quat()[[3, 0, 1, 2]]
    np.testing.assert_allclose(call(r).transported_ball_offset_xy_m, [0.03, 0.8], atol=1e-12)


def test_identity_keeps_original_point_exactly():
    t = call(fixture(), target=0)
    assert t.transported_ball_offset_xy_m == t.original_ball_offset_xy_m
    assert t.source_translation_in_original_heading_m == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "kind", ["nan", "quat", "dtype", "shape", "frame_bool", "frame_outside", "far"]
)
def test_invalid_or_out_of_bound_transport_fails_closed(kind):
    r = fixture()
    kw = {}
    if kind == "nan":
        r[1, 0] = np.nan
    elif kind == "quat":
        r[1, 3] = 0.5
    elif kind == "dtype":
        r = r.astype(np.float32)
    elif kind == "shape":
        r = r[:, :6]
    elif kind == "frame_bool":
        kw["target"] = True
    elif kind == "frame_outside":
        kw["target"] = 2
    else:
        r[1, 0] = 3.0
    with pytest.raises(ValueError):
        call(r, **kw)
