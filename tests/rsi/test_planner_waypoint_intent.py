"""Pure model-input fixtures; no planner inference or physical success."""

from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.rsi.planner_waypoint_intent import (
    RootWaypointIntent,
    apply_waypoint_intent,
    past_context_at_30hz,
    validate_planner_output,
)


def fixture():
    c = np.zeros((1, 4, 36), dtype=np.float32)
    c[:, :, 3] = 1.0
    c[:, :, 2] = 0.793
    feed = dict(
        context_mujoco_qpos=c,
        has_specific_target=np.zeros((1, 1), dtype=np.int64),
        specific_target_positions=np.zeros((1, 4, 3), dtype=np.float32),
        specific_target_headings=np.zeros((1, 4), dtype=np.float32),
        random_seed=np.array([30300], dtype=np.int64),
    )
    i = RootWaypointIntent((0.03, -0.04, 0.793), 0.0, "sha256:" + "a" * 64, "sha256:" + "b" * 64)
    return feed, i


def test_goal_is_owned_input_change_not_body_motion_or_default_activation():
    f, i = fixture()
    sham = apply_waypoint_intent(f, i, activate=False)
    active = apply_waypoint_intent(f, i, activate=True)
    assert not f["has_specific_target"].any() and active["has_specific_target"].item() == 1
    assert not sham["has_specific_target"].any()
    np.testing.assert_array_equal(
        active["specific_target_positions"], sham["specific_target_positions"]
    )
    for k in f:
        assert not np.shares_memory(active[k], f[k])
    active["context_mujoco_qpos"][0, 0, 0] = 5.0
    assert f["context_mujoco_qpos"][0, 0, 0] == 0
    assert not i.report()["dynamically_qualified"] and not i.report()["hardware_authorized"]


@pytest.mark.parametrize(
    "change", ["nan", "far", "height", "yaw", "hash", "list", "flag", "context", "dtype"]
)
def test_bad_goal_or_tensor_fails_before_nn(change):
    f, i = fixture()
    with pytest.raises(ValueError):
        if change == "nan":
            i = replace(i, world_target_xyz_m=(float("nan"), 0.0, 0.793))
        elif change == "far":
            i = replace(i, world_target_xyz_m=(1.0, 0.0, 0.793))
        elif change == "height":
            i = replace(i, world_target_xyz_m=(0.0, 0.0, 0.0))
        elif change == "yaw":
            i = replace(i, world_target_yaw_rad=4.0)
        elif change == "hash":
            i = replace(i, goal_basis_hash="none")
        elif change == "list":
            i = replace(i, world_target_xyz_m=[0.03, -0.04, 0.793])
        elif change == "context":
            f["context_mujoco_qpos"][0, 0, 3] = 0.1
        elif change == "dtype":
            f["specific_target_positions"] = f["specific_target_positions"].astype(np.float64)
        apply_waypoint_intent(f, i, activate=1 if change == "flag" else True)


def test_context_uses_only_past_and_nn_quaternion_prep_does_not_change_actual_history():
    q = np.zeros((100, 43))
    q[:, 0] = np.arange(100)
    q[:, 3] = 1.0
    q[57:59, 3] = -1.0
    q[61:] = np.nan
    before = q.copy()
    c = past_context_at_30hz(q, 60)
    np.testing.assert_allclose(c[0, :, 0], [55.0, 56.0 + 2.0 / 3.0, 58.0 + 1.0 / 3.0, 60.0])
    np.testing.assert_allclose(np.linalg.norm(c[0, :, 3:7], axis=1), 1.0)
    np.testing.assert_array_equal(q, before)
    with pytest.raises(ValueError):
        past_context_at_30hz(q, 61)


def test_native_joint_violation_is_reported_not_clipped_or_called_dynamic_success():
    out = np.zeros((1, 16, 36), dtype=np.float32)
    out[:, :, 3] = 1.0
    out[:, :, 7] = 4.0
    saved = out.copy()
    r = validate_planner_output(out, np.array([16]), np.tile([-3.2, 3.2], (29, 1)))
    assert r["maximum_native_joint_limit_violation_rad"] == pytest.approx(0.8)
    assert not r["reference_within_native_joint_limits"] and not r["dynamic_feasibility_proven"]
    np.testing.assert_array_equal(out, saved)
    out[0, 0, 3] = 0.0
    with pytest.raises(ValueError):
        validate_planner_output(out, np.array([16]), np.tile([-3.2, 3.2], (29, 1)))
