import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_action_observations import observe_postcontact_separation


def measured():
    return dict(
        force_n=np.zeros((300, 1, 6)),
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.7),
        ball_position_after_step_m=np.zeros((300, 1, 3)),
        navigation_command=np.zeros((300, 1, 3)),
        root_pose_xyzw_m=np.zeros((300, 1, 7)),
        root_velocity_world=np.zeros((300, 1, 6)),
        ball_position_before_step_m=np.zeros((300, 1, 3)),
        ball_linear_velocity_before_step_m_s=np.zeros((300, 1, 3)),
    )


def test_event_snapshots_use_next_pre_step_and_actual_world_velocity():
    trace = measured()
    trace["force_n"][72, 0, 0] = 10.0
    trace["force_n"][206, 0, 4] = 20.0
    trace["root_pose_xyzw_m"][73, 0, :3] = [1.0, 2.0, 0.7]
    trace["ball_position_before_step_m"][73, 0] = [1.3, 1.8, 0.11]
    trace["root_velocity_world"][73, 0] = [1.0, 0.5, 0.1, 99, 99, 99]
    trace["ball_linear_velocity_before_step_m_s"][73, 0] = [2.0, 0.0, 0.2]
    trace["navigation_command"][:] = 1.4
    row = observe_postcontact_separation(trace)
    snap = row["first_post_foot_contact"]
    assert snap["frame"] == 73
    assert snap["ball_minus_pelvis_world_position_m"] == pytest.approx([0.3, -0.2, -0.59])
    assert snap["ball_minus_root_world_linear_velocity_m_s"] == pytest.approx([1, -0.5, 0.1])
    assert row["ten_frames_after_foot_contact"]["frame"] == 82
    assert row["pre_delayed_nonfoot_contact"]["frame"] == 206
    assert row["causation_proven"] is False
    assert row["new_physics_executions"] == 0


@pytest.mark.parametrize("frame", [None, 299])
def test_missing_followthrough_is_never_filled(frame):
    trace = measured()
    if frame is not None:
        trace["force_n"][frame, 0, 0] = 10.0
    row = observe_postcontact_separation(trace)
    assert row["first_post_foot_contact"] is None
    assert row["ten_frames_after_foot_contact"] is None
    assert row["pre_delayed_nonfoot_contact"] is None


@pytest.mark.parametrize("nonfoot", [20, 40])
def test_nonfoot_at_or_before_foot_not_reported_as_delayed(nonfoot):
    trace = measured()
    trace["force_n"][40, 0, 0] = 5.0
    trace["force_n"][nonfoot, 0, 4] = 5.0
    assert observe_postcontact_separation(trace)["pre_delayed_nonfoot_contact"] is None


@pytest.mark.parametrize(
    "name",
    [
        "root_pose_xyzw_m",
        "root_velocity_world",
        "ball_position_before_step_m",
        "ball_linear_velocity_before_step_m_s",
    ],
)
def test_incomplete_or_nonfinite_separation_trace_rejected(name):
    trace = measured()
    trace[name][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="separation"):
        observe_postcontact_separation(trace)
    trace = measured()
    trace[name] = trace[name][:-1]
    with pytest.raises(ValueError, match="separation"):
        observe_postcontact_separation(trace)
