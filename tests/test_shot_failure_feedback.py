import numpy as np
import pytest

from rosclaw_soccer.growth.shot_failure_feedback import diagnose_shot_options

IDS = tuple(
    f"{team}.{role}"
    for team in ("blue", "red")
    for role in ("defender", "finisher", "goalkeeper", "playmaker")
)


def trace_fixture():
    n = 8
    trace = {
        key: np.zeros(n, dtype=np.int64)
        for key in (
            "option_agent_code",
            "ball_contact_agent_code",
            "ball_contact_effector_code",
            "ball_nonfoot_contact_agent_code",
        )
    }
    trace.update(
        time=np.arange(n) * 0.02,
        ball_contact_force_n=np.zeros(n),
        ball_nonfoot_contact_force_n=np.zeros(n),
        option_target_position_m=np.zeros((n, 3)),
        ball_pose=np.column_stack((np.arange(n) * 0.2, np.zeros(n), np.full(n, 0.3))),
    )
    trace["option_agent_code"][:4] = 6
    trace["option_target_position_m"][:4] = [1.0, 0.0, 0.3]
    trace["ball_contact_agent_code"][1] = 6
    trace["ball_contact_effector_code"][1] = 2
    trace["ball_contact_force_n"][1] = 10.0
    return trace


def test_concurrent_opponent_contact_is_not_claimed_as_clean_foot_precision():
    trace = trace_fixture()
    trace["ball_nonfoot_contact_agent_code"][:2] = [6, 4]
    trace["ball_nonfoot_contact_force_n"][:2] = [5.0, 8.0]
    result = diagnose_shot_options(trace, IDS, goal_planes_x_m=(-1.0, 1.0))[0]
    assert result["body_contacts"][0]["relative_to_first_foot"] == "BEFORE"
    concurrent = result["body_contacts"][1]
    assert concurrent["relative_to_first_foot"] == "SAME_CONTROL_FRAME_ORDER_UNKNOWN"
    assert not concurrent["same_team"]
    crossing = result["observed_centre_plane_crossing"]
    assert crossing["intervening_or_concurrent_contact"]
    assert crossing["target_error_m"] == 0
    assert "NOT_GOAL_ADJUDICATION" in crossing["claim"]
    assert not result["promotion_eligible"]


def test_other_foot_interruption_and_missing_source_contact_are_distinct():
    trace = trace_fixture()
    trace["ball_contact_agent_code"][3] = 2
    trace["ball_contact_effector_code"][3] = 1
    trace["ball_contact_force_n"][3] = 5.0
    result = diagnose_shot_options(trace, IDS, goal_planes_x_m=(-1.0, 1.0))[0]
    assert result["observed_centre_plane_crossing"]["intervening_or_concurrent_contact"]
    trace["ball_contact_force_n"][1] = 0.0
    result = diagnose_shot_options(trace, IDS, goal_planes_x_m=(-1.0, 1.0))[0]
    assert result["first_foot_contact_sec"] is None
    assert result["observed_centre_plane_crossing"] is None


def test_waypoints_and_changed_or_nonfinite_targets_are_not_silently_scored():
    trace = trace_fixture()
    trace["option_target_position_m"][:4, 0] = 0.7
    assert diagnose_shot_options(trace, IDS, goal_planes_x_m=(-1.0, 1.0)) == []
    trace = trace_fixture()
    trace["option_target_position_m"][2, 1] = 0.1
    with pytest.raises(ValueError, match="target changed"):
        diagnose_shot_options(trace, IDS, goal_planes_x_m=(-1.0, 1.0))
    trace = trace_fixture()
    trace["ball_pose"][3, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        diagnose_shot_options(trace, IDS, goal_planes_x_m=(-1.0, 1.0))


def test_measured_crossing_error_and_zero_force_body_records():
    trace = trace_fixture()
    trace["ball_pose"][:, 1] = 0.25
    trace["ball_nonfoot_contact_agent_code"][2] = 4  # No measured force.
    result = diagnose_shot_options(trace, IDS, goal_planes_x_m=(-1.0, 1.0))[0]
    assert result["body_contacts"] == []
    assert result["observed_centre_plane_crossing"]["target_error_m"] == 0.25
    assert not result["observed_centre_plane_crossing"]["intervening_or_concurrent_contact"]
