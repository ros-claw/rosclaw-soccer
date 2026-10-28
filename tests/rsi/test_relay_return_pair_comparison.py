import numpy as np
import pytest

from rosclaw_soccer.rsi.relay_return_pair_comparison import compare_return_pair


def trajectory(n=4):
    return {
        "qpos": np.zeros((n, 10, 9)),
        "qvel": np.zeros((n, 10, 8)),
        "torque": np.zeros((n, 10, 6)),
        "ball_forces": np.zeros((n, 10, 2, 3)),
    }


def compare(a, b, prefix=2):
    return compare_return_pair(
        a, b, prefix_controls=prefix, ball_qpos=3, return_goal=np.array([2.0, 0, 0])
    )


def test_only_post_intervention_ball_data_is_scored():
    a, b = trajectory(), trajectory()
    a["qpos"][2:, :, 3] = 0.5
    b["qpos"][2:, :, 3] = 1.0
    result = compare(a, b)
    assert result["baseline_min_ball_goal_distance_m"] == 1.5
    assert result["candidate_min_ball_goal_distance_m"] == 1.0
    assert result["distance_delta_m"] == -0.5
    assert result["goal_proximity_is_not_pass_success"] is True


@pytest.mark.parametrize("field", ["qpos", "qvel", "torque", "ball_forces"])
def test_rejects_unequal_causal_prefix(field):
    a, b = trajectory(), trajectory()
    b[field][1, 1].flat[0] = 0.1
    with pytest.raises(ValueError, match="prefix differs"):
        compare(a, b)


def test_no_handoff_has_no_goal_distance():
    result = compare(trajectory(2), trajectory(2), prefix=2)
    assert result["distance_delta_m"] is None


def test_post_handoff_receiver_contact_substeps_are_not_counted_as_kicks():
    a, b = trajectory(), trajectory()
    a["ball_forces"][2, 0, 1, 0] = 2.0
    b["ball_forces"][2, 1:3, 1, 2] = 3.0
    result = compare(a, b)
    assert result["baseline_receiver_contact_substeps"] == {
        "left_foot": 1,
        "right_foot": 0,
        "nonfoot": 0,
    }
    assert result["candidate_receiver_contact_substeps"] == {
        "left_foot": 0,
        "right_foot": 0,
        "nonfoot": 2,
    }
    assert result["baseline_first_foot_followthrough"]["status"] == "RIGHT_CENSORED"
    assert result["candidate_first_foot_followthrough"]["status"] == "NO_FOOT_CONTACT"


def test_complete_followthrough_uses_actual_substeps():
    a, b = trajectory(64), trajectory(64)
    for d in (a, b):
        d["ball_forces"][2, 0, 1, 0] = 2.0
    b["qpos"][62, 0, 3] = 1.0
    result = compare(a, b)
    assert result["candidate_first_foot_followthrough"] == {
        "status": "OBSERVED",
        "forward_60_controls_m": 1.0,
    }
    assert result["followthrough_includes_subsequent_contacts"] is True


def test_rejects_nonfinite_or_incomplete_trajectory():
    a, b = trajectory(), trajectory()
    b["qpos"][3, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compare(a, b)
    b = trajectory()
    b["torque"] = b["torque"][:2]
    with pytest.raises(ValueError, match="lengths differ"):
        compare(a, b)
