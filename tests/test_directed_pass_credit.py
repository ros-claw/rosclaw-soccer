"""The opt-in reward must not manufacture a directed physical pass."""

import numpy as np
import pytest
from test_s224_launch_credit import delayed_pass_trace

from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.training.near_ball_residual_ppo import physical_rewards, train


def fixture():
    ids, trace = delayed_pass_trace()
    n = len(trace["time"])
    trace["time"] = np.arange(n) * 0.02
    for key in ("pass_source_agent_code", "pass_target_agent_code"):
        trace[key] = trace[key].astype(int)
    trace["pass_feedback_launch_relative"] = np.ones(n, dtype=bool)
    trace["ball_control_entry_velocity"] = np.zeros((n, 6))
    trace["ball_velocity"][:, 0] = 0.5
    for agent in ids:
        key = agent.replace(".", "_")
        trace[key + "_joint_safety_margin_rad"] = np.full((n, 29), 0.5)
        trace[key + "_intent_code"] = np.full(n, list(TacticalIntent).index(TacticalIntent.HOLD))
        trace[key + "_pelvis_pose"] = np.tile([0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0], (n, 1))
    trace["red_finisher_pelvis_pose"][:, 0] = 2.0
    return ids, trace


def reward_difference(ids, trace):
    return physical_rewards(
        trace, ids, reward_shaping="directed_pass_credit_v1"
    ) - physical_rewards(trace, ids, reward_shaping="pass_impulse_alignment_v1")


def test_matching_commitment_and_directed_velocity_preserve_credit():
    ids, trace = fixture()
    np.testing.assert_array_equal(reward_difference(ids, trace), np.zeros((61, 8)))


@pytest.mark.parametrize("failure", ["wrong_direction", "missing_commitment"])
def test_reception_without_directed_committed_launch_earns_no_pass_bonus(failure):
    ids, trace = fixture()
    if failure == "wrong_direction":
        trace["ball_velocity"][20, 0] = -0.5
    else:
        trace["pass_source_agent_code"][19] = 0
    expected = np.zeros((61, 8))
    expected[45, [5, 7]] = -1.0
    np.testing.assert_allclose(reward_difference(ids, trace), expected, atol=1e-12)


def test_strict_telemetry_required_and_bad_pose_fails_closed():
    ids, trace = fixture()
    del trace["pass_feedback_launch_relative"]
    with pytest.raises(ValueError, match="strict physical handoff"):
        physical_rewards(trace, ids, reward_shaping="directed_pass_credit_v1")
    trace["pass_feedback_launch_relative"] = np.ones(61, dtype=bool)
    trace["red_finisher_pelvis_pose"][19, 0] = np.nan
    with pytest.raises(ValueError, match="finite receiver poses"):
        physical_rewards(trace, ids, reward_shaping="directed_pass_credit_v1")


def test_training_rejects_missing_handoff_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="bounded online training"):
        train(
            assets=tmp_path,
            output=tmp_path / "out",
            iterations=1,
            duration=5.0,
            reward_shaping="directed_pass_credit_v1",
        )
    assert not (tmp_path / "out").exists()
