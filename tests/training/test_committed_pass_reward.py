"""Immutable target attribution must fail closed and leave legacy untouched."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.training.committed_pass_reward import (
    committed_directed_physical_rewards,
    committed_pass_shaping_delta,
)

IDS = (
    "blue.defender",
    "blue.finisher",
    "blue.goalkeeper",
    "blue.playmaker",
    "red.defender",
    "red.finisher",
    "red.goalkeeper",
    "red.playmaker",
)


def example():
    trace = {
        "time": np.array([0.02, 0.04]),
        "pass_entry_time_sec": np.array([0.0, 0.02]),
        "pass_entry_source_code": np.array([3, 0]),
        "pass_entry_receiver_code": np.array([1, 0]),
        "pass_entry_created_sec": np.array([0.0, -1.0]),
        "pass_entry_target_valid": np.array([True, False]),
        "pass_entry_target_xy_m": np.array([[1.0, 0.0], [0.0, 0.0]]),
        "pass_entry_ball_velocity_xy_mps": np.zeros((2, 2)),
        "pass_entry_ball_xy_m": np.zeros((2, 2)),
        "team_native_pre_control_qpos": np.zeros((2, 8, 43)),
        "ball_pose": np.zeros((2, 7)),
        "ball_control_entry_velocity": np.zeros((2, 6)),
        "ball_velocity": np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6]),
        "pass_source_agent_code": np.array([3, 0]),
        "pass_target_agent_code": np.array([1, 0]),
        "ball_contact_agent_code": np.array([3, 0]),
        "ball_contact_effector_code": np.array([1, 0]),
        "ball_contact_force_n": np.array([10.0, 0.0]),
        "ball_nonfoot_contact_force_n": np.zeros(2),
    }
    for agent in IDS:
        trace[agent.replace(".", "_") + "_intent_code"] = np.zeros(2, dtype=int)
        trace[agent.replace(".", "_") + "_target_position"] = np.array([[0.0, 1.0, 0.0]] * 2)
    return trace


def test_direction_uses_fixed_committed_target_only_for_sender():
    trace = example()
    snapshot = copy.deepcopy(trace)
    delta = committed_pass_shaping_delta(trace, IDS)
    assert delta.shape == (2, 8)
    assert delta[0, 2] == pytest.approx(0.08)
    delta[0, 2] = 0
    assert not delta.any()
    for name in trace:
        np.testing.assert_array_equal(trace[name], snapshot[name])


@pytest.mark.parametrize(
    "key,value",
    [
        ("pass_entry_source_code", 4),
        ("pass_entry_receiver_code", 2),
        ("pass_entry_created_sec", 0.01),
        ("pass_entry_time_sec", 0.01),
        ("pass_entry_target_valid", False),
    ],
)
def test_inconsistent_entry_fails_closed(key, value):
    trace = example()
    trace[key][0] = value
    with pytest.raises(ValueError):
        committed_pass_shaping_delta(trace, IDS)


def test_missing_telemetry_fails_closed():
    trace = example()
    del trace["pass_entry_target_xy_m"]
    with pytest.raises(ValueError, match="missing"):
        committed_pass_shaping_delta(trace, IDS)


def test_mismatched_entry_velocity_fails_closed():
    trace = example()
    trace["pass_entry_ball_velocity_xy_mps"][0, 0] = 0.1
    with pytest.raises(ValueError, match="velocity|entry"):
        committed_pass_shaping_delta(trace, IDS)


def test_mismatched_entry_position_fails_closed():
    trace = example()
    trace["pass_entry_ball_xy_m"][0, 0] = 0.1
    with pytest.raises(ValueError, match="position"):
        committed_pass_shaping_delta(trace, IDS)


def test_wrapper_preserves_legacy_reward_and_applies_only_correction(monkeypatch):
    from rosclaw_soccer.training import near_ball_residual_ppo

    legacy = np.full((2, 8), 0.5)

    def fake_legacy(trace, ids, *, reward_shaping, gamma):
        assert reward_shaping == "directed_pass_credit_v1"
        assert ids == IDS
        assert gamma == 0.9
        return legacy.copy()

    monkeypatch.setattr(near_ball_residual_ppo, "physical_rewards", fake_legacy)
    corrected = committed_directed_physical_rewards(example(), IDS, gamma=0.9)
    assert corrected[0, 2] == pytest.approx(0.58)
    corrected[0, 2] = 0.5
    np.testing.assert_array_equal(corrected, legacy)
