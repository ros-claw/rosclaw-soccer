"""Synthetic clocks exercise the credit contract, not a physical pass claim."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.training import committed_team_capture_credit as module

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


def sample():
    n = 130
    trace = {
        "time": np.arange(1, n + 1, dtype=float) * 0.02,
        "pass_entry_time_sec": np.arange(n, dtype=float) * 0.02,
        "pass_entry_source_code": np.zeros(n, dtype=int),
        "pass_entry_receiver_code": np.zeros(n, dtype=int),
        "pass_entry_created_sec": np.full(n, -1.0),
        "pass_entry_target_valid": np.zeros(n, dtype=bool),
        "pass_entry_target_xy_m": np.zeros((n, 2)),
        "pass_entry_ball_xy_m": np.zeros((n, 2)),
        "pass_entry_ball_velocity_xy_mps": np.zeros((n, 2)),
        "team_native_pre_control_qpos": np.zeros((n, 8, 43)),
        "ball_pose": np.zeros((n, 7)),
        "ball_control_entry_velocity": np.zeros((n, 6)),
        "ball_contact_agent_code": np.zeros(n, dtype=int),
        "ball_contact_effector_code": np.zeros(n, dtype=int),
        "ball_contact_force_n": np.zeros(n),
        "ball_nonfoot_contact_force_n": np.zeros(n),
        "pass_source_agent_code": np.zeros(n, dtype=int),
        "pass_target_agent_code": np.zeros(n, dtype=int),
    }
    f = 10
    trace["pass_entry_target_valid"][f] = True
    trace["pass_entry_source_code"][f] = trace["pass_source_agent_code"][f] = 3
    trace["pass_entry_receiver_code"][f] = trace["pass_target_agent_code"][f] = 1
    trace["pass_entry_created_sec"][f] = 0.1
    trace["pass_entry_target_xy_m"][f] = [1.0, 0.0]
    trace["ball_contact_agent_code"][f] = 3
    trace["ball_contact_effector_code"][f] = 1
    trace["ball_contact_force_n"][f] = 3
    trace["ball_contact_agent_code"][20] = 1
    trace["ball_contact_effector_code"][20] = 2
    trace["ball_contact_force_n"][20] = 3
    trace["ball_pose"][20, 0] = 1
    return trace


@pytest.fixture(autouse=True)
def isolated_physical_scorer(monkeypatch):
    monkeypatch.setattr(
        module,
        "sampled_team_episode_cut",
        lambda *a, **k: SimpleNamespace(first_terminal_frame=None),
    )
    monkeypatch.setattr(
        module,
        "explain_receiving_window",
        lambda *a, **k: {"controlled_reception": True, "post_contact_observation_sec": 1.96},
    )


def test_credit_only_after_controlled_window_and_for_both_teammates():
    trace = sample()
    before = copy.deepcopy(trace)
    credit, events = module.committed_team_capture_credit(trace, IDS)
    assert events[0]["outcome"] == "CONTROLLED_SAMPLED" and events[0]["credit_frame"] == 118
    assert credit[118, 2] == credit[118, 0] == 0.1 and credit.sum() == pytest.approx(0.2)
    assert not credit[:118].any()
    assert all(np.array_equal(trace[k], before[k]) for k in trace)


def test_interference_and_unknown_contact_do_not_reward():
    for who in (0, 5):
        trace = sample()
        trace["ball_contact_agent_code"][15] = who
        trace["ball_contact_effector_code"][15] = 1
        trace["ball_contact_force_n"][15] = 3
        credit, events = module.committed_team_capture_credit(trace, IDS)
        assert events[0]["outcome"] == "OTHER_CONTACT" and not credit.any()


def test_incomplete_tail_is_censored(monkeypatch):
    monkeypatch.setattr(
        module, "sampled_team_episode_cut", lambda *a, **k: SimpleNamespace(first_terminal_frame=50)
    )
    credit, events = module.committed_team_capture_credit(sample(), IDS)
    assert events[0]["outcome"] == "CENSORED_CAPTURE" and not credit.any()


def test_no_receiver_before_early_terminal_is_censored(monkeypatch):
    monkeypatch.setattr(
        module, "sampled_team_episode_cut", lambda *a, **k: SimpleNamespace(first_terminal_frame=15)
    )
    credit, events = module.committed_team_capture_credit(sample(), IDS)
    assert events[0]["outcome"] == "CENSORED_FLIGHT" and not credit.any()


@pytest.mark.parametrize("receiver_x", [0.1, 2.0])
def test_short_or_off_target_touch_never_receives_team_credit(receiver_x):
    trace = sample()
    trace["ball_pose"][20, 0] = receiver_x
    credit, events = module.committed_team_capture_credit(trace, IDS)
    assert events[0]["outcome"] == "GEOMETRY_REJECTED" and not credit.any()


def test_uncontrolled_receiver_is_not_rewarded(monkeypatch):
    monkeypatch.setattr(
        module, "explain_receiving_window", lambda *a, **k: {"controlled_reception": False}
    )
    credit, events = module.committed_team_capture_credit(sample(), IDS)
    assert events[0]["outcome"] == "UNCONTROLLED" and not credit.any()


@pytest.mark.parametrize(
    "mutation", ["missing", "future", "other_team", "mismatched_ball", "wrong_receiver"]
)
def test_missing_or_inconsistent_evidence_fails_closed(mutation):
    trace = sample()
    if mutation == "missing":
        del trace["pass_entry_target_valid"]
    elif mutation == "future":
        trace["pass_entry_created_sec"][10] = 10
    elif mutation == "other_team":
        trace["pass_entry_receiver_code"][10] = 5
    elif mutation == "mismatched_ball":
        trace["pass_entry_ball_xy_m"][10, 0] = 1
    else:
        trace["pass_target_agent_code"][10] = 2
    with pytest.raises(ValueError):
        module.committed_team_capture_credit(trace, IDS)


def test_one_commitment_never_double_counts():
    trace = sample()
    for key in (
        "pass_entry_source_code",
        "pass_entry_receiver_code",
        "pass_entry_created_sec",
        "pass_entry_target_valid",
        "pass_entry_target_xy_m",
        "pass_source_agent_code",
        "pass_target_agent_code",
        "ball_contact_agent_code",
        "ball_contact_effector_code",
        "ball_contact_force_n",
    ):
        trace[key][11] = trace[key][10]
    credit, events = module.committed_team_capture_credit(trace, IDS)
    assert len(events) == 1 and credit.sum() == pytest.approx(0.2)


def test_same_handoff_cannot_change_committed_target():
    trace = sample()
    for key in (
        "pass_entry_source_code",
        "pass_entry_receiver_code",
        "pass_entry_created_sec",
        "pass_entry_target_valid",
    ):
        trace[key][11] = trace[key][10]
    trace["pass_entry_target_xy_m"][11] = [2.0, 0.0]
    with pytest.raises(ValueError, match="target changed"):
        module.committed_team_capture_credit(trace, IDS)


def test_control_score_must_describe_the_receiver_touch(monkeypatch):
    monkeypatch.setattr(
        module,
        "explain_receiving_window",
        lambda *a, **k: {"controlled_reception": True, "post_contact_observation_sec": 1.94},
    )
    with pytest.raises(ValueError, match="not bound"):
        module.committed_team_capture_credit(sample(), IDS)
