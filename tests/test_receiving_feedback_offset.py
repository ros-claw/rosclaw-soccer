import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.receiving_feedback_offset import (
    MeasuredContactOffset,
    bounded_contact_offset,
)
from rosclaw_soccer.skills.team.motor_option import (
    TeamMotorFoundation,
    TeamMotorObservation,
    TeamMotorPhysicsObservation,
    TeamMotorTarget,
)


def test_offset_respects_original_envelope_and_does_not_mutate_inputs():
    torque, kp, previous = np.full(29, 20.0), np.full(29, 10.0), np.zeros(29)
    for _ in range(100):
        next_value = bounded_contact_offset(torque, kp, previous)
        assert np.max(abs(next_value - previous)) <= 0.020000001
        assert np.max(abs(next_value)) <= 0.100000001
        previous = next_value
    np.testing.assert_allclose(previous, 0.1)
    np.testing.assert_array_equal(torque, np.full(29, 20.0))
    assert np.isfinite(bounded_contact_offset(torque, np.zeros(29), previous)).all()
    np.testing.assert_allclose(bounded_contact_offset(torque, np.zeros(29), previous), 0.08)


@pytest.mark.parametrize(
    "which,value", [(0, float("nan")), (0, 21.0), (1, -1.0), (1, 301.0), (2, 0.11)]
)
def test_invalid_offsets_rejected(which, value):
    values = [np.zeros(29), np.ones(29), np.zeros(29)]
    values[which][0] = value
    with pytest.raises(ValueError):
        bounded_contact_offset(*values)


@pytest.fixture
def option():
    root = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if not root:
        pytest.skip("qualified external G1 FK assets required")
    return MeasuredContactOffset(Path(root), "red.playmaker", start_frame=0)


def observation(frame=0):
    target = TeamMotorTarget((0.0,) * 29, (100.0,) * 29, (5.0,) * 29)
    foundation = TeamMotorFoundation(
        "red.playmaker", frame, target, (0.0,) * 29, "sha256:" + "1" * 64, "sha256:" + "2" * 64
    )
    q = (
        (0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0)
        + (0.0,) * 29
        + (0.3, -0.08, 0.115, 1.0, 0.0, 0.0, 0.0)
    )
    v = (0.0,) * 35 + (-0.75, 0.0, 0.0, 0.0, 0.0, 0.0)
    return TeamMotorObservation(
        "red.playmaker",
        frame,
        frame * 0.02,
        "other",
        True,
        q,
        v,
        (0.0, 0.0, 0.0),
        foundation=foundation,
    )


def test_private_fk_feedback_is_bounded_and_original_observation_unchanged(option):
    state = observation()
    before = state.qpos
    target = option.propose(state)
    assert target is not None and state.qpos == before
    assert np.max(abs(np.asarray(target.target_rad))) <= 0.020000001
    assert target.kp == state.foundation.target.kp
    assert option.records[0]["active"]
    with pytest.raises(ValueError, match="missing or stale"):
        option.propose(observation(1))
    with pytest.raises(ValueError, match="latched"):
        option.propose(observation(1))


def test_actual_completed_observation_is_required_and_policy_cannot_swap(option):
    state = observation()
    option.propose(state)
    option.observe_physics(
        TeamMotorPhysicsObservation(
            0.02, state.qpos, state.qvel, True, 0.0, 0.0, "red.playmaker", True, ()
        )
    )
    next_state = observation(1)
    option.propose(next_state)
    option.observe_physics(
        TeamMotorPhysicsObservation(
            0.04, state.qpos, state.qvel, True, 0.0, 0.0, "red.playmaker", True, ()
        )
    )
    changed = observation(2)
    changed = replace(
        changed, foundation=replace(changed.foundation, policy_hash="sha256:" + "3" * 64)
    )
    with pytest.raises(ValueError, match="identity changed"):
        option.propose(changed)


def test_foreign_contact_latches_without_inventing_contact(option):
    state = observation()
    with pytest.raises(ValueError, match="attributed"):
        option.observe_physics(
            TeamMotorPhysicsObservation(
                0.002, state.qpos, state.qvel, True, 0.0, 0.0, "blue.playmaker", True, ()
            )
        )
    with pytest.raises(ValueError, match="latched"):
        option.propose(state)


def test_disabled_option_is_exact_foundation_target(option):
    root = Path(os.environ["ROSCLAW_G1_ASSET_ROOT"])
    disabled = MeasuredContactOffset(root, "red.playmaker", start_frame=0, enabled=False)
    state = observation()
    assert disabled.propose(state) == state.foundation.target
    assert disabled.contract_hash != option.contract_hash


def test_nonunit_pose_rejected_before_private_fk(option):
    state = observation()
    q = list(state.qpos)
    q[3] = 0.0
    with pytest.raises(ValueError):
        option.propose(replace(state, qpos=tuple(q)))
