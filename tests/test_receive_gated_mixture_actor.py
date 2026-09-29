"""Measured-state, bounded SIM_ONLY mixture preserves its zero-gate parent."""

from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.team_receive_gated_mixture_actor import TeamReceiveGatedMixtureActor
from rosclaw_soccer.rsi.team_receive_phase_actor import TeamReceivePhaseActor
from rosclaw_soccer.skills.team.navigation_option import NavigationSlot
from tests.test_navigation_option import observation


def _observation(frame: int):
    return replace(
        observation(frame),
        agent_id="red.finisher",
        role="finisher",
        intent="receive",
        committed_receiver=True,
        body_pose=(1.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0),
        ball_position=(0.5, -0.1, 0.115),
        ball_velocity=(0.8, 0.2, 0.0),
        effector_positions=(("left_foot", 1.0, 0.1, 0.1), ("right_foot", 0.9, -0.15, 0.1)),
        effector_velocities=(("left_foot", 0.0, 0.0, 0.0), ("right_foot", 0.0, -0.3, 0.0)),
        body_angular_velocity=(0.0, 0.0, 0.0),
        joint_positions_rad=(0.0,) * 29,
        joint_velocities_radps=(0.0,) * 9 + (2.0,) + (0.0,) * 19,
    )


def test_zero_gate_is_bitwise_parent_and_active_gate_is_bounded():
    retained = (
        -0.1853410496026612,
        -0.2120161425477506,
        -0.10038265224958914,
        0.1304744032562346,
        0.16730409297832222,
    )
    foundation = "sha256:" + "a" * 64
    config = "sha256:" + "b" * 64
    parent = NavigationSlot(TeamReceivePhaseActor("red.finisher", foundation, config, retained))
    zero = TeamReceiveGatedMixtureActor(
        "red.finisher",
        foundation,
        config,
        retained,
        alternative_weights=(0.5,) * 5,
        gate_weights=(-12.0, 0.0, 0.0, 0.0),
    )
    zero_slot = NavigationSlot(zero)
    active = TeamReceiveGatedMixtureActor(
        "red.finisher",
        foundation,
        config,
        retained,
        alternative_weights=(0.5,) * 5,
        gate_weights=(8.0, 0.0, 0.0, 0.0),
    )
    active_slot = NavigationSlot(active)
    for frame in range(3):
        value = _observation(frame)
        assert zero_slot.propose(value) == parent.propose(value)
        delta = active_slot.propose(value)
        assert (delta[0] ** 2 + delta[1] ** 2) ** 0.5 <= 0.25
    assert zero.maximum_gate == 0.0 and zero.alternative_active_frames == 0
    assert active.maximum_gate == 1.0 and active.alternative_active_frames == 3
    assert not zero_slot.faulted and not active_slot.faulted


def test_mixture_rejects_unbounded_or_nonfinite_parameters():
    with pytest.raises(ValueError):
        TeamReceiveGatedMixtureActor(
            "red.finisher",
            "sha256:" + "a" * 64,
            "sha256:" + "b" * 64,
            alternative_weights=(float("nan"),) * 5,
        )
    with pytest.raises(ValueError):
        TeamReceiveGatedMixtureActor(
            "red.finisher",
            "sha256:" + "a" * 64,
            "sha256:" + "b" * 64,
            gate_weights=(13.0, 0.0, 0.0, 0.0),
        )
