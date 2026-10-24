import numpy as np
import pytest
from rosclaw.simforge.g1_neural_torque import (
    G1TorqueControlFrame,
    G1TorqueSafetyConfig,
    G1TorqueSafetyProjector,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

from rosclaw_soccer.rsi.guarded_torque_mix import guarded_torque_mix


def frame(velocity=0.0):
    return G1TorqueControlFrame(
        joint_position=np.zeros(29),
        joint_velocity=np.full(29, velocity),
        joint_lower_limits=np.full(29, -2.0),
        joint_upper_limits=np.full(29, 2.0),
        torso_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        pelvis_position=np.array([0.0, 0.0, 0.8]),
        ball_position=np.array([1.0, 0.0, 0.11]),
        ball_velocity=np.zeros(3),
        target_y_m=0.0,
        target_z_m=0.11,
        policy_phase=0.3,
        left_contact=True,
        right_contact=True,
    )


@pytest.mark.parametrize("fraction", [0.125, 0.25])
def test_postmix_cannot_undo_rate_envelope_or_expand_effective_parent_budget(fraction):
    safety = G1TorqueSafetyConfig(torque_guard_scale=0.85, maximum_projection_ratio=1.0)
    parent, previous = np.zeros(29), np.zeros(29)
    parent[0] = 10.0
    first = G1TorqueSafetyProjector(safety).project(
        np.zeros(29), parent=parent, previous=previous, frame=frame()
    )
    assert not first.used_parent
    old_mix = parent + fraction * (first.torque - parent)
    hard = np.asarray(G1_HARD_TORQUE_LIMITS)
    assert np.any(np.abs(old_mix - previous) / hard > safety.max_delta_ratio_per_step)
    corrected = guarded_torque_mix(
        first.torque, parent, previous, frame=frame(), safety=safety, fraction=fraction
    )
    assert corrected.used_parent
    np.testing.assert_array_equal(corrected.torque, parent)


@pytest.mark.parametrize("fraction", [0.125, 0.25])
def test_feasible_mixture_stays_inside_all_final_vector_constraints(fraction):
    safety = G1TorqueSafetyConfig(torque_guard_scale=0.85, maximum_projection_ratio=1.0)
    parent = np.full(29, 0.1)
    previous = np.zeros(29)
    first = G1TorqueSafetyProjector(safety).project(
        np.zeros(29), parent=parent, previous=previous, frame=frame()
    )
    mixed = guarded_torque_mix(
        first.torque, parent, previous, frame=frame(), safety=safety, fraction=fraction
    )
    hard = np.asarray(G1_HARD_TORQUE_LIMITS)
    assert not mixed.used_parent
    assert np.all(np.abs(mixed.torque - previous) / hard <= safety.max_delta_ratio_per_step + 1e-10)
    assert np.all(
        np.abs(mixed.torque - parent) / hard
        <= fraction * safety.maximum_parent_deviation_ratio + 1e-10
    )


def test_power_correction_cannot_escape_the_narrow_effective_budget():
    hard = np.asarray(G1_HARD_TORQUE_LIMITS)
    parent = hard * 0.3
    result = guarded_torque_mix(
        parent * 0.99,
        parent,
        parent.copy(),
        frame=frame(100.0),
        safety=G1TorqueSafetyConfig(torque_guard_scale=0.85, maximum_projection_ratio=1.0),
        fraction=0.125,
    )
    assert result.used_parent
    np.testing.assert_array_equal(result.torque, parent)


@pytest.mark.parametrize("fraction", [0.0, 1.0, np.nan, np.inf, -0.1])
def test_only_explicit_partial_composition(fraction):
    with pytest.raises(ValueError):
        guarded_torque_mix(
            np.zeros(29),
            np.zeros(29),
            np.zeros(29),
            frame=frame(),
            safety=G1TorqueSafetyConfig(),
            fraction=fraction,
        )
