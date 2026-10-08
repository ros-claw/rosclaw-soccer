"""Reward redistribution tests, never substitutes for physical audits."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.contextual_first_touch_option import first_touch_reward
from rosclaw_soccer.rsi.independent_first_touch_bank import post_contact_displacement
from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from rosclaw_soccer.rsi.temporal_contact_rewards import redistribute_contact_rewards


def course(contact=70, body=0):
    trace = {
        "force_n": np.zeros((300, 1, 6)),
        "pelvis_z_per_substep_m": np.full((300, 1, 10), 0.75),
        "ball_position_after_step_m": np.zeros((300, 1, 3)),
    }
    if contact is not None:
        trace["force_n"][contact, 0, body] = 5
        trace["ball_position_after_step_m"][contact:, 0, 0] = np.minimum(
            np.arange(300 - contact) * 0.02, 1.2
        )
    return trace


def outcome(trace):
    force = trace["force_n"][:, 0]
    ids = np.flatnonzero(np.any(force > 1, axis=1))
    first = int(ids[0]) if len(ids) else None
    bodies = np.flatnonzero(np.max(force, axis=0) > 1).tolist()
    ball = trace["ball_position_after_step_m"][:, 0]
    clean = first is not None and set(bodies) <= {0, 1}
    displacement = post_contact_displacement(ball, first)
    forward, lateral = displacement["forward_60_m"], displacement["lateral_60_m"]
    excursion = float(np.max(np.abs(ball[:, 1])))
    minimum = float(trace["pelvis_z_per_substep_m"].min())
    value = dict(
        first_contact_frame=first,
        contact_body_indices=bodies,
        clean_foot_only=clean,
        minimum_pelvis_z_m=minimum,
        maximum_lateral_excursion_m=excursion,
        safety_passed=minimum >= 0.65,
        high_quality=bool(
            clean
            and forward is not None
            and lateral is not None
            and forward >= 1
            and abs(lateral) / max(forward, 0.01) <= 0.3
            and excursion <= 4
        ),
        **displacement,
    )
    value["reward"] = first_touch_reward({**value, "max_lateral_excursion_m": excursion})
    return value


def test_clean_quality_awarded_only_at_fixed_measured_horizon():
    trace = course()
    value = outcome(trace)
    result = redistribute_contact_rewards(trace, value)
    assert result.evaluation_frame == 130 and result.first_contact_frame == 70
    assert np.count_nonzero(result.component_rewards[:, 1]) == 1
    assert result.component_rewards[130, 1] == 10
    np.testing.assert_allclose(result.rewards.sum(), terminal_return(value))
    assert not result.rewards.flags.writeable and not result.component_rewards.flags.writeable


def test_later_dirty_contact_revokes_quality_and_prior_progress():
    trace = course()
    trace["force_n"][180, 0, 5] = 5
    result = redistribute_contact_rewards(trace, outcome(trace))
    assert result.first_dirty_frame == 180
    assert result.component_rewards[130, 1] == 10 and result.component_rewards[180, 1] == -10
    assert result.component_rewards[180, 2] == -8
    assert result.undiscounted_return == -10


def test_knee_and_later_height_failure_keep_all_penalties():
    trace = course(body=5)
    trace["pelvis_z_per_substep_m"][98, 0, 8] = 0.647
    result = redistribute_contact_rewards(trace, outcome(trace))
    assert result.first_dirty_frame == 70 and result.first_unsafe_frame == 98
    assert result.component_rewards[70, 2] == -8 and result.component_rewards[98, 4] == -100
    assert result.component_rewards[98, 0] == -8
    assert result.undiscounted_return == -118


def test_quality_and_safety_are_separate_as_in_original_audit():
    trace = course()
    trace["pelvis_z_per_substep_m"][180, 0, 0] = 0.5
    value = outcome(trace)
    assert value["high_quality"] and not value["safety_passed"]
    result = redistribute_contact_rewards(trace, value)
    assert result.component_rewards[130, 1] == 10 and result.component_rewards[180, 4] == -100
    assert result.undiscounted_return == -100


def test_late_out_revokes_quality_and_applies_boundary_cost():
    trace = course()
    trace["ball_position_after_step_m"][170:, 0, 1] = 4.1
    result = redistribute_contact_rewards(trace, outcome(trace))
    assert result.first_out_frame == 170
    assert result.component_rewards[170, 1] == -10 and result.component_rewards[170, 3] == -20
    assert result.undiscounted_return == -24


def test_no_contact_failure_is_resolved_at_endpoint_without_fabricated_touch():
    trace = course(contact=None)
    result = redistribute_contact_rewards(trace, outcome(trace))
    assert result.first_contact_frame is None and result.evaluation_frame is None
    assert np.count_nonzero(result.rewards) == 1 and result.rewards[-1] == -11


def test_pre_control_failure_is_not_moved_to_later_actions():
    trace = course(contact=None)
    trace["pelvis_z_per_substep_m"][10, 0, 0] = 0.4
    result = redistribute_contact_rewards(trace, outcome(trace))
    assert result.first_unsafe_frame == 10 and result.rewards[10] == -110
    assert result.rewards[-1] == -8 and result.undiscounted_return == -118


def test_late_contact_cannot_invent_missing_sixty_frame_measurement():
    trace = course(contact=260)
    forged = outcome(course())
    forged["first_contact_frame"] = 260
    with pytest.raises(ValueError):
        redistribute_contact_rewards(trace, forged)


@pytest.mark.parametrize(
    "field",
    [
        "reward",
        "minimum_pelvis_z_m",
        "forward_60_m",
        "maximum_lateral_excursion_m",
        "high_quality",
        "clean_foot_only",
        "safety_passed",
        "contact_body_indices",
        "first_contact_frame",
    ],
)
def test_forged_audit_labels_rejected(field):
    trace = course()
    value = outcome(trace)
    value[field] = (
        False
        if isinstance(value[field], bool)
        else ([5] if isinstance(value[field], list) else -999)
    )
    with pytest.raises(ValueError):
        redistribute_contact_rewards(trace, value)


@pytest.mark.parametrize(
    "field", ["force_n", "pelvis_z_per_substep_m", "ball_position_after_step_m"]
)
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_all_nonfinite_samples_rejected(field, invalid):
    trace = course()
    value = outcome(trace)
    trace[field].flat[-1] = invalid
    with pytest.raises(ValueError):
        redistribute_contact_rewards(trace, value)


def test_negative_force_norm_rejected():
    trace = course()
    value = outcome(trace)
    trace["force_n"][0, 0, 0] = -1
    with pytest.raises(ValueError):
        redistribute_contact_rewards(trace, value)


def test_input_arrays_are_not_changed():
    trace = course()
    originals = {k: v.copy() for k, v in trace.items()}
    redistribute_contact_rewards(trace, outcome(trace))
    for k in trace:
        np.testing.assert_array_equal(trace[k], originals[k])


def test_future_failures_do_not_change_earlier_reward_labels():
    parent = course()
    original = redistribute_contact_rewards(parent, outcome(parent))
    for failure in ("dirty", "unsafe", "out"):
        child = {k: v.copy() for k, v in parent.items()}
        if failure == "dirty":
            child["force_n"][180, 0, 5] = 5
        elif failure == "unsafe":
            child["pelvis_z_per_substep_m"][180, 0, 0] = 0.4
        else:
            child["ball_position_after_step_m"][180:, 0, 1] = 5
        result = redistribute_contact_rewards(child, outcome(child))
        np.testing.assert_array_equal(
            original.component_rewards[:180], result.component_rewards[:180]
        )


def test_boolean_body_index_cannot_impersonate_integer_zero():
    trace = course()
    value = outcome(trace)
    value["contact_body_indices"] = [False]
    with pytest.raises(ValueError):
        redistribute_contact_rewards(trace, value)


@pytest.mark.parametrize("height,unsafe", [(0.65, False), (0.65 - 1e-10, True)])
def test_original_strict_height_boundary_retained(height, unsafe):
    trace = course()
    trace["pelvis_z_per_substep_m"][180, 0, 0] = height
    result = redistribute_contact_rewards(trace, outcome(trace))
    assert (result.first_unsafe_frame == 180) == unsafe


@pytest.mark.parametrize("lateral,out", [(4.0, False), (4.0 + 1e-10, True)])
def test_original_strict_boundary_crossing_retained(lateral, out):
    trace = course(contact=None)
    trace["ball_position_after_step_m"][180:, 0, 1] = lateral
    result = redistribute_contact_rewards(trace, outcome(trace))
    assert (result.first_out_frame == 180) == out


def test_exact_contact_threshold_is_not_contact():
    trace = course(contact=None)
    trace["force_n"][70, 0, 0] = 1.0
    result = redistribute_contact_rewards(trace, outcome(trace))
    assert result.first_contact_frame is None and result.undiscounted_return == -11
