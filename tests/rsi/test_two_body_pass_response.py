from copy import deepcopy

import numpy as np
import pytest

from rosclaw_soccer.rsi.two_body_pass_response import measured_response, new_distinct_proposals


def fixture():
    positions = np.column_stack((np.arange(10) * 0.5, np.zeros(10), np.full(10, 0.11)))
    velocities = np.tile([3.4, 0.0, 0.0], (10, 1))
    initial = np.array([0.0, 0.0, 0.11])
    goal = np.array([3.0, 0.0, 0.11])
    o = dict(
        formal_native_steps=10,
        incoming_closest_substep=4,
        incoming_arrival_speed_mps=3.4,
        incoming=dict(
            incoming_error_observed=True,
            A_first_foot_strike_substep=2,
            B_first_ball_contact_substep=5,
            incoming_window_end_exclusive=5,
            incoming_target_error_3d_before_B_contact_m=1.0,
        ),
    )
    return positions, velocities, initial, goal, o


def test_response_uses_actual_pre_receiver_contact_not_later_better_point():
    result = measured_response(*fixture())
    np.testing.assert_allclose(result, [0.0, -1.0, 0.0, 0.6], rtol=0, atol=1e-15)


def test_response_is_invariant_under_global_planar_rotation_and_translation():
    p, v, i, g, o = fixture()
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    for x in (p, i, g):
        x[..., :2] = x[..., :2] @ rot.T + np.array([5.0, -2.0])
    v[:, :2] = v[:, :2] @ rot.T
    np.testing.assert_allclose(
        measured_response(p, v, i, g, o), [0.0, -1.0, 0.0, 0.6], rtol=0, atol=1e-15
    )


@pytest.mark.parametrize(
    "fault", ["late", "error", "speed", "shortened_window", "bad_shape", "nonfinite"]
)
def test_raw_contradiction_or_retroactive_receiving_help_is_not_admitted(fault):
    p, v, i, g, o = fixture()
    if fault == "late":
        o["incoming_closest_substep"] = 6
    elif fault == "error":
        o["incoming"]["incoming_target_error_3d_before_B_contact_m"] = 0.01
    elif fault == "speed":
        o["incoming_arrival_speed_mps"] = 3.3
    elif fault == "shortened_window":
        o["incoming"]["incoming_window_end_exclusive"] = 4
    elif fault == "bad_shape":
        p = p[:, :2]
    else:
        v[0, 0] = np.nan
    with pytest.raises(ValueError):
        measured_response(p, v, i, g, o)


def test_missing_delivery_remains_missing_not_a_zero_error_sample():
    p, v, i, g, o = fixture()
    o["incoming"].update(
        incoming_error_observed=False, incoming_target_error_3d_before_B_contact_m=None
    )
    o["incoming_closest_substep"] = o["incoming_arrival_speed_mps"] = None
    assert measured_response(p, v, i, g, o) is None
    o["incoming"]["incoming_target_error_3d_before_B_contact_m"] = 0.0
    with pytest.raises(ValueError):
        measured_response(p, v, i, g, o)


def test_deduplicate_only_controls_without_removing_or_resampling_physical_states():
    z = np.zeros((2, 6))
    a = np.full((2, 6), 0.1)
    b = np.full((2, 6), 0.2)
    candidates = [dict(latent=x.tolist()) for x in (z, -z, a, a, b)]
    before = deepcopy(candidates)
    fresh, duplicates = new_distinct_proposals(candidates, [z])
    assert [x["candidate_index"] for x in fresh] == [2, 4]
    assert [x["candidate_index"] for x in duplicates] == [0, 1, 3]
    assert candidates == before


@pytest.mark.parametrize("z", [np.full((2, 6), 1.01), np.full((2, 6), np.inf), np.zeros((1, 6))])
def test_invalid_local_control_proposals_fail_before_any_physics(z):
    with pytest.raises(ValueError):
        new_distinct_proposals([dict(latent=z.tolist())], [])
