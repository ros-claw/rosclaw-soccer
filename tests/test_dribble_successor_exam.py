from dataclasses import replace

import pytest

from rosclaw_soccer.training.dribble_successor_exam import (
    DribbleSample,
    evaluate_dribble_successor,
)


def samples(count=100):
    rows = []
    for i in range(count + 1):
        t = 1.0 + i * 0.02
        p = (-0.3 * (t - 1), 0.0, 0.115)
        rows.append(
            DribbleSample(
                50 + i,
                t,
                p,
                (-0.3, 0.0, 0.0),
                ((p[0] + 0.2, 0.0, 0.03), (p[0] + 0.2, 0.1, 0.03)),
                2.2 if t >= 2.2 else (1.2 if t >= 1.2 else 0.9),
                None,
            )
        )
    return tuple(rows)


@pytest.mark.parametrize("profile,count", [("one_second", 50), ("two_second", 100)])
def test_valid_continuous_success_is_not_promotion(profile, count):
    result = evaluate_dribble_successor(
        samples(count), direction=-1, episode_safe=True, profile=profile
    )
    assert result.passed and not result.reasons
    assert not result.successor_ready_verified and not result.training_authorized
    assert result.activation_ceiling == "SIM_ONLY"


@pytest.mark.parametrize(
    "fault",
    [
        "gap",
        "duplicate",
        "clock",
        "nan",
        "future_contact",
        "lost_contact",
        "regressing_contact",
        "missing",
        "future_interruption",
        "lost_interruption",
    ],
)
def test_corrupted_or_incomplete_evidence_rejected(fault):
    rows = list(samples())
    if fault == "gap":
        rows[30] = replace(rows[30], frame=81)
    if fault == "duplicate":
        rows[30] = rows[29]
    if fault == "clock":
        rows[30] = replace(rows[30], time_sec=1.601)
    if fault == "nan":
        rows[30] = replace(rows[30], ball_velocity_mps=(float("nan"), 0.0, 0.0))
    if fault == "future_contact":
        rows[30] = replace(rows[30], last_own_contact_sec=10.0)
    if fault == "lost_contact":
        rows[30] = replace(rows[30], last_own_contact_sec=None)
    if fault == "regressing_contact":
        rows[30] = replace(rows[30], last_own_contact_sec=0.5)
    if fault == "missing":
        rows.pop(30)
    if fault == "future_interruption":
        rows[30] = replace(rows[30], last_interruption_sec=10.0)
    if fault == "lost_interruption":
        rows[29] = replace(rows[29], last_interruption_sec=1.5)
    with pytest.raises(ValueError):
        evaluate_dribble_successor(
            tuple(rows), direction=-1, episode_safe=True, profile="two_second"
        )


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("unsafe", "unsafe_episode"),
        ("distance", "ball_out_of_reach"),
        ("height", "ball_height"),
        ("speed", "terminal_ball_speed"),
        ("progress", "insufficient_progress"),
        ("touch", "no_second_half_foot_touch"),
        ("interruption", "contact_interruption"),
    ],
)
def test_valid_physical_failures_are_not_invalid_evidence(fault, reason):
    rows = list(samples())
    if fault == "distance":
        rows[30] = replace(rows[30], foot_positions_m=((2.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    if fault == "height":
        rows[30] = replace(rows[30], ball_position_m=(-0.18, 0.0, 0.21))
    if fault == "speed":
        rows[-1] = replace(rows[-1], ball_velocity_mps=(-0.81, 0.0, 0.0))
    if fault == "progress":
        rows = [replace(r, ball_position_m=(0.0, 0.0, 0.115)) for r in rows]
    if fault == "touch":
        rows = [replace(r, last_own_contact_sec=min(r.last_own_contact_sec, 1.2)) for r in rows]
    if fault == "interruption":
        rows = [replace(r, last_interruption_sec=1.4 if r.time_sec >= 1.4 else None) for r in rows]
    result = evaluate_dribble_successor(
        tuple(rows), direction=-1, episode_safe=fault != "unsafe", profile="two_second"
    )
    assert not result.passed and reason in result.reasons


@pytest.mark.parametrize("direction,safe", [(True, True), (-1, 1), (0, True)])
def test_invalid_call_contract(direction, safe):
    with pytest.raises(ValueError):
        evaluate_dribble_successor(
            samples(), direction=direction, episode_safe=safe, profile="two_second"
        )


def test_extreme_integer_rejected_without_float_overflow():
    rows = list(samples())
    rows[20] = replace(rows[20], time_sec=10**1000)
    with pytest.raises(ValueError):
        evaluate_dribble_successor(
            tuple(rows), direction=-1, episode_safe=True, profile="two_second"
        )
