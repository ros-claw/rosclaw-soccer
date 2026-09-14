import pytest

from rosclaw_soccer.training.ballistic_launch_guidance import ballistic_launch_guidance


def cue(**changes):
    args = dict(
        launch_position_m=(0.0, 0.0, 0.15),
        launch_velocity_m_s=(9.0, -1.0, 3.0),
        target_position_m=(9.0, -1.25, 1.75),
        complete_clean_episode=True,
    )
    args.update(changes)
    return ballistic_launch_guidance(**args)


def test_measured_low_launch_gets_continuous_upward_guidance():
    result = cue()
    assert result.prediction_available
    assert result.estimated_flight_time_s == 1
    assert result.estimated_required_vertical_speed_m_s == pytest.approx(6.505)
    assert result.estimated_required_lateral_speed_m_s == -1.25
    better = cue(launch_velocity_m_s=(9.0, -1.25, 4.0))
    assert better.auxiliary_penalty > result.auxiliary_penalty
    ideal = cue(launch_velocity_m_s=(9.0, -1.25, 6.505))
    assert ideal.auxiliary_penalty == pytest.approx(0)
    # Zero model-based velocity gap is deliberately NOT a goal verdict.
    assert not hasattr(ideal, "goal_scored")


@pytest.mark.parametrize(
    "changes",
    [
        dict(complete_clean_episode=False),
        dict(launch_velocity_m_s=(0.0, 0.0, 0.0)),
        dict(launch_velocity_m_s=(-2.0, 0.0, 5.0)),
        dict(target_position_m=(-1.0, 0.0, 1.75)),
        dict(target_position_m=(100.0, 0.0, 1.75)),
        dict(target_position_m=(0.01, 0.0, 1.75)),
    ],
)
def test_unsupported_prediction_or_failed_episode_has_maximum_cost(changes):
    result = cue(**changes)
    assert not result.prediction_available and result.auxiliary_penalty == -4
    assert result.velocity_gap_m_s is None


@pytest.mark.parametrize(
    "changes",
    [
        dict(launch_position_m=(float("nan"), 0.0, 0.0)),
        dict(launch_velocity_m_s=(float("inf"), 0.0, 0.0)),
        dict(target_position_m=(True, 0.0, 0.0)),
        dict(target_position_m=[9.0, 0.0, 1.75]),
        dict(launch_position_m=(0.0, 0.0)),
        dict(launch_velocity_m_s=(101.0, 0.0, 0.0)),
        dict(complete_clean_episode=1),
    ],
)
def test_invalid_inputs_rejected_not_sanitized_into_credit(changes):
    with pytest.raises(ValueError):
        cue(**changes)


def test_penalty_is_bounded_and_does_not_reward_overshoot():
    assert cue(launch_velocity_m_s=(9.0, 100.0, 100.0)).auxiliary_penalty == -4
    assert (
        cue(launch_velocity_m_s=(9.0, -1.25, 6.505)).auxiliary_penalty
        > cue(launch_velocity_m_s=(9.0, -1.25, 10.0)).auxiliary_penalty
    )
