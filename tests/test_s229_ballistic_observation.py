import numpy as np
import pytest

from rosclaw_soccer.skills.goalkeeper_v2.observations import GoalkeeperActorObserver


def observe(observer, t, p):
    return observer.observe(
        timestamp_sec=t,
        ball_relative_position_m=np.array(p),
        gravity_orientation=np.array((0, 0, -1)),
        root_linear_velocity_mps=np.zeros(3),
        angular_velocity_rad_s=np.zeros(3),
        joint_position_rad=np.zeros(29),
        joint_velocity_rad_s=np.zeros(29),
        previous_action_rad=np.zeros(29),
    )


def test_current_ballistic_velocity_and_intercept_are_causal_and_separately_bound():
    old, corrected = (
        GoalkeeperActorObserver(),
        GoalkeeperActorObserver(ballistic_airborne_velocity=True),
    )
    # Ball starts 3 m ahead, 0.3 m high, reaches keeper in .5 s at z=1.5.
    vz = (1.5 - 0.3 + 4.905 * 0.5**2) / 0.5
    for i in range(1, 13):
        t = i * 0.02
        p = (3 - 6 * t, 0.2, 0.3 + vz * t - 4.905 * t * t)
        a, b = observe(old, t, p), observe(corrected, t, p)
    assert b.ball_history_ready
    assert b.estimated_ball_velocity_mps[2] == pytest.approx(vz - 9.81 * t, abs=1e-12)
    assert b.estimated_intercept[2] == pytest.approx(1.5, abs=1e-12)
    assert a.estimated_intercept[2] > 1.65
    assert a.actor_contract_hash == old.spec.actor_contract_hash
    assert b.actor_contract_hash != a.actor_contract_hash
    # Public features carry exactly the same corrected estimates as the decoder fields.
    np.testing.assert_allclose(b.values[24:27], b.estimated_ball_velocity_mps)
    np.testing.assert_allclose(b.values[27:30], b.estimated_intercept)


def test_rolling_history_does_not_get_a_fictitious_downward_velocity():
    observer = GoalkeeperActorObserver(ballistic_airborne_velocity=True)
    for i in range(12):
        result = observe(observer, i * 0.02, (2 - i * 0.04, 0, 0.115))
    assert result.estimated_ball_velocity_mps[2] == 0
