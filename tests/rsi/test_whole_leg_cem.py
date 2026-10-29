import numpy as np
import pytest

from rosclaw_soccer.rsi.whole_leg_cem import (
    case_score,
    initial_population,
    pair_rank,
    population,
    update_elites,
)


def trace(controls=300):
    return dict(
        qpos=np.zeros((controls * 10, 43)),
        ball_forces=np.zeros((controls * 10, 1, 3)),
        controls=controls,
        runtime_failure=None,
        body_failed=False,
    )


def test_fixed_rng_and_elite_ties():
    rng, mean, sigma = initial_population()
    raw, bounded = population(rng, mean, sigma)
    other, other_mean, other_sigma = initial_population()
    np.testing.assert_array_equal(raw, population(other, other_mean, other_sigma)[0])
    assert bounded.shape == (16, 12) and np.abs(bounded).max() <= 0.25
    mean2, sigma2, ids = update_elites(bounded, [(1, 1, 0, 0.0, 3)] * 16)
    assert ids == [0, 1, 2, 3]
    np.testing.assert_allclose(mean2, bounded[:4].mean(axis=0))
    assert np.all(sigma2 >= 0.025)


def test_safety_and_full_horizon_veto_false_success():
    safe = trace()
    safe["ball_forces"][100, 0, 0] = 11
    safe["qpos"][600, 36] = 1
    score = case_score(safe, 36)
    assert score["strong_foot"] == 1 and score["forward_displacement_m"] == 1
    unsafe = trace()
    unsafe["ball_forces"][100, 0, 0] = 11
    unsafe["ball_forces"][200, 0, 2] = 1.1
    unsafe_score = case_score(unsafe, 36)
    assert unsafe_score["safe"] == unsafe_score["strong_foot"] == 0
    assert pair_rank(score, score) > pair_rank(unsafe_score, unsafe_score)
    partial = case_score(trace(30), 36)
    assert partial["complete"] == partial["safe"] == 0 and partial["first_nonfoot_index"] == 0


def test_late_contact_is_censored_and_nonfinite_rejected():
    x = trace()
    x["ball_forces"][2900, 0, 1] = 11
    assert case_score(x, 36)["successor_censored"] is True
    x["qpos"][0, 0] = np.nan
    with pytest.raises(ValueError):
        case_score(x, 36)
