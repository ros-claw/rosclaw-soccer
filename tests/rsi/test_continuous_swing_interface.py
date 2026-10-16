import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_campaign_worker import (
    initial_campaign_population,
    preflight_population,
)
from rosclaw_soccer.rsi.taskspace_swing_probe import (
    MAX_JOINT_DELTA_RAD,
    continuous_swing_joint_delta,
    swing_joint_delta,
    validate_continuous_swing_parameters,
)


def inputs(rng):
    foot = rng.normal(size=3)
    ball = foot + rng.uniform(-1.0, 1.0, size=3)
    baseline = rng.uniform(-2.2, 2.2, size=6)
    return foot, ball, rng.normal(size=(3, 6)), baseline, np.tile([-2.0, 2.0], (6, 1))


def test_original_defaults_remain_bit_exact_including_joint_boundary_cases():
    rng = np.random.default_rng(42)
    for _ in range(1000):
        args = inputs(rng)
        kwargs = dict(forward_cap_m=0.08, lateral_cap_m=0.15, vertical_offset_m=0.04)
        assert np.array_equal(
            swing_joint_delta(*args, **kwargs), continuous_swing_joint_delta(*args, **kwargs)
        )


def test_previously_fatal_candidate_now_uses_explicit_continuous_interface():
    args = (
        np.zeros(3),
        np.asarray([0.5, 0.1, 0.1]),
        np.eye(3, 6),
        np.zeros(6),
        np.tile([-2.0, 2.0], (6, 1)),
    )
    kwargs = dict(
        forward_cap_m=0.07284813897250697,
        lateral_cap_m=0.05411615028054481,
        vertical_offset_m=0.012225575636274745,
    )
    with pytest.raises(ValueError):
        swing_joint_delta(*args, **kwargs)  # old protocol is not relaxed
    delta = continuous_swing_joint_delta(*args, **kwargs)
    assert np.isfinite(delta).all() and np.max(np.abs(delta)) <= MAX_JOINT_DELTA_RAD


def test_continuous_domain_preserves_joint_guards():
    rng = np.random.default_rng(43)
    for _ in range(1000):
        args = inputs(rng)
        delta = continuous_swing_joint_delta(
            *args,
            forward_cap_m=float(rng.uniform(0.02, 0.1)),
            lateral_cap_m=float(rng.uniform(0.04, 0.15)),
            vertical_offset_m=float(rng.uniform(0.0, 0.06)),
        )
        baseline, limits = args[3:]
        assert np.max(np.abs(delta)) <= MAX_JOINT_DELTA_RAD
        inside = (baseline >= limits[:, 0]) & (baseline <= limits[:, 1])
        assert np.all(baseline[inside] + delta[inside] >= limits[inside, 0])
        assert np.all(baseline[inside] + delta[inside] <= limits[inside, 1])
        assert np.all(delta[baseline < limits[:, 0]] >= 0)
        assert np.all(delta[baseline > limits[:, 1]] <= 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("forward_cap_m", 0.16),
        ("lateral_cap_m", 0.2),
        ("vertical_offset_m", -0.04),
        ("forward_cap_m", float("nan")),
        ("lateral_cap_m", float("inf")),
        ("vertical_offset_m", True),
    ],
)
def test_new_interface_does_not_expand_declared_parameter_bounds(field, value):
    kwargs = dict(forward_cap_m=0.08, lateral_cap_m=0.15, vertical_offset_m=0.04)
    kwargs[field] = value
    with pytest.raises(ValueError):
        validate_continuous_swing_parameters(**kwargs)


def test_all_predeclared_candidates_reach_actual_contact_interface():
    for seed in range(202610150, 202610182):
        preflight_population(initial_campaign_population(seed))


def test_preflight_does_not_only_exercise_default_candidate(monkeypatch):
    import rosclaw_soccer.rsi.first_touch_campaign_worker as worker

    seen = []

    def reject_random(*args, **kwargs):
        seen.append(kwargs["forward_cap_m"])
        if len(seen) == 3:
            raise ValueError("candidate2 incompatibility")
        return np.zeros(6)

    monkeypatch.setattr(worker, "continuous_swing_joint_delta", reject_random)
    with pytest.raises(ValueError, match="candidate2"):
        preflight_population(initial_campaign_population(202610150))
    assert len(seen) == 3
