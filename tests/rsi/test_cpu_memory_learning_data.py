import numpy as np
import pytest

from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return
from scripts.rsi_cpu_memory_learning_data import ordered_cpu_arrays


def values():
    return (
        np.zeros((270, 134)),
        np.repeat(np.arange(3), 90),
        [(np.zeros(12), -1.0) for _ in range(270)],
    )


def outcome():
    return dict(
        reward=0.0,
        high_quality=False,
        clean_foot_only=False,
        minimum_pelvis_z_m=0.7,
        forward_60_m=1.0,
        lateral_over_forward_60=0.2,
        maximum_lateral_excursion_m=1.0,
    )


def test_complete_failed_cpu_trajectory_kept_with_unchanged_terminal_objective():
    x, p, draws = values()
    arrays = ordered_cpu_arrays(x, p, draws, outcome=outcome(), std=0.1, group=2)
    assert arrays["observation"].shape == (270, 134)
    assert np.array_equal(arrays["trajectory_index"], np.full(270, 2))
    assert np.all(arrays["terminal_return"] == terminal_return(outcome()))
    x[:] = 1
    p[:] = 2
    assert not np.any(arrays["observation"])
    assert arrays["phase_index"][0] == 0


@pytest.mark.parametrize(
    "fault", ["short", "nan", "phase", "phase_float", "draws", "latent", "logp", "std", "group"]
)
def test_partial_nonfinite_or_relabelled_cpu_data_rejected(fault):
    x, p, draws = values()
    std = 0.1
    group = 0
    if fault == "short":
        x = x[:-1]
    elif fault == "nan":
        x[7, 3] = np.nan
    elif fault == "phase":
        p[3] = 3
    elif fault == "phase_float":
        p = p.astype(float)
    elif fault == "draws":
        draws.pop()
    elif fault == "latent":
        draws[4] = (np.zeros(11), -1.0)
    elif fault == "logp":
        draws[3] = (np.zeros(12), np.inf)
    elif fault == "std":
        std = 0.05
    else:
        group = True
    with pytest.raises(ValueError):
        ordered_cpu_arrays(x, p, draws, outcome=outcome(), std=std, group=group)
