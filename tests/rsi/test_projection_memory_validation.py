import numpy as np
import pytest

from rosclaw_soccer.rsi.projection_memory_validation import validate_projection_memory


def sample(reflect):
    raw = np.full((3, 29), 2.0, dtype=np.float32)
    mask = np.zeros((3, 29), dtype=bool)
    mask[:, 1] = True
    targets = np.ones((3, 29), dtype=np.float64)
    targets[:, 1] = 0.5
    memory = raw.copy()
    if reflect:
        memory[:, 1] = 1.0
    obs = np.zeros((3, 98), dtype=np.float32)
    obs[1:, -29:] = memory[:-1]
    return dict(
        observations=obs,
        raw_actions=raw,
        targets=targets,
        changed_axes=mask,
        memory_after=memory,
        offset=np.zeros(29),
        scale=np.full(29, 0.5),
        reflect_projection=reflect,
    )


@pytest.mark.parametrize("reflect", [False, True])
def test_actual_next_observations_match_declared_memory(reflect):
    result = validate_projection_memory(**sample(reflect))
    assert result["verified_next_observations"] == 2
    assert result["projection_changed_controls"] == 3
    assert result["max_normalized_raw_vs_executed_delta"] == 1.0


def test_incorrect_next_observation_is_rejected():
    args = sample(True)
    args["observations"][1, -28] = 7
    with pytest.raises(ValueError):
        validate_projection_memory(**args)


def test_misreported_memory_semantics_are_rejected():
    args = sample(True)
    args["memory_after"][0, 1] = 2
    with pytest.raises(ValueError):
        validate_projection_memory(**args)
