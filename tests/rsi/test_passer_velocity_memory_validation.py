import numpy as np
import pytest

from rosclaw_soccer.rsi.passer_velocity_memory_validation import validate_passer_velocity_history


def fixture():
    active = np.array([False, False, True, True], dtype=bool)
    proposed = np.full((4, 29), 0.5, dtype=np.float32)
    targets = np.full((4, 29), 0.2, dtype=np.float64)
    targets[2:] = 0.5
    memory = np.full((4, 29), 0.2, dtype=np.float32)
    memory[2:] = 0.5
    obs = np.zeros((4, 98), dtype=np.float32)
    obs[1:, -29:] = memory[:-1]
    return dict(
        observations=obs,
        proposed=proposed,
        actual_targets=targets,
        memory_after=memory,
        active=active,
        offset=np.zeros(29, np.float64),
        scale=np.ones(29, np.float64),
    )


def test_actual_target_is_the_shadow_history_then_own_action_is_active():
    result = validate_passer_velocity_history(**fixture())
    assert result["shadow_controls"] == 2 and result["active_controls"] == 2
    assert result["verified_next_observations"] == 3


@pytest.mark.parametrize("fault", ["shadow", "observation", "active_target", "reentry"])
def test_reject_wrong_history_or_invisible_transition(fault):
    row = fixture()
    if fault == "shadow":
        row["memory_after"][0, 0] = 0.5
    elif fault == "observation":
        row["observations"][1, -29] = 0.5
    elif fault == "active_target":
        row["actual_targets"][2, 0] = 0.9
    else:
        row["active"][-1] = False
    with pytest.raises(ValueError):
        validate_passer_velocity_history(**row)
