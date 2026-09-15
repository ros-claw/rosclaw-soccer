import numpy as np
import pytest

from rosclaw_soccer.growth.per_player_motor_trace import per_player_motor_columns


def test_parallel_targets_are_independent_and_legacy_remains_optional():
    assert per_player_motor_columns({}, "red.finisher", agent_code=1, frames=3) is None
    trace = {"per_player_motor_contract": np.ones(3, dtype=bool)}
    for _code, team, direction in ((1, "red", 1), (2, "blue", -1)):
        trace[f"{team}_finisher_motor_option_active"] = np.array([True, True, False])
        trace[f"{team}_finisher_motor_option_target_m"] = np.array(
            [[direction * 7.5, 0, 1.5], [direction * 7.5, 0, 1.5], [0, 0, 0]]
        )
    for code, team, direction in ((1, "red", 1), (2, "blue", -1)):
        codes, targets = per_player_motor_columns(
            trace, f"{team}.finisher", agent_code=code, frames=3
        )
        np.testing.assert_array_equal(codes, [code, code, 0])
        assert targets[0, 0] == direction * 7.5


@pytest.mark.parametrize("damage", ["missing", "integer_mask", "nan", "inactive_target"])
def test_malformed_actor_evidence_fails_closed(damage):
    trace = {
        "per_player_motor_contract": np.ones(2, dtype=bool),
        "red_finisher_motor_option_active": np.array([True, False]),
        "red_finisher_motor_option_target_m": np.array([[7.5, 0, 1.5], [0, 0, 0]]),
    }
    if damage == "missing":
        del trace["red_finisher_motor_option_target_m"]
    elif damage == "integer_mask":
        trace["red_finisher_motor_option_active"] = np.array([1, 0])
    elif damage == "nan":
        trace["red_finisher_motor_option_target_m"][0, 0] = np.nan
    else:
        trace["red_finisher_motor_option_target_m"][1, 0] = 1
    with pytest.raises(ValueError):
        per_player_motor_columns(trace, "red.finisher", agent_code=1, frames=2)
