import numpy as np
import pytest

from rosclaw_soccer.rsi.native_team_intent import ROLES, command_toward, role_target


@pytest.mark.parametrize("role", ROLES)
def test_team_role_law_is_half_turn_symmetric(role):
    ball = np.array([1.0, 0.7])
    np.testing.assert_array_equal(role_target("red", role, ball), -role_target("blue", role, -ball))


def test_commands_use_actual_body_frame_and_qualified_domain():
    red = command_toward(np.zeros(2), 0.0, np.array([2.0, 1.0]), np.array([3.0, 1.0]))
    blue = command_toward(np.zeros(2), np.pi, np.array([-2.0, -1.0]), np.array([-3.0, -1.0]))
    np.testing.assert_allclose(red, blue, atol=1e-6)
    assert -0.5 <= red[0] <= 1 and -0.5 <= red[1] <= 0.5 and -1 <= red[2] <= 1
    assert command_toward(np.zeros(2), 0.0, np.zeros(2), np.array([1.0, 1.0]))[0] == 0


def test_unknown_role_or_nonfinite_state_is_not_silently_coached():
    with pytest.raises(ValueError):
        role_target("red", "nobody", np.zeros(2))
    with pytest.raises(ValueError):
        command_toward(np.zeros(2), float("nan"), np.ones(2), np.ones(2))
