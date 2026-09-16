import numpy as np
import pytest

from rosclaw_soccer.providers.g1.keeper_recovery_observation import (
    RELEASE_RESIDUAL_EPSILON_RAD,
    release_only_arm_observation_gate,
)


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("value", [0.0, 1e-6, -1e-6, 0.04, -0.8])
def test_release_only_not_active_reach(active, value):
    residual = np.zeros(29)
    residual[15] = value
    original = residual.copy()
    assert release_only_arm_observation_gate(
        reach_active=active, previous_residual_rad=residual
    ) is (not active and abs(value) > RELEASE_RESIDUAL_EPSILON_RAD)
    np.testing.assert_array_equal(residual, original)


@pytest.mark.parametrize("active", [None, 0, 1, "false", np.bool_(False)])
def test_invalid_reach_state(active):
    with pytest.raises(ValueError):
        release_only_arm_observation_gate(reach_active=active, previous_residual_rad=np.zeros(29))


@pytest.mark.parametrize(
    "residual",
    [
        np.zeros(28),
        np.zeros((1, 29)),
        np.zeros(30),
        np.zeros(29, dtype=bool),
        np.zeros(29, dtype=complex),
        ["0"] * 29,
        [None] * 29,
        [0.0] * 28 + [float("nan")],
        [0.0] * 28 + [float("inf")],
        [0.0] * 28 + [-float("inf")],
        [0.0] * 28 + [10.01],
        [0.1] + [0.0] * 28,
    ],
)
def test_invalid_history_fails_even_during_reach(residual):
    for active in (False, True):
        with pytest.raises(ValueError):
            release_only_arm_observation_gate(reach_active=active, previous_residual_rad=residual)


def test_history_is_explicit_private_and_has_no_latch():
    released, empty = np.zeros(29), np.zeros(29)
    released[28] = 0.15
    for _ in range(2):
        assert release_only_arm_observation_gate(reach_active=False, previous_residual_rad=released)
        assert not release_only_arm_observation_gate(
            reach_active=False, previous_residual_rad=empty
        )
        assert not release_only_arm_observation_gate(
            reach_active=True, previous_residual_rad=released
        )
