import math
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupConfig, G1SonicRunupController


def test_existing_straight_ahead_configuration_hashes_are_unchanged():
    assert (
        G1SonicRunupConfig().config_hash
        == "sha256:46a6112294bd467f42e904228f9b80ee8539916e4666f98e7ae27ceda1a4a7d9"
    )
    assert (
        G1SonicRunupConfig(model_variant="sonic_v1_1").config_hash
        == "sha256:cf11d8f19ac909e36a49f9a1979e779469faf258d5272df7b2b8932ff30e6537"
    )
    assert (
        G1SonicRunupConfig(movement_heading_rad=0.2).config_hash != G1SonicRunupConfig().config_hash
    )
    assert (
        G1SonicRunupConfig(facing_heading_rad=0.2).config_hash
        != G1SonicRunupConfig(movement_heading_rad=0.2).config_hash
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), math.pi + 0.01, True])
@pytest.mark.parametrize("field", ["movement_heading_rad", "facing_heading_rad"])
def test_invalid_planning_headings_are_rejected(field, value):
    with pytest.raises(ValueError, match="heading"):
        G1SonicRunupConfig(**{field: value})


def test_measured_task_directions_reach_planner_without_mutating_body_state():
    controller = object.__new__(G1SonicRunupController)
    controller.config = G1SonicRunupConfig(
        movement_heading_rad=-math.pi / 2, facing_heading_rad=math.pi / 2
    )
    initial = np.zeros(36)
    initial[2:4] = (0.75, 1)
    original = initial.copy()
    calls = []

    def run(_, feed):
        calls.append(feed)
        return np.repeat(initial[None, None, :], 16, axis=1), np.array([16])

    controller._planner = SimpleNamespace(run=run)
    reference = controller._generate_reference(initial)
    assert reference.shape[1] == 36 and len(calls) == 4
    np.testing.assert_array_equal(initial, original)
    for feed in calls:
        np.testing.assert_allclose(feed["movement_direction"], [[0, -1, 0]], atol=1e-7)
        np.testing.assert_allclose(feed["facing_direction"], [[0, 1, 0]], atol=1e-7)
