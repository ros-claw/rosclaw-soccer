import numpy as np
import pytest
from test_shared_robonaldo_player import setup

from rosclaw_soccer.rsi.cftrack_live_history import CFtrackLiveHistory


def test_five_real_owned_boundaries_are_read_only_and_include_actual_targets():
    world = setup()
    history = CFtrackLiveHistory(world, "a/")
    for i in range(5):
        targets = {"a/": np.full(29, i * 0.001), "b/": np.zeros(29)}
        world.advance(targets)
        before = (
            world.data.qpos.copy(),
            world.data.qvel.copy(),
            world.data.ctrl.copy(),
            world.data.time,
        )
        sample = history.capture()
        np.testing.assert_array_equal(sample.previous_executed_target, targets["a/"])
        np.testing.assert_array_equal(
            sample.cached_ball_position, world.data.xpos[world.layouts["a/"].ball]
        )
        for expected, actual in zip(
            before,
            (world.data.qpos, world.data.qvel, world.data.ctrl, world.data.time),
            strict=True,
        ):
            np.testing.assert_array_equal(expected, actual)
    samples = history.at_current_boundary()
    assert tuple(x.control for x in samples) == (1, 2, 3, 4, 5)
    assert world.control_steps == 5


@pytest.mark.parametrize("skip", [False, True])
def test_repeat_or_missing_boundary_poison_collector(skip):
    world = setup()
    history = CFtrackLiveHistory(world, "a/")
    world.advance({"a/": np.zeros(29), "b/": np.zeros(29)})
    history.capture()
    if skip:
        world.advance({"a/": np.zeros(29), "b/": np.zeros(29)})
        world.advance({"a/": np.zeros(29), "b/": np.zeros(29)})
    with pytest.raises(ValueError):
        history.capture()
    with pytest.raises(ValueError):
        history.at_current_boundary()


def test_cannot_read_stale_history_after_world_advances():
    world = setup()
    history = CFtrackLiveHistory(world, "a/")
    for _ in range(5):
        world.advance({"a/": np.zeros(29), "b/": np.zeros(29)})
        history.capture()
    world.advance({"a/": np.zeros(29), "b/": np.zeros(29)})
    with pytest.raises(ValueError):
        history.at_current_boundary()
