"""Tiny compiled-world control tests, not G1 football success evidence."""

import numpy as np
import pytest
import torch
from test_shared_native_neural_player import players

from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter
from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer


def make_adapter_player(world, original, actor):
    return SharedNativeNeuralPlayer(
        world,
        original.prefix,
        actor,
        clip=original.clip,
        torso_index=original.torso_index,
        build_observation=original.build_observation,
        goal=original.goal,
        offset=original.offset,
        scale=original.scale,
        order=original.order,
    )


def test_frozen_head_actor_runs_shared_pd_boundary_and_carries_stand_history():
    world, original = players()
    adapted = {}
    for prefix, player in original.items():
        actor = ConsolidatedMotorAdapter(player.actor, head_bound_rad=0.2).requires_grad_(False)
        adapted[prefix] = make_adapter_player(world, player, actor)
    targets = {}
    for prefix, player in adapted.items():
        expected, _ = original[prefix].propose()
        actual, _ = player.propose()
        assert np.array_equal(expected, actual)
        targets[prefix] = actual
    assert adapted["a/"].state[0].data_ptr() != adapted["b/"].state[0].data_ptr()
    world.advance(targets)
    for player in adapted.values():
        player.commit()
        old = tuple(x.clone() for x in player.state)
        player.begin_standing()
        assert all(torch.equal(a, b) for a, b in zip(old, player.state, strict=True))
    world.advance({prefix: player.propose()[0] for prefix, player in adapted.items()})
    for player in adapted.values():
        player.commit()
    assert world.native_steps == 20


def test_new_class_does_not_enable_duck_typed_or_trainable_native_actor():
    world, original = players()
    actor = ConsolidatedMotorAdapter(original["a/"].actor)
    with pytest.raises(ValueError):
        make_adapter_player(world, original["a/"], actor)

    class UnadmittedSubclass(ConsolidatedMotorAdapter):
        pass

    actor = UnadmittedSubclass(original["a/"].actor).requires_grad_(False)
    with pytest.raises(ValueError):
        make_adapter_player(world, original["a/"], actor)
    assert world.native_steps == 0
