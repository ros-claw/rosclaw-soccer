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


def test_supported_neural_actor_records_own_activation_and_keeps_actual_pd_targets():
    from rosclaw_soccer.rsi.feature_support_gate import FeatureSupportGate
    from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter
    from rosclaw_soccer.rsi.supported_refinement_motor import SupportedRefinementMotor

    world, original = players()
    masked, targets = {}, {}
    for prefix, player in original.items():
        learned = RefinementMotorAdapter(ConsolidatedMotorAdapter(player.actor))
        with torch.no_grad():
            learned.head[-1].bias.fill_(2)
        gate = FeatureSupportGate(
            torch.full((1, 288), 100.0),
            torch.ones(288, dtype=torch.float64),
            torch.full((1,), 0.01, dtype=torch.float64),
        )
        masked[prefix] = make_adapter_player(world, player, SupportedRefinementMotor(learned, gate))
        expected, _ = player.propose()
        actual, trace = masked[prefix].propose(with_reference_evidence=True)
        assert np.array_equal(expected, actual)
        assert trace["NN_support_activation"].shape == (1,)
        assert trace["NN_support_activation"].item() == 0
        targets[prefix] = actual
    world.advance(targets)
    for player in masked.values():
        player.commit()
        before = tuple(t.clone() for t in player.state)
        player.begin_standing()
        assert all(torch.equal(x, y) for x, y in zip(before, player.state, strict=True))
    world.advance({prefix: player.propose()[0] for prefix, player in masked.items()})
    for player in masked.values():
        player.commit()
    assert world.native_steps == 20
