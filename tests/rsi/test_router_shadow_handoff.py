import numpy as np
import pytest
import torch
from test_dual_recurrent_motor import ToyBase
from test_paid_ancestor_router_student import model as router_model
from test_shared_native_neural_player import players

from rosclaw_soccer.rsi.paid_ancestor_router_student import FrozenAncestralRouterActor
from rosclaw_soccer.rsi.shared_shadow_motor import SharedShadowMotor


def test_router_shadow_uses_own_actual_action_and_carries_own_memory():
    world, existing = players()
    actor = FrozenAncestralRouterActor(ToyBase(), router_model(), torch.full((29,), 0.5))
    actor.requires_grad_(False).eval()
    previous = existing["b/"]
    shadow = SharedShadowMotor(
        world,
        "b/",
        actor,
        clip=previous.clip,
        torso_index=previous.torso_index,
        build_observation=previous.build_observation,
        goal=previous.goal,
        offset=previous.offset,
        scale=previous.scale,
        order=previous.order,
    )
    trace = shadow.observe()
    assert trace["h_after"].shape == (2, 1, 128)
    with pytest.raises(ValueError):
        shadow.commit_actual()
    targets = {prefix: p.propose()[0] for prefix, p in existing.items()}
    world.advance(targets)
    shadow.commit_actual()
    for p in existing.values():
        p.commit()
    with pytest.raises(ValueError):
        shadow.handoff(goal=previous.goal, carry_memory=False)
    successor = shadow.handoff(goal=previous.goal, carry_memory=True)
    np.testing.assert_array_equal(successor.state[0].numpy(), trace["h_after"])
    np.testing.assert_allclose(
        successor.offset + successor.last_action[successor.order] * successor.scale,
        world.last_applied_target("b/"),
        rtol=0,
        atol=3e-7,
    )
    assert successor.boundary == world.control_steps


def test_router_shadow_cannot_start_at_an_unobserved_live_boundary():
    world, existing = players()
    world.advance({prefix: p.propose()[0] for prefix, p in existing.items()})
    actor = FrozenAncestralRouterActor(ToyBase(), router_model(), torch.full((29,), 0.5))
    actor.requires_grad_(False).eval()
    p = existing["b/"]
    with pytest.raises(ValueError, match="initial world boundary"):
        SharedShadowMotor(
            world,
            "b/",
            actor,
            clip=p.clip,
            torso_index=p.torso_index,
            build_observation=p.build_observation,
            goal=p.goal,
            offset=p.offset,
            scale=p.scale,
            order=p.order,
        )
