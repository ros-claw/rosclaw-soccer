"""SIM private Dual256 history on ACTUAL controls from another neural skill.

The shadow motor never controls physics. Next observations contain actual
executed target feedback, not its discarded proposal. Handoff is bound to the
same world/player/model and carries only that player's own causal history.
"""

from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor
from rosclaw_soccer.rsi.paid_ancestor_router_student import FrozenAncestralRouterActor
from rosclaw_soccer.rsi.paid_velocity_shadow import policy_action_from_hg_target
from rosclaw_soccer.rsi.shared_native_neural_player import (
    PlayerObservationView,
    SharedNativeNeuralPlayer,
)


class SharedShadowMotor:
    def __init__(
        self,
        world: Any,
        prefix: str,
        actor: DualRecurrentMotor | FrozenAncestralRouterActor,
        *,
        clip: Any,
        torso_index: int,
        build_observation: Any,
        goal: np.ndarray,
        offset: np.ndarray,
        scale: np.ndarray,
        order: np.ndarray,
    ) -> None:
        if (
            type(actor) not in (DualRecurrentMotor, FrozenAncestralRouterActor)
            or actor.training
            or world.failed
            or any(p.requires_grad or p.device.type != "cpu" for p in actor.parameters())
        ):
            raise ValueError("frozen CPU Dual motor and healthy actual world required")
        router = type(actor) is FrozenAncestralRouterActor
        if router and world.control_steps:
            raise ValueError("router shadow must begin at its own actual initial world boundary")
        self.world, self.prefix, self.actor, self.clip = world, prefix, actor, clip
        self.view = PlayerObservationView(world, prefix)
        self.torso_index, self.build_observation = torso_index, build_observation
        self.goal, self.offset, self.scale, self.order = (
            a.copy() for a in (goal, offset, scale, order)
        )
        if (
            self.goal.shape != (3,)
            or not np.isfinite(self.goal).all()
            or any(a.shape != (29,) or not np.isfinite(a).all() for a in (self.offset, self.scale))
            or (self.scale <= 0).any()
            or self.order.dtype.kind not in ("i", "u")
            or sorted(self.order.tolist()) != list(range(29))
        ):
            raise ValueError("exact existing R2 observation/action ABI required")
        if router and not np.array_equal(
            actor.action_scale.detach().numpy(),
            self.scale[np.argsort(self.order)].astype(np.float32),
        ):
            raise ValueError("router policy-order action scale must match actual HG29 codec")
        self.reference = (
            np.asarray(clip.joint_pos[0], np.float32).copy(),
            np.zeros(29, np.float32),
            np.zeros(3, np.float32),
        )
        h, c = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128)) if router else actor.initial_state()
        self.hidden_width = 128 if router else 256
        self.state: tuple[torch.Tensor, torch.Tensor] = (h.detach().clone(), c.detach().clone())
        self.previous_action = np.zeros(29, np.float32)
        if world.control_steps:
            self.previous_action = policy_action_from_hg_target(
                world.last_applied_target(prefix).astype(np.float64),
                self.offset,
                self.scale,
                self.order,
            )
        self.boundary = world.control_steps
        self.pending: tuple[torch.Tensor, torch.Tensor] | None = None
        self.observations = 0

    def observe(self) -> dict[str, np.ndarray]:
        if (
            self.pending is not None
            or self.world.failed
            or self.boundary != self.world.control_steps
        ):
            raise ValueError("one own shadow observation per actual live control boundary required")
        obs = self.build_observation(
            self.view,
            None,
            0,
            self.previous_action,
            *self.reference,
            self.view.ball_pos,
            self.goal,
            self.offset,
            obs_dim=None,
        )
        if obs.shape != (160,) or obs.dtype != np.float32 or not np.isfinite(obs).all():
            raise ValueError("exact actual160 shadow observation required")
        with torch.no_grad():
            mean, after = self.actor(torch.from_numpy(obs).reshape(1, 1, 160), self.state)
        if (
            mean.shape != (1, 1, 29)
            or not torch.isfinite(mean).all()
            or any(
                t.shape != (2, 1, self.hidden_width) or not torch.isfinite(t).all() for t in after
            )
        ):
            raise ValueError("finite shadow mean and own256 history required")
        trace = dict(
            obs=obs.copy(),
            discarded_action=mean[0, 0].numpy().copy(),
            h_before=self.state[0].numpy().copy(),
            c_before=self.state[1].numpy().copy(),
            h_after=after[0].numpy().copy(),
            c_after=after[1].numpy().copy(),
        )
        self.pending = (after[0].detach().clone(), after[1].detach().clone())
        return trace

    def commit_actual(self) -> None:
        if self.pending is None or self.world.control_steps != self.boundary + 1:
            raise ValueError("one actual committed control required after shadow inference")
        self.previous_action = policy_action_from_hg_target(
            self.world.last_applied_target(self.prefix).astype(np.float64),
            self.offset,
            self.scale,
            self.order,
        )
        self.state = self.pending
        self.pending = None
        self.boundary = self.world.control_steps
        self.observations += 1

    def handoff(
        self, *, goal: np.ndarray, carry_memory: bool, reference_bridge_controls: int = 0
    ) -> SharedNativeNeuralPlayer:
        if (
            self.pending is not None
            or self.world.failed
            or not self.observations
            or self.boundary != self.world.control_steps
            or type(carry_memory) is not bool
            or (type(self.actor) is FrozenAncestralRouterActor and not carry_memory)
        ):
            raise ValueError("own current committed healthy shadow history required")
        return SharedNativeNeuralPlayer(
            self.world,
            self.prefix,
            self.actor,
            clip=self.clip,
            torso_index=self.torso_index,
            build_observation=self.build_observation,
            goal=goal,
            offset=self.offset,
            scale=self.scale,
            order=self.order,
            initial_state=self.state if carry_memory else None,
            previous_action=self.previous_action,
            # Both memory arms see the same declared reference goal. Only
            # recurrent history changes in the cold-vs-carry comparison.
            reference_context=self.reference,
            reference_bridge_controls=reference_bridge_controls,
        )
