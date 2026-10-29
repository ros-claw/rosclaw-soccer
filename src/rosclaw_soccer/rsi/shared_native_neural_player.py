"""Private Dual256 player history over a shared, measured MuJoCo world.

No player advances physics by itself. Proposals are committed only after the
shared world's next control boundary. Explicit static standing is a new task,
never padded motion data or a held last action.
"""

from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.committed_neural_motor import CommittedNeuralMotor
from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter
from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor
from rosclaw_soccer.rsi.motor_option_state import MotorOptionState
from rosclaw_soccer.rsi.paid_ancestor_router_student import FrozenAncestralRouterActor
from rosclaw_soccer.rsi.private_motor_history import (
    MotorHistory,
    clone_history,
    history_trace,
    recurrent_history,
)
from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter
from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld
from rosclaw_soccer.rsi.supported_refinement_motor import SupportedRefinementMotor


class PlayerObservationView:
    """Only the measured per-body properties consumed by the pinned160 codec."""

    def __init__(self, world: SharedNativeMotorWorld, prefix: str) -> None:
        self.world, self.layout = world, world.layouts[prefix]

    @property
    def joint_pos(self) -> np.ndarray:
        return np.array(self.world.data.qpos[self.layout.qi], copy=True)

    @property
    def joint_vel(self) -> np.ndarray:
        return np.array(self.world.data.qvel[self.layout.vi], copy=True)

    @property
    def pelvis_pos(self) -> np.ndarray:
        return np.array(self.world.data.xpos[self.layout.pelvis], copy=True)

    @property
    def pelvis_quat(self) -> np.ndarray:
        return np.array(self.world.data.xquat[self.layout.pelvis], copy=True)

    @property
    def base_ang_vel(self) -> np.ndarray:
        start = self.layout.rv + 3
        return np.array(self.world.data.qvel[start : start + 3], copy=True)

    @property
    def ball_pos(self) -> np.ndarray:
        return np.array(self.world.data.xpos[self.layout.ball], copy=True)


class SharedNativeNeuralPlayer:
    def __init__(
        self,
        world: SharedNativeMotorWorld,
        prefix: str,
        actor: DualRecurrentMotor
        | ConsolidatedMotorAdapter
        | RefinementMotorAdapter
        | SupportedRefinementMotor
        | CommittedNeuralMotor
        | FrozenAncestralRouterActor,
        *,
        clip: Any,
        torso_index: int,
        build_observation: Any,
        goal: np.ndarray,
        offset: np.ndarray,
        scale: np.ndarray,
        order: np.ndarray,
        initial_state: MotorHistory | None = None,
        previous_action: np.ndarray | None = None,
        reference_context: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        reference_bridge_controls: int = 0,
        reference_rate: float = 1.0,
    ) -> None:
        if (
            type(actor)
            not in (
                DualRecurrentMotor,
                ConsolidatedMotorAdapter,
                RefinementMotorAdapter,
                SupportedRefinementMotor,
                CommittedNeuralMotor,
                FrozenAncestralRouterActor,
            )
            or actor.training
            or any(p.requires_grad or p.device.type != "cpu" for p in actor.parameters())
            or not 1 <= clip.length <= 300
            or goal.shape != (3,)
            or not np.isfinite(goal).all()
            or order.dtype.kind not in ("i", "u")
            or sorted(order.tolist()) != list(range(29))
            or any(a.shape != (29,) or not np.isfinite(a).all() for a in (offset, scale))
            or (scale <= 0).any()
            or type(reference_rate) not in (int, float)
            or not np.isfinite(reference_rate)
            or not 0.5 <= reference_rate <= 1.0
        ):
            raise ValueError(
                "frozen explicit Dual256 motor, real reference and HG29 codec required"
            )
        router = type(actor) is FrozenAncestralRouterActor
        width = 128 if router else 256
        if router and not np.array_equal(
            actor.action_scale.detach().numpy(), scale[np.argsort(order)].astype(np.float32)
        ):
            raise ValueError("router policy-order action scale must match actual HG29 codec")
        if router and world.control_steps and initial_state is None:
            raise ValueError("live router requires explicitly supplied own 128-wide history")
        if initial_state is not None:
            recurrent_history(initial_state, hidden_width=width)
            if (type(actor) is CommittedNeuralMotor) != (type(initial_state) is MotorOptionState):
                raise ValueError("explicit actor-bound private history representation required")
            if (
                type(actor) is CommittedNeuralMotor
                and type(initial_state) is MotorOptionState
                and initial_state.generation_hash != actor.generation_hash
            ):
                raise ValueError("handoff history belongs to a different model generation")
        if previous_action is not None and (
            previous_action.shape != (29,)
            or previous_action.dtype != np.float32
            or not np.isfinite(previous_action).all()
        ):
            raise ValueError("finite actual previous R2 action required")
        if world.control_steps and (
            previous_action is None
            or not np.allclose(
                offset + previous_action[order] * scale,
                world.last_applied_target(prefix),
                rtol=0,
                atol=3e-7,
            )
        ):
            raise ValueError("live skill handoff must carry the ACTUAL last executed target")
        if reference_context is not None and (
            len(reference_context) != 3
            or any(
                a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all()
                for a, shape in zip(reference_context, ((29,), (29,), (3,)), strict=True)
            )
        ):
            raise ValueError("explicit finite static-reference handoff context required")
        self.world, self.prefix, self.actor, self.clip = world, prefix, actor, clip
        self.hidden_width = width
        self.view = PlayerObservationView(world, prefix)
        if type(reference_bridge_controls) is not int or not 0 <= reference_bridge_controls <= 50:
            raise ValueError("explicit bounded0..50 reference teacher controls required")
        self.reference_bridge = None
        if reference_bridge_controls:
            from rosclaw_soccer.rsi.causal_reference_bridge import CausalReferenceBridge

            self.reference_bridge = CausalReferenceBridge(
                self.view.joint_pos[np.argsort(order)], controls=reference_bridge_controls
            )
        self.torso_index, self.build_observation = torso_index, build_observation
        self.goal, self.offset, self.scale, self.order = (
            a.copy() for a in (goal, offset, scale, order)
        )
        if initial_state is None:
            initial_state = (
                (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
                if router
                else actor.initial_state()
            )
        self.state = clone_history(initial_state, hidden_width=width)
        self.last_action = (
            np.zeros(29, np.float32) if previous_action is None else previous_action.copy()
        )
        self.ref_pos, self.ref_vel = np.zeros(29, np.float32), np.zeros(29, np.float32)
        self.ref_ang = np.zeros(3, np.float32)
        if reference_context is not None:
            self.ref_pos, self.ref_vel, self.ref_ang = (a.copy() for a in reference_context)
        self.frame, self.decisions = 0, 0
        self.reference_rate = float(reference_rate)
        self.reference_position = 0.0
        self.phase = "MOTION"
        self.boundary = world.control_steps
        self.pending: dict[str, Any] | None = None

    def begin_standing(self) -> None:
        if (
            self.phase != "MOTION_COMPLETE"
            or self.pending is not None
            or self.world.failed
            or self.boundary != self.world.control_steps
        ):
            raise ValueError("actual healthy complete motion boundary required for new stand task")
        self.ref_vel = np.zeros(29, np.float32)
        self.ref_ang = np.zeros(3, np.float32)
        self.phase = "STANDING"

    def propose(
        self, *, with_reference_evidence: bool = False
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if (
            self.pending is not None
            or self.world.failed
            or self.boundary != self.world.control_steps
            or self.phase not in ("MOTION", "STANDING")
        ):
            raise ValueError("one own proposal at the current live shared boundary required")
        reference = (self.ref_pos, self.ref_vel, self.ref_ang)
        if self.reference_bridge is not None:
            reference = self.reference_bridge.reference(*reference, frame=self.decisions)
        if type(with_reference_evidence) is not bool:
            raise ValueError("explicit simulation teacher evidence mode required")
        if type(self.actor) is FrozenAncestralRouterActor and with_reference_evidence:
            raise ValueError("router adapter has no qualified motor teacher authority")
        obs = self.build_observation(
            self.view,
            None,
            0,
            self.last_action,
            *reference,
            self.view.ball_pos,
            self.goal,
            self.offset,
            obs_dim=None,
        )
        if obs.shape != (160,) or obs.dtype != np.float32 or not np.isfinite(obs).all():
            raise ValueError("exact actual per-player160 observation required; no padding")
        nominal: np.ndarray | None = None
        support_activation: np.ndarray | None = None
        state: MotorHistory
        with torch.no_grad():
            if isinstance(self.actor, CommittedNeuralMotor):
                if not isinstance(self.state, MotorOptionState):
                    raise ValueError("explicit committed model history required")
                mean, state, evidence = self.actor.forward_with_evidence(
                    torch.from_numpy(obs).reshape(1, 1, 160), self.state
                )
                if with_reference_evidence:
                    nominal = evidence["reference_nominal"][0, 0].numpy().copy()
            elif with_reference_evidence or type(self.actor) is SupportedRefinementMotor:
                if not isinstance(self.state, tuple):
                    raise ValueError("legacy motor requires explicit two-tensor history")
                mean, state, evidence = self.actor.forward_with_evidence(
                    torch.from_numpy(obs).reshape(1, 1, 160), self.state
                )
                if with_reference_evidence:
                    nominal = evidence["reference_nominal"][0, 0].numpy().copy()
                if type(self.actor) is SupportedRefinementMotor:
                    activation = evidence["support_activation"]
                    if (
                        activation.shape != (1, 1, 1)
                        or not torch.isfinite(activation).all()
                        or ((activation < 0) | (activation > 1)).any()
                    ):
                        raise ValueError("finite own-state applicability evidence required")
                    support_activation = activation[0, 0].numpy().copy()
            else:
                mean, state = self.actor(torch.from_numpy(obs).reshape(1, 1, 160), self.state)
        if mean.shape != (1, 1, 29) or not torch.isfinite(mean).all():
            raise ValueError("finite own HG29 action and private Dual256 history required")
        private_trace = history_trace(self.state, state, hidden_width=self.hidden_width)
        action = mean[0, 0].numpy().copy()
        trace = dict(
            NN_obs=obs.copy(),
            NN_action=action.copy(),
            **private_trace,
        )
        if nominal is not None:
            trace["NN_reference_nominal"] = nominal.copy()
        if support_activation is not None:
            trace["NN_support_activation"] = support_activation
        self.pending = dict(
            action=action,
            state=clone_history(state, hidden_width=self.hidden_width),
            reference_nominal=nominal,
        )
        return self.offset + action[self.order] * self.scale, trace

    def commit(self, *, executed_action: np.ndarray | None = None) -> None:
        """Commit only action feedback bound to this body's ACTUAL last target.

        Explicit teacher actions require own frozen-reference evidence and the
        unchanged total±.2rad authority. No reverse-normalization rounding or
        discarded mean is silently substituted for the executed policy action.
        """
        self._commit(executed_action=executed_action, transition_target=None)

    def commit_sim_transition(self, *, executed_action: np.ndarray, target: np.ndarray) -> None:
        """Record a separately selected simulated transition, not teacher authority.

        The caller owns the transition and must execute this exact target in the
        shared MuJoCo world. This does not modify the bounded teacher path.
        """
        if self.phase != "STANDING" or not isinstance(target, np.ndarray):
            raise ValueError("standing SIM transition and explicit executed target required")
        self._commit(executed_action=executed_action, transition_target=target)

    def commit_motion_sim_transition(
        self, *, executed_action: np.ndarray, target: np.ndarray
    ) -> None:
        """Commit an exact SIM transition target while advancing a motion reference.

        This exposes no hardware authority and does not certify a safe blend.
        The own physical target must match exactly, as in the standing variant.
        """
        if self.phase != "MOTION" or not isinstance(target, np.ndarray):
            raise ValueError("motion SIM transition and explicit executed target required")
        self._commit(executed_action=executed_action, transition_target=target)

    def _commit(
        self, *, executed_action: np.ndarray | None, transition_target: np.ndarray | None
    ) -> None:
        if self.pending is None or self.world.control_steps != self.boundary + 1:
            raise ValueError("own proposal and exactly one executed shared control required")
        action = self.pending["action"] if executed_action is None else executed_action
        if (
            not isinstance(action, np.ndarray)
            or action.shape != (29,)
            or action.dtype != np.float32
            or not np.isfinite(action).all()
        ):
            raise ValueError("finite actual original-unit R2 action required")
        if executed_action is not None and transition_target is None:
            reference = self.pending["reference_nominal"]
            if (
                reference is None
                or np.max(abs((action - reference) * self.scale[np.argsort(self.order)])) > 0.20002
            ):
                raise ValueError(
                    "explicit own reference and unchanged total motor authority required"
                )
        if transition_target is not None and not np.array_equal(
            transition_target, self.world.last_applied_target(self.prefix)
        ):
            raise ValueError("SIM transition must identify the ACTUAL executed target")
        if not np.array_equal(
            self.offset + action[self.order] * self.scale,
            self.world.last_applied_target(self.prefix),
        ):
            raise ValueError("feedback must reconstruct this body's ACTUAL executed target exactly")
        self.state = self.pending["state"]
        self.last_action = action.copy()
        self.pending = None
        if self.phase == "MOTION":
            if self.reference_rate == 1.0:
                self.ref_pos = self.clip.joint_pos[self.frame].copy()
                self.ref_vel = self.clip.joint_vel[self.frame].copy()
                self.ref_ang = self.clip.body_ang_vel_w[self.frame, self.torso_index].copy()
            else:
                from rosclaw_soccer.rsi.motion_reference_tempo import sample_reference

                self.ref_pos, self.ref_vel, self.ref_ang = sample_reference(
                    self.clip, self.torso_index, self.reference_position, self.reference_rate
                )
            self.reference_position += self.reference_rate
            self.frame = min(int(self.reference_position), self.clip.length)
            if self.reference_position >= self.clip.length:
                self.phase = "MOTION_COMPLETE"
        self.boundary = self.world.control_steps
        self.decisions += 1
