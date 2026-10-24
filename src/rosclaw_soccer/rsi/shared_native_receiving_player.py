"""SIM-only receiving motor with own observations and post-execution feedback.

No physics stepping, transport, reset, checkpoint loading or learning here.
The shared world owns execution. Every proposal queries the original ONNX and
both whole Torch networks on this body's current observation, never source
actions or a teammate's state/history. Three B network evaluations per control.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.providers.g1.velocity_policy import (
    G1VelocityPolicy,
    observation98,
    phase_delta,
    target29,
)
from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
from rosclaw_soccer.rsi.bounded_motor_exploration import motor_box
from rosclaw_soccer.rsi.receiver_body_context import receiver_context14
from rosclaw_soccer.rsi.relay_audit_clock import RelayAuditClock
from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld


class SharedNativeReceivingPlayer:
    simulation_only = True

    def __init__(
        self,
        world: Any,
        prefix: str,
        actor: AnchoredReceivingMotor,
        *,
        model: Path,
        config: Path,
        expected_model_hash: str,
        expected_config_hash: str,
        horizon_controls: int = 400,
    ) -> None:
        if (
            type(world) not in (SharedNativeMotorWorld, RelayAuditClock)
            or prefix not in world.layouts
            or world.failed
            or world.control_steps != 0
            or type(horizon_controls) is not int
            or not 1 <= horizon_controls <= 400
            or type(actor) is not AnchoredReceivingMotor
            or actor.training
            or any(p.requires_grad or p.device.type != "cpu" for p in actor.parameters())
        ):
            raise ValueError(
                "unused shared SIM world and frozen explicit CPU receiving motor required"
            )
        # Immutable CPU loader only; never invoke or extend the legacy H300 loop.
        loaded = G1VelocityPolicy(
            model,
            config,
            expected_model_hash=expected_model_hash,
            expected_config_hash=expected_config_hash,
        )
        self.params, self.session = loaded.params, loaded.session
        self.scale = torch.tensor(self.params.action_scale, dtype=torch.float32)
        if not torch.equal(actor.action_scale, self.scale):
            raise ValueError("unchanged HG29 receiving codec required")
        self.world, self.prefix, self.actor = world, prefix, actor
        self.effort = world.parameters[prefix].effort.copy()
        self._check_profile()
        self.layout = world.layouts[prefix]
        self.horizon_controls = horizon_controls
        self.phase = phase_delta()
        self.last_action = np.zeros(29, np.float32)
        self.last_observation = np.zeros(98, np.float32)
        self.foot_contact_seen = self.other_body_contact_seen = False
        self.boundary = self.decisions = 0
        self.reference_neural_calls = self.whole_mlp_neural_calls = self.attempted_neural_calls = 0
        self.faulted = False
        self.pending: dict[str, Any] | None = None

    def _check_profile(self) -> None:
        pd = self.world.parameters[self.prefix]
        if (
            not np.array_equal(pd.stiffness, np.array(self.params.stiffness))
            or not np.array_equal(pd.damping, np.array(self.params.damping))
            or not np.array_equal(pd.effort, self.effort)
        ):
            raise ValueError(
                "receiving profile must match original velocity gains and fixed effort"
            )

    def propose(
        self, command_body: np.ndarray, anchor_world: np.ndarray, *, frame: int
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if self.faulted:
            raise ValueError("receiving motor fault latched; no implicit retry")
        try:
            w = self.world
            self._check_profile()
            if (
                self.pending is not None
                or w.failed
                or w.control_steps != self.boundary
                or type(frame) is not int
                or frame != self.decisions
                or not 0 <= frame < self.horizon_controls
                or self.actor.training
                or any(p.requires_grad for p in self.actor.parameters())
                or not torch.equal(self.actor.action_scale, self.scale)
            ):
                raise ValueError(
                    "one own proposal at the actual consecutive shared boundary required"
                )
            q, v = self.layout.canonical(w.data)
            obs, phase = observation98(
                q,
                v,
                self.layout.angular_velocity_body(w.data),
                command_body,
                self.last_action,
                self.phase,
                self.params,
            )
            cues = receiver_context14(
                q,
                v,
                anchor_world,
                foot_contact_seen=self.foot_contact_seen,
                other_body_contact_seen=self.other_body_contact_seen,
            )
            self.attempted_neural_calls += 1
            reference = self.session.run(["actions"], {"obs": obs[None]})[0]
            self.reference_neural_calls += 1
            if (
                type(reference) is not np.ndarray
                or reference.shape != (1, 29)
                or reference.dtype != np.float32
            ):
                raise ValueError("exact original CPU receiving reference required")
            target29(
                reference[0], self.params
            )  # Reject a failed original reference, never repair it.
            inputs = np.concatenate((obs, cues))[None]
            self.attempted_neural_calls += 2
            with torch.no_grad():
                means, evidence = self.actor.forward_with_evidence(
                    torch.from_numpy(inputs), torch.from_numpy(reference.copy())
                )
            self.whole_mlp_neural_calls += 2
            mean = means[0].numpy().copy()
            target = target29(mean, self.params)
            self.pending = dict(
                mean=mean.copy(), reference=reference[0].copy(), obs=obs.copy(), phase=phase
            )
            trace = dict(
                obs=obs.copy(),
                action=mean.copy(),
                command=command_body.copy(),
                phase=np.array(phase),
                control=np.array(frame),
                context14=cues.copy(),
                reference_action=reference[0].copy(),
                realized_delta_radians=evidence["realized_delta_radians"][0].numpy().copy(),
                contact_flags_before=np.array(
                    [self.foot_contact_seen, self.other_body_contact_seen]
                ),
            )
            return target, trace
        except Exception:
            self.faulted = True
            raise

    def commit(self, *, executed_action: np.ndarray | None = None) -> None:
        if self.faulted:
            raise ValueError("receiving motor fault latched; no implicit retry")
        try:
            self._check_profile()
            if self.pending is None or self.world.control_steps != self.boundary + 1:
                raise ValueError("own proposal and exactly one actual shared control required")
            actual = self.pending["mean"] if executed_action is None else executed_action
            if (
                type(actual) is not np.ndarray
                or actual.shape != (29,)
                or actual.dtype != np.float32
                or not np.isfinite(actual).all()
            ):
                raise ValueError("finite actual executed HG29 receiving action required")
            lower, upper = motor_box(torch.from_numpy(self.pending["reference"]), self.scale)
            value = torch.from_numpy(actual).double()
            if ((value < lower) | (value > upper)).any():
                raise ValueError("actual receiving action exceeded original declared box")
            if not np.array_equal(
                target29(actual, self.params), self.world.last_applied_target(self.prefix)
            ):
                raise ValueError("receiving feedback must match actual applied target exactly")
            feet = {self.prefix + "left_ankle_roll_link", self.prefix + "right_ankle_roll_link"}
            foot, other = self.foot_contact_seen, self.other_body_contact_seen
            for name, force in self.world.last_ball_contact_bodies[self.prefix].items():
                if not name.startswith(self.prefix) or not np.isfinite(force) or force <= 1:
                    raise ValueError("actual >1N own contact witnesses required")
                foot = foot or name in feet
                other = other or name not in feet
            self.last_action = actual.copy()
            self.last_observation = self.pending["obs"].copy()
            self.phase = self.pending["phase"]
            self.foot_contact_seen, self.other_body_contact_seen = foot, other
            self.pending = None
            self.boundary = self.world.control_steps
            self.decisions += 1
        except Exception:
            self.faulted = True
            raise
