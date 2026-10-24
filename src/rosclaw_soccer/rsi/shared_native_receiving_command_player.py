"""Own-body neural receiving commands, followed by the qualified SIM motor.

No stepping, resets, teacher actions, optimization or source-history playback.
Every command policy evaluation uses current OWN state; all previous motor
actions/contact cues advance only after the underlying shared world executes.
"""

import numpy as np
import torch

from rosclaw_soccer.rsi.receiving_command_actor import ReceivingCommandActor, command_inputs
from rosclaw_soccer.rsi.shared_native_receiving_player import SharedNativeReceivingPlayer

KIND = "receiving-neural-command-bc-native-v1"


def admit_command_actor(actor: ReceivingCommandActor) -> None:
    if (
        type(actor) is not ReceivingCommandActor
        or actor.training
        or any(
            p.requires_grad
            or p.device.type != "cpu"
            or p.dtype != torch.float32
            or not torch.isfinite(p).all()
            for p in actor.parameters()
        )
    ):
        raise ValueError("frozen finite CPU command-role snapshot required")


class SharedNativeReceivingCommandPlayer:
    simulation_only = True

    def __init__(
        self,
        motor: SharedNativeReceivingPlayer,
        actor: ReceivingCommandActor,
        *,
        home_xy: np.ndarray,
        pass_direction: np.ndarray,
    ) -> None:
        admit_command_actor(actor)
        if (
            type(motor) is not SharedNativeReceivingPlayer
            or motor.decisions != 0
            or motor.pending is not None
            or motor.faulted
            or motor.world.control_steps != 0
            or any(
                x.shape != (2,) or x.dtype != np.float64 or not np.isfinite(x).all()
                for x in (home_xy, pass_direction)
            )
            or abs(np.linalg.norm(pass_direction) - 1) > 1e-10
        ):
            raise ValueError(
                "unused own SIM motor and original declared receiving orientation required"
            )
        self.motor, self.actor = motor, actor
        self.home_xy, self.pass_direction = home_xy.copy(), pass_direction.copy()
        self.launch_seen = False
        self.pending_launch: bool | None = None
        self.neural_command_calls = 0
        self.faulted = False

    def propose(
        self, command_body: np.ndarray, anchor_world: np.ndarray, *, frame: int
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if self.faulted:
            raise ValueError("receiving command player fault latched")
        try:
            admit_command_actor(self.actor)
            body = self.motor
            if self.pending_launch is not None or body.world.control_steps != body.boundary:
                raise ValueError("one own actual shared command boundary required")
            q, v = body.layout.canonical(body.world.data)
            features, baseline, next_phase, launched = command_inputs(
                q,
                v,
                body.layout.angular_velocity_body(body.world.data),
                params=body.params,
                last_action=body.last_action,
                phase=body.phase,
                home_xy=self.home_xy,
                pass_direction=self.pass_direction,
                anchor_world=anchor_world,
                launch_seen=self.launch_seen,
                foot_contact_seen=body.foot_contact_seen,
                other_body_contact_seen=body.other_body_contact_seen,
            )
            if (
                type(command_body) is not np.ndarray
                or command_body.dtype != np.float32
                or not np.array_equal(command_body, baseline)
            ):
                raise ValueError("current own pursuit proposal required, never teacher command")
            self.neural_command_calls += 1
            with torch.no_grad():
                command = self.actor(torch.from_numpy(features[None]))[0].numpy().copy()
            target, trace = body.propose(command, anchor_world, frame=frame)
            if next_phase != trace["phase"]:
                raise ValueError("role proposal and actual motor share one original phase clock")
            trace.update(
                command_inputs112=features.copy(),
                command_baseline3=baseline.copy(),
                command_policy_action3=command.copy(),
                command_launch_seen_before=np.array(self.launch_seen),
                command_launch_seen_proposed=np.array(launched),
            )
            self.pending_launch = launched
            return target, trace
        except Exception:
            self.faulted = True
            raise

    def commit(self, *, executed_action: np.ndarray | None = None) -> None:
        if self.faulted:
            raise ValueError("receiving command player fault latched")
        try:
            if self.pending_launch is None:
                raise ValueError("actual proposed own neural command required")
            self.motor.commit(executed_action=executed_action)
            self.launch_seen = self.pending_launch
            self.pending_launch = None
        except Exception:
            self.faulted = True
            raise


def command_metadata(controls: int) -> dict[str, str | int | bool]:
    if type(controls) is not int or not 1 <= controls <= 400:
        raise ValueError("bounded actual neural role controls required")
    return dict(
        controller_kind=KIND,
        receiver_command_neural=True,
        receiver_command_neural_calls=controls,
        receiver_command_updates_during_episode=0,
        receiver_command_teacher_actions_executed=False,
        receiver_command_own_body_and_actual_feedback=True,
    )
