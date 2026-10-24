"""Own-body neural receiving commands, followed by the qualified SIM motor.

No stepping, resets, teacher actions, optimization or source-history playback.
Every command policy evaluation uses current OWN state; all previous motor
actions/contact cues advance only after the underlying shared world executes.
"""

import numpy as np
import torch

from rosclaw_soccer.rsi.receiving_command_actor import ReceivingCommandActor, command_inputs
from rosclaw_soccer.rsi.receiving_task_command_actor import (
    ReceivingTaskCommandActor,
    receiving_task_context6,
)
from rosclaw_soccer.rsi.shared_native_receiving_player import SharedNativeReceivingPlayer

KIND = "receiving-neural-command-bc-native-v1"
TASK_KIND = "receiving-neural-task-command-dagger-native-v2"


def admit_command_actor(actor: ReceivingCommandActor | ReceivingTaskCommandActor) -> None:
    if (
        type(actor) not in (ReceivingCommandActor, ReceivingTaskCommandActor)
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
        actor: ReceivingCommandActor | ReceivingTaskCommandActor,
        *,
        home_xy: np.ndarray,
        pass_direction: np.ndarray,
        capture_goal_world: np.ndarray | None = None,
    ) -> None:
        admit_command_actor(actor)
        if type(actor) is ReceivingTaskCommandActor:
            if (
                capture_goal_world is None
                or capture_goal_world.shape != (3,)
                or capture_goal_world.dtype != np.float64
                or not np.isfinite(capture_goal_world).all()
            ):
                raise ValueError(
                    "explicit communicated capture goal for task-conditioned actor required"
                )
        elif capture_goal_world is not None:
            raise ValueError("legacy112 actor cannot implicitly consume a task goal")
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
        self.capture_goal_world = None if capture_goal_world is None else capture_goal_world.copy()
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
            policy_inputs = features
            task_context = None
            if self.capture_goal_world is not None:
                task_context = receiving_task_context6(
                    q[:7], self.capture_goal_world, self.pass_direction
                )
                policy_inputs = np.r_[features, task_context]
            with torch.no_grad():
                command = self.actor(torch.from_numpy(policy_inputs[None]))[0].numpy().copy()
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
            if task_context is not None:
                trace.update(
                    command_inputs118=policy_inputs.copy(),
                    command_task_context6=task_context.copy(),
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


def command_metadata(
    controls: int, *, task_conditioned: bool = False
) -> dict[str, str | int | bool]:
    if type(controls) is not int or not 1 <= controls <= 400 or type(task_conditioned) is not bool:
        raise ValueError("bounded actual neural role controls required")
    result: dict[str, str | int | bool] = dict(
        controller_kind=TASK_KIND if task_conditioned else KIND,
        receiver_command_neural=True,
        receiver_command_neural_calls=controls,
        receiver_command_updates_during_episode=0,
        receiver_command_teacher_actions_executed=False,
        receiver_command_own_body_and_actual_feedback=True,
    )
    if task_conditioned:
        result["receiver_command_task_conditioned"] = True
    return result
