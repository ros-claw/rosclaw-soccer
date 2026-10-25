"""Bounded own-state multi-control velocity SIM futures, without skill switching.

Every predicted step recomputes the measured approach command and original NN.
No recorded future commands, targets, body states or ball forcing are accepted.
"""

import copy
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.experimental_velocity_envelope import (
    ExperimentalVelocityEnvelopeStream,
)
from rosclaw_soccer.providers.g1.velocity_stream_branch import (
    VelocityStream,
    fork_velocity_stream,
    velocity_stream_state_hash,
)
from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_target_forecast import SharedTargetForecaster, compiled_model_hash
from rosclaw_soccer.rsi.walk_kick_approach_command import approach_command


class SharedVelocityTrajectoryForecaster:
    def __init__(
        self,
        world: SharedNativeMotorWorld,
        stream: VelocityStream,
        *,
        expected_model_hash: str,
        native_step_budget: int,
    ) -> None:
        if len(world.layouts) != 1 or world._private_policy_owners:
            raise ValueError("one plain current locomotion SIM body required")
        velocity_stream_state_hash(stream, completed_control=world.control_steps)
        self.world, self.stream = world, stream
        self.prefix = next(iter(world.layouts))
        self.native = SharedTargetForecaster(
            world, expected_model_hash=expected_model_hash, native_step_budget=native_step_budget
        )
        self.neural_calls = 0
        self.last_attempt: dict[str, Any] | None = None
        self._require_velocity_pd()

    def _require_velocity_pd(self) -> None:
        pd, params = self.world.parameters[self.prefix], self.stream.params
        if not np.array_equal(pd.stiffness, params.stiffness) or not np.array_equal(
            pd.damping, params.damping
        ):
            raise ValueError("multi-control locomotion future requires unchanged velocity PD")

    def counters(self) -> dict[str, int]:
        return dict(self.native.counters(), future_velocity_nn_calls=self.neural_calls)

    def forecast(
        self,
        *,
        controls: int,
        previous_command: np.ndarray,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        import mujoco

        donor, stream = self.world, self.stream
        donor._require_current_boundary()
        origin = donor.control_steps
        before = velocity_stream_state_hash(stream, completed_control=origin)
        self._require_velocity_pd()
        if (
            type(controls) is not int
            or not 1 <= controls <= 32
            or origin + controls > stream.horizon_controls
            or type(previous_command) is not np.ndarray
            or previous_command.shape != (3,)
            or previous_command.dtype != np.float32
            or not np.isfinite(previous_command).all()
        ):
            raise ValueError(
                "bounded1..32 future controls and actual prior float32 command required"
            )
        if donor._torque_layer_owners or compiled_model_hash(donor.model) != self.native.model_hash:
            raise ValueError("unchanged compiled native body and original PD law required")
        if (
            self.native.attempted_steps + controls * 10 > self.native.budget
            or self.neural_calls + controls > self.native.budget // 10
        ):
            raise RuntimeError("multi-control velocity budget exhausted; no retry extension")
        ghost = fork_velocity_stream(stream, completed_control=origin)
        mujoco.mj_copyData(self.native.scratch, donor.model, donor.data)
        owned = SharedNativeMotorWorld(
            donor.model,
            self.native.scratch,
            players=donor.parameters,
            ball_body_name=donor.model.body(donor.ball).name,
            measure_interplayer_contact=donor.measure_interplayer_contact,
        )
        lo = owned.layouts[self.prefix]
        cfg = copy.deepcopy(parameters)
        previous = previous_command.copy()
        raw: dict[str, list[np.ndarray]] = {
            k: [] for k in ("qpos", "qvel", "torque", "ball_forces")
        }
        targets, commands, observations, actions, phases = [], [], [], [], []
        self.native.calls += 1
        self.native.attempted_steps += controls * 10
        self.last_attempt = None
        try:
            for local in range(controls):
                q, v = lo.canonical(owned.data)
                command = approach_command(
                    q=q,
                    v=v,
                    rotation=owned.data.xmat[lo.pelvis].reshape(3, 3),
                    cached_ball_position=owned.data.xpos[lo.ball],
                    previous=previous,
                    parameters=cfg,
                    fast=type(stream) is ExperimentalVelocityEnvelopeStream,
                    feedback=cfg.get("approach_feedback", False),
                )
                target = ghost.propose(
                    q,
                    v,
                    lo.angular_velocity_body(owned.data),
                    command,
                    frame=origin + local,
                )
                result = owned.advance({self.prefix: target})
                for key in raw:
                    raw[key].append(result[key])
                targets.append(target.copy())
                commands.append(command.copy())
                observations.append(ghost.last_observation.copy())
                actions.append(ghost.last_action.copy())
                phases.append(ghost.phase)
                previous = command.copy()
                if owned.failed:
                    break
        finally:
            self.native.integrated_steps += owned.native_steps
            self.neural_calls += ghost.neural_calls - stream.neural_calls
            # Also retained if an NN/contract exception interrupts this attempt.
            self.last_attempt = dict(
                **{k: np.concatenate(v) for k, v in raw.items() if v},
                joint_targets=np.asarray(targets),
                commands=np.asarray(commands),
                observations=np.asarray(observations),
                actions=np.asarray(actions),
                phases=np.asarray(phases),
                policy_ticks=np.arange(origin, origin + len(targets), dtype=np.int64),
                controls=owned.control_steps,
                native_steps=owned.native_steps,
                requested_controls=controls,
                donor_control_frame=origin,
                body_failed=owned.failed,
                neural_faulted=ghost.faulted,
                completed_control_raw_native_steps=10 * len(targets),
                native_substep_exception_raw_complete=owned.native_steps == 10 * len(targets),
                actual_completed_control=stream.next_frame,
                copied_next_frame=ghost.next_frame,
                donor_stream_hash=before,
                terminal_qpos=owned.data.qpos.copy(),
                terminal_qvel=owned.data.qvel.copy(),
            )
            donor._require_current_boundary()
            if velocity_stream_state_hash(stream, completed_control=origin) != before:
                raise ValueError("actual velocity NN history mutated during private trajectory")
        return dict(
            **self.last_attempt,
            copied_stream_after_hash=velocity_stream_state_hash(
                ghost, completed_control=origin + owned.control_steps
            ),
            donor_unchanged=True,
            prediction_is_not_actual_execution=True,
            future_skill_switch_prediction_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
