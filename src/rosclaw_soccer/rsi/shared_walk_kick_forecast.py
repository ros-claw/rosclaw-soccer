"""Current locomotion -> future measured-history kick, one owned native SIM body.

No physical reset or recorded action playback. A cold immutable policy template
shares only CPU ORT weights; private software history is seeded at future entry.
"""

import copy
from collections import deque
from importlib import import_module
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.experimental_velocity_envelope import (
    ExperimentalVelocityEnvelopeStream,
)
from rosclaw_soccer.providers.g1.kick_measured_history import KickMeasuredSample
from rosclaw_soccer.providers.g1.velocity_stream_branch import (
    VelocityStream,
    fork_velocity_stream,
    velocity_stream_state_hash,
)
from rosclaw_soccer.rsi.kick_reference_phase import nearest_kick_reference_phase
from rosclaw_soccer.rsi.shared_kick_history import measured_kick_sample
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_nn_forecast import copy_sim_policy, policy_state_hash
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer
from rosclaw_soccer.rsi.shared_target_forecast import SharedTargetForecaster, compiled_model_hash
from rosclaw_soccer.rsi.walk_kick_approach_command import approach_command


class _KickCalls:
    """Private observer around stateless CPU ORT; never changes inputs/outputs."""

    def __init__(self, session: Any, maximum_calls: int) -> None:
        import onnxruntime as ort

        if type(session) is not ort.InferenceSession or session.get_providers() != [
            "CPUExecutionProvider"
        ]:
            raise ValueError("original stateless CPU inference session required")
        self.session = session
        self.maximum_calls = maximum_calls
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.attempted = 0

    def get_inputs(self) -> Any:
        return self.session.get_inputs()

    def get_outputs(self) -> Any:
        return self.session.get_outputs()

    def run(self, names: Any, feed: Any) -> Any:
        if self.attempted >= self.maximum_calls:
            raise RuntimeError("future kick NN call budget exhausted")
        if (
            names != ["actions"]
            or type(feed) is not dict
            or set(feed) != {"obs"}
            or type(feed["obs"]) is not np.ndarray
            or feed["obs"].shape != (1, 547)
            or feed["obs"].dtype != np.float32
            or not np.isfinite(feed["obs"]).all()
        ):
            raise ValueError(
                "one original finite547 observation per future kick inference required"
            )
        self.attempted += 1
        self.observations.append(feed["obs"][0].copy())
        result = self.session.run(names, feed)
        self.actions.append(result[0][0].copy())
        return result


class SharedWalkKickForecaster:
    def __init__(
        self,
        world: SharedNativeMotorWorld,
        stream: VelocityStream,
        cold_policy: Any,
        *,
        expected_model_hash: str,
        native_step_budget: int,
    ) -> None:
        if len(world.layouts) != 1 or world._private_policy_owners or cold_policy.time_step != 0:
            raise ValueError(
                "plain current locomotion body and unused cold SIM kick template required"
            )
        velocity_stream_state_hash(stream, completed_control=world.control_steps)
        copy_sim_policy(cold_policy)
        self.world, self.stream, self.template = world, stream, cold_policy
        self.prefix = next(iter(world.layouts))
        self.native = SharedTargetForecaster(
            world, expected_model_hash=expected_model_hash, native_step_budget=native_step_budget
        )
        self.velocity_calls = self.kick_calls = self.attempted_kick_calls = 0
        self.last_attempt: dict[str, Any] | None = None

    def counters(self) -> dict[str, int]:
        return dict(
            self.native.counters(),
            future_velocity_nn_calls=self.velocity_calls,
            future_kick_nn_calls=self.kick_calls,
            attempted_future_kick_nn_calls=self.attempted_kick_calls,
        )

    def forecast(
        self,
        *,
        velocity_controls: int,
        kick_controls: int,
        previous_command: np.ndarray,
        parameters: dict[str, Any],
        history: tuple[KickMeasuredSample, ...],
        prior_contact: bool,
    ) -> dict[str, Any]:
        import mujoco

        donor, stream = self.world, self.stream
        donor._require_current_boundary()
        origin = donor.control_steps
        before = velocity_stream_state_hash(stream, completed_control=origin)
        template_hash = policy_state_hash(self.template)
        if (
            type(velocity_controls) is not int
            or not 0 <= velocity_controls <= 16
            or type(kick_controls) is not int
            or not 1 <= kick_controls <= 128
            or not 5 <= origin <= 239
            or origin + velocity_controls > 239
            or origin + velocity_controls + kick_controls > 360
            or prior_contact is not False
            or type(previous_command) is not np.ndarray
            or previous_command.shape != (3,)
            or previous_command.dtype != np.float32
            or not np.isfinite(previous_command).all()
            or type(history) is not tuple
            or len(history) != 5
            or any(type(s) is not KickMeasuredSample for s in history)
            or tuple(s.frame for s in history) != tuple(range(origin - 4, origin + 1))
            or history[-1] != measured_kick_sample(donor, self.prefix)
            or any(
                abs(b.time_sec - a.time_sec - 0.02) > 1e-6
                for a, b in zip(history, history[1:], strict=False)
            )
        ):
            raise ValueError(
                "bounded current precontact body and true five-frame measured history required"
            )
        pd = donor.parameters[self.prefix]
        if (
            donor._torque_layer_owners
            or compiled_model_hash(donor.model) != self.native.model_hash
            or not np.array_equal(pd.stiffness, stream.params.stiffness)
            or not np.array_equal(pd.damping, stream.params.damping)
        ):
            raise ValueError("unchanged current native velocity PD and compiled body required")
        requested = velocity_controls + kick_controls
        if self.native.attempted_steps + requested * 10 > self.native.budget:
            raise RuntimeError(
                "composed native prediction budget exhausted; no automatic extension"
            )
        ghost = fork_velocity_stream(stream, completed_control=origin)
        policy = copy_sim_policy(self.template)
        calls = _KickCalls(policy.ort_session, kick_controls)
        policy.ort_session = calls
        mujoco.mj_copyData(self.native.scratch, donor.model, donor.data)
        owned = SharedNativeMotorWorld(
            donor.model,
            self.native.scratch,
            players=donor.parameters,
            ball_body_name=donor.model.body(donor.ball).name,
        )
        lo = owned.layouts[self.prefix]
        measured = deque(history, maxlen=5)
        cfg = copy.deepcopy(parameters)
        previous = previous_command.copy()
        raw: dict[str, list[np.ndarray]] = {
            k: [] for k in ("qpos", "qvel", "torque", "ball_forces")
        }
        targets, commands, phases, velobs, velactions, kickactions, kickticks = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        kick = None
        receipt = None
        admitted = None
        entry_samples = None
        admission_rejection = None
        velocity_contact = False
        reference_phase = None
        self.native.calls += 1
        self.native.attempted_steps += 10 * requested
        self.last_attempt = None
        try:
            for local in range(requested):
                q, v = lo.canonical(owned.data)
                if local == velocity_controls:
                    delta = owned.data.xmat[lo.pelvis].reshape(3, 3).T @ (
                        owned.data.xpos[lo.ball] - q[:3]
                    )
                    if (
                        velocity_contact
                        or not 50 <= origin + local <= 239
                        or not 0 < delta[0] <= cfg["entry_gap_m"]
                        or abs(delta[1]) > 0.25
                    ):
                        admission_rejection = "CONTACT_OR_ORIGINAL_MEASURED_ENTRY_GEOMETRY"
                        break
                    reference_phase = cfg["reference_frame"]
                    if reference_phase is None or cfg.get("entry_phase_feedback", False):
                        mapping = np.asarray(import_module(type(policy).__module__).MUJOCO_TO_ISAAC)
                        reference_phase = nearest_kick_reference_phase(
                            q[7:36],
                            v[6:35],
                            policy.motion_joint_pos[:, mapping],
                            policy.motion_joint_vel[:, mapping],
                            minimum_frame=180 if cfg.get("entry_phase_feedback", False) else 0,
                        )["reference_frame"]
                    switch = owned.switch_pd_profile(
                        self.prefix,
                        NativePDParameters(np.array(policy.kps), np.array(policy.kds), pd.effort),
                    )
                    if not switch["physics_unchanged"]:
                        raise ValueError("future skill switch modified physical state")
                    kick = SharedRoboNaldoPlayer(
                        owned, self.prefix, policy, goal=np.array([8.0, 0.0, 0.11])
                    )
                    kick._forecast_control_origin = origin
                    entry_samples = np.stack([s.numeric_record() for s in measured])
                    receipt = kick.install_measured_entry(
                        tuple(measured), reference_frame=reference_phase
                    )
                    admitted = origin + local
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
                if kick is None:
                    target = ghost.propose(
                        q, v, lo.angular_velocity_body(owned.data), command, frame=origin + local
                    )
                    velobs.append(ghost.last_observation.copy())
                    velactions.append(ghost.last_action.copy())
                    phases.append(ghost.phase)
                else:
                    target, proposal = kick.propose()
                    if proposal["warmup_without_NN"]:
                        raise ValueError(
                            "future measured entry may not bypass full NN with interpolation"
                        )
                    kickactions.append(policy.last_action_il.copy())
                    kickticks.append(proposal["policy_tick_before"])
                result = owned.advance({self.prefix: target})
                if kick is not None:
                    kick.commit()
                for key in raw:
                    raw[key].append(result[key])
                targets.append(target.copy())
                commands.append(command.copy())
                previous = command.copy()
                if kick is None:
                    velocity_contact |= bool(owned.last_ball_contact_bodies[self.prefix])
                if owned.failed:
                    break
                measured.append(measured_kick_sample(owned, self.prefix, control_origin=origin))
        finally:
            self.native.integrated_steps += owned.native_steps
            self.velocity_calls += ghost.neural_calls - stream.neural_calls
            self.kick_calls += len(calls.actions)
            self.attempted_kick_calls += calls.attempted
            self.last_attempt = dict(
                **{k: np.concatenate(v) for k, v in raw.items() if v},
                joint_targets=np.asarray(targets),
                commands=np.asarray(commands),
                velocity_observations=np.asarray(velobs),
                velocity_actions=np.asarray(velactions),
                velocity_phases=np.asarray(phases),
                kick_observations=np.asarray(calls.observations),
                kick_raw_actions=np.asarray(calls.actions),
                kick_clipped_actions=np.asarray(kickactions),
                kick_policy_ticks=np.asarray(kickticks, dtype=np.int64),
                policy_ticks=np.arange(origin, origin + len(targets), dtype=np.int64),
                controls=owned.control_steps,
                native_steps=owned.native_steps,
                requested_controls=requested,
                donor_control_frame=origin,
                velocity_controls=velocity_controls,
                kick_controls=kick_controls,
                admission_control=admitted,
                admission_rejection=admission_rejection,
                measured_entry_receipt=receipt,
                measured_entry_samples=entry_samples,
                admitted_reference_phase=reference_phase,
                body_failed=owned.failed,
                velocity_contact_observed=velocity_contact,
                source_prefix_controls_not_new=origin,
                completed_control_raw_native_steps=10 * len(targets),
                native_substep_exception_raw_complete=owned.native_steps == 10 * len(targets),
                terminal_qpos=owned.data.qpos.copy(),
                terminal_qvel=owned.data.qvel.copy(),
            )
            donor._require_current_boundary()
            if (
                velocity_stream_state_hash(stream, completed_control=origin) != before
                or policy_state_hash(self.template) != template_hash
            ):
                raise ValueError("composed future mutated donor NN history or cold template")
        return dict(
            **self.last_attempt,
            donor_unchanged=True,
            prediction_is_not_actual_execution=True,
            task_effectiveness_proven=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
