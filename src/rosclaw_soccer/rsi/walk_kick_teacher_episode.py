"""One native physical evaluation of a declared whole-body teacher proposal."""

import contextlib
import io
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
from rosclaw_soccer.providers.g1.kick_measured_history import KickMeasuredSample
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.kick_reference_phase import nearest_kick_reference_phase
from rosclaw_soccer.rsi.shared_kick_history import measured_kick_sample
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer
from rosclaw_soccer.rsi.walk_kick_approach_command import approach_command


def evaluate_episode(
    *,
    model: Any,
    assets: dict[str, Any],
    policy_types: tuple[Any, ...],
    course: tuple[float, float, float],
    parameters: dict[str, Any],
    hard_limits: np.ndarray,
    player_type: Any = SharedRoboNaldoPlayer,
    post_contact_exit: bool = False,
    protect_passing_entry_prediction: bool = False,
    velocity_boundary_probe: Callable[..., None] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """No copied rollout state or action playback; every call creates fresh NN state.

    Initialization ends at world construction. All subsequent physics belongs
    to the shared clock; every joint target is executed under unchanged caps.
    """
    import mujoco

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

    state_type, output_type, policy_type, mapping = policy_types
    if (
        type(post_contact_exit) is not bool
        or type(protect_passing_entry_prediction) is not bool
        or (protect_passing_entry_prediction and not post_contact_exit)
        or (velocity_boundary_probe is not None and not callable(velocity_boundary_probe))
        or (velocity_boundary_probe is not None and post_contact_exit)
    ):
        raise ValueError("explicit post-contact exit research switch required")
    fast = assets.get("experimental_fast", False)
    feedback = parameters.get("approach_feedback", False)
    phase_feedback = parameters.get("entry_phase_feedback", False)
    if (
        type(feedback) is not bool
        or type(phase_feedback) is not bool
        or (phase_feedback and not feedback)
    ):
        raise ValueError("explicit measured-state approach hypothesis switch required")
    if type(fast) is not bool:
        raise ValueError("explicit experimental speed mode required")
    stream_args: dict[str, Any] = dict(
        expected_model_hash=assets["velocity_model_hash"],
        expected_config_hash=assets["velocity_config_hash"],
        horizon_controls=360,
    )
    stream: Any
    if fast:
        from rosclaw_soccer.providers.g1.experimental_velocity_envelope import (
            ExperimentalVelocityEnvelopeStream,
        )

        stream = ExperimentalVelocityEnvelopeStream(
            assets["velocity_model"],
            assets["velocity_config"],
            **stream_args,
            reference_source=assets["reference_source"],
            expected_reference_hash=assets["reference_hash"],
        )
    else:
        stream = ContinuousVelocityStream(
            assets["velocity_model"],
            assets["velocity_config"],
            **stream_args,
        )
    d = mujoco.MjData(model)
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    x, y, vx = course
    d.qpos[lo.rq : lo.rq + 7] = [0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0]
    d.qpos[lo.qi] = np.asarray(stream.params.defaults)
    d.qpos[lo.bq : lo.bq + 7] = [x, y, 0.11, 1.0, 0.0, 0.0, 0.0]
    d.qvel[lo.bv] = vx
    d.qvel[lo.bv + 4] = vx / 0.11  # actual no-slip sphere angular launch, not external force
    mujoco.mj_forward(model, d)
    vp = NativePDParameters(
        np.array(stream.params.stiffness),
        np.array(stream.params.damping),
        0.85 * np.asarray(hard_limits),
    )
    w = SharedNativeMotorWorld(model, d, players={"a/": vp}, ball_body_name="soccer_ball")
    raw: dict[str, list[np.ndarray]] = {k: [] for k in ("qpos", "qvel", "torque", "ball_forces")}
    targets, commands, ball_controls, body_controls = [], [], [], []
    velocity_active = []
    history: deque[KickMeasuredSample] = deque(maxlen=5)
    previous = np.zeros(3, np.float32)
    kick, entry, phase, history_receipt = None, None, None, None
    contacted = False
    entry_state = None
    continuation = None
    if post_contact_exit:
        from rosclaw_soccer.rsi.observed_velocity_continuation import (
            ObservedVelocityContinuation,
        )

        continuation = ObservedVelocityContinuation(stream)
    exit_frame = None
    observed_foot = False
    continuation_receipts, shadow_observations = [], []
    for frame in range(360):
        q, v = lo.canonical(d)
        rot = d.xmat[lo.pelvis].reshape(3, 3)
        delta = rot.T @ (d.xpos[lo.ball] - q[:3])
        if (
            entry is None
            and not contacted
            and 50 <= frame <= 239
            and 0 < delta[0] <= parameters["entry_gap_m"]
            and abs(delta[1]) <= 0.25
        ):
            entry = frame
            entry_state = dict(
                qpos=q.tolist(),
                qvel=v.tolist(),
                measured_history=[s.numeric_record().tolist() for s in history],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                policy = policy_type(state_type(29), output_type(29))
            phase = parameters["reference_frame"]
            if phase is None or phase_feedback:
                phase = nearest_kick_reference_phase(
                    q[7:36],
                    v[6:35],
                    policy.motion_joint_pos[:, mapping],
                    policy.motion_joint_vel[:, mapping],
                    minimum_frame=180 if phase_feedback else 0,
                )["reference_frame"]
            kp = NativePDParameters(np.array(policy.kps), np.array(policy.kds), vp.effort)
            switch = w.switch_pd_profile("a/", kp)
            if not switch["physics_unchanged"]:
                raise ValueError("gain switch modified physical state")
            kick = player_type(w, "a/", policy, goal=np.array([8.0, 0.0, 0.11]))
            history_receipt = kick.install_measured_entry(tuple(history), reference_frame=phase)
        exit_allowed = True
        if (
            protect_passing_entry_prediction
            and kick is not None
            and exit_frame is None
            and observed_foot
        ):
            from rosclaw_soccer.rsi.observed_velocity_continuation import (
                exit_after_entry_abstention,
            )

            exit_allowed = exit_after_entry_abstention(kick.events)
        if (
            post_contact_exit
            and kick is not None
            and exit_frame is None
            and observed_foot
            and exit_allowed
        ):
            exit_frame = frame
            switch = w.switch_pd_profile("a/", vp)
            if not switch["physics_unchanged"]:
                raise ValueError("exit gain switch modified physical state")
        command = approach_command(
            q=q,
            v=v,
            rotation=rot,
            cached_ball_position=d.xpos[lo.ball],
            previous=previous,
            parameters=parameters,
            fast=fast,
            feedback=feedback,
            halt=exit_frame is not None,
        )
        previous = command.copy()
        shadow = None
        if continuation is not None:
            shadow = continuation.propose(q, v, lo.angular_velocity_body(d), command, frame=frame)
            shadow_observations.append(stream.last_observation.copy())
        if continuation is not None and (kick is None or exit_frame is not None):
            assert shadow is not None
            target = shadow
        elif kick is None:
            if velocity_boundary_probe is not None:
                velocity_boundary_probe(stream=stream, world=w, command=command.copy(), frame=frame)
            target = stream.propose(q, v, lo.angular_velocity_body(d), command, frame=frame)
        else:
            target = kick.propose()[0]
        result = w.advance({"a/": target})
        if kick is not None and exit_frame is None:
            kick.commit()
        if continuation is not None:
            continuation_receipts.append(
                continuation.observe_executed(target, completed_control=w.control_steps)
            )
        if not w.failed:
            history.append(measured_kick_sample(w, "a/"))
        contacted |= bool(w.last_ball_contact_bodies["a/"])
        observed_foot |= bool((result["ball_forces"][:, 0, :2] > 1).any())
        for key in raw:
            raw[key].append(result[key])
        targets.append(target.copy())
        commands.append(command.copy())
        velocity_active.append(kick is None or exit_frame is not None)
        # Explicit actual qpos addresses, never infer positions from array order.
        ball_controls.append(result["qpos"][-1, lo.bq : lo.bq + 3].copy())
        body_controls.append(result["qpos"][:, lo.rq + 2].copy())
        if w.failed:
            break
    arrays = {k: np.concatenate(v) for k, v in raw.items()}
    arrays.update(
        joint_targets=np.stack(targets),
        commands=np.stack(commands),
        ball_position_after_step_m=np.stack(ball_controls),
        pelvis_z_per_substep_m=np.stack(body_controls),
        velocity_control_active=np.asarray(velocity_active, dtype=bool),
    )
    if continuation is not None:
        arrays["velocity_shadow_observations"] = np.stack(shadow_observations)
    if w.control_steps >= 300:
        # All non-foot bodies are aggregated in measured force slot2. Empty
        # slots3..5 denote no extra disjoint body categories, not missing data.
        force = np.zeros((300, 1, 6), np.float64)
        force[:, 0, :3] = arrays["ball_forces"][:3000, 0].reshape(300, 10, 3).max(axis=1)
        outcome = label_first_touch(
            dict(
                force_n=force,
                pelvis_z_per_substep_m=arrays["pelvis_z_per_substep_m"][:300, None],
                ball_position_after_step_m=arrays["ball_position_after_step_m"][:300, None],
            )
        ).to_dict()
    else:
        # Legal physical termination is retained, never padded to a fake300.
        feet = np.flatnonzero(arrays["ball_forces"][:, 0, :2].max(axis=1) > 1)
        outcome = dict(
            first_foot_contact_frame=int(feet[0] // 10) if len(feet) else None,
            event_status="PHYSICALLY_TERMINATED",
            measurement_status="INCOMPLETE_PHYSICAL_EPISODE",
            quality_observed=False,
            forward_60_m=None,
            lateral_60_m=None,
            task_pass=False,
            safe=False,
            training_reward=-10.0,
        )
    row = dict(
        parameters=parameters,
        course=course,
        controls=w.control_steps,
        native_steps=w.native_steps,
        healthy_full_horizon=w.control_steps == 360 and not w.failed,
        entry_control=entry,
        entry_state=entry_state,
        admitted_reference_phase=phase,
        measured_history_receipt=history_receipt,
        outcome=outcome,
        velocity_neural_calls=stream.neural_calls,
        kick_neural_proposals=kick.proposals if kick else 0,
        native_nonfoot_contact_samples=int((arrays["ball_forces"][:, 0, 2] > 1).sum()),
        native_nonfoot_peak_force_n=float(arrays["ball_forces"][:, 0, 2].max()),
        ball_qpos_address=lo.bq,
        robot_qpos_address=lo.rq,
        optimizer_updates=0,
        experimental_fast_approach=fast,
        feedback_approach=feedback,
        feedback_reference_phase=phase_feedback,
        physical_teacher_qualified=False,
    )
    if continuation is not None:
        row.update(
            post_contact_exit=True,
            protect_passing_entry_prediction=protect_passing_entry_prediction,
            exit_gate_uses_source_course_or_outcome=False,
            exit_control=exit_frame,
            exit_trigger="completed_previous_control_foot_force_gt1N",
            exit_command_intent="zero_velocity_existing_rate_limit",
            observed_velocity_continuation=continuation_receipts,
            new_neural_optimizer_updates=0,
        )
    return row, arrays
