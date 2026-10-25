"""One native physical evaluation of a declared whole-body teacher proposal."""

import contextlib
import io
from collections import deque
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
from rosclaw_soccer.providers.g1.kick_measured_history import KickMeasuredSample
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.kick_reference_phase import nearest_kick_reference_phase
from rosclaw_soccer.rsi.shared_kick_history import measured_kick_sample
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters, SharedNativeMotorWorld
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer


def evaluate_episode(
    *,
    model: Any,
    assets: dict[str, Any],
    policy_types: tuple[Any, ...],
    course: tuple[float, float, float],
    parameters: dict[str, Any],
    hard_limits: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """No copied rollout state or action playback; every call creates fresh NN state.

    Initialization ends at world construction. All subsequent physics belongs
    to the shared clock; every joint target is executed under unchanged caps.
    """
    import mujoco

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

    state_type, output_type, policy_type, mapping = policy_types
    stream = ContinuousVelocityStream(
        assets["velocity_model"],
        assets["velocity_config"],
        expected_model_hash=assets["velocity_model_hash"],
        expected_config_hash=assets["velocity_config_hash"],
        horizon_controls=360,
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
    history: deque[KickMeasuredSample] = deque(maxlen=5)
    previous = np.zeros(3, np.float32)
    kick, entry, phase, history_receipt = None, None, None, None
    contacted = False
    entry_state = None
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
            if phase is None:
                phase = nearest_kick_reference_phase(
                    q[7:36],
                    v[6:35],
                    policy.motion_joint_pos[:, mapping],
                    policy.motion_joint_vel[:, mapping],
                )["reference_frame"]
            kp = NativePDParameters(np.array(policy.kps), np.array(policy.kds), vp.effort)
            switch = w.switch_pd_profile("a/", kp)
            if not switch["physics_unchanged"]:
                raise ValueError("gain switch modified physical state")
            kick = SharedRoboNaldoPlayer(w, "a/", policy, goal=np.array([8.0, 0.0, 0.11]))
            history_receipt = kick.install_measured_entry(tuple(history), reference_frame=phase)
        desired_world = np.array(
            [
                parameters["speed_mps"],
                np.clip(
                    2 * (d.xpos[lo.ball, 1] + parameters["root_lateral_offset_m"] - q[1]), -0.4, 0.4
                ),
            ]
        )
        xy = rot[:2, :2].T @ desired_world
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        wanted = np.array(
            [
                np.clip(xy[0], -0.5, 1.0),
                np.clip(xy[1], -0.5, 0.5),
                np.clip(-2 * yaw, -1, 1),
            ],
            np.float32,
        )
        command = previous + np.clip(wanted - previous, -0.05, 0.05)
        previous = command.copy()
        if kick is None:
            target = stream.propose(q, v, lo.angular_velocity_body(d), command, frame=frame)
        else:
            target = kick.propose()[0]
        result = w.advance({"a/": target})
        if kick is not None:
            kick.commit()
        if not w.failed:
            history.append(measured_kick_sample(w, "a/"))
        contacted |= bool(w.last_ball_contact_bodies["a/"])
        for key in raw:
            raw[key].append(result[key])
        targets.append(target.copy())
        commands.append(command.copy())
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
    )
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
        physical_teacher_qualified=False,
    )
    return row, arrays
