"""Owned current-student numeric sequences, not a physical provenance audit."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.recurrent_sampling_motor import (
    TRACE_FIELDS,
    CompiledRecurrentSamplingMotor,
    make_preview,
)
from rosclaw_soccer.rsi.recurrent_success_motor import STATE_FIELD
from rosclaw_soccer.rsi.step_motor_phase_context import phase_sequence
from rosclaw_soccer.rsi.stochastic_step_execution import features_at_frame

BODY_FIELDS = {
    "joint_position_rad": (300, 1, 29),
    "joint_velocity_rad_s": (300, 1, 29),
    "root_pose_xyzw_m": (300, 1, 7),
    "root_velocity_world": (300, 1, 6),
    "ball_position_before_step_m": (300, 1, 3),
    "ball_linear_velocity_before_step_m_s": (300, 1, 3),
    "foot_geometry_position_before_step_m": (300, 1, 4, 3),
    "pre_motor_joint_target_rad": (300, 1, 29),
    "motor_delta_rad": (300, 1, 12),
    "force_n": (300, 1, 6),
    STATE_FIELD: (300, 1, 64),
}


def extract_sequence(
    view: dict[str, Any], trace: Any, *, motor_limits: Any, terminal_mc_return: float
) -> dict[str, np.ndarray[Any, Any]]:
    """Reconstruct one full actual student's sampled sequence before learning.

    Callers must separately bind the trace/world/report and independent physics
    and Foundation reviews. This function verifies numeric execution, not those
    external identities. Returns are labels only; future rewards never enter
    context, baseline, gates or state. No failed sequence is filtered out.
    """
    if type(trace) is not dict:
        raise ValueError("complete current-student numeric trace required")
    shapes = {
        **BODY_FIELDS,
        **{k: (300, 1, d) for k, d in zip(TRACE_FIELDS, (12, 12, 1, 1), strict=True)},
    }
    if not shapes.keys() <= trace.keys() or {
        k for k in trace if k.startswith("recurrent_sampling_")
    } != set(TRACE_FIELDS):
        raise ValueError("complete unambiguous current-student sampling fields required")
    owned = {}
    for key, shape in shapes.items():
        array = np.asarray(trace[key])
        dtype = np.bool_ if key == TRACE_FIELDS[-1] else np.float64
        if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
            raise ValueError("finite full300 typed current-student trace required")
        if dtype != np.bool_ and np.max(np.abs(array)) > 1e6:
            raise ValueError("bounded current-student trace required")
        owned[key] = array.copy()
    if np.any(owned[TRACE_FIELDS[-1]][:30]) or not np.all(owned[TRACE_FIELDS[-1]][30:]):
        raise ValueError("exact original exploration boundary required")
    limits = np.asarray(motor_limits)
    if (
        limits.shape != (12, 2)
        or limits.dtype.kind not in "fiu"
        or not np.isfinite(limits).all()
        or np.max(np.abs(limits.astype(np.float64))) > 1e6
        or np.any(limits[:, 0] >= limits[:, 1])
        or type(terminal_mc_return) not in (int, float)
        or not np.isfinite(terminal_mc_return)
        or abs(terminal_mc_return) > 1e6
    ):
        raise ValueError("explicit finite actual motor limits and frozen MC label required")
    limits = np.array(limits, dtype=np.float64, copy=True)
    policy = make_preview(view)
    decoder = CompiledRecurrentSamplingMotor(policy)
    phases = phase_sequence(owned["force_n"][:, 0])
    context, baseline, gates = [], [], []
    previous = np.zeros(12)
    for frame in range(300):
        nominal = owned["pre_motor_joint_target_rad"][frame, 0]
        forces = owned["force_n"][frame - 1, 0] if frame else np.zeros(6)
        if frame >= 30:
            observation = features_at_frame(
                owned,
                frame=frame,
                nominal_target=nominal,
                previous=previous,
                previous_contact_forces=forces,
            )
            phase = int(phases[frame])
            current = np.concatenate((decoder.features(observation)[:134], [phase]))
            # Query only the unchanged feedforward parent. Never query the
            # student's mean outside delta_at_frame: that would advance GRU.
            baseline.append(decoder._behavior.raw_mean(observation, phase).copy())
            gates.append(float(decoder._guard.gate(current)))
            context.append(current)
        delta = decoder.delta_at_frame(
            policy,
            owned,
            frame=frame,
            nominal_target=nominal,
            baseline=nominal[:12],
            limits=limits,
            previous=previous,
            previous_contact_forces=forces,
        )
        if not np.array_equal(delta, owned["motor_delta_rad"][frame, 0]):
            raise ValueError("current student's actual executed action must reconstruct")
        if not np.array_equal(decoder.hidden_state, owned[STATE_FIELD][frame, 0]):
            raise ValueError("current student's actual causal hidden state must reconstruct")
        for key, value in decoder.sampled_transition.items():
            if not np.array_equal(value, owned[key][frame, 0]):
                raise ValueError("current student's actual mean, draw and density must reconstruct")
        previous = delta
    return {
        "context": np.stack(context),
        "baseline": np.stack(baseline),
        "gates": np.asarray(gates, dtype=np.float64),
        "latent_actions": owned[TRACE_FIELDS[1]][30:, 0].copy(),
        "behavior_log_probabilities": owned[TRACE_FIELDS[2]][30:, 0, 0].copy(),
        "returns": np.full(270, terminal_mc_return, dtype=np.float64),
    }
