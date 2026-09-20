"""Measured receiving traces to generic Practice payloads, never skill labels.

The source records completed control intervals. PD targets and actuator ctrl
are from the final physics substep, not averaged torque, sensor torque, or a
complete command history. Missing arrays remain explicitly unknown per joint.
"""

import re
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES


def receiving_feedback_payloads(
    trace: dict[str, Any],
    *,
    agent_id: str,
    body_id: str,
    frame_prefix: str,
    source_file_hash: str,
) -> tuple[dict[str, Any], ...]:
    """Convert aligned evidence without inferring commands from measured poses.

    The caller authenticates the supplied trace against the file hash and its
    pinned recorder implementation; this function cannot authenticate a file
    from an in-memory dictionary. No outcome or teacher qualification is added.
    """
    for value in (agent_id, body_id, frame_prefix):
        if type(value) is not str or re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", value) is None:
            raise ValueError("bounded named feedback identities required")
    if (
        type(source_file_hash) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", source_file_hash) is None
    ):
        raise ValueError("source trajectory hash required")
    time = np.asarray(trace["time"])
    if (
        time.ndim != 1
        or not 1 <= len(time) <= 50000
        or time.dtype.kind not in "fiu"
        or not np.isfinite(time).all()
        or np.any(time < 0)
        or not np.allclose(np.diff(time), 0.02, atol=1e-8, rtol=0)
    ):
        raise ValueError("completed 50 Hz control timeline required")
    key = agent_id.replace(".", "_")

    def column(suffix: str, *, required: bool = False) -> np.ndarray | None:
        name = key + suffix
        if name not in trace and not required:
            return None
        values = np.asarray(trace[name])
        if (
            values.shape != (len(time), 29)
            or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()
        ):
            raise ValueError("finite aligned 29-joint evidence required")
        return values.astype(np.float64, copy=True)

    actual = column("_joint_position", required=True)
    assert actual is not None
    target = column("_applied_pd_target")
    command = column("_joint_torque")
    with np.errstate(over="ignore", invalid="ignore"):
        error = None if target is None else target - actual
    if error is not None and not np.isfinite(error).all():
        raise ValueError("finite derived position error required")

    def named(row: np.ndarray | None) -> dict[str, float | None]:
        return {
            name: None if row is None else float(row[i])
            for i, name in enumerate(G1_DDS_JOINT_NAMES)
        }

    return tuple(
        dict(
            frame_id=f"{frame_prefix}_{i:05d}",
            body_id=body_id,
            timestamp=float(stamp),
            actual=named(actual[i]),
            target=named(None if target is None else target[i]),
            position_error=named(None if error is None else error[i]),
            primary_event="measured_simulation_state",
            metadata=dict(
                source_trace_sha256=source_file_hash,
                agent_id=agent_id,
                simulation_only=True,
                position_unit="rad",
                sample_semantics="completed_control_interval",
                target_semantics="last_physics_substep_pd_target",
                position_error_semantics="target_minus_completed_position_not_control_start_error",
                commanded_torque_nm=named(None if command is None else command[i]),
                torque_semantics="last_physics_substep_actuator_ctrl_not_measured_torque",
                target_recorded=target is not None,
                commanded_torque_recorded=command is not None,
                complete_command_history=False,
            ),
        )
        for i, stamp in enumerate(time)
    )
