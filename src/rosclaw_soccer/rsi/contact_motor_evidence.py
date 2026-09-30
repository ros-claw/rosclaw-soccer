"""Independently reconstruct every bilateral motor target from causal state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.bootstrap_motor_execution import audit_preview
from rosclaw_soccer.rsi.contact_motor_contract import motor_delta, validate_policy
from rosclaw_soccer.rsi.contact_motor_primitive import JOINT_NAMES
from rosclaw_soccer.rsi.taskspace_swing_evidence import audit_taskspace_swing_trace
from rosclaw_soccer.sim.contracts import hash_bytes


def audit_motor_arrays(
    motor: Any, body: Any, swing: Any, physics: Any, report: dict[str, Any]
) -> dict[str, Any]:
    knots, policy_hash = validate_policy(report["contact_motor_policy"])
    order = report.get("taskspace_joint_order")
    if (
        report.get("contact_motor_policy_hash") != policy_hash
        or report.get("frames") != 300
        or len(report.get("environments", [])) != 1
        or not isinstance(order, list)
        or len(order) != 29
        or set(order) != set(G1_DDS_JOINT_NAMES)
    ):
        raise ValueError("invalid motor execution contract")
    ids = [order.index(name) for name in JOINT_NAMES]
    dds_ids = [G1_DDS_JOINT_NAMES.index(name) for name in JOINT_NAMES]
    shapes = {
        "baseline_joint_target_rad": (300, 1, 12),
        "applied_joint_delta_rad": (300, 1, 12),
        "joint_limits_rad": (1, 12, 2),
    }
    if any(
        k not in motor or motor[k].shape != s or not np.isfinite(motor[k]).all()
        for k, s in shapes.items()
    ):
        raise ValueError("invalid motor physical trace")
    audit_taskspace_swing_trace(swing, report, frames=300, count=1)
    baseline = motor["baseline_joint_target_rad"]
    delta = motor["applied_joint_delta_rad"]
    limits = motor["joint_limits_rad"]
    if not np.allclose(limits, swing["taskspace_joint_limits_rad"][:, ids], atol=0, rtol=0):
        raise ValueError("motor limits differ from frozen body")
    if not np.allclose(
        baseline, swing["executed_taskspace_joint_target_rad"][:, :, ids], atol=1e-7, rtol=0
    ):
        raise ValueError("motor did not compose with audited swing actor")
    expected_full = swing["executed_taskspace_joint_target_rad"][
        :, :, [order.index(n) for n in G1_DDS_JOINT_NAMES]
    ].copy()
    expected_full[:, :, dds_ids] += delta
    if not np.allclose(expected_full, body["joint_target_rad"], atol=2e-5, rtol=0):
        raise ValueError("executed body target differs from composed motor policy")
    force = physics["ball_body_contact_force_peak_n"]
    if not np.array_equal(force, swing["observed_ball_body_contact_force_peak_n"]):
        raise ValueError("motor contact event not bound to physics")
    root = body["root_pose_xyzw_m"]
    ball = body["ball_position_before_step_m"]
    if root.shape != (300, 1, 7) or ball.shape != (300, 1, 3) or force.shape != (300, 1, 6):
        raise ValueError("invalid causal motor observations")
    neural_preview = "bootstrap_proof" in report["contact_motor_policy"]
    if neural_preview:
        audit_preview(report["contact_motor_policy"], body, force)
    previous = np.zeros(12)
    contact_delta = np.zeros(12)
    contact_frame = None
    for frame in range(300):
        expected = motor_delta(
            report["contact_motor_policy"],
            knots,
            float(ball[frame, 0, 0] - root[frame, 0, 0]),
            baseline[frame, 0],
            limits[0],
            previous,
            contact_delta,
            frame - contact_frame if contact_frame is not None else None,
        )
        if neural_preview and frame < 30:
            expected = np.zeros(12)
        expected = (baseline[frame, 0] + expected).astype(np.float32).astype(float) - baseline[
            frame, 0
        ]
        if not np.allclose(delta[frame, 0], expected, atol=2e-5, rtol=0):
            raise ValueError(f"motor policy differs from causal state at frame {frame}")
        previous = delta[frame, 0].copy()
        if contact_frame is None and np.any(force[frame, 0] > 1):
            contact_frame = frame
            contact_delta = previous.copy()
    return {
        "contact_motor_action_audited": True,
        "contact_motor_policy_hash": policy_hash,
        "contact_motor_active_frames": int(
            np.count_nonzero(np.max(np.abs(delta[:, 0]), axis=1) > 1e-6)
        ),
        "contact_motor_max_delta_rad": float(np.max(np.abs(delta))),
    }


def audit_motor_execution(folder: Path, report: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "contact_motor_trace.npz": "contact_motor_trace_hash",
        "body_trace.npz": "body_trace_hash",
        "late_swing_action_trace.npz": "late_swing_action_trace_hash",
        "trace.npz": "trace_hash",
    }
    if any(
        not (folder / p).is_file() or hash_bytes((folder / p).read_bytes()) != report.get(key)
        for p, key in paths.items()
    ):
        raise ValueError("unbound motor/body/swing/physics evidence")
    with (
        np.load(folder / "contact_motor_trace.npz", allow_pickle=False) as motor,
        np.load(folder / "body_trace.npz", allow_pickle=False) as body,
        np.load(folder / "late_swing_action_trace.npz", allow_pickle=False) as swing,
        np.load(folder / "trace.npz", allow_pickle=False) as physics,
    ):
        return audit_motor_arrays(motor, body, swing, physics, report)
