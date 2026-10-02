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
    motor: Any,
    body: Any,
    swing: Any,
    physics: Any,
    report: dict[str, Any],
    *,
    decoder_sink: list[Any] | None = None,
) -> dict[str, Any]:
    if decoder_sink is not None and (type(decoder_sink) is not list or decoder_sink):
        raise ValueError("empty decoder capture sink required")
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
    online_preview = "online_motor_proof" in report["contact_motor_policy"]
    progressive_preview = "progressive_motor_proof" in report["contact_motor_policy"]
    step_preview = "step_motor_proof" in report["contact_motor_policy"]
    if sum((neural_preview, online_preview, progressive_preview, step_preview)) > 1:
        raise ValueError("ambiguous motor model backend")
    if (
        report["contact_motor_policy"].get("execution_profile")
        == "causal_per_frame_neural_residual"
    ) != step_preview:
        raise ValueError("unbound per-frame execution profile")
    if progressive_preview:
        from rosclaw_soccer.rsi.progressive_motor_actor import audit_preview as audit_progressive

        audit_progressive(report["contact_motor_policy"], body, force)
    if online_preview:
        from rosclaw_soccer.rsi.online_motor_actor_critic import audit_preview as audit_online

        if neural_preview:
            raise ValueError("ambiguous motor model backend")
        audit_online(report["contact_motor_policy"], body, force)
    previous = np.zeros(12)
    contact_delta = np.zeros(12)
    contact_frame = None
    compiled: Any = None
    if "current_memory_motor_proof" in report["contact_motor_policy"]:
        from rosclaw_soccer.rsi.current_memory_motor import CompiledCurrentMemoryMotor

        if not step_preview:
            raise ValueError("unbound current-memory guarded motor")
        compiled = CompiledCurrentMemoryMotor(report["contact_motor_policy"])
    elif "consolidated_smooth_motor_proof" in report["contact_motor_policy"]:
        from rosclaw_soccer.rsi.consolidated_smooth_motor import CompiledConsolidatedSmoothMotor

        if not step_preview:
            raise ValueError("unbound consolidated current-parent motor")
        compiled = CompiledConsolidatedSmoothMotor(report["contact_motor_policy"])
    elif "smooth_memory_motor_proof" in report["contact_motor_policy"]:
        from rosclaw_soccer.rsi.smooth_memory_motor import CompiledSmoothMemoryMotor

        if not step_preview:
            raise ValueError("unbound smooth-memory motor")
        compiled = CompiledSmoothMemoryMotor(report["contact_motor_policy"])
    elif "output_memory_motor_proof" in report["contact_motor_policy"]:
        from rosclaw_soccer.rsi.output_memory_step_motor import CompiledOutputMemoryMotor

        if not step_preview:
            raise ValueError("unbound output-memory motor")
        compiled = CompiledOutputMemoryMotor(report["contact_motor_policy"])
    elif "replay_motor_proof" in report["contact_motor_policy"]:
        from rosclaw_soccer.rsi.kernel_replay_motor import CompiledReplayStepMotor

        if not step_preview:
            raise ValueError("unbound replay motor")
        compiled = CompiledReplayStepMotor(report["contact_motor_policy"])
    elif "selective_memory_motor_proof" in report["contact_motor_policy"]:
        from rosclaw_soccer.rsi.selective_phase_memory import CompiledSelectivePhaseMemory

        if not step_preview:
            raise ValueError("unbound selective-memory motor")
        compiled = CompiledSelectivePhaseMemory(report["contact_motor_policy"])
    elif "memory_phase_motor_proof" in report["contact_motor_policy"]:
        from rosclaw_soccer.rsi.memory_guarded_phase_transfer import CompiledMemoryPhaseMotor

        if not step_preview:
            raise ValueError("unbound memory phase transfer")
        compiled = CompiledMemoryPhaseMotor(report["contact_motor_policy"])
    elif "kernel_motor_proof" in report["contact_motor_policy"]:
        from rosclaw_soccer.rsi.kernel_guarded_step_execution import CompiledKernelStepMotor

        if not step_preview:
            raise ValueError("unbound kernel-protected motor execution")
        compiled = CompiledKernelStepMotor(report["contact_motor_policy"])
    elif "protected_phase_motor_proof" in report["contact_motor_policy"]:
        from rosclaw_soccer.rsi.protected_phase_step_execution import CompiledProtectedPhaseMotor

        if not step_preview:
            raise ValueError("unbound protected phase motor execution")
        compiled = CompiledProtectedPhaseMotor(report["contact_motor_policy"])
    elif "compiled_motor_proof" in report["contact_motor_policy"]:
        from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor

        if not step_preview:
            raise ValueError("unbound compiled motor execution")
        compiled = CompiledStepMotor(report["contact_motor_policy"])
    elif step_preview:
        from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor

        compiled = CompiledStepMotor.from_legacy_preview(report["contact_motor_policy"])
    for frame in range(300):
        if step_preview:
            from rosclaw_soccer.rsi.step_motor_execution import delta_at_frame

            if compiled is not None:
                delta_at_frame = compiled.delta_at_frame
            elif "online_step_motor_proof" in report["contact_motor_policy"]:
                from rosclaw_soccer.rsi.online_step_execution import (
                    delta_at_frame as online_step_delta_at_frame,
                )

                delta_at_frame = online_step_delta_at_frame
            elif "stochastic_motor_proof" in report["contact_motor_policy"]:
                from rosclaw_soccer.rsi.stochastic_step_execution import (
                    delta_at_frame as sampled_delta_at_frame,
                )

                delta_at_frame = sampled_delta_at_frame

            expected = delta_at_frame(
                report["contact_motor_policy"],
                body,
                frame=frame,
                nominal_target=swing["executed_taskspace_joint_target_rad"][
                    frame, 0, [order.index(n) for n in G1_DDS_JOINT_NAMES]
                ],
                baseline=baseline[frame, 0],
                limits=limits[0],
                previous=previous,
                previous_contact_forces=force[frame - 1, 0] if frame else np.zeros(6),
            )
        else:
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
        if (neural_preview or online_preview or progressive_preview) and frame < 30:
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
    if decoder_sink is not None:
        if compiled is None:
            raise ValueError("compiled causal decoder required for latent replay")
        # Expose only the internally constructed, fully audited decoder. Its
        # phase memory has advanced; explicit-phase latent_sample is stateless.
        decoder_sink.append(compiled)
    return {
        "contact_motor_action_audited": True,
        "contact_motor_policy_hash": policy_hash,
        "contact_motor_active_frames": int(
            np.count_nonzero(np.max(np.abs(delta[:, 0]), axis=1) > 1e-6)
        ),
        "contact_motor_max_delta_rad": float(np.max(np.abs(delta))),
    }


def audit_motor_execution(
    folder: Path, report: dict[str, Any], *, decoder_sink: list[Any] | None = None
) -> dict[str, Any]:
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
        # NpzFile indexing decompresses a whole member on every access. The
        # causal per-frame decoder repeatedly reads the same observations;
        # materialize each numeric array once without reducing audited frames.
        arrays = [{key: data[key] for key in data.files} for data in (motor, body, swing, physics)]
        return audit_motor_arrays(
            arrays[0], arrays[1], arrays[2], arrays[3], report, decoder_sink=decoder_sink
        )
