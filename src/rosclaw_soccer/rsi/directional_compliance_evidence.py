"""Validate recorded gain authority; actual substep dynamics still need replay."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_exit_torque_evidence import compare_plans
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json

PREFIX = "directional_compliance_"


def compare_compliance_plans(actual: dict[str, Any], recorded: dict[str, Any]) -> None:
    # Matrix bytes are authenticated separately, then numerically reconstructed.
    compare_plans(
        {k: v for k, v in actual.items() if k != "stiffness_matrix_hash"},
        {k: v for k, v in recorded.items() if k != "stiffness_matrix_hash"},
    )


def require_directional_capture(
    report: dict[str, Any], commitment: dict[str, Any], trace: dict[str, Any]
) -> None:
    contract = commitment["directional_contact_compliance"]
    if (
        report.get("directional_contact_compliance") != contract
        or contract.get("schema") != "soccer.rsi.directional_contact_compliance.v1"
        or contract.get("contract_hash")
        != hash_json({k: v for k, v in contract.items() if k != "contract_hash"})
        or contract.get("world_hash") != commitment.get("compiled_model_hash")
        or contract.get("entry_frame") != 0
        or contract.get("forecast_budget_steps") != 700
        or contract.get("contraction_fraction") != 0.5
        or contract.get("maximum_relative_matrix_change_per_frame") != 0.1
        or contract.get("relative_stiffness_eigenvalue_bounds") != [0.5, 1.0]
        or contract.get("activation_ceiling") != "SIM_ONLY"
        or any(
            contract.get(k) is not False
            for k in (
                "trained_policy",
                "teacher_qualified",
                "passivity_proven",
                "floating_base_stability_proven",
                "damping_changed",
                "physical_hard_torque_limits_changed",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or any(
            k in commitment
            for k in (
                "contact_exit_torque_mpc",
                "swing_release_ablation",
                "cerebellar_torque_contract",
                "generated_seed_episode_binding",
                "foundation_approach_ablation",
                "postcontact_navigation_probe",
                "locomotion_mode_probe",
            )
        )
        or any(
            k not in commitment
            for k in (
                "bounded_control_profile",
                "foundation_observation_capture",
                "mesh_task_contact_protocol",
            )
        )
        or commitment.get("execution_profile") != "taskspace_plus_motor"
    ):
        raise ValueError("unmixed source-bound SIM directional gain contract required")
    shapes = {
        PREFIX + "stiffness_nm_per_rad": (300, 1, 29, 29),
        PREFIX + "requested_delta_nm": (300, 1, 10, 29),
        PREFIX + "applied_delta_nm": (300, 1, 10, 29),
        PREFIX + "forecast_steps": (300, 1, 1),
    }
    if {k for k in trace if k.startswith(PREFIX)} != set(shapes):
        raise ValueError("complete exact directional gain fields required")
    for name, shape in shapes.items():
        a = trace[name]
        dtype = np.int64 if name.endswith("forecast_steps") else np.float64
        if a.shape != shape or a.dtype != dtype or not np.isfinite(a).all():
            raise ValueError("finite whole typed gain capture required")
    kp = np.asarray(contract.get("kp"))
    kd = np.asarray(contract.get("kd"))
    if (
        kp.shape != (29,)
        or kd.shape != (29,)
        or not np.isfinite(kp).all()
        or not np.isfinite(kd).all()
        or np.any(kp <= 0)
        or np.any(kd < 0)
        or contract["kp"] != commitment["physics"]["canonical_pd_kp"]
        or contract["kd"] != commitment["physics"]["canonical_pd_kd"]
    ):
        raise ValueError("unchanged original physical gains required")
    plans = report.get("directional_compliance_plans")
    if not isinstance(plans, list) or len(plans) != 300:
        raise ValueError("all 300 sealed causal gain plans required")
    previous = np.eye(29)
    hard = np.asarray(G1_HARD_TORQUE_LIMITS)
    for frame, plan in enumerate(plans):
        matrix = trace[PREFIX + "stiffness_nm_per_rad"][frame, 0]
        if not np.array_equal(matrix, matrix.T):
            raise ValueError("symmetric actual gain matrix required")
        relative = matrix / np.sqrt(kp)[:, None] / np.sqrt(kp)[None, :]
        relative = (relative + relative.T) * 0.5
        eig = np.linalg.eigvalsh(relative)
        step = float(np.max(abs(np.linalg.eigvalsh(relative - previous))))
        if eig.min() < 0.5 - 1e-9 or eig.max() > 1 + 1e-9 or step > 0.1 + 1e-9:
            raise ValueError("gain contraction or spectral slew exceeded")
        if not np.array_equal(matrix[12:], np.diag(kp)[12:]):
            raise ValueError("torso and arms must retain exact original stiffness")
        previous = relative
        identity = bool(np.array_equal(matrix, np.diag(kp)))
        steps = int(trace[PREFIX + "forecast_steps"][frame, 0, 0])
        if (
            plan.get("schema") != "soccer.rsi.directional_compliance_plan.v1"
            or type(plan.get("frame")) is not int
            or plan["frame"] != frame
            or plan.get("report_hash")
            != hash_json({k: v for k, v in plan.items() if k != "report_hash"})
            or plan.get("stiffness_matrix_hash") != hash_bytes(matrix.tobytes())
            or plan.get("identity") is not identity
            or plan.get("hardware_authorized") is not False
            or plan.get("promotion_authorized") is not False
            or type(plan.get("forecast_substeps")) is not int
            or plan["forecast_substeps"] != steps
            or steps != (0 if identity else 10)
            or (plan.get("expected_end_state_hash") is None) != identity
            or plan.get("mode") not in {"INACTIVE", "COMPLY", "RESTORE"}
            or (plan["mode"] == "INACTIVE" and (not identity or plan.get("side") is not None))
            or (plan["mode"] == "RESTORE" and (identity or plan.get("side") is not None))
            or (plan["mode"] == "COMPLY" and type(plan.get("side")) is not int)
        ):
            raise ValueError("sealed matrix/plan/forecast correspondence required")
        raw = trace["pd_requested_torque_nm"][frame, 0]
        delta = trace[PREFIX + "requested_delta_nm"][frame, 0]
        torque = trace["torque_nm"][frame, 0]
        if identity and (np.any(delta) or np.any(trace[PREFIX + "applied_delta_nm"][frame, 0])):
            raise ValueError("identity law must have exact zero intervention")
        expected = np.clip(raw + delta, -hard, hard)
        applied = torque - np.clip(raw, -hard, hard)
        if not np.allclose(expected, torque, atol=1e-10, rtol=0) or not np.allclose(
            applied, trace[PREFIX + "applied_delta_nm"][frame, 0], atol=1e-10, rtol=0
        ):
            raise ValueError("recorded actual gain authority differs from bounded PD projection")
        first_raw = (
            matrix @ (trace["joint_target_rad"][frame, 0] - trace["joint_position_rad"][frame, 0])
            - trace["joint_velocity_rad_s"][frame, 0] * kd
        )
        if not np.allclose(first_raw - raw[0], delta[0], atol=1e-8, rtol=0):
            raise ValueError("gain-modulated first substep does not reconstruct")
    counters = report.get("directional_compliance_counters", {})
    total = int(trace[PREFIX + "forecast_steps"].sum())
    expected = dict(
        frames_planned=300,
        forecast_attempts=total // 10,
        reserved_forecast_steps=total,
        attempted_forecast_steps=total,
        integrated_forecast_steps=total,
        maximum_forecast_steps=700,
    )
    if (
        counters != expected
        or any(type(v) is not int for v in counters.values())
        or not 0 <= total <= 700
    ):
        raise ValueError("all gain forecasts and complete sequential frames must be charged")
