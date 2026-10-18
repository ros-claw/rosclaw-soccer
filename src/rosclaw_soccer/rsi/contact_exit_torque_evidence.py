"""Validate bounded SIM teacher records and reconstruct optimization decisions.

Optimization reconstruction uses the pinned teacher, not an independent solver.
Actual body dynamics and Foundation decisions retain their independent audits.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_json

PREFIX = "contact_exit_mpc_"


def compare_plans(actual: Any, recorded: Any, *, path: str = "plan") -> None:
    """Allow replay roundoff, never different choices or ignored physical fields."""
    if isinstance(actual, dict):
        if not isinstance(recorded, dict) or actual.keys() != recorded.keys():
            raise ValueError(f"contact-exit optimization structure differs: {path}")
        for key in actual:
            if key not in {
                "donor_state_hash",
                "end_state_hash",
                "expected_end_state_hash",
                "report_hash",
            }:
                compare_plans(actual[key], recorded[key], path=f"{path}.{key}")
    elif isinstance(actual, list):
        if not isinstance(recorded, list) or len(actual) != len(recorded):
            raise ValueError(f"contact-exit optimization length differs: {path}")
        for i, (a, b) in enumerate(zip(actual, recorded, strict=True)):
            compare_plans(a, b, path=f"{path}[{i}]")
    elif isinstance(actual, float):
        if (
            type(recorded) not in {float, int}
            or not np.isfinite(actual)
            or not np.isfinite(recorded)
            or not np.isclose(actual, recorded, rtol=0, atol=1e-8)
        ):
            raise ValueError(f"contact-exit optimization value differs: {path}")
    elif type(actual) is not type(recorded) or actual != recorded:
        raise ValueError(f"contact-exit optimization decision differs: {path}")


def require_contact_exit_capture(
    report: dict[str, Any], commitment: dict[str, Any], trace: dict[str, Any]
) -> None:
    contract = commitment["contact_exit_torque_mpc"]
    if (
        report.get("contact_exit_torque_mpc") != contract
        or contract.get("schema") != "soccer.rsi.contact_exit_torque_mpc.v1"
        or contract.get("world_hash") != commitment.get("compiled_model_hash")
        or contract.get("contract_hash")
        != hash_json({k: v for k, v in contract.items() if k != "contract_hash"})
        or any(
            contract.get(key) is not False
            for key in (
                "trained_policy",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or any(
            key in commitment
            for key in (
                "swing_release_ablation",
                "cerebellar_torque_contract",
                "generated_seed_episode_binding",
                "foundation_approach_ablation",
                "postcontact_navigation_probe",
                "locomotion_mode_probe",
            )
        )
        or "bounded_control_profile" not in commitment
        or "foundation_observation_capture" not in commitment
        or "mesh_task_contact_protocol" not in commitment
        or commitment.get("execution_profile") != "taskspace_plus_motor"
    ):
        raise ValueError("unmixed captured SIM contact-exit contract required")
    shapes = {
        PREFIX + "requested_offset_nm": (300, 1, 29),
        PREFIX + "applied_offset_nm": (300, 1, 10, 29),
        PREFIX + "forecast_steps": (300, 1, 1),
    }
    if {k for k in trace if k.startswith(PREFIX)} != set(shapes):
        raise ValueError("exact contact-exit capture fields required")
    for name, shape in shapes.items():
        a = trace[name]
        dtype = np.int64 if name.endswith("forecast_steps") else np.float64
        if a.shape != shape or a.dtype != dtype or not np.isfinite(a).all():
            raise ValueError("complete finite typed contact-exit capture required")
    offsets = trace[PREFIX + "requested_offset_nm"][:, 0]
    caps = np.minimum(12.0, 0.2 * np.asarray(G1_HARD_TORQUE_LIMITS))
    if np.any(abs(offsets) > caps) or np.any(
        abs(np.diff(np.vstack((np.zeros((1, 29)), offsets)), axis=0)) > 2.0 + 1e-12
    ):
        raise ValueError("contact-exit requested torque cap or slew exceeded")
    pd = trace["pd_requested_torque_nm"]
    hard = np.asarray(G1_HARD_TORQUE_LIMITS)
    expected = np.clip(pd + offsets[:, None, None, :], -hard, hard)
    applied = expected - np.clip(pd, -hard, hard)
    if not np.allclose(expected, trace["torque_nm"], rtol=0, atol=1e-12) or not np.allclose(
        applied, trace[PREFIX + "applied_offset_nm"], rtol=0, atol=1e-12
    ):
        raise ValueError("contact-exit actual applied torque differs from bounded request")
    plans = report.get("contact_exit_mpc_plans")
    if not isinstance(plans, list) or len(plans) != 300:
        raise ValueError("all 300 contact-exit plan records required")
    for frame, plan in enumerate(plans):
        if (
            plan.get("schema") != "soccer.rsi.contact_exit_torque_plan.v1"
            or type(plan.get("frame")) is not int
            or plan["frame"] != frame
            or plan.get("report_hash")
            != hash_json({k: v for k, v in plan.items() if k != "report_hash"})
            or plan.get("hardware_authorized") is not False
            or plan.get("promotion_authorized") is not False
            or type(plan.get("forecast_substeps")) is not int
            or not 0 <= plan["forecast_substeps"] <= 70
            or plan["forecast_substeps"] % 10
            or plan["forecast_substeps"] != int(trace[PREFIX + "forecast_steps"][frame, 0, 0])
        ):
            raise ValueError("sealed causal contact-exit plan and bounded step count required")
        if plan.get("mode") == "INACTIVE":
            if (
                plan.get("side") is not None
                or plan.get("choices") != []
                or plan.get("selected_index") is not None
                or plan.get("expected_end_state_hash") is not None
                or plan["forecast_substeps"] != 0
                or np.any(offsets[frame])
                or plan.get("no_safe_choice_fallback") is not False
            ):
                raise ValueError("inactive contact-exit controller must be exactly inert")
        else:
            choices, selected = plan.get("choices"), plan.get("selected_index")
            if (
                plan.get("mode") not in {"PRESSURE_EXIT", "RAMP_OUT"}
                or not isinstance(choices, list)
                or not 1 <= len(choices) <= 6
                or type(selected) is not int
                or not 0 <= selected < len(choices)
                or plan["forecast_substeps"] < 10
                or not np.array_equal(np.asarray(choices[selected]["offset_nm"]), offsets[frame])
                or choices[selected]["end_state_hash"] != plan.get("expected_end_state_hash")
            ):
                raise ValueError("selected bounded physical contact-exit forecast required")
    counters = report.get("contact_exit_mpc_counters", {})
    total = int(trace[PREFIX + "forecast_steps"].sum())
    if (
        any(type(v) is not int for v in counters.values())
        or set(counters)
        != {
            "forecast_attempts",
            "attempted_forecast_steps",
            "integrated_forecast_steps",
            "maximum_forecast_steps",
        }
        or counters["attempted_forecast_steps"] != total
        or counters["integrated_forecast_steps"] != total
        or counters["forecast_attempts"] * 10 != total
        or counters["maximum_forecast_steps"] != contract.get("forecast_budget_steps")
        or not 0 <= total <= counters["maximum_forecast_steps"]
    ):
        raise ValueError("all attempted and integrated contact-exit forecasts must be charged")
