"""Local private-future diagnostics, explicitly not the complete H300 exam."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cftrack_motor_residual_plan import CFtrackMotorResidualPlan

MOTOR_AXES = (
    ("right_hip_yaw_joint", 8),
    ("right_hip_roll_joint", 7),
    ("right_ankle_roll_joint", 11),
    ("waist_yaw_joint", 12),
)


def selected_motor_plan(report: dict[str, Any]) -> CFtrackMotorResidualPlan:
    """Decode one frozen pilot nomination, never a promotion or hardware permit."""
    if (
        report.get("status") != "CFTRACK_MOTOR_AUTHORITY_LOCAL_RESCUE"
        or report.get("private_post60_is_not_full_h300_exam") is not True
        or report.get("actual_native_steps") != 0
        or report.get("neural_optimizer_updates") != 0
    ):
        raise ValueError("closed local-only motor pilot nomination required")
    records = report.get("records")
    expected = ["parent", *[f"arm-{i:02d}" for i in range(8)]]
    if (
        type(records) is not list
        or len(records) != 9
        or [r.get("name") for r in records] != expected
        or report.get("selected_local_candidate") not in expected[1:]
    ):
        raise ValueError("complete declared nine-arm nomination required")
    index = expected.index(report["selected_local_candidate"]) - 1
    row = records[index + 1]
    if (
        row["metrics"].get("local_post60_conditions_met") is not True
        or row["future"].get("controls") != 160
        or row["future"].get("runtime_failure") is not None
        or row["future"].get("body_failed") is not False
    ):
        raise ValueError("complete safe local nominee required")
    raw = row["motor_plan"]
    plan = CFtrackMotorResidualPlan(
        raw["origin_counter"],
        raw["start_control"],
        raw["peak_control"],
        tuple(raw["delta_hg"]),
        raw["donor_policy_hash"],
    )
    plan.offsets()
    first = records[0]["metrics"]["first_foot_control"]
    expected_delta = [0.0] * 29
    expected_delta[MOTOR_AXES[index // 2][1]] = 0.15 if index % 2 else -0.15
    if (
        type(first) is not int
        or not 25 <= first <= 99
        or plan.origin_counter != 0
        or plan.start_control != first - 25
        or plan.peak_control != 24
        or plan.delta_hg != tuple(expected_delta)
    ):
        raise ValueError("nominee differs from predeclared motor axis/window")
    return plan


def summarize_motor_future(
    future: dict[str, Any], *, ball_address: int, root_address: int
) -> dict[str, Any]:
    controls = future["controls"]
    if (
        type(controls) is not int
        or not 0 <= controls <= 160
        or any(type(a) is not int or not 0 <= a <= 36 for a in (ball_address, root_address))
        or abs(ball_address - root_address) < 7
    ):
        raise ValueError(
            "bounded private control count and separate native free-joint addresses required"
        )
    for key, shape in [
        ("qpos", (controls * 10, 43)),
        ("ball_forces", (controls * 10, 1, 3)),
        ("joint_targets", (controls, 29)),
        ("nn_decoded_targets", (controls, 29)),
    ]:
        value = future[key]
        if (
            type(value) is not np.ndarray
            or value.shape != shape
            or value.dtype.kind != "f"
            or not np.isfinite(value).all()
        ):
            raise ValueError("complete finite private future arrays required: " + key)
    forces = future["ball_forces"][:, 0]
    if (forces < 0).any():
        raise ValueError("nonnegative native contact forces required")
    feet = np.flatnonzero((forces[:, :2] > 1).any(axis=1))
    first = int(feet[0] // 10) if len(feet) else None
    observed = first is not None and first + 60 < controls
    q = future["qpos"].reshape(controls, 10, 43)
    displacement = (
        q[first + 60, -1, ball_address : ball_address + 3]
        - q[first, -1, ball_address : ball_address + 3]
        if observed and first is not None
        else None
    )
    forward = float(displacement[0]) if displacement is not None else None
    lateral = float(displacement[1]) if displacement is not None else None
    ratio = (
        abs(lateral) / max(forward, 0.01) if lateral is not None and forward is not None else None
    )
    minimum = float(q[:, :, root_address + 2].min()) if controls else None
    nf = float(forces[:, 2].max()) if controls else None
    safe = (
        minimum is not None
        and minimum >= 0.65
        and nf is not None
        and nf <= 1
        and not future["body_failed"]
        and future["runtime_failure"] is None
    )
    return dict(
        first_foot_control=first,
        post60_observed=observed,
        forward_60_m=forward,
        lateral_60_m=lateral,
        lateral_forward_ratio=ratio,
        minimum_root_z_m=minimum,
        maximum_nonfoot_force_n=nf,
        private_safe=safe,
        local_post60_conditions_met=bool(
            safe
            and observed
            and forward is not None
            and forward >= 1
            and ratio is not None
            and ratio <= 0.3
        ),
        maximum_applied_motor_offset_rad=float(
            abs(future["joint_targets"] - future["nn_decoded_targets"]).max()
        )
        if controls
        else None,
        official_h300_exam=False,
        full_task_pass_not_evaluated=True,
        hardware_authorized=False,
    )
