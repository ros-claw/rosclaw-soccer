"""Causal foot-heading composition/capture and CPU-state reconstruction.

No source-file authentication, integration, policy admission or hardware calls.
Old original_ik_planner_evidence source and schemas remain unchanged.
"""

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi import original_ik_planner_evidence as original
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.foot_heading_control import FootHeadingContactProbe, require_heading_request
from rosclaw_soccer.rsi.foot_heading_nullspace import MAX_HEADING_REQUEST_RAD, propose_foot_heading
from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

TAG = "foot_heading_contact_probe"
PREFIX = "foot_heading_"
FIELDS = {
    "pre_increment_rad": 6,
    "combined_increment_rad": 6,
    "additional_request_rad": 6,
    "additional_applied_rad": 6,
    "forward_axis_world": 3,
    "linear_predicted_displacement_m": 3,
    "state": 10,
}
STATUS = {
    s: i
    for i, s in enumerate(
        (
            "INACTIVE",
            "OFF_UNCHANGED",
            "BASELINE_OUTSIDE_LIMITS_UNCHANGED",
            "OUTSIDE_DECLARED_HEADING_DOMAIN_UNCHANGED",
            "ALREADY_ALIGNED_UNCHANGED",
            "NO_POSITION_NULLSPACE_YAW_AUTHORITY_UNCHANGED",
            "BOUNDED_KINEMATIC_PROPOSAL_NOT_PHYSICS_QUALIFIED",
            "NUMERICAL_POSITION_LEAK_UNCHANGED",
            "NO_JOINT_LIMIT_ROOM_UNCHANGED",
        )
    )
}


def compose_heading_step(
    probe: FootHeadingContactProbe,
    model: Any,
    data: Any,
    *,
    frame: int,
    side: int,
    contact_frame: int | None,
    navigation_target: np.ndarray,
    increment: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compose only a current pre-contact swing; release memory is caller-owned."""
    import mujoco

    if (
        type(probe) is not FootHeadingContactProbe
        or type(frame) is not int
        or not 0 <= frame < 300
        or type(side) is not int
        or side not in (-1, 0, 1)
        or (
            contact_frame is not None
            and (type(contact_frame) is not int or not 0 <= contact_frame < frame)
        )
        or navigation_target.shape != (29,)
        or navigation_target.dtype != np.float64
        or increment.shape != (6,)
        or increment.dtype != np.float64
        or not np.isfinite(navigation_target).all()
        or not np.isfinite(increment).all()
        or np.max(np.abs(increment)) > 0.35 + 1e-12
    ):
        raise ValueError("typed current-frame causal heading composition inputs required")
    probe.__post_init__()
    phase = 0 if frame < 30 else 2 if contact_frame is not None else 1 if side >= 0 else 3
    record = {PREFIX + k: np.zeros(width) for k, width in FIELDS.items()}
    record[PREFIX + "pre_increment_rad"] = increment.copy()
    record[PREFIX + "combined_increment_rad"] = increment.copy()
    record[PREFIX + "state"] = np.asarray(
        [frame, side, phase, float(probe.enabled), 0, 0, 0, 0, 0, -1], dtype=np.float64
    )
    combined = increment
    if phase == 1:
        foot = int(model.body(("left_ankle_roll_link", "right_ankle_roll_link")[side]).id)
        linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        # data is the actual native/CPU cached collision-stage state. Do not
        # refresh it with mj_forward or kinematics here: that changes its clock.
        mujoco.mj_jacBody(model, data, linear, angular, foot)
        joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
        vi = np.asarray([int(model.jnt_dofadr[j]) for j in joints])
        ids = np.arange(6 * side, 6 * side + 6)
        forward = data.xmat[foot].reshape(3, 3)[:, 0].copy()
        combined, p = propose_foot_heading(
            linear[:, vi[ids]],
            angular[:, vi[ids]],
            forward,
            navigation_target[ids],
            increment,
            model.jnt_range[np.asarray(joints)[ids]].copy(),
            enabled=probe.enabled,
        )
        record[PREFIX + "combined_increment_rad"] = combined.copy()
        for field, key in (
            ("additional_request_rad", "additional_joint_request_rad"),
            ("additional_applied_rad", "additional_joint_applied_rad"),
            ("linear_predicted_displacement_m", "predicted_position_increment_m"),
        ):
            record[PREFIX + field] = np.asarray(p[key], dtype=np.float64)
        record[PREFIX + "forward_axis_world"] = forward
        record[PREFIX + "state"][4:] = [
            STATUS[p["status"]],
            p["measured_foot_heading_rad"],
            p["requested_heading_change_rad"],
            p["predicted_additional_heading_change_rad"],
            p["uniform_limit_scale"],
            -1 if p["position_jacobian_rank"] is None else p["position_jacobian_rank"],
        ]
    return combined, record


def require_heading_capture(
    report: dict[str, Any], commitment: dict[str, Any], trace: dict[str, Any]
) -> dict[str, Any]:
    if (
        TAG not in commitment
        or {k for k in trace if k.startswith(PREFIX)} != {PREFIX + k for k in FIELDS}
        or any(k.startswith(("reference_student_", "planner_pulse_")) for k in trace)
        or "reference_student_model" in commitment
    ):
        raise ValueError("explicit unmixed heading declaration and complete capture required")
    contract = commitment[TAG]
    probe = FootHeadingContactProbe(**contract["parameters"])
    if probe.contract() != contract or report.get(TAG) != contract:
        raise ValueError("same pinned heading parameters and sources required")
    navigation_review = original.require_original_ik_capture(report, commitment, trace)
    require_heading_request(
        probe,
        pulse=OriginalIkPlannerPulse(**commitment[original.TAG]["parameters"]),
        reference_model=None,
        profile=BoundedFirstTouchControl(**commitment["bounded_control_profile"]["parameters"]),
        record_planner_events=True,
    )
    data = {}
    for k, width in FIELDS.items():
        a = np.asarray(trace.get(PREFIX + k))
        if (
            a.shape != (300, 1, width)
            or a.dtype != np.float64
            or not np.isfinite(a).all()
            or np.max(np.abs(a)) > 1e6
        ):
            raise ValueError("whole finite float64 heading capture required")
        data[k] = a[:, 0]
    forces = np.asarray(trace.get("force_n"))
    if (
        forces.shape != (300, 1, 6)
        or forces.dtype != np.float64
        or not np.isfinite(forces).all()
        or np.any(forces < 0)
    ):
        raise ValueError("whole causal original six-channel robot contact history required")
    contact_indices = np.flatnonzero(np.any(forces[:, 0] > 1, axis=1))
    first = int(contact_indices[0]) if len(contact_indices) else None
    pre, combined, requested, applied = (
        data[k]
        for k in (
            "pre_increment_rad",
            "combined_increment_rad",
            "additional_request_rad",
            "additional_applied_rad",
        )
    )
    state = data["state"]
    if (
        not np.array_equal(state[:, 0], np.arange(300, dtype=np.float64))
        or not np.isin(state[:, 1], [-1, 0, 1]).all()
        or not np.isin(state[:, 2], [0, 1, 2, 3]).all()
        or np.any(state[:, 3] != float(probe.enabled))
        or not np.isin(state[:, 4], list(STATUS.values())).all()
        or not np.isin(state[:, 9], [-1, 0, 1, 2, 3]).all()
        or np.any(state[:, 8] < 0)
        or np.any(state[:, 8] > 1)
        or np.max(np.abs(pre)) > 0.35 + 1e-12
        or np.max(np.abs(combined)) > 0.35 + 1e-12
        or np.max(np.abs(applied)) > 0.1 + 1e-12
        or not np.allclose(combined, pre + applied, atol=1e-15, rtol=0)
        or not np.allclose(applied, requested * state[:, 8, None], atol=1e-15, rtol=0)
        or np.max(np.abs(data["linear_predicted_displacement_m"])) > 1e-8
    ):
        raise ValueError("bounded recorded heading equations and exact causal clock required")
    for f in range(300):
        phase = (
            0 if f < 30 else 2 if first is not None and f > first else 1 if state[f, 1] >= 0 else 3
        )
        if state[f, 2] != phase:
            raise ValueError("recorded heading phase differs from causal contact history")
        if phase != 1:
            if state[f, 4:].tobytes() != np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            ).tobytes() or any(
                np.any(data[k][f] != 0)
                for k in (
                    "additional_request_rad",
                    "additional_applied_rad",
                    "forward_axis_world",
                    "linear_predicted_displacement_m",
                )
            ):
                raise ValueError(
                    "inactive and post-contact phases cannot propose new heading actions"
                )
        else:
            if not np.isclose(
                np.linalg.norm(data["forward_axis_world"][f]), 1.0, atol=1e-8, rtol=0
            ):
                raise ValueError("active heading uses an actual unit foot forward axis")
            if not probe.enabled and (
                state[f, 4] != STATUS["OFF_UNCHANGED"]
                or np.any(applied[f] != 0)
                or np.any(requested[f] != 0)
            ):
                raise ValueError("heading OFF must retain the exact original increment")
            expected_request = (
                -float(np.clip(state[f, 5], -MAX_HEADING_REQUEST_RAD, MAX_HEADING_REQUEST_RAD))
                if probe.enabled
                else 0.0
            )
            if state[f, 6] != expected_request:
                raise ValueError("fixed bounded heading request required")
    result = dict(
        schema="soccer.rsi.foot_heading_capture_review.v1",
        contract_hash=contract["contract_hash"],
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        frames_reconstructed=300,
        navigation_review_hash=navigation_review["report_hash"],
        causal_phase_and_projection_equations_checked=True,
        actual_cpu_geometry_reconstructed_here=False,
        input_files_authenticated_here=False,
        new_physical_steps=0,
        optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))


class FootHeadingCpuReconstruction(original.OriginalIkCpuReconstruction):
    """Own causal state, actual CPU kinematics, original IK and combined release."""

    def __init__(self, model: Any, probe: FootHeadingContactProbe) -> None:
        if type(probe) is not FootHeadingContactProbe:
            raise ValueError("exact fixed heading reconstruction intent required")
        probe.__post_init__()
        super().__init__(model)
        self.probe = probe
        self.last_record: dict[str, np.ndarray] | None = None

    def before_step(
        self, data: Any, frame: int, positions: np.ndarray, navigation_target: np.ndarray
    ) -> np.ndarray:
        full = super().before_step(data, frame, positions, navigation_target)
        assert self.pending is not None
        combined, record = compose_heading_step(
            self.probe,
            self.model,
            data,
            frame=frame,
            side=self.side,
            contact_frame=self.contact,
            navigation_target=navigation_target,
            increment=self.pending[1],
        )
        if self.side >= 0:
            full[6 * self.side : 6 * self.side + 6] = combined
        self.pending = (frame, combined.copy())
        self.last_record = record
        return full

    def review(self) -> dict[str, Any]:
        if self.next_frame != 300 or self.pending is not None:
            raise ValueError("whole300 causal combined IK/heading CPU reconstruction required")
        result = dict(
            schema="soccer.rsi.foot_heading_cpu_reconstruction.v1",
            contract_hash=self.probe.contract()["contract_hash"],
            frames_reconstructed=300,
            original_ik_from_actual_cpu_state_reconstructed=True,
            heading_and_combined_release_from_actual_cpu_state_reconstructed=True,
            source_hash=hash_bytes(Path(__file__).read_bytes()),
            original_ik_source_hash=hash_bytes(Path(original.__file__).read_bytes()),
            independent_ik_implementation=False,
            independent_heading_implementation=False,
            input_files_authenticated_here=False,
            capture_comparison_performed_here=False,
            new_physical_steps=0,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        return dict(result, report_hash=hash_json(result))


def require_heading_reviews(
    scalar: Any,
    capture: Any,
    cpu: Any,
    *,
    pulse_hash: str,
    heading_hash: str,
    frames_compared: Any,
) -> None:
    """Bind complete numerical reviews, not input file identity or execution authority."""
    if type(frames_compared) is not int or frames_compared != 300:
        raise ValueError("all300 actual CPU heading captures must be compared")
    for d, schema, source_hash, contract_hash, truths, falsehoods in (
        (
            scalar,
            "soccer.rsi.original_ik_pre_planner_scalar_review.v1",
            hash_bytes(Path(original.__file__).read_bytes()),
            pulse_hash,
            (
                "original_bounded_navigation_reconstructed",
                "pulse_composition_reconstructed",
                "forward_and_yaw_unchanged",
                "original_ik_intent_checked",
                "learned_reference_arrays_absent",
            ),
            (
                "original_ik_targets_reconstructed_here",
                "source_trace_authentication_performed_here",
                "planner_and_encoder_effect_verified_here",
            ),
        ),
        (
            capture,
            "soccer.rsi.foot_heading_capture_review.v1",
            hash_bytes(Path(__file__).read_bytes()),
            heading_hash,
            ("causal_phase_and_projection_equations_checked",),
            ("actual_cpu_geometry_reconstructed_here", "input_files_authenticated_here"),
        ),
        (
            cpu,
            "soccer.rsi.foot_heading_cpu_reconstruction.v1",
            hash_bytes(Path(__file__).read_bytes()),
            heading_hash,
            (
                "original_ik_from_actual_cpu_state_reconstructed",
                "heading_and_combined_release_from_actual_cpu_state_reconstructed",
            ),
            (
                "independent_ik_implementation",
                "independent_heading_implementation",
                "input_files_authenticated_here",
                "capture_comparison_performed_here",
            ),
        ),
    ):
        if (
            type(d) is not dict
            or d.get("schema") != schema
            or d.get("source_hash") != source_hash
            or d.get("contract_hash") != contract_hash
            or type(d.get("frames_reconstructed")) is not int
            or d["frames_reconstructed"] != 300
            or any(d.get(k) is not True for k in truths)
            or any(d.get(k) is not False for k in falsehoods)
            or any(
                d.get(k, False) is not False
                for k in (
                    "hardware_authorized",
                    "promotion_authorized",
                    "training_authorized",
                    "teacher_qualified",
                    "runtime_execution_authorized",
                )
            )
            or any(
                type(d.get(k)) is not int or d[k] != 0
                for k in ("new_physical_steps", "optimizer_updates")
            )
            or d.get("report_hash") != hash_json({k: v for k, v in d.items() if k != "report_hash"})
        ):
            raise ValueError(
                "whole current-source navigation, heading capture and actual CPU reviews required"
            )
    if capture.get("navigation_review_hash") != scalar["report_hash"] or cpu.get(
        "original_ik_source_hash"
    ) != hash_bytes(Path(original.__file__).read_bytes()):
        raise ValueError("heading reviews must bind exact original IK source and navigation review")
