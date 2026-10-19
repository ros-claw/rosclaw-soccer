"""Original-IK pulse capture and CPU-state reconstruction, never admission."""

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.original_ik_planner_pulse import (
    OriginalIkPlannerPulse,
    require_original_ik_planner_request,
)
from rosclaw_soccer.rsi.taskspace_swing_probe import (
    choose_swing_side,
    continuous_swing_joint_delta,
    release_joint_delta,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

TAG = "original_ik_pre_planner_pulse"
PREFIX = "ik_planner_pulse_"
FIELDS = ("baseline_command", "requested_residual", "applied_command")
FORBIDDEN_TAGS = (
    "pre_planner_navigation_pulse",
    "reference_student_model",
    "reference_student_contract",
    "mesh_surface_swing",
    "contact_exit_torque_mpc",
    "directional_contact_compliance",
    "swing_release_ablation",
    "foundation_approach_ablation",
    "cerebellar_torque_contract",
    "postcontact_navigation_probe",
    "locomotion_mode_probe",
)


def require_original_ik_capture(
    report: dict[str, Any], commitment: dict[str, Any], trace: dict[str, Any]
) -> dict[str, Any]:
    if (
        TAG not in commitment
        or {k for k in trace if k.startswith(PREFIX)} != {PREFIX + k for k in FIELDS}
        or any(k.startswith(("reference_student_", "planner_pulse_")) for k in trace)
        or any(k in d for d in (report, commitment) for k in FORBIDDEN_TAGS)
        or commitment.get("execution_profile") != "taskspace_plus_motor"
        or commitment.get("swing_mode") != "bounded_contact_diagnostic"
        or commitment.get("taskspace_contract")
        != dict(
            forward_cap_m=0.08,
            lateral_cap_m=0.15,
            acquisition_max_gap_m=0.95,
            revalidate_swing_side=True,
            vertical_offset_m=0.04,
        )
    ):
        raise ValueError(
            "explicit unmixed original-IK commitment and complete pulse capture required"
        )
    contract = commitment[TAG]
    pulse = OriginalIkPlannerPulse(**contract["parameters"])
    if contract != pulse.contract() or report.get(TAG) != contract:
        raise ValueError("same exact original-IK pulse source and contract required")
    profile = BoundedFirstTouchControl(**commitment["bounded_control_profile"]["parameters"])
    if commitment["bounded_control_profile"] != profile.contract():
        raise ValueError("original bounded profile source required")
    require_original_ik_planner_request(pulse, None, profile=profile, record_planner_events=True)

    def array(key: str, width: int) -> np.ndarray:
        a = np.asarray(trace.get(key))
        if (
            a.shape != (300, 1, width)
            or a.dtype != np.float64
            or not np.isfinite(a).all()
            or np.max(np.abs(a)) > 1e6
        ):
            raise ValueError("whole finite H300 original-IK pulse arrays required")
        return a[:, 0]

    base, residual, applied = (array(PREFIX + k, 3) for k in FIELDS)
    nav, raw = array("navigation_command", 3), array("approach_requested_navigation_command", 3)
    root, ball = array("root_pose_xyzw_m", 7), array("ball_position_before_step_m", 3)
    velocity, forces = array("ball_linear_velocity_before_step_m_s", 3), array("force_n", 6)
    if np.any(forces < 0):
        raise ValueError("nonnegative original contact force channels required")
    contacts = np.flatnonzero(np.any(forces > 1, axis=1))
    first = int(contacts[0]) if len(contacts) else None
    for f in range(300):
        gap = ball[f] - root[f, :3]
        if first is None or f <= first:
            vx = float(velocity[f, 0]) + 1.2 * (float(gap[0]) - 0.35)
            fade = min(1.0, max(0.0, (float(gap[0]) - 0.35) / (0.95 - 0.35)))
            vy = 1.2 * float(gap[1]) * fade
        else:
            vx, vy = 1.4, 0.0
        expected_raw = np.asarray([vx, vy, 0.0])
        baseline = np.asarray([np.clip(vx, 0.0, 1.4), np.clip(vy, -0.2, 0.2), 0.0])
        # Independent scalar law: do not call the producer composition/navigation.
        delta = pulse.lateral_delta_m_s if 100 <= f < 160 else 0.0
        expected = baseline.copy()
        if delta != 0.0:
            expected[1] = min(0.2, max(-0.2, float(baseline[1]) + delta))
        if (
            raw[f].tobytes() != expected_raw.tobytes()
            or base[f].tobytes() != baseline.tobytes()
            or residual[f].tobytes() != np.asarray([0.0, delta, 0.0]).tobytes()
            or applied[f].tobytes() != expected.tobytes()
            or nav[f].tobytes() != expected.tobytes()
        ):
            raise ValueError("independent original-IK B2 plus pulse reconstruction failed")
    result = dict(
        schema="soccer.rsi.original_ik_pre_planner_scalar_review.v1",
        contract_hash=contract["contract_hash"],
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        frames_reconstructed=300,
        original_bounded_navigation_reconstructed=True,
        pulse_composition_reconstructed=True,
        forward_and_yaw_unchanged=True,
        original_ik_intent_checked=True,
        learned_reference_arrays_absent=True,
        original_ik_targets_reconstructed_here=False,
        source_trace_authentication_performed_here=False,
        planner_and_encoder_effect_verified_here=False,
        new_physical_steps=0,
        optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))


def require_original_ik_parity(
    trace: dict[str, Any], baseline: dict[str, Any], *, frames: int
) -> dict[str, Any]:
    if type(frames) is not int or frames not in (100, 300):
        raise ValueError("exact original-teacher OFF300 or active100 parity required")

    def physical(value: dict[str, Any]) -> dict[str, Any]:
        extra = {k for k in value if k.startswith(PREFIX)}
        if (extra and extra != {PREFIX + k for k in FIELDS}) or any(
            k.startswith(("reference_student_", "planner_pulse_")) for k in value
        ):
            raise ValueError("no mixed or partial IK pulse capture may be hidden")
        arrays = {k: v for k, v in value.items() if k not in extra}
        if len(arrays) != 33 or any(
            not isinstance(a, np.ndarray) or len(a) != 300 for a in arrays.values()
        ):
            raise ValueError("all original33 complete physical arrays required")
        return arrays

    left, right = physical(trace), physical(baseline)
    if left.keys() != right.keys() or any(
        left[k].shape != right[k].shape
        or left[k].dtype != right[k].dtype
        or left[k][:frames].tobytes() != right[k][:frames].tobytes()
        for k in left
    ):
        raise ValueError("all original-teacher physical array bits must match")
    return dict(
        compared_frames=frames,
        physical_array_count=33,
        original_array_bits_equal=True,
        runtime_execution_authorized=False,
    )


class OriginalIkCpuReconstruction:
    """Recompute old IK from the CPU replay's actual cached state, before stepping.

    Shares the original allowlisted numerical IK implementation, not producer
    control captures or a second independent IK algorithm. Owns causal side,
    contact and release history; no future forces or physical integration.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.feet = [
            int(model.body(n).id) for n in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]
        self.ball = int(model.body("ball").id)
        joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
        self.vi = np.asarray([int(model.jnt_dofadr[j]) for j in joints])
        self.limits = model.jnt_range[joints].copy()
        self.side = -1
        self.contact: int | None = None
        self.next_frame = 0
        self.contact_delta = np.zeros(6)
        self.pending: tuple[int, np.ndarray] | None = None

    def before_step(
        self, data: Any, frame: int, positions: np.ndarray, navigation_target: np.ndarray
    ) -> np.ndarray:
        import mujoco

        if (
            type(frame) is not int
            or frame != self.next_frame
            or self.pending is not None
            or frame >= 300
        ):
            raise ValueError("sequential original-IK CPU before/after clock required")
        if (
            navigation_target.shape != (29,)
            or navigation_target.dtype != np.float64
            or not np.isfinite(navigation_target).all()
            or positions.shape != (self.model.nbody, 3)
            or positions.dtype != np.float64
            or not np.isfinite(positions).all()
        ):
            raise ValueError(
                "finite original29-joint target and actual CPU body positions required"
            )
        delta = np.zeros(6)
        full = np.zeros(29)
        if frame >= 30:
            if self.contact is None:
                self.side = choose_swing_side(
                    positions[self.feet],
                    positions[self.ball],
                    self.side,
                    acquisition_max_gap_m=0.95,
                    revalidate_swing_side=True,
                )
            if self.side >= 0:
                ids = np.arange(6 * self.side, 6 * self.side + 6)
                if self.contact is None:
                    jac = np.zeros((3, self.model.nv))
                    mujoco.mj_jacBody(self.model, data, jac, None, self.feet[self.side])
                    delta = continuous_swing_joint_delta(
                        positions[self.feet[self.side]],
                        positions[self.ball],
                        jac[:, self.vi[ids]],
                        navigation_target[ids],
                        self.limits[ids],
                        forward_cap_m=0.08,
                        lateral_cap_m=0.15,
                        vertical_offset_m=0.04,
                    )
                else:
                    delta = release_joint_delta(self.contact_delta, frame - self.contact)
                full[ids] = delta
        self.pending = (frame, delta.copy())
        return full

    def after_step(self, frame: int, forces: np.ndarray) -> None:
        if (
            self.pending is None
            or type(frame) is not int
            or frame != self.pending[0]
            or forces.shape != (6,)
            or forces.dtype != np.float64
            or not np.isfinite(forces).all()
            or np.any(forces < 0)
        ):
            raise ValueError(
                "complete current CPU force observation after pending IK frame required"
            )
        if self.contact is None and np.any(forces > 1):
            self.contact = frame
            self.contact_delta = self.pending[1].copy()
        self.next_frame += 1
        self.pending = None

    def review(self) -> dict[str, Any]:
        if self.next_frame != 300 or self.pending is not None:
            raise ValueError("whole300 original-IK reconstruction required")
        result = dict(
            schema="soccer.rsi.original_ik_cpu_reconstruction.v1",
            frames_reconstructed=300,
            causal_side_contact_and_release_reconstructed=True,
            original_ik_from_actual_cpu_state_reconstructed=True,
            independent_ik_implementation=False,
            source_hash=hash_bytes(Path(__file__).read_bytes()),
            new_physical_steps=0,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        return dict(result, report_hash=hash_json(result))


def require_original_ik_reviews(scalar: Any, reconstruction: Any, *, contract_hash: str) -> None:
    """Validate explicit replay intent, not authenticate input files or authorize use."""
    for document, schema, truths, falsehoods in (
        (
            scalar,
            "soccer.rsi.original_ik_pre_planner_scalar_review.v1",
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
            reconstruction,
            "soccer.rsi.original_ik_cpu_reconstruction.v1",
            (
                "causal_side_contact_and_release_reconstructed",
                "original_ik_from_actual_cpu_state_reconstructed",
            ),
            ("independent_ik_implementation",),
        ),
    ):
        if (
            type(document) is not dict
            or document.get("schema") != schema
            or document.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
            or type(document.get("frames_reconstructed")) is not int
            or document["frames_reconstructed"] != 300
            or any(document.get(k) is not True for k in truths)
            or any(
                document.get(k) is not False
                for k in (*falsehoods, "hardware_authorized", "promotion_authorized")
            )
            or any(
                type(document.get(k)) is not int or document[k] != 0
                for k in ("new_physical_steps", "optimizer_updates")
            )
            or document.get("report_hash")
            != hash_json({k: v for k, v in document.items() if k != "report_hash"})
        ):
            raise ValueError(
                "whole sealed current-source scalar and causal CPU IK reviews required"
            )
    if scalar.get("contract_hash") != contract_hash:
        raise ValueError("same explicit original-IK planner contract required")
