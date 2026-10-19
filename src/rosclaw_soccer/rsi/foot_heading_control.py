"""Explicit fixed OFF/ON contact-heading SIM intent, never an execution permit."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi import foot_heading_nullspace as numeric
from rosclaw_soccer.rsi.original_ik_planner_pulse import (
    OriginalIkPlannerPulse,
    require_original_ik_planner_request,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@dataclass(frozen=True, slots=True)
class FootHeadingContactProbe:
    enabled: bool

    def __post_init__(self) -> None:
        if type(self) is not FootHeadingContactProbe or type(self.enabled) is not bool:
            raise ValueError("exact fixed heading OFF/ON probe required")

    def contract(self) -> dict[str, Any]:
        self.__post_init__()
        value = dict(
            schema="soccer.rsi.foot_heading_contact_probe.v1",
            parameters=dict(enabled=self.enabled),
            source_hash=hash_bytes(Path(__file__).read_bytes()),
            numerical_source_hash=hash_bytes(Path(numeric.__file__).read_bytes()),
            max_heading_request_rad=numeric.MAX_HEADING_REQUEST_RAD,
            heading_deadband_rad=numeric.HEADING_DEADBAND_RAD,
            heading_law="deadband_then_clipped_error.v1",
            max_additional_joint_rad=numeric.MAX_ADDITIONAL_JOINT_RAD,
            max_total_ik_increment_rad=numeric.MAX_TOTAL_INCREMENT_RAD,
            target_heading_world_rad=0.0,
            diagnostic_course_forward_world_positive_x=True,
            instantaneous_body_origin_position_nullspace=True,
            mesh_contact_point_not_position_constrained=True,
            original_positive_ik_planner_pulse_required=True,
            original_bounded_ik_and_parent_required=True,
            first_possible_frame=30,
            original_causal_airborne_swing_side_required=True,
            original_six_channel_contact_gate_unchanged=True,
            first_any_monitored_robot_contact_ends_heading_updates=True,
            monitored_robot_bodies=[
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_ankle_pitch_link",
                "right_ankle_pitch_link",
                "left_knee_link",
                "right_knee_link",
            ],
            dense_unmonitored_contact_review_required=True,
            combined_increment_released_by_original_20_frame_law=True,
            off_matches_original_positive_ik_all300_arrays=True,
            on_matches_original_positive_ik_first30_arrays=True,
            future_contact_time_or_outcome_input_forbidden=True,
            native_request_and_independent_qualification_required=True,
            trained_policy=False,
            physics_qualified=False,
            runtime_execution_authorized=False,
            training_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        return dict(value, contract_hash=hash_json(value))


def require_heading_request(
    probe: Any,
    *,
    pulse: Any,
    reference_model: Any,
    profile: Any,
    record_planner_events: Any,
) -> None:
    """Values-only one-factor request; callers still enforce the complete native boundary."""
    if type(probe) is not FootHeadingContactProbe:
        raise ValueError("explicit typed contact-heading probe required")
    probe.__post_init__()
    require_original_ik_planner_request(
        pulse, reference_model, profile=profile, record_planner_events=record_planner_events
    )
    if type(pulse) is not OriginalIkPlannerPulse or pulse.lateral_delta_m_s != 0.08:
        raise ValueError("same fixed positive original-IK pulse required in both heading arms")
