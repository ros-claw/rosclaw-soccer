"""Fail-closed SIM check of the actual state before a staged reference proceeds."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cftrack_alignment_diagnostic import direct_approach_geometry


@dataclass(frozen=True)
class CFtrackStageBoundary:
    control: int
    waypoint_xy: tuple[float, float]
    reference_ball_xyz: tuple[float, float, float]
    actual_goal_xyz: tuple[float, float, float]
    planned_contact_horizon_s: float
    waypoint_tolerance_m: float = 0.08
    ball_prediction_tolerance_m: float = 0.08

    def __post_init__(self) -> None:
        if (
            type(self.control) is not int
            or not 1 <= self.control <= 100
            or type(self.waypoint_xy) is not tuple
            or len(self.waypoint_xy) != 2
            or type(self.reference_ball_xyz) is not tuple
            or len(self.reference_ball_xyz) != 3
            or type(self.actual_goal_xyz) is not tuple
            or len(self.actual_goal_xyz) != 3
            or any(
                type(x) is not float or not np.isfinite(x)
                for x in (*self.waypoint_xy, *self.reference_ball_xyz, *self.actual_goal_xyz)
            )
            or type(self.waypoint_tolerance_m) is not float
            or not 0.0 < self.waypoint_tolerance_m <= 0.08
            or type(self.ball_prediction_tolerance_m) is not float
            or not 0.0 < self.ball_prediction_tolerance_m <= 0.08
            or type(self.planned_contact_horizon_s) is not float
            or not 1.0 <= self.planned_contact_horizon_s <= 2.0
        ):
            raise ValueError("explicit bounded SIM stage commitment required")

    def check(self, world: Any, layout: Any) -> dict[str, Any]:
        if world.control_steps != self.control:
            raise ValueError("exact actual SIM stage boundary required")
        world._require_current_boundary()
        root = world.data.qpos[layout.rq : layout.rq + 7].copy()
        ball = world.data.qpos[layout.bq : layout.bq + 3].copy()
        velocity = world.data.qvel[layout.bv : layout.bv + 3].copy()
        remaining = max(0.0, self.planned_contact_horizon_s - self.control * 0.02)
        observed_prediction = ball + velocity * remaining
        prediction_error = float(
            np.linalg.norm(observed_prediction - np.array(self.reference_ball_xyz))
        )
        geometry = direct_approach_geometry(
            root[:3], root[3:7], np.array(self.reference_ball_xyz), np.array(self.actual_goal_xyz)
        )
        waypoint_error = float(np.linalg.norm(root[:2] - np.array(self.waypoint_xy)))
        ready = bool(
            waypoint_error <= self.waypoint_tolerance_m
            and prediction_error <= self.ball_prediction_tolerance_m
            and geometry["heading_within_guard"]
            and geometry["forward_within_guard"]
            and geometry["lateral_within_guard"]
            and root[2] >= 0.65
            and not world.failed
        )
        return dict(
            schema="soccer.cftrack.actual_stage_boundary.v1",
            control=self.control,
            measured_root_pose=root.tolist(),
            measured_ball_position=ball.tolist(),
            measured_ball_velocity=velocity.tolist(),
            current_constant_velocity_projection=observed_prediction.tolist(),
            ball_projection_error_m=prediction_error,
            ball_projection_tolerance_m=self.ball_prediction_tolerance_m,
            waypoint_error_m=waypoint_error,
            waypoint_tolerance_m=self.waypoint_tolerance_m,
            measured_geometry_to_planned_intercept=geometry,
            ready=ready,
            geometry_only_not_a_complete_physical_safety_proof=True,
            actual_body_state_not_reset=True,
            hardware_authorized=False,
        )
