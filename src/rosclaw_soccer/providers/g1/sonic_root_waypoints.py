"""Explicit SIM research adapter for a fixed known world-root stance goal.

Default G1SonicNavigation remains unchanged. This subclass adds no physics or
hardware access. Dynamic intercept/ball tracking and forecast cloning are not
qualified by this fixed-goal pilot. The caller owns every native step.
"""

import copy
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.sonic_navigation import (
    G1SonicNavigation,
    SonicNavigationConfig,
    _StreamingBackend,
)
from rosclaw_soccer.providers.g1.sonic_root_waypoint_inputs import (
    RootWaypointIntent,
    apply_waypoint_intent,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation, TeamMotorTarget


class _RootWaypointBackend(_StreamingBackend):
    def __init__(
        self,
        model_root: Path,
        config: SonicNavigationConfig,
        native_joint_limits: tuple[tuple[float, float], ...],
    ) -> None:
        self.active_goal: RootWaypointIntent | None = None
        self.waypoint_events: list[dict[str, Any]] = []
        self.active_observation_hash: str | None = None
        self.native_joint_limits = native_joint_limits
        super().__init__(model_root, config)

    def _transform_planner_feed(
        self, feed: dict[str, np.ndarray], frame: int
    ) -> dict[str, np.ndarray]:
        if self.active_observation_hash is None:
            raise ValueError("root waypoint planner requires this tick current-state intent")
        actual = (
            apply_waypoint_intent(feed, self.active_goal, activate=True)
            if self.active_goal is not None
            else feed
        )
        self.waypoint_events.append(
            dict(
                frame=frame,
                intent=self.active_goal.report() if self.active_goal is not None else None,
                intent_hash=self.active_goal.contract_hash
                if self.active_goal is not None
                else None,
                observation_hash=self.active_observation_hash,
                fixed_goal_expired=self.active_goal is None,
                actual_planner_inputs={k: v.tolist() for k, v in actual.items()},
                actual_planner_input_dtypes={k: str(v.dtype) for k, v in actual.items()},
            )
        )
        return actual

    def plan(self, context: np.ndarray, frame: int) -> np.ndarray:
        result = super().plan(context, frame)
        self.waypoint_events[-1]["original_planner_record"] = self.events[-1].copy()
        return result

    def _validate_planner_segment(self, segment: np.ndarray) -> None:
        limits = np.asarray(self.native_joint_limits, dtype=np.float64)
        violation = float(
            max(0.0, np.maximum(limits[:, 0] - segment[:, 7:], segment[:, 7:] - limits[:, 1]).max())
        )
        self.waypoint_events[-1]["raw_planner_segment"] = segment.tolist()
        self.waypoint_events[-1]["maximum_native_joint_limit_violation_rad"] = violation
        if violation > 1e-6:
            raise ValueError(
                "waypoint planner reference violates bound native joint metadata; no clipping"
            )


class G1SonicRootWaypointNavigation(G1SonicNavigation):
    """A separately bound fixed-goal option, not an implicit default switch."""

    def _make_backend(self, model_root: Path, config: SonicNavigationConfig) -> _StreamingBackend:
        return _RootWaypointBackend(model_root, config, self.native_joint_limits)

    def __init__(
        self,
        model_root: Path,
        agent_id: str,
        config: SonicNavigationConfig,
        *,
        fixed_world_goal_xy_m: tuple[float, float],
        goal_basis_hash: str,
        goal_deadline_frame: int,
        native_joint_limits: tuple[tuple[float, float], ...],
    ) -> None:
        expected = SonicNavigationConfig(
            maximum_frames=300,
            model_variant="low_latency",
            inference_threads=1,
            record_default_planner_events=True,
        )
        if type(config) is not SonicNavigationConfig or asdict(config) != asdict(expected):
            raise ValueError("unmixed frozen low-latency H300 configuration required")
        if type(fixed_world_goal_xy_m) is not tuple or len(fixed_world_goal_xy_m) != 2:
            raise ValueError("fixed known world XY root target required")
        if type(goal_deadline_frame) is not int or not 0 <= goal_deadline_frame < 300:
            raise ValueError("explicit bounded fixed-goal deadline required")
        if (
            type(native_joint_limits) is not tuple
            or len(native_joint_limits) != 29
            or any(
                type(p) is not tuple
                or len(p) != 2
                or any(type(v) is not float or not math.isfinite(v) for v in p)
                or p[0] >= p[1]
                for p in native_joint_limits
            )
        ):
            raise ValueError("immutable original native29 joint-limit metadata required")
        # Reuse input validation before NN/session allocation.
        checked = RootWaypointIntent(
            (*fixed_world_goal_xy_m, 0.793), 0.0, goal_basis_hash, goal_basis_hash
        )
        self.fixed_world_goal_xy_m = fixed_world_goal_xy_m
        self.goal_basis_hash = checked.goal_basis_hash
        self.goal_deadline_frame = goal_deadline_frame
        self.native_joint_limits = native_joint_limits
        super().__init__(model_root, agent_id, config)
        self.contract_hash = hash_json(
            dict(
                schema="soccer.g1.sonic_fixed_root_waypoint_navigation.v1",
                base_navigation_hash=self.contract_hash,
                fixed_world_goal_xy_m=fixed_world_goal_xy_m,
                goal_basis_hash=goal_basis_hash,
                goal_yaw_rad=0.0,
                goal_deadline_frame=goal_deadline_frame,
                native_joint_limits=native_joint_limits,
                inputs_only_no_body_state_mutation=True,
                forecast_clone_qualified=False,
                activation_ceiling="SIM_ONLY",
                hardware_authorized=False,
            )
        )

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget:
        if not isinstance(self.backend, _RootWaypointBackend):
            raise ValueError("explicit root waypoint backend required")
        try:
            current = hash_json(
                dict(
                    frame=observation.frame,
                    qpos=list(observation.qpos),
                    qvel=list(observation.qvel),
                )
            )
            self.backend.active_observation_hash = current
            self.backend.active_goal = (
                RootWaypointIntent(
                    (*self.fixed_world_goal_xy_m, float(observation.qpos[2])),
                    0.0,
                    current,
                    self.goal_basis_hash,
                )
                if observation.frame <= self.goal_deadline_frame
                else None
            )
            return super().propose(observation)
        except Exception:
            self._faulted = True
            raise
        finally:
            self.backend.active_goal = None
            self.backend.active_observation_hash = None

    @property
    def waypoint_events(self) -> list[dict[str, Any]]:
        if not isinstance(self.backend, _RootWaypointBackend):
            raise ValueError("explicit root waypoint backend required")
        return copy.deepcopy(self.backend.waypoint_events)
