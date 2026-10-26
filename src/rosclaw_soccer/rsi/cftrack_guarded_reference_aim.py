"""Choose a bounded SIM reference aim with the existing alignment fallback."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cftrack_alignment_diagnostic import direct_approach_geometry
from rosclaw_soccer.rsi.cftrack_velocity_intercept import (
    condition_velocity_intercept,
    predict_initial_intercept,
)
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer


def select_guarded_aim(
    *,
    ball: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
    base: np.ndarray,
    quaternion: np.ndarray,
    requested_offset_y_m: float,
) -> dict[str, Any]:
    if (
        type(requested_offset_y_m) is not float
        or not np.isfinite(requested_offset_y_m)
        or abs(requested_offset_y_m) > 0.7
    ):
        raise ValueError("bounded explicit reference-only goal offset required")
    predicted, prediction = predict_initial_intercept(ball, velocity, goal)
    # Margin is declared before testing, so float encoding cannot accidentally
    # pass a state just outside the original direct-approach guard.
    margin = 1e-6

    def inspect(offset: float) -> tuple[dict[str, float | bool], bool]:
        planned_goal = goal.astype(np.float64, copy=True)
        planned_goal[1] += offset
        geometry = direct_approach_geometry(base, quaternion, predicted, planned_goal)
        allowed = bool(
            float(geometry["heading_error_rad"]) <= 0.2 - margin
            and 0.3 + margin <= float(geometry["forward_offset_m"]) <= 1.5 - margin
            and abs(float(geometry["lateral_offset_m"])) <= 0.2 - margin
        )
        return geometry, allowed

    requested, requested_allowed = inspect(requested_offset_y_m)
    fallback, fallback_allowed = inspect(0.0)
    if not requested_allowed and not fallback_allowed:
        raise ValueError("neither requested nor original reference satisfies the existing guard")
    chosen = requested_offset_y_m if requested_allowed else 0.0
    return dict(
        schema="soccer.cftrack.guarded_reference_aim.v1",
        requested_offset_y_m=requested_offset_y_m,
        selected_offset_y_m=chosen,
        fallback_to_original=not requested_allowed,
        requested_geometry=requested,
        original_geometry=fallback,
        declared_margin=margin,
        prediction=prediction,
        original_joint_pose_and_native_guards_still_apply=True,
        selector_is_not_a_trained_neural_policy=True,
        hardware_authorized=False,
    )


def condition_guarded_reference_aim(
    player: SharedOmniContactPlayer, *, requested_offset_y_m: float
) -> dict[str, Any]:
    world, state, layout = player.world, player.policy.state_cmd, player.layout
    world._require_current_boundary()
    if not np.array_equal(state.ball_pos, world.data.xpos[layout.ball]):
        raise ValueError("actual current ball observation required for selection")
    decision = select_guarded_aim(
        ball=np.asarray(state.ball_pos),
        velocity=world.data.qvel[layout.bv : layout.bv + 3].copy(),
        goal=np.asarray(player.policy.goal_pos),
        base=np.asarray(state.base_pos),
        quaternion=np.asarray(state.base_quat),
        requested_offset_y_m=requested_offset_y_m,
    )
    reference = condition_velocity_intercept(
        player, reference_goal_offset_y_m=decision["selected_offset_y_m"]
    )
    return dict(selection=decision, reference=reference, hardware_authorized=False)
