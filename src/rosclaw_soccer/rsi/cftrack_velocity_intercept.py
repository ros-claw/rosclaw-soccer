"""SIM privileged initial interception reference from measured rolling velocity.

The prediction changes the planned reference only. Physical state and measured
ball observations remain actual; no second policy entry, transport, or motor
command is created. Generalization and actual contact require native trials.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cftrack_direct_approach import condition_direct_kick_approach
from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer


def predict_initial_intercept(
    ball_position: np.ndarray, ball_velocity: np.ndarray, goal: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    arrays = (ball_position, ball_velocity, goal)
    if any(
        type(x) is not np.ndarray
        or x.shape != (3,)
        or x.dtype.kind != "f"
        or not np.isfinite(x).all()
        for x in arrays
    ):
        raise ValueError("finite measured ball, velocity and goal vectors required")
    if np.linalg.norm(ball_velocity[:2]) > 0.2 or abs(ball_velocity[2]) > 1e-12:
        raise ValueError("bounded ground-ball velocity up to 0.2m/s required")
    direction = goal[:2] - ball_position[:2]
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        raise ValueError("nondegenerate goal direction required")
    direction = direction / length
    longitudinal = float(ball_velocity[:2] @ direction)
    # Nominal first contact was frame72 at 50Hz. 0.8m/s is a declared
    # assumed approach speed, not a claim about measured real robot speed.
    horizon = 1.44 / (1.0 - longitudinal / 0.8)
    predicted = ball_position.astype(np.float64, copy=True) + ball_velocity * horizon
    displacement = float(np.linalg.norm(predicted - ball_position))
    if not 1.0 <= horizon <= 2.0 or displacement > 0.3:
        raise ValueError("bounded constant-velocity intercept prediction required")
    return predicted, dict(
        measured_ball_position=ball_position.tolist(),
        measured_ball_velocity=ball_velocity.tolist(),
        predicted_intercept_position=predicted.tolist(),
        nominal_contact_time_s=1.44,
        assumed_approach_speed_m_s=0.8,
        predicted_horizon_s=horizon,
        maximum_prediction_displacement_m=0.3,
        prediction_is_not_a_measured_future=True,
        privileged_sim_velocity_input=True,
    )


def condition_velocity_intercept(
    player: SharedOmniContactPlayer, *, reference_goal_offset_y_m: float = 0.0
) -> dict[str, Any]:
    """Regenerate a cold reference once from a bounded predicted ball location."""
    from policy.omnicontact.CFgen_reference import (  # type: ignore[import-not-found]
        plan_cfgen_reference,
    )

    w, p = player.world, player.policy
    try:
        w._require_current_boundary()
    except BaseException:
        player.failed = True
        raise
    if (
        type(player) is not SharedOmniContactPlayer
        or w.native_steps != 0
        or w.control_steps != 0
        or player.failed
        or player.pending is not None
        or player.motor_plan is not None
        or w._torque_layer_owners
        or p.counter_step != 0
        or p.task != "kickball"
        or p.reference_source != "CFgen"
        or p.async_stage_thread is not None
        or np.any(p.obs_history_buffer)
        or np.any(p.action)
        or type(reference_goal_offset_y_m) is not float
        or not np.isfinite(reference_goal_offset_y_m)
        or abs(reference_goal_offset_y_m) > 0.7
    ):
        raise ValueError("unexecuted private CFgen SIM policy required")
    s = p.state_cmd
    ball = np.asarray(s.ball_pos).copy()
    obj = np.asarray(s.obj_pos).copy()
    actual_goal = np.asarray(p.goal_pos).copy()
    if not np.array_equal(ball, w.data.xpos[player.layout.ball]) or not np.array_equal(ball, obj):
        raise ValueError("actual observed current ball position required")
    velocity = w.data.qvel[player.layout.bv : player.layout.bv + 3].copy()
    prediction, forecast = predict_initial_intercept(ball, velocity, np.asarray(p.goal_pos))
    old_policy = cftrack_state_hash(p)
    before_q, before_v, before_time = w.data.qpos.copy(), w.data.qvel.copy(), float(w.data.time)
    try:
        if np.any(velocity) or reference_goal_offset_y_m:
            # Restricted to plan generation. The state is restored before
            # any inference or integration, including on exceptions.
            s.ball_pos = prediction.copy()
            s.obj_pos = prediction.copy()
            p.goal_pos = actual_goal.copy()
            p.goal_pos[1] += reference_goal_offset_y_m
            fk = p.kinematics.forward(s.q, s.base_pos, s.base_quat)
            plan_cfgen_reference(p, fk)
        direct = condition_direct_kick_approach(p)
    except BaseException:
        player.failed = True
        raise
    finally:
        s.ball_pos = ball
        s.obj_pos = obj
        p.goal_pos = actual_goal
    try:
        w._require_current_boundary()
    except BaseException:
        player.failed = True
        raise
    if (
        not np.array_equal(w.data.qpos, before_q)
        or not np.array_equal(w.data.qvel, before_v)
        or float(w.data.time) != before_time
        or p.counter_step != 0
        or p.async_stage_thread is not None
        or not np.array_equal(s.ball_pos, ball)
        or not np.array_equal(s.obj_pos, obj)
        or not np.array_equal(p.goal_pos, actual_goal)
    ):
        player.failed = True
        raise ValueError("actual body and measured observations must remain unchanged")
    receipt = dict(
        schema="soccer.cftrack.velocity_intercept_reference.v1",
        prediction=forecast,
        direct_approach=direct,
        previous_policy_hash=old_policy,
        conditioned_policy_hash=cftrack_state_hash(p),
        physical_state_and_clock_unchanged=True,
        measured_ball_observations_restored_before_inference=True,
        no_second_policy_entry=True,
        neural_optimizer_updates=0,
        hardware_authorized=False,
    )
    if reference_goal_offset_y_m:
        receipt["reference_aim"] = dict(
            planning_only_goal_offset_y_m=reference_goal_offset_y_m,
            measured_task_goal_restored_before_inference=True,
            h300_target_and_scoring_unchanged=True,
        )
    return receipt
