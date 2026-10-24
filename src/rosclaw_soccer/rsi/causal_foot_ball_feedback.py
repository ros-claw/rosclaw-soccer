"""Bounded current-state foot-point feedback for privileged SIM teacher research.

No neural policy training, simulator stepping, future trajectories, force
estimates, or runtime/hardware interface. A qualified caller owns integration.
"""

import math
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_stroke_intent import LEG_SUFFIXES


def feedback_parameters(variant: int, *, anticipation_seconds: float = 0.0) -> dict[str, Any]:
    if type(variant) is not int or not 0 <= variant <= 3:
        raise ValueError("one of four declared causal feedback variants required")
    if type(anticipation_seconds) is not float or anticipation_seconds not in (0.0, 0.12):
        raise ValueError("explicit original zero or preregistered120ms anticipation required")
    cfg = dict(
        variant=variant,
        gain=[0.0, 0.25, 0.5, 1.0][variant],
        desired_closing_speed_mps=3.0,
        velocity_lead_seconds=0.04,
        lateral_position_gain=0.5,
        maximum_task_displacement_m=0.025,
        maximum_joint_correction_rad=0.12,
        jacobian_damping_m_per_rad=0.03,
        point_body_local_m=[0.13, -0.02, -0.025],
        first_control=88,
        last_control_exclusive=120,
        fade_controls=4,
        near_distance_m=0.35,
        direction_target="ORIGINAL_PASS_TARGET",
        total_authority_rad=0.2,
    )
    if anticipation_seconds:
        cfg.update(
            anticipation_seconds=anticipation_seconds,
            distance_gate="CURRENT_RELATIVE_VELOCITY_CLOSEST_APPROACH_NOT_SIMULATION",
        )
    return cfg


def closest_approach(
    relative_position: np.ndarray, relative_velocity: np.ndarray, *, seconds: float
) -> tuple[float, float]:
    """Linear current-state gate heuristic, NEVER a verified future trajectory.

    Both vectors are ball minus foot-point in world coordinates. Include all
    three axes, bound time to the declared horizon, and do not extrapolate a
    receding pair backwards. No contact, control, or safety truth is inferred.
    """
    if (
        type(seconds) is not float
        or seconds not in (0.0, 0.12)
        or any(
            not isinstance(x, np.ndarray)
            or x.shape != (3,)
            or x.dtype != np.float64
            or not np.isfinite(x).all()
            for x in (relative_position, relative_velocity)
        )
    ):
        raise ValueError("finite current relative state and declared linear gate horizon required")
    speed_squared = float(relative_velocity @ relative_velocity)
    time = (
        0.0
        if speed_squared <= 1e-12
        else float(np.clip(-(relative_position @ relative_velocity) / speed_squared, 0, seconds))
    )
    distance = float(np.linalg.norm(relative_position + time * relative_velocity))
    if not math.isfinite(distance):
        raise ValueError("finite bounded linear gate estimate required")
    return distance, time


def task_displacement(
    point: np.ndarray,
    ball: np.ndarray,
    point_velocity: np.ndarray,
    ball_velocity: np.ndarray,
    goal: np.ndarray,
    *,
    variant: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Current planar alignment/relative-speed request, not a predicted impact.

    Keep vertical motion with the learned motor. Do not servo longitudinal
    position into the ball: only brake excessive forward relative velocity.
    All returns are fresh arrays and input observations remain unchanged.
    """
    cfg = feedback_parameters(variant)
    if any(
        not isinstance(x, np.ndarray)
        or x.shape != (3,)
        or x.dtype != np.float64
        or not np.isfinite(x).all()
        for x in (point, ball, point_velocity, ball_velocity, goal)
    ):
        raise ValueError("finite measured world-frame float64 three-vectors required")
    direction = goal - ball
    direction[2] = 0
    distance = np.linalg.norm(direction)
    if distance < 0.1:
        raise ValueError("distinct original target required")
    direction /= distance
    lateral = np.array([-direction[1], direction[0], 0.0], dtype=np.float64)
    relative = point_velocity - ball_velocity
    closing = float(relative @ direction)
    lateral_gap = float((ball - point) @ lateral)
    lateral_speed = float(relative @ lateral)
    task = cfg["gain"] * (
        (cfg["lateral_position_gain"] * lateral_gap - cfg["velocity_lead_seconds"] * lateral_speed)
        * lateral
        - cfg["velocity_lead_seconds"]
        * max(0.0, closing - cfg["desired_closing_speed_mps"])
        * direction
    )
    norm = float(np.linalg.norm(task))
    if norm > cfg["maximum_task_displacement_m"]:
        task *= cfg["maximum_task_displacement_m"] / norm
    return task, np.array([closing, lateral_gap, lateral_speed], dtype=np.float64)


def damped_joint_delta(jacobian: np.ndarray, task: np.ndarray) -> np.ndarray:
    """Damped minimum-norm correction in the six measured leg hinge coordinates."""
    if (
        jacobian.shape != (3, 6)
        or task.shape != (3,)
        or jacobian.dtype != np.float64
        or task.dtype != np.float64
        or not np.isfinite(jacobian).all()
        or not np.isfinite(task).all()
    ):
        raise ValueError("finite measured3x6 Jacobian and task delta required")
    cfg = feedback_parameters(1)
    delta = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + cfg["jacobian_damping_m_per_rad"] ** 2 * np.eye(3), task
    )
    return np.asarray(
        np.clip(delta, -cfg["maximum_joint_correction_rad"], cfg["maximum_joint_correction_rad"]),
        dtype=np.float64,
    )


class CausalFootBallFeedback:
    """Scratch-free measured Jacobian adapter for the exact shared two-G1 model."""

    def __init__(
        self,
        world: Any,
        ep: Any,
        *,
        goal: np.ndarray,
        variant: int,
        anticipation_seconds: float = 0.0,
    ) -> None:
        from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES

        self.cfg = feedback_parameters(variant, anticipation_seconds=anticipation_seconds)
        self.anticipation_seconds = anticipation_seconds
        self.world, self.ep = world, ep
        if goal.shape != (3,) or goal.dtype != np.float64 or not np.isfinite(goal).all():
            raise ValueError("finite actual original pass target required")
        self.goal = goal.copy()
        model = world.model
        if (model.nq, model.nv, model.nu) != (79, 76, 58) or model.opt.timestep != 0.002:
            raise ValueError("qualified unchanged2ms two-G1 shared world required")
        self.layout = world.layouts["a/"]
        self.foot = int(model.body("a/left_ankle_roll_link").id)
        self.indices = np.array(
            [G1_DDS_JOINT_NAMES.index(f"left_{s}_joint") for s in LEG_SUFFIXES], dtype=np.int64
        )

    def apply(
        self, proposal: np.ndarray, reference: np.ndarray, *, frame: int, standing: bool
    ) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
        """Return action, HG29 correction, authority clip, 24 or26 diagnostics.

        Diagnostics: point3,ball3,point_velocity3,ball_velocity3,
        closing/lateral_gap/lateral_speed3, task3, gate, joint_norm,
        actual_extra_joint_norm, total_authority_max, task_norm, variant.
        Anticipatory mode appends heuristic closest distance and time, NOT
        measured future contact or a verified forward simulation.
        Exact zero outside the declared window. Extra correction is never
        mislabelled as learned inference or a measured contact force.
        """
        import mujoco

        from rosclaw_soccer.rsi.own_state_motor_assistance import blend_actual_motor

        if type(frame) is not int or not 0 <= frame <= 400 or type(standing) is not bool:
            raise ValueError("explicit actual bounded control and phase required")
        if any(
            x.shape != (29,) or x.dtype != np.float32 or not np.isfinite(x).all()
            for x in (proposal, reference)
        ):
            raise ValueError("finite original policy-unit proposal and nominal required")
        cfg = self.cfg
        delta, diagnostics = np.zeros(29), np.zeros(26 if self.anticipation_seconds else 24)
        if (
            standing
            or not cfg["first_control"] <= frame < cfg["last_control_exclusive"]
            or not cfg["gain"]
        ):
            return proposal.copy(), delta, False, diagnostics
        w, ep = self.world, self.ep
        point = w.data.xpos[self.foot] + w.data.xmat[self.foot].reshape(3, 3) @ np.array(
            cfg["point_body_local_m"]
        )
        ball = w.data.xpos[self.layout.ball].copy()
        jac = np.zeros((3, w.model.nv))
        mujoco.mj_jac(w.model, w.data, jac, None, point, self.foot)
        pv = jac @ w.data.qvel
        bv = w.data.qvel[self.layout.bv : self.layout.bv + 3].copy()
        task, measured = task_displacement(point, ball, pv, bv, self.goal, variant=cfg["variant"])
        phase = min(
            1.0,
            (frame - cfg["first_control"] + 1) / cfg["fade_controls"],
            (cfg["last_control_exclusive"] - frame) / cfg["fade_controls"],
        )
        gate = 0.5 * (1 - math.cos(math.pi * phase))
        separation = float(np.linalg.norm(ball - point))
        time_to_closest = 0.0
        if self.anticipation_seconds:
            separation, time_to_closest = closest_approach(
                ball - point, bv - pv, seconds=self.anticipation_seconds
            )
        if separation >= cfg["near_distance_m"]:
            gate = 0.0
        elif separation > 0.25:
            gate *= (cfg["near_distance_m"] - separation) / (cfg["near_distance_m"] - 0.25)
        leg_delta = damped_joint_delta(jac[:, self.layout.vi[self.indices]], task * gate)
        delta[self.indices] = leg_delta
        oracle = proposal.copy()
        oracle[ep.order] = (oracle[ep.order] + delta / ep.scale).astype(np.float32)
        action, info = blend_actual_motor(
            proposal, oracle, reference, ep.scale[np.argsort(ep.order)], share=1.0
        )
        changed = np.zeros(29, dtype=bool)
        changed[ep.order] = delta != 0
        action[~changed] = proposal[~changed]
        actual_extra = (action[ep.order] - proposal[ep.order]) * ep.scale
        total = (action[ep.order] - reference[ep.order]) * ep.scale
        diagnostics[:24] = np.concatenate(
            [
                point,
                ball,
                pv,
                bv,
                measured,
                task,
                [
                    gate,
                    np.linalg.norm(leg_delta),
                    np.linalg.norm(actual_extra),
                    np.max(np.abs(total)),
                    np.linalg.norm(task),
                    cfg["variant"],
                ],
            ]
        )
        if self.anticipation_seconds:
            diagnostics[24:] = [separation, time_to_closest]
        return action, delta, info["oracle_authority_clipped"], diagnostics
