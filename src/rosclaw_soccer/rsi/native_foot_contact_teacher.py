"""Privileged SIM contact-authority probe over an unchanged frozen gait.

Only a bounded HG29 joint-target correction is proposed. This is neither a
trained policy nor a transport, simulator step, external ball force or motion
authorization. The base gait retains its own previous-network-action memory;
the caller must separately record nominal and actually applied joint targets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FootContactTeacherConfig:
    strength: float = 1.0
    velocity_match_gain: float = 1.0
    approach_gain_per_sec: float = 2.0
    activation_gap_m: float = 0.35
    maximum_delta_rad: float = 0.08
    maximum_slew_rad_per_control: float = 0.015
    damping_m_per_rad: float = 0.04

    def __post_init__(self) -> None:
        if (
            any(type(v) not in (int, float) or not np.isfinite(v) for v in asdict(self).values())
            or not 0 <= self.strength <= 1
            or not 0 <= self.velocity_match_gain <= 1
            or not 0 <= self.approach_gain_per_sec <= 4
            or not 0.05 <= self.activation_gap_m <= 0.5
            or not 0 < self.maximum_delta_rad <= 0.12
            or not 0 < self.maximum_slew_rad_per_control <= 0.03
            or not 0.01 <= self.damping_m_per_rad <= 0.1
        ):
            raise ValueError("finite bounded preregistered SIM teacher parameters required")


def bounded_contact_target(
    *,
    nominal: np.ndarray,
    previous_delta: np.ndarray,
    joint_ranges: np.ndarray,
    jacobian_hg: np.ndarray,
    relative_velocity: np.ndarray,
    direction_to_ball: np.ndarray,
    surface_gap: float,
    foot: int,
    enabled: bool,
    config: FootContactTeacherConfig,
) -> dict[str, Any]:
    """Damped least-squares foot displacement with explicit target authority.

    ``relative_velocity`` is ball translation minus actual foot point velocity,
    in world coordinates. Jacobian columns use the canonical HG29 dof order.
    The selected leg's six joints alone may receive a correction. Any infeasible
    target/authority/slew intersection explicitly falls back to the parent.
    """
    if (
        type(config) is not FootContactTeacherConfig
        or type(enabled) is not bool
        or type(foot) is not int
        or foot not in (0, 1)
        or type(surface_gap) not in (int, float)
        or not np.isfinite(surface_gap)
        or not -1 <= surface_gap <= 2
        or any(
            not isinstance(a, np.ndarray) or a.shape != shape or not np.isfinite(a).all()
            for a, shape in (
                (nominal, (29,)),
                (previous_delta, (29,)),
                (joint_ranges, (29, 2)),
                (jacobian_hg, (3, 29)),
                (relative_velocity, (3,)),
                (direction_to_ball, (3,)),
            )
        )
        or nominal.dtype != np.float64
        or (joint_ranges[:, 0] >= joint_ranges[:, 1]).any()
        or np.any(abs(previous_delta) > config.maximum_delta_rad + 1e-12)
        or np.any(previous_delta[np.r_[12:29]] != 0)
        or np.linalg.norm(direction_to_ball) > 1 + 1e-10
    ):
        raise ValueError("finite aligned measured foot context and own HG29 parent required")
    active = enabled and surface_gap <= config.activation_gap_m and config.strength > 0
    leg = slice(6 * foot, 6 * foot + 6)
    # An episode latches one foot. Refuse an accidental cross-leg history.
    other = slice(6 * (1 - foot), 6 * (1 - foot) + 6)
    if np.any(previous_delta[other]):
        raise ValueError("single latched foot's private correction history required")
    requested = np.zeros(29, np.float64)
    if active:
        desired = (
            0.02
            * config.strength
            * (
                config.velocity_match_gain * relative_velocity
                + config.approach_gain_per_sec * max(0.0, surface_gap) * direction_to_ball
            )
        )
        # A single local linearization is deliberately limited to 2cm. This
        # limits the probe, not a claim that larger motions are unreachable.
        norm = np.linalg.norm(desired)
        if norm > 0.02:
            desired *= 0.02 / norm
        j = jacobian_hg[:, leg]
        requested[leg] = j.T @ np.linalg.solve(
            j @ j.T + config.damping_m_per_rad**2 * np.eye(3), desired
        )
    if not np.isfinite(requested).all():
        raise ValueError("nonfinite DLS correction; no physical retry")
    delta = np.zeros(29, np.float64)
    fallback = False
    if np.any(requested) or np.any(previous_delta):
        lower = np.maximum.reduce(
            (
                np.full(6, -config.maximum_delta_rad),
                previous_delta[leg] - config.maximum_slew_rad_per_control,
                joint_ranges[leg, 0] - nominal[leg],
            )
        )
        upper = np.minimum.reduce(
            (
                np.full(6, config.maximum_delta_rad),
                previous_delta[leg] + config.maximum_slew_rad_per_control,
                joint_ranges[leg, 1] - nominal[leg],
            )
        )
        fallback = bool(np.any(lower > upper))
        if not fallback:
            delta[leg] = np.clip(requested[leg], lower, upper)
    target = nominal.copy() if not np.any(delta) else nominal + delta
    # Actual difference includes the final float64 addition's rounding.
    applied = target - nominal
    return dict(
        target=target,
        requested_delta=requested,
        applied_delta=applied,
        active=active,
        fallback=fallback,
        estimated_local_foot_shift=jacobian_hg @ applied,
    )


class NativeFootContactTeacher:
    """One SIM player's privileged geometry and private latched-foot memory."""

    def __init__(self, config: FootContactTeacherConfig) -> None:
        if type(config) is not FootContactTeacherConfig:
            raise ValueError("explicit bounded teacher configuration required")
        self.config = config
        self.previous_delta = np.zeros(29, np.float64)
        self.foot: int | None = None
        self.next_frame = 0

    def propose(
        self,
        *,
        model: Any,
        snapshot: Any,
        layout: Any,
        nominal: np.ndarray,
        ball_geom: int,
        foot_bodies: list[int],
        enabled: bool,
        frame: int,
    ) -> dict[str, Any]:
        import mujoco

        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

        if (
            type(frame) is not int
            or frame != self.next_frame
            or not 0 <= frame < 600
            or type(layout) is not NativeVelocityLayout
            or layout.model is not model
            or snapshot.qpos.shape != (model.nq,)
            or snapshot.qvel.shape != (model.nv,)
        ):
            raise ValueError("own consecutive bounded snapshot and canonical SIM layout required")

        if len(foot_bodies) != 2 or any(
            model.body(body).name != layout.robot_prefix + side + "_ankle_roll_link"
            for side, body in zip(("left", "right"), foot_bodies, strict=True)
        ):
            raise ValueError("own actual left/right ankle bodies required")
        if int(model.geom_bodyid[ball_geom]) != layout.ball:
            raise ValueError("actual common soccer ball geom required")
        measured = []
        for body in foot_bodies:
            geoms = np.flatnonzero(
                (model.geom_bodyid == body)
                & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
            )
            if len(geoms) == 0:
                raise ValueError("own physical foot collision geometry required")
            candidates = []
            for geom in geoms:
                segment = np.zeros(6, np.float64)
                gap = float(mujoco.mj_geomDistance(model, snapshot, geom, ball_geom, 1.0, segment))
                candidates.append((gap, segment[:3].copy()))
            measured.append(min(candidates, key=lambda x: x[0]))
        if (
            self.foot is None
            and enabled
            and min(x[0] for x in measured) <= self.config.activation_gap_m
        ):
            self.foot = int(np.argmin([x[0] for x in measured]))
        foot = 0 if self.foot is None else self.foot
        gap, point = measured[foot]
        # Distance cutoff returns no segment. Retain a real current foot point
        # for inactive diagnostics rather than a fabricated origin contact.
        if gap >= 1.0:
            point = snapshot.xpos[foot_bodies[foot]].copy()
        jac = np.zeros((3, model.nv), np.float64)
        mujoco.mj_jac(model, snapshot, jac, None, point, foot_bodies[foot])
        velocity = jac @ snapshot.qvel
        direction = snapshot.qpos[layout.bq : layout.bq + 3] - point
        length = np.linalg.norm(direction)
        direction = direction / length if length > 1e-12 else np.zeros(3)
        result = bounded_contact_target(
            nominal=nominal,
            previous_delta=self.previous_delta,
            joint_ranges=model.jnt_range[layout.joints].copy(),
            jacobian_hg=jac[:, layout.vi].copy(),
            relative_velocity=snapshot.qvel[layout.bv : layout.bv + 3] - velocity,
            direction_to_ball=direction,
            surface_gap=gap,
            foot=foot,
            enabled=enabled and self.foot is not None,
            config=self.config,
        )
        self.previous_delta = result["applied_delta"].copy()
        self.next_frame += 1
        result.update(
            foot_index=-1 if self.foot is None else self.foot,
            surface_gap=gap,
            gap_censored_at_one_meter=gap >= 1.0,
            point=point,
            point_local=snapshot.xmat[foot_bodies[foot]].reshape(3, 3).T
            @ (point - snapshot.xpos[foot_bodies[foot]]),
            foot_velocity=velocity,
            ball_velocity=snapshot.qvel[layout.bv : layout.bv + 3].copy(),
            jacobian_hg=jac[:, layout.vi].copy(),
        )
        return result
