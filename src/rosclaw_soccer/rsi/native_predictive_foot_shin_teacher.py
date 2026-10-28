"""Measured-geometry adapter for a bounded, unlearned SIM contact teacher."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.ball_surface_tasks import measure_ball_surfaces
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.predictive_foot_shin_target import predictive_foot_shin_target


class NativePredictiveFootShinTeacher:
    """One player's own measured state and private correction history only.

    No simulation step, body/ball state edit, future trajectory, or learned
    weights. Physical contact and balance must be independently verified.
    """

    def __init__(self, *, axis_local_fallback: bool = False) -> None:
        if type(axis_local_fallback) is not bool:
            raise ValueError("explicit axis-local fallback flag required")
        self.axis_local_fallback = axis_local_fallback
        self.next_frame = 0
        self.foot: int | None = None
        self.previous_delta = np.zeros(29, dtype=np.float64)

    def propose(
        self,
        *,
        model: Any,
        snapshot: Any,
        layout: NativeVelocityLayout,
        nominal: np.ndarray,
        enabled: bool,
        frame: int,
    ) -> dict:
        if (
            type(layout) is not NativeVelocityLayout
            or layout.model is not model
            or type(frame) is not int
            or frame != self.next_frame
            or not 0 <= frame < 600
            or type(enabled) is not bool
            or not isinstance(nominal, np.ndarray)
            or nominal.shape != (29,)
            or nominal.dtype != np.float64
            or not np.isfinite(nominal).all()
        ):
            raise ValueError("own consecutive SIM snapshot and finite HG target required")
        surfaces = measure_ball_surfaces(model, snapshot, layout)["surfaces"]
        feet = [surfaces[side + "_foot"] for side in ("left", "right")]
        foot_gaps = [x["gap_m"] if x["available"] else 2.0 for x in feet]
        if self.foot is None and enabled and min(foot_gaps) <= 0.55:
            self.foot = int(np.argmin(foot_gaps))
        foot = self.foot if self.foot is not None else 0
        foot_surface = feet[foot]
        if not foot_surface["available"]:
            result = predictive_foot_shin_target(
                nominal=nominal,
                previous_delta=self.previous_delta,
                joint_ranges=model.jnt_range[layout.joints].copy(),
                foot_jacobian=np.zeros((3, 29)),
                foot_displacement=np.zeros(3),
                shin_gap=np.full(2, 2.0),
                shin_gap_rate=np.zeros(2),
                shin_gap_jacobian=np.zeros((2, 29)),
                foot=foot,
                enabled=False,
                axis_local_fallback=self.axis_local_fallback,
            )
            self.previous_delta = result["applied_delta"].copy()
            self.next_frame += 1
            result.update(
                foot_index=-1 if self.foot is None else self.foot,
                foot_gap=2.0,
                shin_gap=np.full(2, 2.0),
                shin_gap_rate=np.zeros(2),
                nonsmooth_abstention=False,
                geometry_censored=True,
                active=False,
            )
            return result
        shins = [surfaces[side + "_shin"] for side in ("left", "right")]
        # Penetration has no smooth gap derivative; do not invent one.
        nonsmooth = not foot_surface["smooth_separated"] or any(
            s["available"] and not s["smooth_separated"] for s in shins
        )
        active = bool(enabled and self.foot is not None and not nonsmooth)
        ball_velocity = snapshot.qvel[layout.bv : layout.bv + 3].copy()
        direction = foot_surface["normal_toward_ball"] if not nonsmooth else np.zeros(3)
        delta = 0.02 * (
            0.6 * (ball_velocity - foot_surface["point_velocity"])
            + 2.0 * max(0.0, foot_surface["gap_m"]) * direction
        )
        norm = float(np.linalg.norm(delta))
        if norm > 0.08:
            delta *= 0.08 / norm
        gaps = np.array([s["gap_m"] if s["available"] else 2.0 for s in shins], dtype=np.float64)
        rates = np.array(
            [
                float(s["normal_toward_ball"] @ (ball_velocity - s["point_velocity"]))
                if s["available"] and s["smooth_separated"]
                else 0.0
                for s in shins
            ],
            dtype=np.float64,
        )
        gradients = np.stack(
            [
                s["gap_jacobian_hg"] if s["available"] and s["smooth_separated"] else np.zeros(29)
                for s in shins
            ]
        ).astype(np.float64)
        result = predictive_foot_shin_target(
            nominal=nominal,
            previous_delta=self.previous_delta,
            joint_ranges=model.jnt_range[layout.joints].copy(),
            foot_jacobian=foot_surface["jacobian_hg"].copy(),
            foot_displacement=delta,
            shin_gap=gaps,
            shin_gap_rate=rates,
            shin_gap_jacobian=gradients,
            foot=foot,
            enabled=active,
            axis_local_fallback=self.axis_local_fallback,
        )
        self.previous_delta = result["applied_delta"].copy()
        self.next_frame += 1
        result.update(
            foot_index=-1 if self.foot is None else self.foot,
            foot_gap=float(foot_surface["gap_m"]),
            shin_gap=gaps,
            shin_gap_rate=rates,
            nonsmooth_abstention=nonsmooth,
            geometry_censored=False,
            active=active,
        )
        return result
