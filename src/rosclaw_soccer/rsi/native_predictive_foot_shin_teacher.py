"""Measured-geometry adapter for a bounded, unlearned SIM contact teacher."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.ball_surface_tasks import measure_ball_surfaces
from rosclaw_soccer.rsi.early_contact_intercept import early_contact_intercept
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.predictive_foot_shin_target import predictive_foot_shin_target


class NativePredictiveFootShinTeacher:
    """One player's own measured state and private correction history only.

    No simulation step, body/ball state edit, future trajectory, or learned
    weights. Physical contact and balance must be independently verified.
    """

    def __init__(
        self,
        *,
        axis_local_fallback: bool = False,
        early_intercept: bool = False,
        contact_lead: bool = False,
        all_knee_colliders: bool = False,
        max_delta_rad: float = 0.16,
        joint_margin_rad: float = 0.0,
        parent_relative_envelope: bool = False,
    ) -> None:
        if (
            type(axis_local_fallback) is not bool
            or type(early_intercept) is not bool
            or type(contact_lead) is not bool
            or type(all_knee_colliders) is not bool
            or type(parent_relative_envelope) is not bool
            or type(max_delta_rad) is not float
            or not 0 < max_delta_rad <= 0.2
            or type(joint_margin_rad) is not float
            or not 0 <= joint_margin_rad <= 0.2
            or (contact_lead and not early_intercept)
        ):
            raise ValueError("explicit axis-local fallback and early-intercept flags required")
        self.axis_local_fallback = axis_local_fallback
        self.early_intercept = early_intercept
        self.contact_lead = contact_lead
        self.all_knee_colliders = all_knee_colliders
        self.max_delta_rad = max_delta_rad
        self.joint_margin_rad = joint_margin_rad
        self.parent_relative_envelope = parent_relative_envelope
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
        surfaces = measure_ball_surfaces(
            model, snapshot, layout, all_knee_colliders=self.all_knee_colliders
        )["surfaces"]
        feet = [surfaces[side + "_foot"] for side in ("left", "right")]
        foot_gaps = [x["gap_m"] if x["available"] else 2.0 for x in feet]
        candidates = [None, None]
        if self.early_intercept and enabled:
            ball_pos = snapshot.qpos[layout.bq : layout.bq + 3]
            ball_vel = snapshot.qvel[layout.bv : layout.bv + 3]
            for index, surface in enumerate(feet):
                if surface["available"] and surface["smooth_separated"]:
                    candidates[index] = early_contact_intercept(
                        np.asarray(ball_pos - surface["world_point"], dtype=np.float64),
                        np.asarray(ball_vel - surface["point_velocity"], dtype=np.float64),
                        contact_lead_m=0.1 if self.contact_lead else 0.0,
                    )
        eligible = [i for i, item in enumerate(candidates) if item is not None and item["eligible"]]
        if self.foot is None and eligible:
            self.foot = min(
                eligible, key=lambda i: (candidates[i]["miss_m"], candidates[i]["eta_s"])
            )
        if self.foot is None and enabled and min(foot_gaps) <= 0.55:
            self.foot = int(np.argmin(foot_gaps))
        foot = self.foot if self.foot is not None else 0
        ranges = model.jnt_range[layout.joints].copy()
        ranges[:, 0] += self.joint_margin_rad
        ranges[:, 1] -= self.joint_margin_rad
        prediction = candidates[foot]
        early_active = bool(prediction is not None and prediction["eligible"])
        foot_surface = feet[foot]
        if not foot_surface["available"]:
            result = predictive_foot_shin_target(
                nominal=nominal,
                previous_delta=self.previous_delta,
                joint_ranges=ranges,
                foot_jacobian=np.zeros((3, 29)),
                foot_displacement=np.zeros(3),
                shin_gap=np.full(2, 2.0),
                shin_gap_rate=np.zeros(2),
                shin_gap_jacobian=np.zeros((2, 29)),
                foot=foot,
                enabled=False,
                max_delta_rad=self.max_delta_rad,
                axis_local_fallback=self.axis_local_fallback,
                parent_relative_envelope=self.parent_relative_envelope,
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
                early_active=early_active,
                early_eta_s=prediction["eta_s"] if prediction else 0.0,
                early_miss_m=prediction["miss_m"] if prediction else 2.0,
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
        if early_active:
            # Aim at a causal constant-velocity closest point; cap the actual
            # displacement below, and judge only from subsequent physics.
            delta = np.asarray(prediction["displacement"], dtype=np.float64).copy()
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
            joint_ranges=ranges,
            foot_jacobian=foot_surface["jacobian_hg"].copy(),
            foot_displacement=delta,
            shin_gap=gaps,
            shin_gap_rate=rates,
            shin_gap_jacobian=gradients,
            foot=foot,
            enabled=active,
            max_delta_rad=self.max_delta_rad,
            axis_local_fallback=self.axis_local_fallback,
            parent_relative_envelope=self.parent_relative_envelope,
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
            early_active=early_active,
            early_eta_s=prediction["eta_s"] if prediction else 0.0,
            early_miss_m=prediction["miss_m"] if prediction else 2.0,
        )
        return result
