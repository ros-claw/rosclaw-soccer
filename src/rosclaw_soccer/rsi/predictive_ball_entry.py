"""Bounded privileged entry-command MPC using own-neural native previews.

Not a trained actor or a contact-skill predictor. Only the navigation prefix
before first predicted skill readiness is scored; its later navigation-only
continuation is not confused with execution of the contact reference.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
from rosclaw_soccer.physics.sonic_navigation_forecast import (
    SonicNavigationForecast,
    SonicNavigationForecaster,
)
from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation
from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance, ball_entry_decision


def entry_candidates(
    q: np.ndarray, v: np.ndarray, *, frame: int, affordance: SkillEntryAffordance
) -> list[tuple[str, tuple[float, float, float]]]:
    nominal = ball_entry_decision(q, v, frame=frame, affordance=affordance).command
    velocity = np.asarray(nominal[:2])

    def bounded(xy: np.ndarray, yaw: float) -> tuple[float, float, float]:
        xy = xy * min(1.0, 0.6 / max(float(np.linalg.norm(xy)), 1e-12))
        return float(xy[0]), float(xy[1]), float(yaw)

    return [
        ("nominal", nominal),
        ("half", bounded(0.5 * velocity, nominal[2])),
        ("zero", (0.0, 0.0, 0.0)),
        ("damped", bounded(velocity - 0.75 * v[:2], nominal[2])),
        ("strong_damped", bounded(0.75 * velocity - 1.5 * v[:2], nominal[2])),
        ("brake", bounded(-v[:2], nominal[2])),
    ]


def score_entry_forecast(
    model: Any, forecast: SonicNavigationForecast, *, frame: int, affordance: SkillEntryAffordance
) -> dict[str, Any]:
    arrays = forecast.arrays
    q = np.vstack(
        (arrays["canonical_qpos"], arrays["target_body_canonical_qpos_after_substep"][-1, -1])
    )
    v = np.vstack(
        (arrays["canonical_qvel"], arrays["target_body_canonical_qvel_after_substep"][-1, -1])
    )
    if (
        q.shape != (41, 43)
        or v.shape != (41, 41)
        or not np.isfinite(q).all()
        or not np.isfinite(v).all()
    ):
        raise ValueError("whole40-frame own-physics forecast boundaries required")
    decisions = [
        ball_entry_decision(a, b, frame=frame + i, affordance=affordance)
        for i, (a, b) in enumerate(zip(q, v, strict=True))
    ]
    ready = [i for i, d in enumerate(decisions) if d.ready]
    prefix = ready[0] if ready else 40
    root = model.body_rootid[model.body("pelvis").id]
    feet = [
        model.body(side + suffix).id
        for side in ("left", "right")
        for suffix in ("_ankle_roll_link", "_ankle_pitch_link")
    ]
    nonfeet = np.flatnonzero((model.body_rootid == root) & ~np.isin(np.arange(model.nbody), feet))
    if root == 0 or not len(nonfeet):
        raise ValueError("complete robot nonfoot-body classification required")
    forces = arrays["ball_contact_force_by_body_n"]
    if forces.shape != (40, 10, model.nbody) or not np.isfinite(forces).all():
        raise ValueError("full native per-body ball-contact capture required")
    peak_nonfoot = float(forces[:prefix, :, nonfeet].max()) if prefix else 0.0
    heights = arrays["pelvis_z_per_substep_m"][:prefix]
    lowest = float(heights.min()) if prefix else float(q[0, 2])
    rotations = arrays["target_body_canonical_qpos_after_substep"][:prefix, :, 3:7]
    upright = (
        float((1 - 2 * (rotations[:, :, 1] ** 2 + rotations[:, :, 2] ** 2)).min())
        if prefix
        else 1.0
    )
    safe = peak_nonfoot <= 1 and lowest >= 0.65 and upright >= 0.9
    distances = [d.horizontal_error_m for d in decisions]
    speed2 = float(np.dot(v[prefix, :2], v[prefix, :2]))
    cost = (
        0.01 * prefix / 40 + 0.02 * speed2
        if ready
        else 1 + 10 * min(distances) ** 2 + 5 * distances[-1] ** 2 + 0.02 * speed2
    )
    return dict(
        predicted_safe=bool(safe),
        cost=float(cost) if safe else None,
        first_ready_offset=ready[0] if ready else None,
        evaluated_navigation_prefix_frames=prefix,
        maximum_prefix_nonfoot_force_n=peak_nonfoot,
        minimum_prefix_pelvis_z_m=lowest,
        minimum_prefix_upright=upright,
        minimum_entry_error_m=min(distances),
        end_entry_error_m=distances[-1],
        root_speed_at_evaluation_mps=float(np.sqrt(speed2)),
        forecast_after_readiness_is_not_contact_skill_prediction=True,
    )


class PredictiveBallEntry:
    """At most4x6 native previews; caller retains the actual physics owner."""

    def __init__(
        self,
        *,
        world_hash: str,
        recorder: Callable[[int, int, SonicNavigationForecast, dict[str, Any]], None] | None = None,
    ) -> None:
        self.forecaster = SonicNavigationForecaster(world_hash=world_hash, max_native_steps=9600)
        self.recorder = recorder
        self.last_frame = -1

    def __call__(
        self, nav: G1SonicNavigation, body: G1TargetEpisode, affordance: SkillEntryAffordance
    ) -> tuple[tuple[float, float, float], dict[str, Any]]:
        frame = body.next_frame
        if frame not in (20, 40, 60, 80) or frame <= self.last_frame:
            raise ValueError("one bounded planning decision per declared navigation boundary")
        q, v = body.canonical()
        if ball_entry_decision(q, v, frame=frame, affordance=affordance).ready:
            raise ValueError("actual ready state must hand off before command search")
        candidates = entry_candidates(q, v, frame=frame, affordance=affordance)
        before = self.forecaster.integrated_native_steps
        rows = []
        for index, (name, candidate) in enumerate(candidates):

            def command(
                bq: np.ndarray,
                bv: np.ndarray,
                tick: int,
                held: tuple[float, float, float] = candidate,
            ) -> tuple[float, float, float]:
                if tick > 90:
                    return 0.0, 0.0, 0.0
                return (
                    held
                    if tick < frame + 20
                    else ball_entry_decision(bq, bv, frame=tick, affordance=affordance).command
                )

            forecast = self.forecaster.forecast(nav, body, command)
            score = score_entry_forecast(body.model, forecast, frame=frame, affordance=affordance)
            row = dict(
                index=index,
                name=name,
                command=list(candidate),
                **score,
                new_native_steps=forecast.new_integrated_native_steps,
                donor_body_hash=forecast.donor_integration_hash,
                donor_navigation_hash=forecast.donor_navigation_hash,
            )
            rows.append(row)
            if self.recorder is not None:
                self.recorder(frame, index, forecast, row)
        safe = [i for i, row in enumerate(rows) if row["predicted_safe"]]
        selected = min(safe, key=lambda i: rows[i]["cost"]) if safe else 2
        self.last_frame = frame
        return candidates[selected][1], dict(
            schema="soccer.rsi.predictive_ball_entry_decision.v1",
            frame=frame,
            candidates=rows,
            selected_index=selected,
            selected_name=rows[selected]["name"],
            all_candidates_predicted_unsafe=not bool(safe),
            selection_predicted_safe=bool(safe),
            fallback_if_all_unsafe="explicit_zero_command_NOT_safety_certificate",
            new_forecast_native_steps=self.forecaster.integrated_native_steps - before,
            cumulative_forecast_native_steps=self.forecaster.integrated_native_steps,
            candidate_law="held_to_next20_or_deadline90_then_greedy_own_state",
            future_physics_available_to_teacher_only=True,
            optimizer_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        )
