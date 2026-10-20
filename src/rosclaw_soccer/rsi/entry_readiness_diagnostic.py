"""Recompute initiation predicates from historical measured states, zero dynamics."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance, ball_entry_decision
from rosclaw_soccer.rsi.full_entry_forecast import feedback_entry_laws


def diagnose_readiness(
    raw: dict[str, np.ndarray],
    entry: SkillEntryAffordance,
    *,
    law_index: int,
    expected_handoff_frame: int | None = None,
) -> dict[str, Any]:
    if type(law_index) is not int or law_index not in (0, 3, 4):
        raise ValueError("one fixed declared entry law required")
    q, v = raw["canonical_qpos"], raw["canonical_qvel"]
    if (
        q.shape != (300, 43)
        or v.shape != (300, 41)
        or q.dtype != np.float64
        or v.dtype != np.float64
        or not np.isfinite(q).all()
        or not np.isfinite(v).all()
    ):
        raise ValueError("complete finite measured H300 states required")
    rows = []
    handoff = np.flatnonzero(raw["entry_stage"] == 1)
    if (int(handoff[0]) if len(handoff) else None) != expected_handoff_frame or (
        len(handoff) and not np.all(raw["entry_stage"][handoff[0] :] == 1)
    ):
        raise ValueError("recorded stage must agree with independently bound handoff report")
    stop = int(handoff[0]) if len(handoff) else 300
    for frame in range(stop):
        decision = ball_entry_decision(q[frame], v[frame], frame=frame, affordance=entry)
        command = (
            decision.command
            if frame < 20
            else feedback_entry_laws()[law_index].command(
                q[frame],
                v[frame],
                frame=frame,
                affordance=entry,
            )
        )
        if (
            not np.array_equal(command, raw["entry_command"][frame])
            or not np.array_equal(decision.target_root_xy_m, raw["entry_target_root_xy_m"][frame])
            or decision.horizontal_error_m != raw["entry_error_m"][frame]
            or raw["entry_stage"][frame] != (2 if frame > 90 else 0)
            or decision.ready
        ):
            raise ValueError("native decision/command/stage differs before source handoff")
        if 30 <= frame <= 90:
            w, x, y, z = q[frame, 3:7]
            yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            rows.append(
                dict(
                    frame=frame,
                    distance_m=decision.horizontal_error_m,
                    yaw_rad=yaw,
                    upright=float(1 - 2 * (x * x + y * y)),
                    pelvis_z_m=float(q[frame, 2]),
                    root_velocity_xy_mps=v[frame, :2].tolist(),
                )
            )
    if not rows:
        raise ValueError("actual navigation readiness-window observations required")
    closest = min(rows, key=lambda r: r["distance_m"])
    return dict(
        source_handoff_frame=int(handoff[0]) if len(handoff) else None,
        compared_navigation_frames=stop,
        readiness_window_frames=len(rows),
        closest=closest,
        distance_predicate_pass_frames=sum(r["distance_m"] <= 0.04 for r in rows),
        yaw_predicate_pass_frames=sum(abs(r["yaw_rad"]) <= 0.1 for r in rows),
        height_predicate_pass_frames=sum(r["pelvis_z_m"] >= 0.65 for r in rows),
        upright_predicate_pass_frames=sum(r["upright"] >= 0.9 for r in rows),
        distance_threshold_m=0.04,
        readiness_deadline_frame=90,
        failure_stage="navigation_before_reference_handoff"
        if not len(handoff)
        else "after_handoff_not_diagnosed_here",
        new_native_steps=0,
        neural_training_updates=0,
    )
