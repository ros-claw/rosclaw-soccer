"""Audit the measured, SIM_ONLY approach command before first ball contact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_json


def audit_lateral_approach(folder: Path) -> dict[str, Any]:
    physical = audit_vector_first_touch(folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    gain = report.get("navigation_lateral_ball_gain")
    negative_only = report.get("navigation_lateral_negative_only", False)
    if (
        gain not in (0.0, 0.8)
        or type(negative_only) is not bool
        or (negative_only and gain != 0.8)
        or report.get("frames") != 300
        or len(report["environments"]) != 1
    ):
        raise ValueError("single-instance bounded approach contract required")
    first = report["environments"][0]["first_contact_frame"]
    if first is not None and (type(first) is not int or not 0 <= first < 300):
        raise ValueError("invalid first-contact frame")
    with np.load(folder / "body_trace.npz", allow_pickle=False) as trace:
        root = trace["root_pose_xyzw_m"]
        ball = trace["ball_position_before_step_m"]
        command = trace["navigation_lateral_speed_mps"]
        if (
            root.shape != (300, 1, 7)
            or ball.shape != (300, 1, 3)
            or command.shape != (300, 1)
            or not all(np.isfinite(item).all() for item in (root, ball, command))
        ):
            raise ValueError("finite aligned body/ball/command trace required")
        gap = ball[:, 0, 0] - root[:, 0, 0]
        lateral = ball[:, 0, 1] - root[:, 0, 1]
        effective_gain = gain if not negative_only or lateral[0] < 0 else 0.0
        before_contact = np.arange(300) <= (299 if first is None else first)
        expected = np.where(
            before_contact & (gap > 0.95), np.clip(effective_gain * lateral, -0.2, 0.2), 0.0
        )
        max_error = float(np.max(np.abs(command[:, 0] - expected)))
        if max_error > 1e-7:
            raise ValueError("lateral approach command diverged from bounded measured feedback")
        active_frames = int(np.count_nonzero(np.abs(command[:, 0]) > 1e-8))
        maximum_command_mps = float(np.max(np.abs(command[:, 0])))
    result: dict[str, Any] = {
        "schema": "rsi_isaac_lateral_approach_command_audit_v1",
        "activation_ceiling": "SIM_ONLY",
        "source_report_hash": report["report_hash"],
        "physical_audit_hash": physical["report_hash"],
        "navigation_lateral_ball_gain": gain,
        "navigation_lateral_negative_only": negative_only,
        "active_frames": active_frames,
        "maximum_abs_command_mps": maximum_command_mps,
        "maximum_reconstruction_error_mps": max_error,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result
