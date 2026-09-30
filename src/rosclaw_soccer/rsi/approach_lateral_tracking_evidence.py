"""Audit the measured, SIM_ONLY approach command before first ball contact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.conservative_approach_rectangle import ApproachRectangle
from rosclaw_soccer.rsi.precontact_proprio_policy import (
    proprio_vector,
    risk_probability,
    validate_policy,
)
from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_json

SEALED_POLICY_HASH = "sha256:12b3f8fea5aec4db3cb38aa7e289a03bc3fd4d3e1d31b50ed0b70f1ae297922a"
SEALED_X_MAX_M = 2.5524194955825807
SEALED_Y_MIN_M = -0.09402785405516624


def audit_lateral_approach(folder: Path) -> dict[str, Any]:
    physical = audit_vector_first_touch(folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    gain = report.get("navigation_lateral_ball_gain")
    negative_only = report.get("navigation_lateral_negative_only", False)
    rectangle_hash = report.get("navigation_rectangle_policy_hash")
    proprio_hash = report.get("navigation_proprio_risk_policy_hash")
    early_switch = report.get("navigation_lateral_early_switch", False)
    if rectangle_hash is not None and (
        rectangle_hash != SEALED_POLICY_HASH
        or report.get("navigation_rectangle_x_max_m") != SEALED_X_MAX_M
        or report.get("navigation_rectangle_y_min_m") != SEALED_Y_MIN_M
        or gain != 0.8
        or negative_only
    ):
        raise ValueError("unsealed learned approach policy in physical report")
    if (
        gain not in (0.0, 0.8, 1.0, 1.2)
        or type(negative_only) is not bool
        or (negative_only and gain == 0.0)
        or report.get("frames") != 300
        or len(report["environments"]) != 1
    ):
        raise ValueError("single-instance bounded approach contract required")
    if proprio_hash is not None and (
        gain != 1.2
        or negative_only is not True
        or rectangle_hash is not None
        or report.get("navigation_proprio_risk_policy") is None
    ):
        raise ValueError("invalid precontact proprioceptive approach contract")
    if type(early_switch) is not bool or (
        early_switch
        and (
            gain != 1.2
            or not negative_only
            or rectangle_hash is not None
            or proprio_hash is not None
        )
    ):
        raise ValueError("invalid early-switch approach diagnostic")
    if proprio_hash is None and any(
        key in report
        for key in (
            "navigation_proprio_risk_policy",
            "navigation_proprio_risk_probability",
            "navigation_proprio_risk_vetoed",
        )
    ):
        raise ValueError("unbound precontact proprioceptive decision")
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
        if rectangle_hash is not None:
            rectangle = ApproachRectangle(SEALED_X_MAX_M, SEALED_Y_MIN_M, 0, 0.0)
            effective_gain = gain if rectangle.choose(float(gap[0]), float(lateral[0])) else 0.0
        if early_switch:
            effective_gain = np.full(300, effective_gain, dtype=np.float64)
            effective_gain[10:] = 0.8 if lateral[0] < 0 else 0.0
        if proprio_hash is not None:
            policy, policy_hash = validate_policy(report["navigation_proprio_risk_policy"])
            if policy_hash != proprio_hash or (first is not None and first <= 30):
                raise ValueError("unsealed or noncausal proprioceptive decision")
            features = proprio_vector(
                root[30, 0],
                trace["root_velocity_world"][30, 0],
                ball[20, 0],
                ball[30, 0],
                trace["ball_linear_velocity_before_step_m_s"][30, 0],
                trace["foot_geometry_position_before_step_m"][20, 0],
                trace["foot_geometry_position_before_step_m"][30, 0],
            )
            probability = risk_probability(policy, features)
            veto = probability >= policy["threshold"]
            recorded_probability = report.get("navigation_proprio_risk_probability")
            recorded_veto = report.get("navigation_proprio_risk_vetoed")
            if (
                not isinstance(recorded_probability, list)
                or len(recorded_probability) != 1
                or type(recorded_probability[0]) not in (int, float)
                or not np.isfinite(recorded_probability[0])
                or abs(recorded_probability[0] - probability) > 1e-7
                or recorded_veto != [veto]
            ):
                raise ValueError("precontact proprioceptive decision diverged from body state")
            effective_gain = np.full(300, effective_gain, dtype=np.float64)
            if veto:
                effective_gain[30:] = policy["fallback_gain"] if lateral[0] < 0 else 0.0
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
        "navigation_rectangle_policy_hash": rectangle_hash,
        "active_frames": active_frames,
        "maximum_abs_command_mps": maximum_command_mps,
        "maximum_reconstruction_error_mps": max_error,
        "promotion_authorized": False,
    }
    if proprio_hash is not None:
        result["navigation_proprio_risk_policy_hash"] = proprio_hash
    if early_switch:
        result["navigation_lateral_early_switch"] = True
    result["report_hash"] = hash_json(result)
    return result
