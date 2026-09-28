"""Independent SIM_ONLY audit of full-episode late-swing actor execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.contact_time_phase_features import (
    current_context,
    gait_phase_features,
    predict_contact_time,
)
from rosclaw_soccer.rsi.late_swing_memory import load_late_swing_actor
from rosclaw_soccer.rsi.taskspace_gate_memory import select_taskspace_gate
from rosclaw_soccer.rsi.taskspace_swing_evidence import audit_taskspace_swing_trace
from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_bytes


def audit_full_episode_late_swing(
    folder: Path, actor_path: Path, parent_folder: Path
) -> dict[str, Any]:
    """Recompute frame-30 gate and every bounded foot target from recorded physics."""
    audit = audit_vector_first_touch(folder)
    parent_audit = audit_vector_first_touch(parent_folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    parent = json.loads((parent_folder / "report.json").read_text(encoding="utf-8"))
    actor = load_late_swing_actor(actor_path)
    trace_path = folder / "late_swing_action_trace.npz"
    if (
        report.get("schema") != "rsi_isaac_vector_first_touch_late_swing_v1"
        or parent.get("schema") != "rsi_isaac_vector_first_touch_smoke_v1"
        or report.get("parent_report_hash") != parent_audit["source_report_hash"]
        or report.get("late_swing_actor_hash") != actor["actor_hash"]
        or report.get("late_swing_action_trace_hash") != hash_bytes(trace_path.read_bytes())
        or report.get("training_course_seed") != parent.get("training_course_seed")
        or report.get("course_catalog_hash") != parent.get("course_catalog_hash")
        or report.get("asset_hash") != parent.get("asset_hash")
        or report.get("sonic_qualification_hash") != parent.get("sonic_qualification_hash")
        or report.get("frames") != 300
        or len(report.get("environments", [])) != 16
        or report.get("torch_batch_plan_only") is not True
        or parent.get("torch_batch_plan_only") is not True
    ):
        raise ValueError("unbound full-episode actor or Parent")
    courses = [row["course"] for row in report["environments"]]
    if courses != [row["course"] for row in parent["environments"]]:
        raise ValueError("full-episode courses differ from Parent")
    with (
        np.load(folder / "body_trace.npz", allow_pickle=False) as body,
        np.load(trace_path, allow_pickle=False) as trace,
    ):
        expected_keys = {
            "pre_step_foot_link_position_w",
            "pre_step_foot_linear_jacobian_w",
            "taskspace_selected_side",
            "applied_taskspace_joint_delta_rad",
            "baseline_taskspace_joint_target_rad",
            "executed_taskspace_joint_target_rad",
            "taskspace_joint_limits_rad",
            "pre_step_ball_position_local_m",
            "observed_ball_body_contact_force_peak_n",
            "predicted_baseline_joint_target_rad",
            "frame30_gate_features",
        }
        if set(trace.files) != expected_keys:
            raise ValueError("full-episode action trace contract changed")
        action_audit = audit_taskspace_swing_trace(trace, report, frames=300, count=16)
        lanes = np.arange(16) * 8.0
        root = body["root_pose_xyzw_m"][30].copy()
        ball = body["ball_position_before_step_m"][30].copy()
        root[:, 1] -= lanes
        ball[:, 1] -= lanes
        raw = current_context(
            root,
            body["root_velocity_world"][30],
            ball,
            body["ball_linear_velocity_before_step_m_s"][30],
        )
        features = gait_phase_features(
            raw, predict_contact_time(raw, np.asarray(actor["contact_time_weights"]))
        )
        mask = select_taskspace_gate(
            features,
            np.asarray(actor["memory_features"]),
            np.asarray(actor["memory_clean"]),
            np.asarray(actor["memory_reward"]),
            np.asarray(actor["memory_groups"]),
            neighbors=actor["neighbors"],
            confidence=actor["confidence"],
            baseline_clean_ceiling=actor["baseline_clean_ceiling"],
        )
        mask &= np.asarray([course["ball_vx_m_s"] < 0 for course in courses])
        if (
            not np.allclose(features, trace["frame30_gate_features"], atol=1e-10, rtol=0)
            or mask.tolist() != report["selected_taskspace_mask"]
            or not np.allclose(
                trace["executed_taskspace_joint_target_rad"][
                    :,
                    :,
                    [report["taskspace_joint_order"].index(name) for name in G1_DDS_JOINT_NAMES],
                ],
                body["joint_target_rad"],
                atol=2e-5,
                rtol=0,
            )
        ):
            raise ValueError("frame-30 gate or executed SONIC target differs from evidence")
    return {
        "schema": "rsi_full_episode_late_swing_audit_v1",
        "report_hash": audit["source_report_hash"],
        "parent_report_hash": parent["report_hash"],
        "late_swing_actor_hash": actor["actor_hash"],
        "selected_incoming_lanes": int(np.count_nonzero(mask)),
        **action_audit,
    }
