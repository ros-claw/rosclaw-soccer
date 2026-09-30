"""The lateral running command must be reconstructible from pre-step state."""

import json

import numpy as np
import pytest

from rosclaw_soccer.rsi import approach_lateral_tracking_evidence as evidence
from rosclaw_soccer.rsi.precontact_proprio_policy import FEATURE_NAMES, V300_HASH
from rosclaw_soccer.sim.contracts import hash_json


def test_approach_command_is_bounded_and_stops_at_contact(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evidence, "audit_vector_first_touch", lambda _: {"report_hash": "physical"})
    report = {
        "frames": 300,
        "environments": [{"first_contact_frame": 5}],
        "navigation_lateral_ball_gain": 0.8,
        "report_hash": "source",
    }
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    root = np.zeros((300, 1, 7))
    ball = np.zeros((300, 1, 3))
    ball[:, 0, 0] = 2.0
    ball[:, 0, 1] = 0.5
    command = np.zeros((300, 1))
    command[:6, 0] = 0.2
    np.savez_compressed(
        tmp_path / "body_trace.npz",
        root_pose_xyzw_m=root,
        ball_position_before_step_m=ball,
        navigation_lateral_speed_mps=command,
    )
    audited = evidence.audit_lateral_approach(tmp_path)
    assert audited["active_frames"] == 6
    assert audited["maximum_abs_command_mps"] == 0.2
    command[6, 0] = 0.2
    np.savez_compressed(
        tmp_path / "body_trace.npz",
        root_pose_xyzw_m=root,
        ball_position_before_step_m=ball,
        navigation_lateral_speed_mps=command,
    )
    with pytest.raises(ValueError, match="diverged"):
        evidence.audit_lateral_approach(tmp_path)


def test_negative_only_mode_protects_positive_initial_ball_side(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evidence, "audit_vector_first_touch", lambda _: {"report_hash": "physical"})
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "frames": 300,
                "environments": [{"first_contact_frame": None}],
                "navigation_lateral_ball_gain": 0.8,
                "navigation_lateral_negative_only": True,
                "report_hash": "source",
            }
        ),
        encoding="utf-8",
    )
    root = np.zeros((300, 1, 7))
    ball = np.zeros((300, 1, 3))
    ball[:, 0, 0] = 2.0
    ball[:, 0, 1] = 0.1
    np.savez_compressed(
        tmp_path / "body_trace.npz",
        root_pose_xyzw_m=root,
        ball_position_before_step_m=ball,
        navigation_lateral_speed_mps=np.zeros((300, 1)),
    )
    assert evidence.audit_lateral_approach(tmp_path)["active_frames"] == 0


def test_intermediate_gain_reconstructs_negative_side_feedback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evidence, "audit_vector_first_touch", lambda _: {"report_hash": "physical"})
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "frames": 300,
                "environments": [{"first_contact_frame": 5}],
                "navigation_lateral_ball_gain": 1.0,
                "navigation_lateral_negative_only": True,
                "report_hash": "source",
            }
        ),
        encoding="utf-8",
    )
    root = np.zeros((300, 1, 7))
    ball = np.zeros((300, 1, 3))
    ball[:, 0, 0] = 2.0
    ball[:, 0, 1] = -0.08
    command = np.zeros((300, 1))
    command[:6, 0] = -0.08
    np.savez_compressed(
        tmp_path / "body_trace.npz",
        root_pose_xyzw_m=root,
        ball_position_before_step_m=ball,
        navigation_lateral_speed_mps=command,
    )
    assert evidence.audit_lateral_approach(tmp_path)["active_frames"] == 6
    command[1, 0] = -0.16
    np.savez_compressed(
        tmp_path / "body_trace.npz",
        root_pose_xyzw_m=root,
        ball_position_before_step_m=ball,
        navigation_lateral_speed_mps=command,
    )
    with pytest.raises(ValueError, match="diverged"):
        evidence.audit_lateral_approach(tmp_path)


def test_sealed_online_rectangle_reconstructs_only_selected_command(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evidence, "audit_vector_first_touch", lambda _: {"report_hash": "physical"})
    report = {
        "frames": 300,
        "environments": [{"first_contact_frame": 5}],
        "navigation_lateral_ball_gain": 0.8,
        "navigation_lateral_negative_only": False,
        "navigation_rectangle_policy_hash": evidence.SEALED_POLICY_HASH,
        "navigation_rectangle_x_max_m": evidence.SEALED_X_MAX_M,
        "navigation_rectangle_y_min_m": evidence.SEALED_Y_MIN_M,
        "report_hash": "source",
    }
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    root = np.zeros((300, 1, 7))
    ball = np.zeros((300, 1, 3))
    ball[:, 0, 0] = 2.3
    ball[:, 0, 1] = -0.04
    command = np.zeros((300, 1))
    command[:6, 0] = -0.032
    np.savez_compressed(
        tmp_path / "body_trace.npz",
        root_pose_xyzw_m=root,
        ball_position_before_step_m=ball,
        navigation_lateral_speed_mps=command,
    )
    assert evidence.audit_lateral_approach(tmp_path)["active_frames"] == 6
    report["navigation_rectangle_x_max_m"] = 2.9
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="unsealed"):
        evidence.audit_lateral_approach(tmp_path)


def test_precontact_proprioceptive_switch_is_reconstructed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evidence, "audit_vector_first_touch", lambda _: {"report_hash": "physical"})
    policy = {
        "schema": "rsi_precontact_proprio_approach_policy_v1",
        "activation_ceiling": "SIM_ONLY",
        "training_report_hash": V300_HASH,
        "feature_names": list(FEATURE_NAMES),
        "decision_frame": 30,
        "threshold": 0.2,
        "aggressive_gain": 1.2,
        "fallback_gain": 0.8,
        "mean": [0.0] * 13,
        "scale": [1.0] * 13,
        "coefficients": [0.0] * 13,
        "intercept": 0.0,
        "promotion_authorized": False,
    }
    policy["policy_hash"] = hash_json(policy)
    report = {
        "frames": 300,
        "environments": [{"first_contact_frame": 40}],
        "navigation_lateral_ball_gain": 1.2,
        "navigation_lateral_negative_only": True,
        "navigation_proprio_risk_policy_hash": policy["policy_hash"],
        "navigation_proprio_risk_policy": policy,
        "navigation_proprio_risk_probability": [0.5],
        "navigation_proprio_risk_vetoed": [True],
        "report_hash": "source",
    }
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    root = np.zeros((300, 1, 7))
    ball = np.zeros((300, 1, 3))
    ball[:, 0, 0] = 2.0
    ball[:, 0, 1] = -0.08
    geometry = np.zeros((300, 1, 4, 3))
    geometry[:, 0, :, 2] = 0.2
    command = np.zeros((300, 1))
    command[:30, 0] = -0.096
    command[30:41, 0] = -0.064
    np.savez_compressed(
        tmp_path / "body_trace.npz",
        root_pose_xyzw_m=root,
        root_velocity_world=np.zeros((300, 1, 6)),
        ball_position_before_step_m=ball,
        ball_linear_velocity_before_step_m_s=np.zeros((300, 1, 3)),
        foot_geometry_position_before_step_m=geometry,
        navigation_lateral_speed_mps=command,
    )
    assert evidence.audit_lateral_approach(tmp_path)["active_frames"] == 41
    report["navigation_proprio_risk_vetoed"] = [False]
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="decision diverged"):
        evidence.audit_lateral_approach(tmp_path)
