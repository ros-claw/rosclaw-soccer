"""The lateral running command must be reconstructible from pre-step state."""

import json

import numpy as np
import pytest

from rosclaw_soccer.rsi import approach_lateral_tracking_evidence as evidence


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
