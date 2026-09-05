from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.causal_transition_growth import CausalTransitionContext
from rosclaw_soccer.training.target_velocity_contact_discovery import (
    TargetVelocityContactProbe,
    run_target_velocity_contact_discovery,
)


def _context(case_id: str) -> CausalTransitionContext:
    return CausalTransitionContext(
        case_id,
        (5.10, -0.16, 0.0),
        -0.10,
        1.33,
        (1.19, -0.15),
        0.80,
        0.09,
    )


def test_target_velocity_probe_builds_three_axis_teacher() -> None:
    probe = TargetVelocityContactProbe(
        _context("s129.target.00"),
        (7.0, -3.0, 3.0),
        12,
        -0.12,
        -0.06,
        foot_yaw_offset_rad=-0.04,
    )
    teacher = probe.teacher_config()

    assert teacher.target_forward_speed_mps == 7.0
    assert teacher.target_lateral_speed_mps == -3.0
    assert teacher.target_vertical_speed_mps == 3.0
    assert teacher.maximum_foot_ball_distance_m == 0.50
    with pytest.raises(ValueError, match="SIM-only envelope"):
        TargetVelocityContactProbe(_context("s129.target.bad"), (7.0, 0.5, 3.0), 12, 0, 0)


def test_target_discovery_checks_failure_memory_before_assets(tmp_path: Path) -> None:
    report = {
        "status": "REJECTED_DUAL_CLOCK_CONTACT_RETENTION",
        "promotion_eligible": False,
        "sealed": True,
        "rows": [],
    }
    report["report_hash"] = hash_json(report)
    path = tmp_path / "failure.json"
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    payload = json.loads(path.read_text())
    payload["sealed"] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    targets = (
        (5.0, -3.0, -1.0),
        (7.0, 3.0, 3.0),
        (5.0, -1.0, 3.0),
        (7.0, 1.0, -1.0),
        (5.0, -2.0, -1.0),
        (7.0, 2.0, 3.0),
        (5.0, -3.0, 3.0),
        (7.0, 3.0, -1.0),
    )
    probes = tuple(
        TargetVelocityContactProbe(_context(f"s129.target.{index:02d}"), target, 0, 0, 0)
        for index, target in enumerate(targets)
    )

    with pytest.raises(ValueError, match="intact failure memory"):
        run_target_velocity_contact_discovery(
            asset_root=tmp_path / "assets",
            source_s95_dir=tmp_path / "source",
            rejected_exam_path=path,
            probes=probes,
            output_dir=tmp_path / "output",
        )
