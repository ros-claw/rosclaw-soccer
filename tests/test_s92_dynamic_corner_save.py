from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.training.dynamic_corner_save import (
    DynamicCornerPortfolioConfig,
    expanded_dynamic_corner_lanes,
    validate_dynamic_corner_evidence,
)


def test_expanded_corner_curriculum_keeps_four_independent_safety_contracts() -> None:
    lanes = expanded_dynamic_corner_lanes()

    assert tuple(lane.lane_id for lane in lanes) == (
        "left-outer",
        "left-inner",
        "right-inner",
        "right-outer",
    )
    assert lanes[0].takeoff_config.minimum_airborne_duration_sec == pytest.approx(0.14)
    assert lanes[0].takeoff_config.lunge_config.outward_punch_force_scale == pytest.approx(
        0.24
    )
    assert lanes[1].takeoff_config.lunge_config.lower_body_scale == pytest.approx(0.776)
    assert lanes[2].takeoff_config.minimum_airborne_duration_sec == pytest.approx(0.15)
    assert lanes[3].takeoff_config.lunge_config.waist_scale == pytest.approx(0.22)
    assert lanes[3].takeoff_config.lunge_config.outward_punch_force_scale == pytest.approx(
        0.30
    )
    with pytest.raises(ValueError, match="four unique lanes"):
        replace(DynamicCornerPortfolioConfig(), lanes=lanes[:3])


def test_dynamic_corner_evidence_is_trajectory_bound(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    cases: dict[str, object] = {}
    for index in range(4):
        trajectory = tmp_path / f"lane-{index}.npz"
        trajectory.write_bytes(f"trajectory-{index}".encode())
        cases[f"lane-{index}"] = {
            "passed": True,
            "trajectory_file": trajectory.name,
            "trajectory_hash": hash_bytes(trajectory.read_bytes()),
        }
    payload = {
        "schema_version": "rosclaw_soccer.dynamic_corner_evidence.v1",
        "passed": True,
        "promotion_status": "FROZEN_RESEARCH_DEMO",
        "claim": "STRICT_MULTI_CORNER_AIRBORNE_SAVE_PORTFOLIO",
        "portfolio_gates": {"all": True},
        "cases": cases,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "pixels_used_for_scoring": False,
        "request_hash": hash_bytes(request.read_bytes()),
    }
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_dynamic_corner_evidence(evidence)["passed"] is True
    (tmp_path / "lane-2.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="binding changed"):
        validate_dynamic_corner_evidence(evidence)
