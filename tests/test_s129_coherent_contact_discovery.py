from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.causal_transition_growth import CausalTransitionContext
from rosclaw_soccer.training.coherent_contact_discovery import (
    CoherentContactProbe,
    run_coherent_contact_discovery,
)


def _context(case_id: str = "s129.test.00") -> CausalTransitionContext:
    return CausalTransitionContext(
        case_id=case_id,
        passer_origin_m=(5.10, -0.16, 0.0),
        receiver_lane_m=-0.10,
        reception_target_x_m=1.33,
        passer_ball_local_xy_m=(1.19, -0.15),
        predecessor_swing_speed_scale=0.80,
        ball_ground_friction=0.09,
    )


def test_coherent_probe_is_content_bound_and_bounded() -> None:
    first = CoherentContactProbe(_context(), 12, -0.12, -0.06, 248, -0.04, 0.01)
    second = CoherentContactProbe(_context(), 12, -0.12, -0.06, 248, 0.04, 0.01)

    assert first.probe_hash == hash_json(asdict(first))
    assert first.probe_hash != second.probe_hash
    with pytest.raises(ValueError, match="SIM-only envelope"):
        CoherentContactProbe(_context(), 24, -0.12, -0.06)
    with pytest.raises(ValueError, match="SIM-only envelope"):
        CoherentContactProbe(_context(), 12, 0.30, -0.06)


def test_discovery_rejects_tampered_failure_memory_before_physics(tmp_path: Path) -> None:
    report = {
        "status": "REJECTED_DUAL_CLOCK_CONTACT_RETENTION",
        "promotion_eligible": False,
        "sealed": True,
        "rows": [],
    }
    report["report_hash"] = hash_json(report)
    path = tmp_path / "exam-report.json"
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    report["sealed"] = False
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="intact rejected sealed exam"):
        run_coherent_contact_discovery(
            asset_root=tmp_path / "missing-assets",
            source_s95_dir=tmp_path / "missing-source",
            rejected_exam_path=path,
            contact_actor_path=tmp_path / "missing-actor.json",
            probes=tuple(
                CoherentContactProbe(_context(f"s129.test.{index:02d}"), 0, 0.04, -0.06)
                for index in range(6)
            ),
            output_dir=tmp_path / "output",
        )
