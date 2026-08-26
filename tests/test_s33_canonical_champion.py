from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rosclaw_soccer.growth.canonical_champion import (
    SoccerChampionEvidenceLayout,
    reconstruct_s33_champion_registry,
)
from rosclaw_soccer.sim.contracts import hash_json


def _digest(label: str) -> str:
    return str(hash_json({"label": label}))


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    payload["report_hash"] = hash_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _exam(checkpoint: str, *, parent: str | None = None) -> dict[str, Any]:
    return {
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "checkpoint_hash": checkpoint,
        "exam_config_hash": _digest(f"config:{checkpoint}"),
        "physics_scene_hash": _digest("scene"),
        "seeds": [1, 2, 3],
        "paired_rollouts": True,
        "passed": True,
        "promotion_status": "PROMOTED_SIM_ONLY",
        "parent_checkpoint_hash": parent,
    }


def _decision(parent: str, candidate: str, suite: str, *, qualified: bool = True) -> dict[str, Any]:
    decision = {
        "replace_champion": False,
        "status": "RETAIN_PARENT_ARCHIVE_CANDIDATE",
        "reasons": ["save_rate_regression_exceeds_budget"],
        "metric_deltas": [["save_rate", -0.1]],
        "parent_artifact_hash": parent,
        "candidate_artifact_hash": candidate,
        "scenario_suite_hash": suite,
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "schema_version": "rosclaw_soccer.paired_champion_decision.v1",
    }
    return {
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "decision": decision,
        "decision_hash": hash_json(decision),
        "candidate": {
            "artifact_hash": candidate,
            "parent_artifact_hash": parent,
            "scenario_suite_hash": suite,
            "qualified": qualified,
        },
    }


def _fixture(root: Path) -> SoccerChampionEvidenceLayout:
    layout = SoccerChampionEvidenceLayout(
        s12_global_exam="s12.json",
        s13_decision="s13.json",
        s15_decision="s15.json",
        s23_decision="s23.json",
        s28_decision="s28.json",
        s31_specialist_exam="s31.json",
    )
    s12 = _digest("s12")
    s13 = _digest("s13")
    s15 = _digest("s15")
    _write_report(root / layout.s12_global_exam, _exam(s12))
    _write_report(root / layout.s13_decision, _decision(s12, s13, _digest("suite13")))
    _write_report(root / layout.s15_decision, _decision(s13, s15, _digest("suite15")))
    _write_report(
        root / layout.s23_decision,
        _decision(s15, _digest("s23"), _digest("suite23"), qualified=False),
    )
    _write_report(
        root / layout.s28_decision,
        _decision(s15, _digest("s28"), _digest("suite28")),
    )
    _write_report(root / layout.s31_specialist_exam, _exam(_digest("s31")))
    return layout


def test_s33_repair_keeps_s12_global_and_isolates_s31_specialist(tmp_path: Path) -> None:
    layout = _fixture(tmp_path)

    report = reconstruct_s33_champion_registry(tmp_path, layout=layout)

    heads = report["registry"]["active_heads"]
    assert heads["global"]["artifact_hash"] == _digest("s12")
    assert heads["goalkeeper.arm.elite"]["artifact_hash"] == _digest("s31")
    audits = {item["stage"]: item for item in report["lineage_audits"]}
    assert audits["S13"]["admitted_to_registry"]
    assert not audits["S15"]["parent_claim"]["valid"]
    assert audits["S15"]["parent_claim"]["reasons"] == ["parent_not_active_track_head"]
    assert not audits["S23"]["admitted_to_registry"]
    assert report["activation_ceiling"] == "SIM_ONLY"


def test_s33_repair_rejects_tampered_sealed_evidence(tmp_path: Path) -> None:
    layout = _fixture(tmp_path)
    path = tmp_path / layout.s13_decision
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["decision"]["status"] = "REPLACE_CHAMPION"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="report hash mismatch"):
        reconstruct_s33_champion_registry(tmp_path, layout=layout)
