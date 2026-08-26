from __future__ import annotations

import json

import pytest

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.recovery_baseline_evidence import (
    aggregate_recovery_physics_reports,
)

_HASH = "sha256:" + "a" * 64


def _report(device: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "rosclaw_soccer.mjlab_getup_physics_probe.v3",
        "contract_hash": _HASH,
        "checkpoint_hash": _HASH,
        "source_hash": _HASH,
        "body_hash": _HASH,
        "physics_scene_hash": _HASH,
        "handoff_config_hash": _HASH,
        "physics_device": device,
        "environment_count": 2,
        "final_stable_recovery_count": 2,
        "final_continuous_stable_sec": [4.0, 5.0],
        "initial_perturbation_scale": 0.1,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    payload["report_hash"] = hash_json(payload)
    return payload


def test_recovery_physics_aggregate_binds_four_devices_and_stays_nonpromotable(
    tmp_path,
) -> None:
    paths = []
    for index in range(4):
        path = tmp_path / f"gpu{index}.json"
        path.write_text(json.dumps(_report(f"cuda:{index}")), encoding="utf-8")
        paths.append(path)
    aggregate = aggregate_recovery_physics_reports(
        report_paths=paths,
        output_path=tmp_path / "aggregate.json",
    )
    assert aggregate["environment_count"] == 8
    assert aggregate["final_stable_recovery_rate"] == 1.0
    assert not aggregate["promotion_eligible"]
    assert aggregate["claim_boundary"].endswith("NOT_TRUE_POST_SAVE_RECOVERY")


def test_recovery_physics_aggregate_rejects_duplicate_device(tmp_path) -> None:
    paths = []
    for index in range(4):
        path = tmp_path / f"gpu{index}.json"
        path.write_text(json.dumps(_report("cuda:0")), encoding="utf-8")
        paths.append(path)
    with pytest.raises(ValueError, match="expected device"):
        aggregate_recovery_physics_reports(
            report_paths=paths,
            output_path=tmp_path / "aggregate.json",
        )
