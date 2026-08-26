from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training import high_corner_reachability_audit
from rosclaw_soccer.training.high_corner_reachability_audit import (
    HighCornerReachabilityAuditConfig,
    classify_high_corner_reachability,
    validate_high_corner_reachability_audit,
)


def _routes(*, save_route: str | None = None, safe: bool = False) -> dict[str, dict[str, object]]:
    names = (
        "bounded-parent",
        "learned-candidate",
        "full-drive-probe",
        "full-drive-lunge-probe",
    )
    return {
        name: {
            "first_save_rate": 0.10 if name == save_route else 0.0,
            "failed_rate": 0.0 if safe else 0.25,
            "maximum_root_angular_speed_rad_s": 3.0 if safe else 6.0,
            "finite_state": True,
            "strict_replay": True,
            "paired_outcome_consistent": True,
            "maximum_paired_replay_metric_delta": 0.0,
            "replay_metrics": {
                "first_save_rate": 0.10 if name == save_route else 0.0,
                "failed_rate": 0.0 if safe else 0.25,
                "maximum_root_angular_speed_rad_s": 3.0 if safe else 6.0,
                "finite_state": True,
            },
            "controller_config_hash": "sha256:" + hashlib.sha256(name.encode()).hexdigest(),
        }
        for name in names
    }


def test_high_corner_audit_routes_action_budget_and_topology_failures() -> None:
    assert (
        classify_high_corner_reachability(_routes())
        == "NEW_LATERAL_LOCOMOTION_DIVE_EXPERT_REQUIRED"
    )
    assert (
        classify_high_corner_reachability(_routes(save_route="full-drive-probe"))
        == "UNSAFE_ACTION_BUDGET_ONLY"
    )
    assert (
        classify_high_corner_reachability(_routes(save_route="learned-candidate", safe=True))
        == "CURRENT_CONTROLLER_FAMILY_REACHABLE"
    )


def test_high_corner_audit_rejects_incomplete_or_nonfinite_metrics() -> None:
    routes = _routes()
    routes.pop("bounded-parent")
    with pytest.raises(ValueError, match="route set"):
        classify_high_corner_reachability(routes)
    routes = _routes()
    routes["bounded-parent"]["failed_rate"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        classify_high_corner_reachability(routes)


def test_high_corner_audit_config_is_sim_only_and_requires_seed_diversity() -> None:
    with pytest.raises(ValueError, match="two distinct"):
        HighCornerReachabilityAuditConfig(seeds=(1, 1))
    with pytest.raises(ValueError, match="SIM_ONLY"):
        HighCornerReachabilityAuditConfig(hardware_authorized=True)


def test_high_corner_audit_manifest_is_content_bound(tmp_path: Path) -> None:
    source = tmp_path / "actor.pt"
    source.write_bytes(b"actor")
    config = asdict(HighCornerReachabilityAuditConfig())
    routes = _routes()
    payload = {
        "schema_version": "rosclaw_soccer.high_corner_reachability_audit.v1",
        "config": config,
        "config_hash": hash_json(config),
        "decision": "NEW_LATERAL_LOCOMOTION_DIVE_EXPERT_REQUIRED",
        "routes": routes,
        "source_files": {str(source): hash_bytes(source.read_bytes())},
        "implementation_hash": hash_bytes(
            Path(high_corner_reachability_audit.__file__).read_bytes()
        ),
        "physics_authority": "DIAGNOSTIC_ONLY",
        "fresh_physics_performed": True,
        "source_actor_frozen": True,
        "candidate_generated": False,
        "promotion_eligible": False,
        "strict_replay": True,
        "paired_outcome_consistent": True,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
    }
    payload["report_hash"] = hash_json(payload)
    manifest = tmp_path / "audit.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_high_corner_reachability_audit(manifest)["decision"].startswith("NEW_")
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source binding"):
        validate_high_corner_reachability_audit(manifest)
